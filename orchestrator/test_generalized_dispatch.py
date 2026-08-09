from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from orchestrator.dispatch_codegen import build_deterministic_dispatcher
from orchestrator.dispatch_signatures import (
    _dispatch_schema_and_features,
    _dispatch_tree_bucket,
    build_generalized_dispatch_plan,
    validate_dispatch_bucket_compatibility,
)
from orchestrator.workload_buckets import WorkloadBucket


def _invocation(*values: list[object]) -> list[object]:
    return ["invocation", list(values), []]


def _tensor(shape: tuple[int, ...], stride: tuple[int, ...]) -> list[object]:
    return [
        "tensor",
        list(shape),
        list(stride),
        "torch.float16",
        "torch.strided",
        False,
    ]


def _gemm_record(index: int, m: int) -> dict:
    return {
        "index": index,
        "id": str(index),
        "init": _invocation(),
        "call": _invocation(
            _tensor((m, 2048), (2048, 1)),
            _tensor((128, 2048), (2048, 1)),
            _tensor((m, 128), (128, 1)),
        ),
    }


class GeneralizedDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        values = (1, 6, 8, 16, 34, 64, 93, 952, 8828, 16294)
        self.records = [_gemm_record(index, value) for index, value in enumerate(values)]
        self.buckets = [
            WorkloadBucket("tiny", (0, 1, 2)),
            WorkloadBucket("small", (3, 4, 5)),
            WorkloadBucket("mid", (6, 7)),
            WorkloadBucket("large", (8, 9)),
        ]
        self.owner = {
            index: bucket.name
            for bucket in self.buckets
            for index in bucket.workload_indices
        }

    def _select(self, m: int) -> str:
        plan, _evidence = build_generalized_dispatch_plan(self.records, self.owner)
        record = _gemm_record(-1, m)
        schema, features, _descriptors = _dispatch_schema_and_features(
            record["init"], record["call"]
        )
        return str(_dispatch_tree_bucket(plan[schema], features))

    def test_unseen_neighbor_shapes_route_without_exact_signature(self) -> None:
        validate_dispatch_bucket_compatibility(self.buckets, self.records)
        _plan, evidence = build_generalized_dispatch_plan(self.records, self.owner)

        self.assertEqual(set(evidence), {"tiny", "small", "mid", "large"})
        self.assertEqual(self._select(7), "tiny")
        self.assertEqual(self._select(35), "small")
        self.assertEqual(self._select(953), "mid")
        self.assertEqual(self._select(16295), "large")

    def test_exact_shape_island_is_rejected(self) -> None:
        records = [_gemm_record(index, value) for index, value in enumerate((1, 2, 3))]
        buckets = [
            WorkloadBucket("outer", (0, 2)),
            WorkloadBucket("exact_middle", (1,)),
        ]
        with self.assertRaisesRegex(ValueError, "no unseen one-step"):
            validate_dispatch_bucket_compatibility(buckets, records)

    def test_generated_dispatcher_contains_range_tree_not_signature_table(self) -> None:
        names = {bucket.name for bucket in self.buckets}
        source = build_deterministic_dispatcher(
            kind="sol",
            signature_records=self.records,
            bucket_by_index=self.owner,
            module_records={name: {"kernel_blob": name} for name in names},
            module_sources={
                name: f"def run(*args, **kwargs):\n    return {name!r}\n"
                for name in names
            },
        )

        compile(source, "<generated-dispatcher>", "exec")
        self.assertIn("_SCHEMA_TO_DISPATCH_TREE", source)
        self.assertNotIn("_SIGNATURE_TO_BUCKET", source)

        class FakeDType:
            pass

        class FakeTensor:
            def __init__(self, shape: tuple[int, ...], stride: tuple[int, ...]):
                self.shape = shape
                self._stride = stride
                self.dtype = "torch.float16"
                self.layout = "torch.strided"
                self.requires_grad = False

            def stride(self) -> tuple[int, ...]:
                return self._stride

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.dtype = FakeDType
        namespace: dict[str, object] = {}
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            exec(source, namespace)

        run = namespace["run"]
        self.assertTrue(callable(run))
        result = run(
            FakeTensor((953, 2048), (2048, 1)),
            FakeTensor((128, 2048), (2048, 1)),
            FakeTensor((953, 128), (128, 1)),
        )
        self.assertEqual(result, "mid")


if __name__ == "__main__":
    unittest.main()
