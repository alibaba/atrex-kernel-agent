from __future__ import annotations

import unittest

import torch

from orchestrator import optimize


def _tensor_signature(shape: list[int]) -> list[object]:
    stride = [1]
    for size in reversed(shape[1:]):
        stride.insert(0, stride[0] * size)
    return ["tensor", shape, stride, "torch.float32", "torch.strided", False]


class AggregateDispatchTest(unittest.TestCase):
    def test_embeds_bucket_sources_in_one_kernel_without_local_imports(self) -> None:
        sources = {
            "short": """
from __future__ import annotations
import torch

BIAS = 1.0

def apply_bias(value):
    return value + BIAS

class Model(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()

    def forward(self, x):
        return apply_bias(x)
""",
            "long": """
from __future__ import annotations
import torch

BIAS = 10.0

def apply_bias(value):
    return value + BIAS

class Model(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()

    def forward(self, x):
        return apply_bias(x)
""",
        }
        records = {
            name: {"embedded": True, "kernel_blob": name}
            for name in sources
        }
        init = ["invocation", [], [["offset", ["int", 0]]]]
        signatures = [
            {
                "index": 0,
                "init": init,
                "call": ["invocation", [], [["x", _tensor_signature([2])]]],
            },
            {
                "index": 1,
                "init": init,
                "call": ["invocation", [], [["x", _tensor_signature([3])]]],
            },
        ]

        generated = optimize.build_deterministic_dispatcher(
            kind="shapes",
            signature_records=signatures,
            bucket_by_index={0: "short", 1: "long"},
            module_records=records,
            module_sources=sources,
        )

        self.assertNotIn("aggregate_kernels", generated)
        self.assertNotIn("_load_bucket_", generated)
        self.assertIn("# BEGIN embedded bucket: short", generated)
        self.assertIn("# BEGIN embedded bucket: long", generated)
        namespace: dict[str, object] = {}
        exec(compile(generated, "kernel.py", "exec"), namespace)
        model = namespace["Model"](offset=0)
        torch.testing.assert_close(model(x=torch.ones(2)), torch.full((2,), 2.0))
        torch.testing.assert_close(model(x=torch.ones(3)), torch.full((3,), 11.0))


if __name__ == "__main__":
    unittest.main()
