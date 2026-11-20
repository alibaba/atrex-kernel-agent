"""Atrex-Bench compatibility adapter; all Shape and Model conventions live here."""
from __future__ import annotations
import json
from aka.contracts.task import Assessment, EvaluationPlan, Metric
from ..content import digest, relative, tree


class Adapter:
    protocol_version = 1
    capabilities = frozenset({"acceptance", "parallel_candidates"})

    def validate(self, contract, config):
        from aka.task.problem.layout import validate_private_shapes
        for name in ("reference.py", "input.py", "shapes.json"):
            if not (contract / name).is_file():
                raise ValueError(f"Atrex-Bench contract requires {name}")
        validate_private_shapes(contract / "shapes.json")
        relative(config.get("entry", "kernel.py"))

    def instructions(self, contract, config):
        return ("Preserve the reference constructor and forward signature. Deliver a callable Model, "
                "or a native solution.json whose entry point generates that interface. "
                "The frozen Atrex-Bench input generator, cases and runner configuration define correctness. "
                + config.get("instructions", ""))

    def prepare(self, context):
        if context.target["backend"] != "agate":
            raise ValueError("The Atrex-Bench adapter requires Agate")
        reference = {"operator": context.config.get("name", "module"),
                     "reference_py": (context.contract / "reference.py").read_text(),
                     "input_py": (context.contract / "input.py").read_text(),
                     "shapes": json.loads((context.contract / "shapes.json").read_text())}
        for name in ("metadata", "roofline"):
            path = context.contract / (name + ".json")
            if path.exists():
                reference[name] = json.loads(path.read_text())
        request = {"name": reference["operator"], "reference": reference,
                   "mode": "full", "lock_clocks": True,
                   "options": {"num_correctness_cases": 5, "bench_iters": 100, "shape_shards": 1,
                               **context.config.get("options", {})},
                   "runner_overrides": context.config.get("runner_overrides", {})}
        for key in ("requirements", "atrex_bench_version"):
            if key in context.config:
                request[key] = context.config[key]
        from aka.task.source.formats.cuda_manifest import execution_source, collect
        from aka.light.evaluation.source_transport import ARCHIVES, ATTACHMENTS
        from aka.light.execution.package import encode_files
        if (context.candidate / "solution.json").is_file():
            # Reuse the existing parser, source declaration and build/dependency
            # transport. This is format compatibility, not a different loop.
            packaged = execution_source(collect(context.candidate), "archives/candidate.tar.gz")
            request["candidate"] = packaged.candidate
            request[ARCHIVES] = packaged.archives
            resources = {}
            for dependency in packaged.candidate.get("dependencies", []):
                if dependency.get("kind") in {"artifact", "weights"}:
                    path = dependency["path"]
                    data = (context.candidate / relative(path)).read_bytes()
                    import hashlib
                    if hashlib.sha256(data).hexdigest() != dependency["sha256"]:
                        raise ValueError("Manifest resource digest differs: " + path)
                    resources[path] = data
            if resources:
                request[ATTACHMENTS] = encode_files(resources)
        else:
            files = tree(context.candidate)
            entry = context.config.get("entry", "kernel.py")
            if set(files) == {entry}:
                request["candidate"] = files[entry].decode()
            else:
                encoded = encode_files(files)
                source = {k: v for k, v in encoded.items() if isinstance(v, str)}
                resources = {k: v for k, v in encoded.items() if not isinstance(v, str)}
                import hashlib
                dependencies = [{"id": "resource-" + hashlib.sha256(k.encode()).hexdigest()[:12],
                    "kind": "artifact", "path": k, "sha256": hashlib.sha256(files[k]).hexdigest()} for k in resources]
                request["candidate"] = {"schema_version": 1, "name": reference["operator"], "version": "1",
                    "entry_point": entry + "::Model", "source": {"type": "inline", "files": source},
                    "dependencies": dependencies}
                if resources:
                    request[ATTACHMENTS] = resources
        return EvaluationPlan("eval", request)

    def interpret(self, context, receipt):
        if receipt.status != "completed":
            return Assessment("unknown", False, reason="Atrex-Bench execution incomplete")
        from aka.light.evaluation.result_conversion import optimizer_result_from_eval
        shapes = list(json.loads((context.contract / "shapes.json").read_text()))
        payload = dict(receipt.payload)
        raw = payload.get("result", payload)
        metadata_path = context.contract / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        result = optimizer_result_from_eval(raw, shapes, metadata)
        values = result.get("latency_us_by_shape", {})
        complete = set(values) == set(shapes) and bool(values)
        metrics = [Metric("latency", value, "us", key) for key, value in values.items()]
        if complete:
            metrics.append(Metric("latency_geomean", result["latency_us_geomean"], "us"))
        environment = raw.get("environment") or payload.get("environment")
        # Unknown runtime identity is never made comparable by the hardware alias alone.
        env_id = digest(environment) if environment else ""
        return Assessment("pass" if result.get("all_pass") is True else "fail", complete,
                          tuple(metrics), env_id, result.get("error") or "", result)

    def delivery(self, contract, config):
        return {"entry": config.get("entry", "kernel.py"), "interface": "Model", **config.get("delivery", {})}
