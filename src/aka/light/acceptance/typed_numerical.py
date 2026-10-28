"""Compiled candidate probes use Agate Eval so Agate owns compilation and dependencies."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

from aka.legacy.evaluation.drivers import numerical_inputs
from aka.light.workspace.files import publish, read_input
from aka.light.workspace.sources import install, snapshot


def input_source(original: bytes, case: dict, seed: int) -> bytes:
    # The generator is trusted Supervisor code. A fixed seed per request avoids
    # depending on the evaluator's input-call count (compile/warmup/correctness).
    algorithms = Path(numerical_inputs.__file__).read_text().replace("from __future__ import annotations\n", "")
    return original + ("\n" + algorithms + "\n__aka_original_inputs = _make_inputs\n"
        "def _make_inputs(**kwargs):\n"
        f"    return numerical_inputs(__aka_original_inputs(**kwargs), {case!r}, {seed!r}, 0)\n").encode()


def run(campaign, workspace, suite, shapes, schedule, directory, *, cancel, max_rel_l2):
    if suite["world_size"] != 1:
        raise ValueError("Typed supplemental probes require one GPU")
    rows = []
    source = snapshot(workspace)
    for plan in schedule:
        if plan.get("status") == "unsupported":
            rows.append(plan)
            continue
        case = next(c for c in suite["cases"] if c["id"] == plan["case_id"])
        for shape_id in plan["shape_ids"]:
            for seed in plan["seeds"]:
                cancel.check()
                with tempfile.TemporaryDirectory(prefix="native-probe-", dir=directory) as temporary:
                    probe = Path(temporary)
                    install(probe, source)
                    publish(probe, "reference.py", read_input(workspace, "reference.py"))
                    publish(probe, "input.py", input_source(read_input(workspace, "input.py"), case, seed))
                    publish(probe, "shapes.json", json.dumps({shape_id: shapes[shape_id]}).encode())
                    response = campaign.measure_for_acceptance(probe, [
                        "--kind", "run", "--no-sync", "--", "python3", "test_kernel.py",
                        "--correctness-only", "--multi-seed", "0",
                        "--correctness-max-rel-l2", str(max_rel_l2),
                    ], execution_timeout=600, parent=cancel.parent, reference_snapshot=probe)
                result, record_id = None, None
                for line in response["stdout"].splitlines():
                    if line.startswith("[test_kernel] RESULT_JSON="):
                        result = json.loads(line.split("=", 1)[1])
                    elif line.startswith("[sandbox] RECORD_JSON="):
                        record_id = json.loads(line.split("=", 1)[1]).get("gateway_record_id")
                diagnostics = result.get("diagnostics", []) if isinstance(result, dict) else []
                failed = [d for d in diagnostics if d.get("stage") == "correctness"
                          and d.get("check") in {"outputs", "mutated_inputs", "unexpected_mutations"}
                          and any(type(d.get(k)) in {int, float} and math.isfinite(d[k])
                                  for k in ("relative_l2", "max_elementwise_abs_diff", "max_elementwise_rel_diff"))]
                # Compile, input generation and transport failures are not a
                # numerical counterexample. Retain them for planner/operator repair.
                status = ("passed" if response["exit_code"] == 0 and result and result.get("all_pass") is True
                          else "needs_repair" if failed and not any(d.get("stage") != "correctness" for d in diagnostics)
                          else "needs_validation")
                rows.append({"case_id": plan["case_id"], "shape_id": shape_id, "seed": seed,
                             "selection_digest": plan["selection_digest"], "gateway_record_id": record_id,
                             "status": status, "diagnostics": diagnostics,
                             "diagnosis": response.get("stderr", "")[-8192:] if result is None else ""})
    status = ("needs_validation" if any(r["status"] == "needs_validation" for r in rows)
              else "needs_repair" if any(r["status"] == "needs_repair" for r in rows)
              else "advisory" if any(r["status"] == "unsupported" for r in rows) else "passed")
    return {"status": status, "probes": rows}
