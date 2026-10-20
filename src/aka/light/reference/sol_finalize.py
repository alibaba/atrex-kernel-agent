#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL-ExecBench output backend: package the workspace's best kernel into a
self-contained, directly-submittable SOL `solution.json`.

Normally you don't need to call this: `test_kernel.py` already writes
`submission.json` on every PASS, so each committed version has a ready-to-submit
artifact that was validated by the same run that produced its metrics. This
script is a stand-alone re-packager (e.g. for the Stop hook / a manual export).

What it does:
  1. Read `<workspace>/solution.json` and inline each `sources[].content` from
     the on-disk file (the working tree = git HEAD = best kernel), producing a
     self-contained submission that no longer depends on the workspace layout.
  2. Write it to `--out` (default `<workspace>/submission.json`).
  3. Packaging only by default. Re-validation (`--validate`) is opt-in and
     usually redundant — the committing iteration already validated this exact
     kernel via `test_kernel.py`.

Usage (cwd anywhere):
    python reference/sol_finalize.py --workspace kernel_opt_<name>            # package only
    python reference/sol_finalize.py --workspace kernel_opt_<name> --validate  # + re-run evaluator
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def _sol_execbench_cmd() -> list[str]:
    exe = shutil.which("sol-execbench")
    if exe:
        return [exe]
    if shutil.which("uv"):
        return ["uv", "run", "sol-execbench"]
    raise SystemExit("sol-execbench CLI not found on PATH.")


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def build_submission(workspace: Path) -> dict:
    """Self-contained solution.json with every source's content inlined from disk."""
    sol = json.loads((workspace / "solution.json").read_text(encoding="utf-8"))
    for src in sol.get("sources", []):
        p = workspace / src["path"]
        if not p.is_file():
            raise SystemExit(f"[sol_finalize] source not found on disk: {p}")
        src["content"] = p.read_text(encoding="utf-8")
    if not sol.get("sources"):
        raise SystemExit("[sol_finalize] solution.json has no sources")
    return sol


def validate(workspace: Path, out: Path) -> bool:
    cmd = _sol_execbench_cmd() + [str(workspace), "--solution", str(out)]
    if (workspace / "config.json").exists():
        cmd += ["--config", str(workspace / "config.json")]
    traces = workspace / ".finalize_traces.jsonl"
    cmd += ["-o", str(traces)]
    print(f"[sol_finalize] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)
    if not traces.exists():
        print("[sol_finalize] evaluator produced no traces", file=sys.stderr)
        return False

    rows = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
    lat_us, fails = [], []
    for t in rows:
        ev = t.get("evaluation") or {}
        if ev.get("status") == "PASSED":
            lat_us.append(float((ev.get("performance") or {}).get("latency_ms") or 0.0) * 1000.0)
        else:
            fails.append(((t.get("workload") or {}).get("uuid") or "?")[:8] + "=" + str(ev.get("status")))
    all_pass = bool(rows) and not fails
    print(f"[sol_finalize] {len(rows) - len(fails)}/{len(rows)} workloads passed | "
          f"geomean latency = {_geomean(lat_us):.1f} us")
    if fails:
        print(f"[sol_finalize] FAILURES: {'; '.join(fails)}", file=sys.stderr)
    return all_pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Package + validate a SOL-ExecBench submission.")
    ap.add_argument("--workspace", default=".", help="Workspace dir (default: cwd).")
    ap.add_argument("--out", default="", help="Submission path (default: <workspace>/submission.json).")
    ap.add_argument("--validate", action="store_true",
                    help="Also re-run the evaluator over all workloads (redundant if this version already passed test_kernel.py).")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not (workspace / "solution.json").exists():
        raise SystemExit(f"[sol_finalize] no solution.json in {workspace}")
    out = Path(args.out).resolve() if args.out else (workspace / "submission.json")

    out.write_text(json.dumps(build_submission(workspace), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sol_finalize] wrote self-contained submission -> {out}")

    if not args.validate:
        print("[sol_finalize] packaged (correctness already verified per-iteration by test_kernel.py; pass --validate to re-check)")
        return 0
    ok = validate(workspace, out)
    print(f"[sol_finalize] {'SUBMITTABLE (all workloads pass)' if ok else 'NOT submittable — fix kernel.py'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
