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

"""Seed a kernel-opt workspace directly from a SOL-ExecBench op directory.

The op dir is the *ground truth*: `definition.json`, `reference.py`,
`workload.jsonl` are copied verbatim and never edited by the optimization loop.
The V0 baseline is a correct, immediately-submittable SOL solution: a
destination-passing-style `run()` in `kernel.py` that wraps the reference logic.
Every workload's shapes, dtypes, input generation and per-workload tolerances
come straight from the SOL files, so a PASS in this workspace == a submittable
solution (validated by the same `sol-execbench` evaluator via `test_kernel.py`).

Produces (under `kernel_opt_<name>/`):
    definition.json, reference.py, workload.jsonl   # ground truth (verbatim)
    config.json         # pinned seed/warmup/iterations/benchmark_reference
    kernel.py           # V0: self-contained DPS wrapper around the reference
    solution.json       # SOL solution (sources reference kernel.py by path)
    test_kernel.py      # the immutable SOL harness (copied from reference/)
    profile_driver.py   # the immutable external profiling entry (copied from reference/)
    CLAUDE.md           # agent constraints (copied from reference/)
    README.md, .gitignore
    memory/v0.json      # baseline metrics (written by test_kernel.py)  [unless --no-bench]

Usage:
    python reference/sol_seed.py --op-dir <sol-op-dir> --name <name> \
        [--framework pytorch] [--platform B200] [--gpu-wiki <path>] [--no-bench]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/reference
GROUND_TRUTH = ("definition.json", "reference.py", "workload.jsonl")

_repo_root = SCRIPT_DIR.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from orchestrator._bootstrap import task_modules
task_modules()
from aka.legacy.task.source.sol.render import _build_kernel, _render_kernel, _solution_json, _readme
from aka.legacy.application.source import source_provider
from aka.legacy.application.workspace import candidate_workspace
from aka.legacy.task.candidate_workspace.initial import GITIGNORE


# Profiling is driven by the external `profile_driver.py` seeded next to kernel.py.
# It is deliberately NOT injected into kernel.py: ncu/rocprofv3 run `python <file>`, and an
# in-kernel `__main__` block is silently lost the first time a session rewrites run().


def main(argv: list[str] | None = None, *, dependencies=None, problem=None) -> int:
    ap = argparse.ArgumentParser(description="Seed a kernel-opt workspace from a SOL-ExecBench op dir.")
    ap.add_argument("--op-dir", required=True, help="SOL op dir (definition.json + reference.py + workload.jsonl).")
    ap.add_argument("--name", required=True, help="Workspace name -> kernel_opt_<name>/.")
    ap.add_argument("--workspace", default="", help="Explicit workspace path (default: ./kernel_opt_<name>).")
    ap.add_argument("--framework", default="pytorch", help="Target framework the loop should migrate to.")
    ap.add_argument("--platform", default="LOCAL", help="Target hardware token, e.g. B200 (default: LOCAL).")
    ap.add_argument("--gpu-wiki", default="", help="Absolute path to gpu-wiki (recorded in README).")
    ap.add_argument("--no-bench", action="store_true", help="Skip the V0 test_kernel.py bench (no memory/v0.json).")
    ap.add_argument("--skip-bench-if-v0-exists", action="store_true",
                    help="If memory/v0.json already exists (e.g. produced by a synthetic seed), "
                         "skip the V0 bench step entirely. Used to unblock the optimizer "
                         "for ops whose torch reference is too slow to bench within a reasonable budget.")
    args = ap.parse_args(argv)

    ws = Path(args.workspace).resolve() if args.workspace else (Path.cwd() / f"kernel_opt_{args.name}")
    if problem is None:
        from aka.legacy.task.problem.legacy import LegacyProblemProvider
        provider = dependencies["problem"] if dependencies is not None else LegacyProblemProvider()
        # Keep the standalone CLI's missing-ground-truth diagnostics on first load.
        if not (ws.parent / ".aka-problems" / args.name).exists():
            for name in GROUND_TRUTH:
                if not (Path(args.op_dir).resolve() / name).is_file():
                    raise SystemExit(f"--op-dir is not a SOL op dir (missing {name}): {Path(args.op_dir).resolve()}")
        problem = provider.load(Path(args.op_dir), snapshot_directory=ws.parent / ".aka-problems" / args.name)
    op = problem.evaluation.directory
    for f in GROUND_TRUTH:
        if not (op / f).is_file():
            raise SystemExit(f"--op-dir is not a SOL op dir (missing {f}): {op}")
    defn = json.loads((op / "definition.json").read_text(encoding="utf-8"))

    if args.no_bench and (ws / ".git").exists():
        history = subprocess.run(
            ["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
            cwd=str(ws), capture_output=True, text=True,
        )
        if history.returncode == 0 and history.stdout.strip():
            source_commit = history.stdout.split()[0]
            # A committed V0 without memory is a measurement retry, not a new
            # seed. Validate provenance before touching anything, including the
            # index; never reset an edited or already optimized kernel to V0.
            if subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=str(ws),
            ).returncode != 0:
                raise SystemExit("refusing to reseed V0 with staged changes")
            for name in (
                *GROUND_TRUTH, "config.json", "kernel.py", "solution.json",
                "test_kernel.py", "profile_driver.py",
            ):
                original = subprocess.run(
                    ["git", "show", f"{source_commit}:{name}"],
                    cwd=str(ws), capture_output=True, check=True,
                ).stdout
                committed = subprocess.run(
                    ["git", "show", f"HEAD:{name}"],
                    cwd=str(ws), capture_output=True, check=True,
                ).stdout
                if (
                    committed != original or not (ws / name).is_file()
                    or (ws / name).read_bytes() != original
                ):
                    raise SystemExit(f"refusing to reseed changed V0 source: {name}")
            for name in GROUND_TRUTH:
                if (ws / name).read_bytes() != (op / name).read_bytes():
                    raise SystemExit(f"V0 operator input changed: {name}")
            if (ws / "kernel.py").read_text(encoding="utf-8") != _render_kernel(
                defn, (op / "reference.py").read_text(encoding="utf-8")
            ):
                raise SystemExit("existing V0 is not the SOL reference wrapper")
            print(f"[sol_seed] reusing V0 source {source_commit}: {ws}")
            return 0
    problem_files = problem.agent.files
    from contextlib import nullcontext
    from aka.contracts.workspace import SourceRequest
    request = SourceRequest("sol", operator_dir=str(op), name=args.name, framework=args.framework,
                            platform=args.platform, gpu_wiki=args.gpu_wiki, initial=problem.initial_source)
    if dependencies is not None:
        content = dependencies["source"].prepare(request)
    else:
        with source_provider("sol", operator_dir=str(op), name=args.name, framework=args.framework,
                         platform=args.platform, gpu_wiki=args.gpu_wiki, definition=defn) as source:
            content = source.prepare(request)
    with (nullcontext(dependencies["workspace"]) if dependencies is not None else candidate_workspace()) as workspace:
        workspace.prepare_sol(ws, content, problem_files=problem_files, reference_dir=SCRIPT_DIR)

    # 5) V0 baseline metrics (real evaluator)
    pre_existing_v0 = (ws / "memory" / "v0.json").exists() and args.skip_bench_if_v0_exists
    if pre_existing_v0:
        print("[sol_seed] skipping V0 bench — memory/v0.json already provided "
              "(synthetic seed); ground-truth files refreshed in-place.", file=sys.stderr)
    elif not args.no_bench:
        r = subprocess.run([sys.executable, str(ws / "test_kernel.py"), "--version", "v0"], cwd=str(ws))
        if r.returncode != 0:
            print("[sol_seed] WARNING: V0 baseline did not pass all workloads — check solution.json / reference.",
                  file=sys.stderr)

    with (nullcontext(dependencies["workspace"]) if dependencies is not None else candidate_workspace()) as workspace:
        workspace.commit_sol(ws)

    print(f"[sol_seed] workspace ready: {ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
