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
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # <repo>/reference
GROUND_TRUTH = ("definition.json", "reference.py", "workload.jsonl")


def _build_kernel(defn: dict) -> str:
    """V0 kernel: inline the reference (rename run->_ref) + a DPS wrapper.

    Staging copies only solution.sources, so kernel.py must be self-contained
    (it cannot `import reference`). Args are definition.inputs then
    definition.outputs (DPS order); the wrapper writes outputs in place.
    """
    inputs = list(defn["inputs"].keys())
    outputs = list(defn["outputs"].keys())
    params = ", ".join(inputs + outputs)
    call = ", ".join(inputs)
    if len(outputs) == 1:
        assign = f"    {outputs[0]}[:] = _out\n"
    else:
        assign = "    if not isinstance(_out, (tuple, list)):\n        _out = (_out,)\n"
        for i, o in enumerate(outputs):
            assign += f"    {o}[:] = _out[{i}]\n"
    header = (
        "# V0 baseline: self-contained destination-passing-style (DPS) wrapper around the\n"
        "# verbatim SOL reference logic. Correct + directly submittable. The optimization\n"
        "# loop rewrites the body of run() toward the target framework; keep it DPS and keep\n"
        "# the argument names/order = definition.inputs then definition.outputs.\n"
        "{REF}\n\n"
        f"def run({params}):\n"
        f"    _out = _ref({call})\n"
        f"{assign}\n"
        + _PROFILE_MAIN
    )
    return header


# Standalone profiling entry: makes `python kernel.py` launch the real kernel under a
# profiler (ncu / rocprofv3, which run `python <kernel_file>`). NOT used by the SOL
# evaluator — it imports run() and never executes this block. Keep it when editing run().
_PROFILE_MAIN = '''
if __name__ == "__main__":
    # Profiling harness (profiler-only): build real inputs via SOL's own gen_inputs and
    # launch run() in a loop so ncu/rocprofv3 capture the kernel. Select the workload with
    # PROFILE_WORKLOAD_IDX (default 0) and iteration count with PROFILE_ITERS (default 10).
    import json as _json, os as _os
    from pathlib import Path as _Path
    import torch as _torch
    from sol_execbench.core.data import Definition as _Def, Workload as _Wl
    from sol_execbench.core.bench.io import gen_inputs as _gen, allocate_outputs as _alloc

    _here = _Path(__file__).resolve().parent
    _defn = _Def(**_json.loads((_here / "definition.json").read_text()))
    _wls = [_Wl(**_json.loads(_l)) for _l in (_here / "workload.jsonl").read_text().splitlines() if _l.strip()]
    _wl = _wls[int(_os.environ.get("PROFILE_WORKLOAD_IDX", "0"))]
    _dev = "cuda:0"
    _inputs = _gen(_defn, _wl, device=_dev)
    _outs = _alloc(_defn, _defn.get_resolved_axes_values(_wl.axes), _dev)
    for _ in range(3):
        run(*_inputs, *_outs)
    _torch.cuda.synchronize()
    for _ in range(int(_os.environ.get("PROFILE_ITERS", "10"))):
        run(*_inputs, *_outs)
    _torch.cuda.synchronize()
'''


def _render_kernel(defn: dict, reference_src: str) -> str:
    ref = re.sub(r"(?m)^def run\(", "def _ref(", reference_src)
    return _build_kernel(defn).replace("{REF}", ref)


def _solution_json(defn: dict, name: str, framework: str, platform: str) -> dict:
    hw = []
    if platform and platform.upper() != "LOCAL":
        hw.append(platform.upper())
    hw.append("LOCAL")
    # V0 is always a pure-PyTorch wrapper (guaranteed correct + submittable).
    # The loop migrates the body to `framework` and updates languages/dependencies then.
    return {
        "name": f"{name}_v0_pytorch",
        "definition": defn["name"],
        "author": "atrex-aka",
        "description": f"V0 baseline (DPS PyTorch wrapper around the reference); target framework: {framework}",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": hw,
            "entry_point": "kernel.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": True,
        },
        # No inline content: the evaluator reads the live kernel.py from disk, so
        # kernel.py stays the single source of truth across iterations.
        "sources": [{"path": "kernel.py"}],
    }


def _readme(name: str, defn: dict, framework: str, platform: str, gpu_wiki: str, n_workloads: int) -> str:
    return (
        f"# kernel_opt_{name}\n\n"
        f"Profile-driven optimization of SOL-ExecBench op **{defn['name']}**.\n\n"
        f"{defn.get('description', '').strip()}\n\n"
        "## Goal\n\n"
        "**Minimize the GEOMEAN of per-workload kernel latency** "
        "(`performance.latency_us` in `memory/v<N>.json`), while keeping ALL workloads "
        "correct under their own SOL tolerances. A version that passes `test_kernel.py` "
        "is directly submittable to SOL-ExecBench.\n\n"
        "## Config\n\n"
        f"- Target platform: `{platform}`\n"
        f"- Target framework: `{framework}` (V0 starts as PyTorch; migrate the body of `run()` in `kernel.py`)\n"
        f"- gpu_wiki_path: `{gpu_wiki}`\n"
        f"- Workloads (shape set): {n_workloads} in `workload.jsonl` (ground truth — do NOT edit)\n\n"
        "## Ground truth (immutable)\n\n"
        "- `definition.json`, `reference.py`, `workload.jsonl` — copied verbatim from the op dir.\n"
        "- `test_kernel.py` — the SOL evaluator harness (immutable methodology).\n\n"
        "## Workflow\n\n"
        "- Edit `kernel.py` only (DPS `run()`; args = definition.inputs then definition.outputs).\n"
        "- When migrating framework, also update `solution.json` `spec.languages` / `dependencies`.\n"
        "- Validate + bench every iteration with `python test_kernel.py --version v<N>`.\n"
    )


GITIGNORE = """__pycache__/
*.pyc
traces.jsonl
.finalize_traces.jsonl
submission.json
*.ncu-rep
profiles/*/att/*.att
profiles/*/att/*.out
profiles/*/att/*.pftrace
profiles/*/att/*.otf2
# orchestrator runtime symlinks (not part of the workspace)
/tools
/reference
/skills
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a kernel-opt workspace from a SOL-ExecBench op dir.")
    ap.add_argument("--op-dir", required=True, help="SOL op dir (definition.json + reference.py + workload.jsonl).")
    ap.add_argument("--name", required=True, help="Workspace name -> kernel_opt_<name>/.")
    ap.add_argument("--workspace", default="", help="Explicit workspace path (default: ./kernel_opt_<name>).")
    ap.add_argument("--framework", default="pytorch", help="Target framework the loop should migrate to.")
    ap.add_argument("--platform", default="LOCAL", help="Target hardware token, e.g. B200 (default: LOCAL).")
    ap.add_argument("--gpu-wiki", default="", help="Absolute path to gpu-wiki (recorded in README).")
    ap.add_argument("--no-bench", action="store_true", help="Skip the V0 test_kernel.py bench (no memory/v0.json).")
    args = ap.parse_args(argv)

    op = Path(args.op_dir).resolve()
    for f in GROUND_TRUTH:
        if not (op / f).is_file():
            raise SystemExit(f"--op-dir is not a SOL op dir (missing {f}): {op}")
    defn = json.loads((op / "definition.json").read_text(encoding="utf-8"))

    ws = Path(args.workspace).resolve() if args.workspace else (Path.cwd() / f"kernel_opt_{args.name}")
    for sub in ("memory", "plans", "profiles"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    # 1) ground truth, verbatim
    for f in GROUND_TRUTH:
        (ws / f).write_text((op / f).read_text(encoding="utf-8"), encoding="utf-8")

    # 2) pinned eval config (matches a submission; stable across versions)
    (ws / "config.json").write_text(
        json.dumps({"seed": 200, "warmup_runs": 10, "iterations": 50, "benchmark_reference": True}, indent=2) + "\n",
        encoding="utf-8",
    )

    # 3) V0 kernel + solution
    (ws / "kernel.py").write_text(_render_kernel(defn, (op / "reference.py").read_text(encoding="utf-8")), encoding="utf-8")
    (ws / "solution.json").write_text(
        json.dumps(_solution_json(defn, args.name, args.framework, args.platform), indent=2) + "\n", encoding="utf-8"
    )

    # 4) harness + constraints + docs (copied from reference/)
    (ws / "test_kernel.py").write_text((SCRIPT_DIR / "test_kernel.py").read_text(encoding="utf-8"), encoding="utf-8")
    claude = SCRIPT_DIR / "CLAUDE.md"
    if claude.exists():
        (ws / "CLAUDE.md").write_text(claude.read_text(encoding="utf-8"), encoding="utf-8")
    n_wl = sum(1 for line in (op / "workload.jsonl").read_text().splitlines() if line.strip())
    (ws / "README.md").write_text(_readme(args.name, defn, args.framework, args.platform, args.gpu_wiki, n_wl), encoding="utf-8")
    (ws / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    # 5) V0 baseline metrics (real evaluator)
    if not args.no_bench:
        r = subprocess.run([sys.executable, str(ws / "test_kernel.py"), "--version", "v0"], cwd=str(ws))
        if r.returncode != 0:
            print("[sol_seed] WARNING: V0 baseline did not pass all workloads — check solution.json / reference.",
                  file=sys.stderr)

    # 6) git init + single baseline commit
    if not (ws / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=str(ws), check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=str(ws), check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "V0: baseline (SOL reference wrapper)"], cwd=str(ws), check=True,
                   stdout=subprocess.DEVNULL)
    # Record the baseline commit hash into memory/v0.json (marks V0 as committed),
    # then fold it into the same commit — matches the iteration commit convention.
    v0 = ws / "memory" / "v0.json"
    if v0.exists():
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ws), capture_output=True, text=True).stdout.strip()
        mem = json.loads(v0.read_text(encoding="utf-8"))
        mem["git_commit_hash"] = h
        mem.setdefault("optimization", {})["action_category"] = "baseline"
        v0.write_text(json.dumps(mem, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "memory/v0.json"], cwd=str(ws), check=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)

    print(f"[sol_seed] workspace ready: {ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
