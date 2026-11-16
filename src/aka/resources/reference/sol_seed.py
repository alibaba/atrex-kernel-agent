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
come straight from the SOL files. The Supervisor measures V0 afterward through
the Runtime-private SOL evaluator.

Produces (under `kernel_opt_<name>/`):
    definition.json, reference.py, workload.jsonl   # ground truth (verbatim)
    config.json         # pinned seed/warmup/iterations/benchmark_reference
    kernel.py           # V0: self-contained DPS wrapper around the reference
    solution.json       # SOL solution (sources reference kernel.py by path)
    CLAUDE.md           # agent constraints (copied from reference/)
    README.md, .gitignore
    # V0 metrics are stored under the Supervisor-private runtime root.

Usage:
    python reference/sol_seed.py --problem <sol-op-dir> --name <name> \
        [--framework pytorch] [--platform B200]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aka.task.candidate_workspace.agent_git import gitignore_contents
from aka.resources.bundles import resource_path

SCRIPT_DIR = resource_path("reference/CLAUDE.md").parent
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
        f"{assign}"
    )
    return header


# Profiling is driven by a trusted harness that Runtime injects only into the
# transient remote request workspace. It is not part of the Agent source tree.


def _render_kernel(defn: dict, reference_src: str) -> str:
    ref = re.sub(r"(?m)^def run\(", "def _ref(", reference_src)
    return _build_kernel(defn).replace("{REF}", ref)


def _solution_json(defn: dict, name: str, framework: str, platform: str) -> dict:
    hw = []
    # SOL-ExecBench's schema only accepts B200 or LOCAL here.  The optimizer's
    # logical platform may be an inventory alias (for example pro5000); actual
    # remote scheduling is selected independently by --sandbox-hardware.
    if platform.upper() == "B200":
        hw.append("B200")
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


def _readme(name: str, defn: dict, framework: str, platform: str, n_workloads: int) -> str:
    return (
        f"# kernel_opt_{name}\n\n"
        f"Evidence-backed optimization of SOL-ExecBench op **{defn['name']}**.\n\n"
        f"{defn.get('description', '').strip()}\n\n"
        "## Goal\n\n"
        "**Maximize performance_score: arithmetic mean of per-workload speedups** "
        "relative to the SOL reference, while keeping ALL workloads correct under their "
        "own SOL tolerances. The Supervisor owns evaluation and final packaging.\n\n"
        "## Config\n\n"
        f"- Target platform: `{platform}`\n"
        f"- Target framework: `{framework}` (V0 starts as PyTorch; migrate the body of `run()` in `kernel.py`)\n"
        f"- Workloads (shape set): {n_workloads} in `workload.jsonl` (ground truth — do NOT edit)\n\n"
        "## Ground truth (immutable)\n\n"
        "- `definition.json`, `reference.py`, `workload.jsonl` — copied verbatim from the op dir.\n"
        "- The Supervisor supplies the official SOL evaluator; never replace it.\n\n"
        "## Workflow\n\n"
        "- Edit `kernel.py` only (DPS `run()`; args = definition.inputs then definition.outputs).\n"
        "- Update `solution.json` only when the current initialization session permits it.\n"
        "- Measure using `python3 tools/sandbox.py --kind run --no-sync`.\n"
        "- Follow skills/runtime-records/SKILL.md for research records and exact-commit `submit-kernel`.\n"
    )


GITIGNORE = gitignore_contents()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a kernel-opt workspace from a SOL-ExecBench op dir.")
    ap.add_argument("--problem", required=True, help="SOL op dir (definition.json + reference.py + workload.jsonl).")
    ap.add_argument("--name", required=True, help="Workspace name -> kernel_opt_<name>/.")
    ap.add_argument("--workspace", default="", help="Explicit workspace path (default: ./kernel_opt_<name>).")
    ap.add_argument("--framework", default="pytorch", help="Target framework the loop should migrate to.")
    ap.add_argument("--platform", default="LOCAL", help="Target hardware token, e.g. B200 (default: LOCAL).")
    args = ap.parse_args(argv)

    op = Path(args.problem).resolve()
    for f in GROUND_TRUTH:
        if not (op / f).is_file():
            raise SystemExit(f"--problem is not a SOL op dir (missing {f}): {op}")
    defn = json.loads((op / "definition.json").read_text(encoding="utf-8"))

    ws = Path(args.workspace).resolve() if args.workspace else (Path.cwd() / f"kernel_opt_{args.name}")
    # Retire the old Agent-visible evaluator before either reuse or creation.
    # Runtime injects the trusted harness only into an ephemeral request tree.
    (ws / "test_kernel.py").unlink(missing_ok=True)
    if (ws / ".git").exists():
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
            for name in (".gitignore", "kernel.py"):
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
    (ws / "scratch").mkdir(parents=True, exist_ok=True)

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

    claude = SCRIPT_DIR / "CLAUDE.md"
    if claude.exists():
        (ws / "CLAUDE.md").write_text(claude.read_text(encoding="utf-8"), encoding="utf-8")
    n_wl = sum(1 for line in (op / "workload.jsonl").read_text().splitlines() if line.strip())
    (ws / "README.md").write_text(_readme(args.name, defn, args.framework, args.platform, n_wl), encoding="utf-8")
    (ws / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    # 5) Git source commit. V0 evidence is produced later through Runtime, so its
    # private harness and result never enter this repository.
    if not (ws / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=str(ws), check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=str(ws), check=True)
    from aka.task.candidate_workspace.agent_git import kernel_git_paths
    source_paths = kernel_git_paths(ws)
    subprocess.run(["git", "add", "--", *source_paths], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "V0: baseline (SOL reference wrapper)"], cwd=str(ws), check=True,
                   stdout=subprocess.DEVNULL)
    print(f"[sol_seed] workspace ready: {ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
