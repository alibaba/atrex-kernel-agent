#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SOL-ExecBench <-> AKA adapter.

Bridges SOL-ExecBench's problem format (definition.json + workload.jsonl, with
an embedded ``reference`` source string and a destination-passing-style ``run``
contract) and AKA's flat ``kernel_opt_<name>/`` workspace.

Subcommands
-----------
materialize <problem_dir> [name]   Create an AKA workspace from a SOL problem:
    reference.py (verbatim from definition.reference), a DPS kernel.py stub, a
    SOL-faithful test_kernel.py (static anti-cheat gate + authoritative SOL CLI
    eval + honest T_b + best-effort dynamic anti-memo/coverage), a frozen
    baseline.json shell, and an honest bench config (benchmark_reference=true).

package <workspace_dir>            Emit a solution.json, labelling spec.languages
    from a light source scan of the kernel.

Anti-cheat is a POLICY documented in CLAUDE.md / SKILL.md (C1-C6), not a code gate:
this adapter no longer runs a static validator. The generated test_kernel.py checks
correctness and reports the SOL leaderboard metrics; it does not reject "cheats".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent

# Anti-cheat is a POLICY (see CLAUDE.md / SKILL.md), not a code gate. `package` labels
# spec.languages from a light source scan below; it does not reject "cheating" kernels.
_LANG_MARKERS = [
    ("triton", ("@triton.jit", "triton.jit", "import triton")),
    ("cute_dsl", ("@cute.jit", "@cute.kernel", "cutlass.cute", "import cutlass")),
    ("cutile", ("cutile", "cuTile")),
    ("cuda_cpp", ("__global__", 'extern "C"', "load_inline", "cpp_extension")),
]


def _detect_languages(kernel_src: str) -> list[str]:
    """Best-effort framework labels from source markers (informational, not enforcement)."""
    langs = [lang for lang, marks in _LANG_MARKERS if any(m in kernel_src for m in marks)]
    return langs


# --------------------------------------------------------------------------- #
# Problem parsing (dependency-free: raw JSON, no SOL import required)
# --------------------------------------------------------------------------- #

def _load_definition(problem_dir: Path) -> dict:
    p = problem_dir / "definition.json"
    if not p.exists():
        raise SystemExit(f"definition.json not found in {problem_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _first_workload(problem_dir: Path) -> dict | None:
    p = problem_dir / "workload.jsonl"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return json.loads(line)
    return None


def _run_signature(definition: dict, dps: bool) -> list[str]:
    params = list(definition.get("inputs", {}).keys())
    if dps:
        params += list(definition.get("outputs", {}).keys())
    return params


def _io_doc(definition: dict) -> str:
    lines = []
    for kind in ("inputs", "outputs"):
        lines.append(f"    {kind}:")
        for name, spec in definition.get(kind, {}).items():
            shape = spec.get("shape")
            lines.append(f"      {name}: shape={shape} dtype={spec.get('dtype')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

KERNEL_STUB = '''\
# AKA workspace for SOL-ExecBench problem: {name}
# Entry point: run({sig})   [destination_passing_style={dps}]
#
# ANTI-CHEAT POLICY: {policy}  (see CLAUDE.md C1-C6 -- policy, not a code gate)
#   Implement the operator as a SELF-WRITTEN GPU kernel reachable from run():
#   Triton (@triton.jit), CuteDSL (@cute.jit/@cute.kernel), cuTile, or inline CUDA.
#   Do NOT delegate to flashinfer/flash_attn/xformers/vllm/aiter or
#   F.scaled_dot_product_attention, camouflage languages, or shape-key memoize.
#   See reference.py for the exact numerical semantics to reproduce.
import torch


def run({sig}):
    raise NotImplementedError(
        "Implement the kernel here. See reference.py for semantics. "
        "{dps_hint}"
    )
'''

README_TMPL = '''\
# {name} (SOL-ExecBench)

{description}

## Contract
- Entry: `run({sig})`  (destination_passing_style = {dps})
- Reference semantics: `reference.py` (verbatim from definition.reference).
- I/O:
{io_doc}

## How this workspace is evaluated
`test_kernel.py` runs, in order:
1. **Authoritative correctness + performance** -- the real `sol-execbench` CLI,
   so correctness uses SOL's exact tolerance and timing matches the leaderboard.
2. **Leaderboard metrics** -- `tools/sol_metrics.py` reports the FOUR SOL-ExecBench
   leaderboard numbers vs a measured library baseline:
     * **Latency**     = median over workloads of per-workload median T_k   (EXACT)
     * **Fast**        = count(T_k < T_b)/N                                 (EXACT vs baseline)
     * **Avg Speedup** = mean(T_b / T_k)  -- vs the Scoring Baseline, NOT the naive ref (EXACT)
     * **SOL Score**   = mean 1/(1+(T_k-T_SOL)/(T_b-T_SOL))  (ESTIMATE from roofline T_SOL; official = submit)

Anti-cheat (no library delegation / camouflage / shape-keyed memo / masked errors) is a POLICY in
`CLAUDE.md` (C1-C6) — follow it; it is not enforced by a code gate here.

## Stop conditions (target = leaderboard top-3)
- Correctness PASSED at SOL tolerance for all workloads.
- **Report all four metrics (SOL Score / Latency / Fast / Avg Speedup) on every result.**
- **Baseline T_b**: the SOL "Scoring Baseline" (SOL Score 0.5 point) -- an optimized library impl.
  Its per-workload values are not public, so measure a library baseline through this harness
  (FlashInfer for attention, DeepGEMM/cuBLAS for GEMM, cuDNN, or torch) and confirm its aggregate
  ~ the leaderboard "Scoring Baseline" row (`fetch_leaderboard.py`). Beating T_b => Fast, AvgSpeedup>1.
- **Target**: match or beat the top-3 leaderboard entries on Latency + Avg Speedup (=> high SOL Score).
  Fetch them with `tools/fetch_leaderboard.py --kernel-id <id> --gpu <gpu>`.
- Never fabricate SOL Score / T_b / T_SOL (anti-cheat C6): T_b is measured, T_SOL is a labelled
  roofline estimate.

## Baseline
`baseline.json` is FROZEN (task identity + scoring). Do not edit it.
'''

# test_kernel.py template -- token-substituted (avoids brace escaping).
TEST_KERNEL_TMPL = r'''#!/usr/bin/env python3
"""SOL-faithful promotion gate for this AKA workspace (auto-generated).

Run inside the SOL-ExecBench environment (e.g. `uv run python test_kernel.py`
from the SOL repo, or with the SOL venv active). Exits non-zero on any cheat or
correctness failure.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
PROBLEM_DIR = Path(r"@@PROBLEM_DIR@@")
TOOLS = Path(r"@@TOOLS@@")
ENTRY = "@@ENTRY@@"
DPS = @@DPS@@
KERNEL = WORKSPACE / "kernel.py"
SOLUTION = WORKSPACE / "solution.json"
CONFIG = WORKSPACE / "aka_bench_config.json"

fail = []
warn = []


def section(msg):
    print("\n" + "=" * 8 + " " + msg + " " + "=" * 8)


# Anti-cheat is a POLICY (see CLAUDE.md / SKILL.md C1-C6), not a gate here.

# --- Gate 1: authoritative SOL correctness + performance -----------------------
section("Gate 1: authoritative sol-execbench eval (+benchmark_reference)")
if not SOLUTION.exists():
    fail.append("solution.json missing -- run `sol_adapter.py package` first")
else:
    sol_cmd = ["sol-execbench", str(PROBLEM_DIR), "--solution", str(SOLUTION),
               "--config", str(CONFIG), "--json"]
    try:
        r2 = subprocess.run(sol_cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        r2 = subprocess.run([sys.executable, "-m", "sol_execbench.cli",
                             str(PROBLEM_DIR), "--solution", str(SOLUTION),
                             "--config", str(CONFIG), "--json"],
                            capture_output=True, text=True, timeout=1800)
    print((r2.stdout or "")[-2000:])
    if r2.returncode != 0:
        print((r2.stderr or "")[-1500:])
        fail.append(f"sol-execbench exited {r2.returncode}")
    # defensively scan the JSON output for status + latencies
    statuses, lat, ref_lat = [], None, None

    def _walk(o):
        global lat, ref_lat
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "status" and isinstance(v, str):
                    statuses.append(v)
                if k in ("latency_ms",) and isinstance(v, (int, float)) and lat is None:
                    lat = float(v)
                if k in ("reference_latency_ms",) and isinstance(v, (int, float)):
                    ref_lat = float(v)
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    for ln in (r2.stdout or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            _walk(json.loads(ln))
        except Exception:
            pass
    if statuses and any(s.upper() != "PASSED" for s in statuses):
        fail.append(f"correctness not PASSED: statuses={set(statuses)}")
    elif statuses:
        print(f"correctness: PASSED ({len(statuses)} workload trace(s))")

    # --- Gate 2: SOL-ExecBench leaderboard metrics (Latency/Fast/AvgSpeedup/SOL Score) -----
    section("Gate 2: SOL leaderboard metrics (Latency / Fast / Avg Speedup / SOL Score)")
    if lat is not None:
        print(f"kernel latency_ms (first workload trace) = {lat}")
    METRICS = TOOLS / "sol_metrics.py"
    BASELINE = WORKSPACE / "baseline" / "solution.json"   # measured library baseline (T_b proxy)
    LEADERBOARD = WORKSPACE / "leaderboard.json"          # from fetch_leaderboard.py (optional)
    TSOL = WORKSPACE / "tsol.json"                        # per-workload roofline T_SOL from Step 0 (optional)
    METRICS_CFG = (WORKSPACE / "dev_config.json") if (WORKSPACE / "dev_config.json").exists() else CONFIG
    if METRICS.exists() and BASELINE.exists():
        mcmd = [sys.executable, str(METRICS), "--problem", str(PROBLEM_DIR),
                "--solution", str(SOLUTION), "--baseline", str(BASELINE),
                "--config", str(METRICS_CFG)]
        td = os.environ.get("FLASHINFER_TRACE_DIR")
        if td:
            mcmd += ["--trace-dir", td]
        if LEADERBOARD.exists():
            mcmd += ["--leaderboard-json", str(LEADERBOARD)]
        if TSOL.exists():
            mcmd += ["--tsol-json", str(TSOL)]
        rm = subprocess.run(mcmd, capture_output=True, text=True)
        print((rm.stdout or "").strip())
        if rm.returncode != 0:
            print((rm.stderr or "")[-800:])
            warn.append("sol_metrics.py failed; see stderr above")
    else:
        warn.append("no measured library baseline at baseline/solution.json -- Fast/Avg Speedup "
                    "cannot be computed. Build a library baseline (FlashInfer/DeepGEMM/cuDNN/torch), "
                    "measure it through this harness, and rerun. Also fetch targets with "
                    "fetch_leaderboard.py. Do NOT fabricate SOL Score / T_b (anti-cheat C6).")
        if ref_lat and lat:
            print(f"(fallback) naive-reference latency_ms = {ref_lat} -> {ref_lat / lat:.2f}x vs naive "
                  f"(NOT the leaderboard Avg Speedup, which is vs the Scoring Baseline)")

# --- Verdict -------------------------------------------------------------------
section("VERDICT")
for w in warn:
    print(f"[WARN] {w}")
for f in fail:
    print(f"[FAIL] {f}")
if fail:
    print("\nRESULT: REJECTED")
    sys.exit(2)
print("\nRESULT: OK" + (" (with warnings)" if warn else ""))
sys.exit(0)
'''


# --------------------------------------------------------------------------- #
# materialize
# --------------------------------------------------------------------------- #

def cmd_materialize(args: argparse.Namespace) -> int:
    problem_dir = Path(args.problem_dir).resolve()
    definition = _load_definition(problem_dir)
    name = args.name or definition.get("name") or problem_dir.name
    dest = Path(args.dest).resolve() if args.dest else Path.cwd()
    ws = dest / f"kernel_opt_{name}"
    if ws.exists() and not args.force:
        raise SystemExit(f"workspace already exists: {ws} (use --force to overwrite)")
    for sub in ("", "memory", "plans", "profiles"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    dps = not args.return_style
    sig = ", ".join(_run_signature(definition, dps))
    dps_hint = ("Write the output tensors in place (last args)."
                if dps else "Return the output tensor(s).")

    # reference.py (verbatim from the embedded definition string)
    ref_src = definition.get("reference")
    if not ref_src:
        raise SystemExit("definition.json has no embedded 'reference' source")
    (ws / "reference.py").write_text(ref_src, encoding="utf-8")

    # kernel.py stub
    (ws / "kernel.py").write_text(
        KERNEL_STUB.format(name=name, sig=sig, dps=dps, policy=args.policy,
                           dps_hint=dps_hint), encoding="utf-8")

    # baseline.json shell (FROZEN scope: task identity + scoring)
    (ws / "baseline.json").write_text(json.dumps({
        "state": "shell",
        "problem": name,
        "frozen": True,
        "note": "Auto-promoted to 'real' on first reference benchmark; never edit by hand.",
        "inputs": list(definition.get("inputs", {}).keys()),
        "outputs": list(definition.get("outputs", {}).keys()),
        "destination_passing_style": dps,
    }, indent=2), encoding="utf-8")

    # honest bench config (benchmark_reference = true -> also times the naive reference)
    (ws / "aka_bench_config.json").write_text(json.dumps({
        "warmup_runs": 10, "iterations": 50,
        "benchmark_reference": True, "lock_clocks": False, "seed": 200,
    }, indent=2), encoding="utf-8")
    # fast dev config (skip the slow naive reference) -- used for leaderboard-metric runs
    (ws / "dev_config.json").write_text(json.dumps({
        "warmup_runs": 10, "iterations": 50,
        "benchmark_reference": False, "lock_clocks": False, "seed": 200,
    }, indent=2), encoding="utf-8")

    # library-baseline scaffold: the SOL "Scoring Baseline" (T_b, SOL Score 0.5) is an optimized
    # library impl whose per-workload values are NOT public. Measure a library proxy through THIS
    # harness so Fast / Avg Speedup / SOL Score can be computed. NOT your submission.
    (ws / "baseline").mkdir(exist_ok=True)
    (ws / "baseline" / "README.md").write_text(
        "# Library baseline (T_b measuring stick) -- NOT the submission\n\n"
        f"Build a `baseline/kernel.py` with entry `run({sig})` that calls an optimized LIBRARY\n"
        "for this op (attention -> flashinfer; GEMM -> deepgemm/cuBLAS; else cuDNN or torch), then\n"
        "emit `baseline/solution.json` (inline the source). `test_kernel.py` / `sol_metrics.py`\n"
        "run it through the SAME sol-execbench harness to get per-workload T_b.\n\n"
        "Sanity-check its AGGREGATE (median latency, avg speedup 1.0x) against the leaderboard\n"
        "'Scoring Baseline' row via `tools/fetch_leaderboard.py --kernel-id <id> --gpu <gpu>`.\n"
        "Realistic library usage: include the library's normal per-call setup (plan/workspace) in\n"
        "the timed path (that is how NVIDIA's baseline is measured -> ~O(100us) on tiny workloads).\n\n"
        "Then: `python <tools>/sol_metrics.py --problem <problem> --solution ../solution.json \\\n"
        "  --baseline solution.json --config ../dev_config.json --trace-dir <SOL_root> \\\n"
        "  [--leaderboard-json ../leaderboard.json] [--tsol-json ../tsol.json]`\n\n"
        "This is a measuring stick only: flashinfer/flash_attn/etc. are fine HERE but BANNED in\n"
        "your real kernel.py/solution.json (anti-cheat C1).\n",
        encoding="utf-8")

    # pointer to the problem dir + materialization metadata
    (ws / ".sol_problem.json").write_text(json.dumps({
        "problem_dir": str(problem_dir), "name": name,
        "destination_passing_style": dps, "entry": "run", "policy": args.policy,
    }, indent=2), encoding="utf-8")

    # test_kernel.py (correctness + leaderboard-metrics harness)
    test_src = (TEST_KERNEL_TMPL
                .replace("@@PROBLEM_DIR@@", str(problem_dir))
                .replace("@@TOOLS@@", str(_TOOLS))
                .replace("@@ENTRY@@", "run")
                .replace("@@DPS@@", "True" if dps else "False"))
    (ws / "test_kernel.py").write_text(test_src, encoding="utf-8")

    # README
    (ws / "README.md").write_text(README_TMPL.format(
        name=name, description=definition.get("description", ""),
        sig=sig, dps=dps, io_doc=_io_doc(definition)), encoding="utf-8")

    print(f"materialized AKA workspace: {ws}")
    print(f"  entry: run({sig})   dps={dps}")
    print(f"  files: reference.py, kernel.py (stub), test_kernel.py, baseline.json,")
    print(f"         aka_bench_config.json, dev_config.json, baseline/ (T_b scaffold),")
    print(f"         README.md, memory/ plans/ profiles/")
    print(f"  anti-cheat: policy in CLAUDE.md (C1-C6), not a code gate")
    print(f"  next: implement kernel.py, then `python {_TOOLS}/sol_adapter.py package {ws}`")
    return 0


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #

def cmd_package(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve()
    kernel_path = ws / "kernel.py"
    if not kernel_path.exists():
        raise SystemExit(f"kernel.py not found in {ws}")
    kernel_src = kernel_path.read_text(encoding="utf-8")

    meta = {}
    meta_path = ws / ".sol_problem.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    problem_dir = Path(args.problem or meta.get("problem_dir", "")).resolve() \
        if (args.problem or meta.get("problem_dir")) else None
    dps = meta.get("destination_passing_style", not args.return_style)
    entry = meta.get("entry", "run")
    definition_name = args.definition_name or meta.get("name")
    if problem_dir and not definition_name:
        definition_name = _load_definition(problem_dir).get("name")
    if not definition_name:
        definition_name = ws.name.replace("kernel_opt_", "")

    # languages from a light source scan (informational; anti-cheat is a CLAUDE.md policy)
    languages = _detect_languages(kernel_src) or ["pytorch"]

    deps = sorted({l for l in ("torch", *( ["triton"] if "triton" in languages else []))})
    solution = {
        "name": args.name or f"{definition_name}_aka",
        "definition": definition_name,
        "author": args.author,
        "description": args.description or f"AKA self-written kernel ({', '.join(languages)})",
        "spec": {
            "languages": languages,
            "target_hardware": [h.strip() for h in args.hardware.split(",")],
            "entry_point": f"kernel.py::{entry}",
            "dependencies": deps,
            "destination_passing_style": bool(dps),
            "binding": None,
        },
        "sources": [{"path": "kernel.py", "content": kernel_src}],
    }
    out = ws / "solution.json"
    out.write_text(json.dumps(solution, indent=2), encoding="utf-8")
    print(f"\nwrote solution.json: {out}")
    print(f"  languages (source scan): {languages}")
    print("  NOTE: anti-cheat is a policy (CLAUDE.md C1-C6), not enforced here.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SOL-ExecBench <-> AKA adapter")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("materialize", help="create an AKA workspace from a SOL problem")
    m.add_argument("problem_dir")
    m.add_argument("name", nargs="?", default=None)
    m.add_argument("--dest", help="parent dir for kernel_opt_<name> (default: cwd)")
    m.add_argument("--return-style", action="store_true",
                   help="generate a return-style run() instead of DPS")
    m.add_argument("--policy", default="require_self_written_kernel",
                   choices=["require_self_written_kernel", "allow_libs"],
                   help="informational only (recorded in the kernel stub); anti-cheat is a CLAUDE.md policy")
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_materialize)

    k = sub.add_parser("package", help="emit a solution.json (languages from a source scan)")
    k.add_argument("workspace_dir")
    k.add_argument("--problem", help="problem dir (default: from .sol_problem.json)")
    k.add_argument("--name")
    k.add_argument("--definition-name")
    k.add_argument("--author", default="gpu-kernel-optimizer")
    k.add_argument("--description")
    k.add_argument("--hardware", default="LOCAL")
    k.add_argument("--return-style", action="store_true")
    k.set_defaults(func=cmd_package)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
