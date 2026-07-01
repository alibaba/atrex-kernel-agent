#!/usr/bin/env python3
"""Fetch the SOL-ExecBench leaderboard for a kernel and derive the optimization target.

Public JSON API (no auth):
  index:       https://research.nvidia.com/benchmarks/sol-execbench/api/kernels
  leaderboard: https://research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/kernel/<id>/<gpu>

Generalizes over ALL cases: pass `--name <case_name>` (e.g. 016_gqa_ragged_prefill_causal_h32_kv4_d128)
and the kernel id is resolved from the index, so you never hard-code an id. `--kernel-id` still works.

What the API exposes (aggregate ONLY — there is NO per-workload T_b / T_SOL anywhere):
  * reference_entries: "SOL Bound" (T_SOL, SOL 1.0), "Scoring Baseline" (T_b, SOL 0.5 -> Avg Speedup
    is measured against this), "Reference Implementation" (the naive ref).
  * rankings: the ranked submissions (usernames map to real people/teams; some open-source their kernels).

Optimization TARGET: beat a named leaderboard entry (default "Recursive") by a margin (default 10%):
  target latency    = entry.latency * (1 - margin)     (10% faster)
  target avgspeedup = entry.avg_speedup * (1 + margin)
  target sol_score  = > entry.sol_score
Use `--out leaderboard.json` to persist the rows + a `targets` block for `sol_metrics.py`.

The "Scoring Baseline" median is your calibration target: your locally-measured library baseline
(baseline/solution.json) should have ~ this median latency, confirming it is a faithful T_b proxy.
"""
import argparse
import json
import sys
import urllib.request

BASE = "https://research.nvidia.com/benchmarks/sol-execbench/api"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def resolve_kernel_id(name: str, gpu: str) -> int:
    """Resolve a case name (exact or unique substring) to its kernel id via the index."""
    kernels = _get(f"{BASE}/kernels").get("data", {}).get("kernels", [])
    exact = [k for k in kernels if k.get("name") == name]
    subs = [k for k in kernels if name in (k.get("name") or "")]
    cands = exact or subs
    cands = [k for k in cands if not k.get("gpu_types") or gpu in k["gpu_types"]] or cands
    if not cands:
        raise SystemExit(f"no kernel matches name '{name}' (gpu {gpu}). Try `--name <substring>`.")
    if len(cands) > 1 and not exact:
        opts = "\n".join(f"    {k['id']}  {k['name']}  {k.get('gpu_types')}" for k in cands[:12])
        raise SystemExit(f"ambiguous name '{name}' -> {len(cands)} matches:\n{opts}\nUse the exact name or --kernel-id.")
    return cands[0]["id"]


def _ref_row(d: dict, needle: str):
    for e in d.get("reference_entries", []):
        if needle.lower() in str(e.get("username", "")).lower():
            return e
    return None


def _ranked_row(d: dict, needle: str):
    for e in d.get("rankings", []):
        if e.get("rank") and needle.lower() in str(e.get("username", "")).lower():
            return e
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch SOL-ExecBench leaderboard + derive the target.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="case name (exact or unique substring), e.g. 016_gqa_ragged_prefill_causal_h32_kv4_d128")
    g.add_argument("--kernel-id", type=int)
    p.add_argument("--gpu", default="B200")
    p.add_argument("--target-user", default="Recursive", help="leaderboard entry to beat (default: Recursive)")
    p.add_argument("--target-margin", type=float, default=0.10, help="beat the target by this fraction (default 0.10 = 10%%)")
    p.add_argument("--out", help="write the leaderboard 'data' + a 'targets' block here (for sol_metrics --leaderboard-json)")
    args = p.parse_args(argv)

    kid = args.kernel_id if args.kernel_id is not None else resolve_kernel_id(args.name, args.gpu)
    d = _get(f"{BASE}/leaderboard/kernel/{kid}/{args.gpu}").get("data", {})
    print(f"kernel: {d.get('kernel_name')}  id={kid}  gpu: {d.get('gpu_type')}  unit: {d.get('latency_unit')}")

    baseline = _ref_row(d, "Baseline")
    print("\nreference rows:")
    for e in d.get("reference_entries", []):
        print(f"  {e.get('username'):<24} SOL={e.get('sol_score')}  Lat={e.get('latency_ms')}ms  "
              f"Fast={e.get('fast_1_count')}/{e.get('fast_1_total')}  AvgSpeedup={e.get('avg_speedup')}x")
    if baseline:
        print(f"  -> baseline CALIBRATION target: your measured library baseline median latency should be "
              f"~ {baseline.get('latency_ms')} ms")

    print("\nrankings:")
    for e in d.get("rankings", []):
        if not e.get("rank"):
            continue
        print(f"  #{e['rank']:<3} {e.get('username','')[:22]:<22} SOL={e.get('sol_score')}  "
              f"Lat={e.get('latency_ms')}ms  Fast={e.get('fast_1_count')}/{e.get('fast_1_total')}  AvgSpeedup={e.get('avg_speedup')}x")

    # ---- derive the optimization target: beat <target-user> by <margin> ----
    tgt = _ranked_row(d, args.target_user)
    targets = None
    m = args.target_margin
    if tgt:
        lat_max = float(tgt["latency_ms"]) * (1 - m)
        spd_min = float(tgt["avg_speedup"]) * (1 + m)
        targets = {
            "target_user": tgt.get("username"), "target_rank": tgt.get("rank"), "margin": m,
            "latency_ms_max": lat_max, "avg_speedup_min": spd_min,
            "sol_score_min": float(tgt["sol_score"]),  # beat (exceed) the target's SOL score
            "target_latency_ms": float(tgt["latency_ms"]),
            "target_avg_speedup": float(tgt["avg_speedup"]),
            "target_sol_score": float(tgt["sol_score"]),
        }
        print(f"\n=== TARGET: beat {tgt.get('username')} (rank {tgt.get('rank')}) by {m*100:.0f}% ===")
        print(f"  Latency    <= {lat_max*1000:.3f} us   ({lat_max:.6f} ms)   [{tgt['latency_ms']}ms * {1-m:.2f}]")
        print(f"  AvgSpeedup >= {spd_min:.2f}x                     [{tgt['avg_speedup']}x * {1+m:.2f}]")
        print(f"  SOL Score  >  {tgt['sol_score']}")
    else:
        avail = [e.get('username') for e in d.get('rankings', []) if e.get('rank')][:8]
        print(f"\n[warn] target user '{args.target_user}' not found in rankings for this case; "
              f"no beat-by-{m*100:.0f}% target derived. Available: {avail}")

    if args.out:
        d["targets"] = targets
        with open(args.out, "w") as f:
            json.dump(d, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
