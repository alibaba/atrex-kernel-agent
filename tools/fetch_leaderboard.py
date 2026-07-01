#!/usr/bin/env python3
"""Fetch the SOL-ExecBench leaderboard for a kernel (targets = top-3, baseline = Scoring Baseline).

The public leaderboard exposes a JSON API:
  https://research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/kernel/<kernel_id>/<gpu>

Each response contains:
  * sol_entry / reference_entries: "SOL Bound" (T_SOL, SOL Score 1.0), "Scoring Baseline"
    (T_b, SOL Score 0.5 -> Avg Speedup is measured against THIS), "Reference Implementation" (naive).
  * rankings: the ranked human/agent submissions (top of board = the target to beat/match).

Only AGGREGATE numbers are public (median Latency, Fast x/N, mean Avg Speedup, SOL Score) — there is
no per-workload T_b or T_SOL. So use this to set the TARGET (match top-3) and to sanity-check that
your measured library baseline's aggregate ~ the "Scoring Baseline" row.

The kernel_id must be discovered from the leaderboard site (e.g. case 016 == kernel_id 225 on B200).
Pass --kernel-id, or --name to search the index if an index endpoint is available.

Usage:
  fetch_leaderboard.py --kernel-id 225 --gpu B200 [--out leaderboard.json]
"""
import argparse
import json
import sys
import urllib.request

API = "https://research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/kernel/{kid}/{gpu}"


def fetch(kernel_id: int, gpu: str) -> dict:
    url = API.format(kid=kernel_id, gpu=gpu)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch SOL-ExecBench leaderboard JSON for a kernel.")
    p.add_argument("--kernel-id", type=int, required=True)
    p.add_argument("--gpu", default="B200")
    p.add_argument("--out", help="write the leaderboard 'data' object here (for sol_metrics --leaderboard-json)")
    args = p.parse_args(argv)

    raw = fetch(args.kernel_id, args.gpu)
    d = raw.get("data", raw)
    print(f"kernel: {d.get('kernel_name')}  gpu: {d.get('gpu_type')}  unit: {d.get('latency_unit')}")
    print("\nreference rows (SOL Bound = T_SOL, Scoring Baseline = T_b for Fast/AvgSpeedup):")
    for e in d.get("reference_entries", []):
        print(f"  {e.get('username'):<24} SOL={e.get('sol_score')}  Lat={e.get('latency_ms')}ms  "
              f"Fast={e.get('fast_1_count')}/{e.get('fast_1_total')}  AvgSpeedup={e.get('avg_speedup')}x")
    print("\nrankings (TARGET = top-3):")
    for e in d.get("rankings", []):
        if not e.get("rank"):
            continue
        print(f"  #{e['rank']:<3} {e.get('username','')[:22]:<22} SOL={e.get('sol_score')}  "
              f"Lat={e.get('latency_ms')}ms  Fast={e.get('fast_1_count')}/{e.get('fast_1_total')}  AvgSpeedup={e.get('avg_speedup')}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(d, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
