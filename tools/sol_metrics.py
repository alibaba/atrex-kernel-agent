#!/usr/bin/env python3
"""Compute the four SOL-ExecBench leaderboard metrics for a solution.

The SOL-ExecBench leaderboard reports FOUR numbers per kernel (verified against the
public API, e.g. research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/kernel/<id>/<gpu>):

  * Latency     = median over workloads of the per-workload median T_k   (ms)
  * Fast        = count(T_k_i < T_b_i) / N                               (x / N)
  * Avg Speedup = mean_i( T_b_i / T_k_i )                                (x)     <- vs the SCORING BASELINE, not the naive ref
  * SOL Score   = mean_i( 1 / (1 + (T_k_i - T_SOL_i)/(T_b_i - T_SOL_i)) )        <- S=0.5 when T_k=T_b, S=1 when T_k=T_SOL

where
  * T_k   = your kernel's per-workload latency (measured here via sol-execbench CUPTI),
  * T_b   = the SCORING BASELINE's per-workload latency. NVIDIA's baseline is an optimized
            library implementation (FlashInfer-like); its per-workload values are NOT public,
            so we measure a library baseline (FlashInfer / DeepGEMM / cuDNN / torch) through the
            SAME harness as a faithful proxy. Its aggregate should match the leaderboard
            "Scoring Baseline" row (fetch it with fetch_leaderboard.py to confirm).
  * T_SOL = the theoretical bound (NVIDIA's SOLAR tool). Per-workload T_SOL is NOT public, so
            SOL Score here is an ESTIMATE from a roofline T_SOL (pass --tsol-json from Step 0's
            roofline: T_SOL_i = max(FLOPs_i/peak_tc, bytes_i/BW)). Clearly labelled as an estimate;
            the official SOL Score only comes from submitting to the leaderboard.

Latency / Fast / Avg Speedup are computed EXACTLY (given a measured baseline). Only SOL Score is
an estimate locally. Never fabricate SOL Score (anti-cheat C6): report it as "est." with its inputs,
or "N/A" if no T_SOL model is supplied.

Usage:
  sol_metrics.py --problem <dir> --solution my.json --baseline base.json \
     [--config cfg.json] [--trace-dir <SOL_repo_root>] \
     [--tsol-json tsol.json] [--leaderboard-json lb.json] \
     [--solution-lat my.lat.json] [--baseline-lat base.lat.json]

If --solution-lat / --baseline-lat are given they are used directly (idx->latency_ms map),
skipping the (slow) sol-execbench runs.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _load_workload_axes(problem_dir: Path):
    wls = [json.loads(l) for l in (problem_dir / "workload.jsonl").read_text().splitlines() if l.strip()]
    uuid2idx = {w["uuid"]: i for i, w in enumerate(wls)}
    axes = {i: w.get("axes", {}) for i, w in enumerate(wls)}
    return wls, uuid2idx, axes


def _find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def run_sol(problem_dir: Path, solution: Path, config: Path | None, trace_dir: str | None):
    """Run sol-execbench --json and return {idx: latency_ms} (per-workload median)."""
    wls, uuid2idx, _ = _load_workload_axes(problem_dir)
    env = dict(os.environ)
    if trace_dir:
        env["FLASHINFER_TRACE_DIR"] = trace_dir
    cmd = ["sol-execbench", str(problem_dir), "--solution", str(solution), "--json"]
    if config:
        cmd += ["--config", str(config)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    except FileNotFoundError:
        cmd[0:1] = [sys.executable, "-m", "sol_execbench.cli"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    lat = {}
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            t = json.loads(ln)
        except Exception:
            continue
        wl = t.get("workload", {})
        uuid = wl.get("uuid") if isinstance(wl, dict) else None
        idx = uuid2idx.get(uuid, -1)
        L = _find(t, "latency_ms")
        st = _find(t, "status")
        if idx >= 0 and L is not None and (st is None or str(st).upper() == "PASSED"):
            lat[idx] = float(L)
    if not lat and r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-1500:] + "\n")
    return lat


def compute_metrics(tk: dict, tb: dict, tsol: dict | None):
    """tk/tb/tsol: {idx: latency_ms}. Returns dict of the four metrics."""
    idxs = sorted(set(tk) & set(tb))
    tk_v = [tk[i] for i in idxs]
    latency_ms = statistics.median(tk_v)
    fast = sum(1 for i in idxs if tk[i] < tb[i])
    avg_speedup = sum(tb[i] / tk[i] for i in idxs) / len(idxs)
    sol = None
    if tsol:
        S = []
        for i in idxs:
            if i not in tsol:
                continue
            s0, b, k = tsol[i], tb[i], tk[i]
            denom = b - s0
            if denom <= 0:
                continue
            S.append(1.0 / (1.0 + max(k - s0, 0.0) / denom))
        if S:
            sol = sum(S) / len(S)
    return {
        "idxs": idxs,
        "latency_ms": latency_ms,
        "fast": fast,
        "n": len(idxs),
        "avg_speedup": avg_speedup,
        "sol_score_est": sol,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute SOL-ExecBench leaderboard metrics (Latency/Fast/AvgSpeedup exact; SOL Score estimated).")
    p.add_argument("--problem", required=True)
    p.add_argument("--solution", required=True)
    p.add_argument("--baseline", help="library baseline solution.json (FlashInfer/DeepGEMM/…). Required unless --baseline-lat given.")
    p.add_argument("--config")
    p.add_argument("--trace-dir", help="FLASHINFER_TRACE_DIR (SOL repo root) so safetensors blobs resolve")
    p.add_argument("--tsol-json", help="per-workload roofline T_SOL {idx: ms} from Step 0 -> enables SOL Score estimate")
    p.add_argument("--leaderboard-json", help="leaderboard.json from fetch_leaderboard.py -> prints top-3 + baseline comparison")
    p.add_argument("--solution-lat", help="precomputed {idx: latency_ms} for the solution (skip sol run)")
    p.add_argument("--baseline-lat", help="precomputed {idx: latency_ms} for the baseline (skip sol run)")
    args = p.parse_args(argv)

    problem = Path(args.problem).resolve()

    def _lat(map_arg, sol_arg, label):
        if map_arg:
            return {int(k): float(v) for k, v in json.loads(Path(map_arg).read_text()).items() if str(k).lstrip("-").isdigit()}
        if not sol_arg:
            raise SystemExit(f"need --{label} or --{label}-lat")
        return run_sol(problem, Path(sol_arg).resolve(), Path(args.config).resolve() if args.config else None, args.trace_dir)

    tk = _lat(args.solution_lat, args.solution, "solution")
    tb = _lat(args.baseline_lat, args.baseline, "baseline")
    tsol = None
    if args.tsol_json:
        tsol = {int(k): float(v) for k, v in json.loads(Path(args.tsol_json).read_text()).items() if str(k).lstrip("-").isdigit()}

    m = compute_metrics(tk, tb, tsol)
    _, _, axes = _load_workload_axes(problem)

    print("\n=== per-workload (T_k = your kernel, T_b = baseline) ===")
    print(f"{'idx':>3} {'axes':>18} {'T_k(us)':>9} {'T_b(us)':>9} {'T_b/T_k':>8} {'<T_b?':>6}")
    for i in m["idxs"]:
        ax = axes.get(i, {})
        al = "/".join(str(v) for v in ax.values())
        print(f"{i:>3} {al:>18} {tk[i]*1000:>9.2f} {tb[i]*1000:>9.2f} {tb[i]/tk[i]:>8.2f} {'yes' if tk[i]<tb[i] else 'no':>6}")

    print("\n=== SOL-ExecBench leaderboard metrics ===")
    print(f"  Latency (median T_k) : {m['latency_ms']*1000:.3f} us  ({m['latency_ms']:.6f} ms)   [EXACT]")
    print(f"  Fast (T_k < T_b)     : {m['fast']}/{m['n']} = {m['fast']/m['n']:.2f}                 [EXACT vs measured baseline]")
    print(f"  Avg Speedup mean(T_b/T_k): {m['avg_speedup']:.2f}x                        [EXACT vs measured baseline]")
    if m["sol_score_est"] is not None:
        print(f"  SOL Score (est.)     : {m['sol_score_est']:.4f}   [ESTIMATE: roofline T_SOL; official = submit]")
    else:
        print(f"  SOL Score            : N/A locally (needs per-workload T_SOL; pass --tsol-json from Step 0 roofline)")

    if args.leaderboard_json:
        lb = json.loads(Path(args.leaderboard_json).read_text())
        print("\n=== leaderboard ===")
        for e in lb.get("rankings", [])[:5]:
            print(f"  {('#'+str(e.get('rank'))) if e.get('rank') else e.get('username'):>16} {e.get('username','')[:18]:<18} "
                  f"SOL={e.get('sol_score')}  Lat={e.get('latency_ms')}ms  Fast={e.get('fast_1_count')}/{e.get('fast_1_total')}  AvgSpd={e.get('avg_speedup')}x")
        for e in lb.get("reference_entries", []):
            if "aseline" in str(e.get("username", "")):
                print(f"  {'baseline':>16} {e.get('username',''):<18} SOL={e.get('sol_score')}  Lat={e.get('latency_ms')}ms  AvgSpd={e.get('avg_speedup')}x")

        # ---- primary target: beat <target-user> by <margin> (from fetch_leaderboard's `targets`) ----
        tg = lb.get("targets")
        if tg:
            lat_us = m["latency_ms"] * 1000
            lat_max_us = tg["latency_ms_max"] * 1000
            lat_ok = lat_us <= lat_max_us
            spd_ok = m["avg_speedup"] >= tg["avg_speedup_min"]
            print(f"\n=== TARGET: beat {tg['target_user']} (rank {tg.get('target_rank')}) by {tg['margin']*100:.0f}% ===")
            print(f"  Latency    : {lat_us:8.3f} us   target <= {lat_max_us:8.3f} us   "
                  f"[{tg['target_latency_ms']*1000:.3f}us x {1-tg['margin']:.2f}]   {'PASS' if lat_ok else 'MISS'}")
            print(f"  Avg Speedup: {m['avg_speedup']:8.2f}x   target >= {tg['avg_speedup_min']:8.2f}x   "
                  f"[{tg['target_avg_speedup']}x x {1+tg['margin']:.2f}]   {'PASS' if spd_ok else 'MISS'}")
            if m["sol_score_est"] is not None:
                print(f"  SOL Score  : {m['sol_score_est']:8.4f}   target >  {tg['sol_score_min']:.4f} (est.)   "
                      f"{'PASS' if m['sol_score_est'] > tg['sol_score_min'] else 'MISS'}")
            print(f"  => TARGET {'MET' if (lat_ok and spd_ok) else 'NOT met'} (Latency + Avg Speedup)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
