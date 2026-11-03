"""Standalone scalar-metric diagnosis shared by NCU report and CSV frontends."""

import math

PATTERNS = {
    "A": ("small-grid", "Small grid / SM idle", "low-sm-utilization"),
    "B": ("tail-effect", "Tail effect (wave quantization)", "tail-effect"),
    "C": ("uncoalesced-loads", "Uncoalesced global loads", "memory-bound"),
    "D": ("sparse-writes", "Sparse writes (low store efficiency)", "memory-bound"),
    "E": ("latency-bound", "Latency-bound (long-scoreboard)", "memory-bound"),
    "F": ("compute-no-tensor", "Compute-bound but not on tensor cores", "compute-bound"),
    "G": ("atomics-contention", "Atomics contention", "compute-bound"),
    "H": ("bank-conflicts", "Shared-memory bank conflicts", "memory-bound"),
    "I": ("sync-overhead", "Synchronization overhead", "pipeline-stalls"),
    "J": ("low-achieved-occupancy", "Low achieved vs theoretical occupancy", "low-sm-utilization"),
    "K": ("register-spill", "Register spill", "register-pressure"),
    "L": ("unintended-fp64", "FP64 used unintentionally", "compute-bound"),
    "M": ("pipeline-bubbles", "Pipeline bubbles (no compute/memory overlap)", "pipeline-stalls"),
    "N": ("warp-divergence", "Warp divergence", "compute-bound"),
    # Supplementary roofline classification, not one of the 14 lettered
    # diagnostic Patterns (so it is excluded from the profile_guide.md playbook
    # pointer below); kept in PATTERNS so its symptom flows into SYMPTOMS/wiki.
    "_mem": ("overall-memory-bound", "Overall memory-bound", "memory-bound"),
}


def _get(metrics, key, default=None):
    v = metrics.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def classify(metrics):
    """Infer matching Pattern list from the metrics JSON.

    Returns list of (pattern_id, label, confidence, evidence).
    """
    findings = []

    # Pattern A — Small grid / SM idle
    waves = _get(metrics, "launch__waves_per_multiprocessor")
    grid = _get(metrics, "launch__grid_size")
    sm_count = _get(metrics, "device__attribute_multiprocessor_count")
    if waves is not None and waves < 0.5:
        findings.append(
            ("A", PATTERNS["A"][1], "high", f"waves_per_multiprocessor={waves:.2f} < 0.5")
        )
    elif grid is not None and sm_count is not None and grid < sm_count:
        findings.append(
            ("A", PATTERNS["A"][1], "medium", f"grid_size={int(grid)} < sm_count={int(sm_count)}")
        )

    # Pattern B — Tail effect / wave quantization.
    # The classic tail (a partial last wave wasting SMs) is detectable from the
    # static launch geometry we already collect: with W waves per SM, execution
    # rounds up to ceil(W) waves, so (ceil(W) - W) / ceil(W) of the last wave is
    # idle. Flag when that idle fraction is significant and W > 1 (W < 1 is the
    # small-grid case owned by Pattern A). The harder data-dependent flavor
    # (variable-length inputs -> per-block duration imbalance) needs per-block /
    # PM timeline data and is not auto-detected here.
    if waves is not None and waves > 1.0:
        full_waves = math.ceil(waves)
        tail_idle = (full_waves - waves) / full_waves
        if tail_idle > 0.15:
            findings.append(
                (
                    "B",
                    PATTERNS["B"][1],
                    "medium",
                    f"waves_per_multiprocessor={waves:.2f} -> last wave "
                    f"~{tail_idle * 100:.0f}% idle (rounds up to "
                    f"{full_waves} waves)",
                )
            )

    # Pattern C — Uncoalesced global loads
    sectors = _get(metrics, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    requests = _get(metrics, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    if sectors is not None and requests is not None and requests > 0:
        ratio = sectors / requests
        if ratio > 5:
            findings.append(
                ("C", PATTERNS["C"][1], "high", f"sectors/requests={ratio:.1f} > 5 (ideal=4)")
            )

    # Pattern D — Sparse writes
    st_bytes_ratio = _get(
        metrics, "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio"
    )
    if st_bytes_ratio is not None and st_bytes_ratio < 16:
        findings.append(
            (
                "D",
                PATTERNS["D"][1],
                "medium",
                f"bytes_per_sector_st={st_bytes_ratio:.1f} < 16 (ideal=32)",
            )
        )

    # Pattern E — Latency-bound (long-scoreboard)
    long_sb = _get(
        metrics, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    )
    dram_rd_pct = _get(metrics, "dram__bytes_read.sum.pct_of_peak_sustained_elapsed")
    if long_sb is not None and long_sb > 3:
        if dram_rd_pct is not None and dram_rd_pct < 10:
            findings.append(
                (
                    "E",
                    PATTERNS["E"][1],
                    "high",
                    f"long_scoreboard_ratio={long_sb:.1f} > 3, "
                    f"dram_rd_pct={dram_rd_pct:.1f}% < 10% → latency-bound, not BW-bound",
                )
            )
        else:
            findings.append(
                ("E", PATTERNS["E"][1], "medium", f"long_scoreboard_ratio={long_sb:.1f} > 3")
            )

    # Pattern F — Compute-bound but not on tensor cores
    fma_pct = _get(metrics, "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active")
    tc_pct = _get(metrics, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    if fma_pct is not None and fma_pct > 50 and (tc_pct is None or tc_pct < 1):
        findings.append(
            (
                "F",
                PATTERNS["F"][1],
                "high",
                f"FMA_pipe={fma_pct:.1f}% > 50%, tensor_core={tc_pct or 0:.1f}% ≈ 0",
            )
        )

    # Pattern G — Atomics contention.
    # No direct "contention" counter exists in a static snapshot, but we can flag
    # the signature: substantial atomic/reduction traffic together with a
    # throttled LSU pipe (lg_throttle), which is what atomics serializing on the
    # memory system look like. Low confidence — confirm with per-PC stalls
    # (extract_stall_hotspots.py) before acting.
    atom = _get(metrics, "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum", 0)
    red = _get(metrics, "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum", 0)
    lg_throttle = _get(
        metrics, "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"
    )
    if (atom + red) > 0 and lg_throttle is not None and lg_throttle > 1.0:
        findings.append(
            (
                "G",
                PATTERNS["G"][1],
                "low",
                f"atomic+red sectors={int(atom + red)}, lg_throttle="
                f"{lg_throttle:.1f} (possible atomics contention — confirm)",
            )
        )

    # Pattern H — Shared-memory bank conflicts
    short_sb = _get(
        metrics, "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
    )
    # Low confidence: short_scoreboard stalls also come from MIO/L1 pressure and
    # non-shared LSU latency, so this is a hint, not a verdict. Confirm with the
    # l1tex bank-conflict counters before acting on it.
    if short_sb is not None and short_sb > 2:
        findings.append(
            (
                "H",
                PATTERNS["H"][1],
                "low",
                f"short_scoreboard_ratio={short_sb:.1f} > 2 "
                f"(heuristic — confirm with l1tex bank-conflict counters)",
            )
        )

    # Pattern I — Synchronization overhead
    barrier = _get(metrics, "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio")
    if barrier is not None and barrier > 2:
        findings.append(("I", PATTERNS["I"][1], "medium", f"barrier_stall_ratio={barrier:.1f} > 2"))

    # Pattern J — Low achieved vs theoretical occupancy
    theoretical_occ = _get(metrics, "sm__maximum_warps_per_active_cycle_pct")
    achieved_occ = _get(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if (
        theoretical_occ is not None
        and achieved_occ is not None
        and theoretical_occ > 50
        and achieved_occ < theoretical_occ * 0.5
    ):
        findings.append(
            (
                "J",
                PATTERNS["J"][1],
                "medium",
                f"theoretical_occ={theoretical_occ:.1f}%, "
                f"achieved_occ={achieved_occ:.1f}% (gap > 50%)",
            )
        )

    # Pattern K — Register spill
    local_ld = _get(metrics, "smsp__sass_inst_executed_op_local_ld.sum", 0)
    local_st = _get(metrics, "smsp__sass_inst_executed_op_local_st.sum", 0)
    regs = _get(metrics, "launch__registers_per_thread")
    if local_ld > 0 or local_st > 0:
        findings.append(
            (
                "K",
                PATTERNS["K"][1],
                "high",
                f"local_ld={int(local_ld)}, local_st={int(local_st)}, "
                f"regs/thread={int(regs) if regs else '?'}",
            )
        )
    elif regs is not None and regs > 128:
        findings.append(
            ("K", PATTERNS["K"][1], "medium", f"regs/thread={int(regs)} > 128 (spill risk)")
        )

    # Pattern L — FP64 unintentional
    # fp64 pipe metric name varies across architectures; trying common names
    fp64 = _get(metrics, "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active")
    # >0% alone over-reports: incidental FP64 (address math, int64 ops, double
    # literals) lights the pipe at a fraction of a percent. Require a meaningful
    # share before flagging unintended FP64.
    if fp64 is not None and fp64 > 1.0:
        findings.append(
            (
                "L",
                PATTERNS["L"][1],
                "medium",
                f"fp64_pipe_active={fp64:.1f}% > 1% in non-FP64 kernel",
            )
        )

    # Pattern M — Pipeline bubbles
    # Requires PM timeline data for precise determination; using rough heuristic here:
    # If long_scoreboard is high and DRAM throughput is also high, load and compute are not overlapping
    if long_sb is not None and long_sb > 2 and dram_rd_pct is not None and dram_rd_pct > 30:
        findings.append(
            (
                "M",
                PATTERNS["M"][1],
                "medium",
                f"long_scoreboard={long_sb:.1f} + dram_rd_pct={dram_rd_pct:.1f}% "
                f"→ load/compute possibly not overlapping",
            )
        )

    # Pattern N — Warp divergence.
    # smsp__thread_inst_executed_per_inst_executed.ratio = average active threads
    # per warp-instruction (32 = fully converged). Sustained values well below 32
    # mean lanes are masked off by divergent branches / predication. Threshold at
    # 24 (75% lane utilization); note tail blocks and boundary masking can also
    # lower this, so treat as medium confidence.
    active_per_warp = _get(metrics, "smsp__thread_inst_executed_per_inst_executed.ratio")
    if active_per_warp is not None and active_per_warp < 24:
        findings.append(
            (
                "N",
                PATTERNS["N"][1],
                "medium",
                f"active_threads_per_warp={active_per_warp:.1f} < 24 "
                f"(of 32 → {active_per_warp / 32 * 100:.0f}% lane utilization)",
            )
        )

    # Supplementary: overall memory-bound vs compute-bound classification
    sm_throughput = _get(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem_throughput = _get(
        metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    )
    if (
        sm_throughput is not None
        and mem_throughput is not None
        and mem_throughput > sm_throughput * 1.5
        and mem_throughput > 30
    ):
        already_has_mem = any(PATTERNS[f[0]][2] == "memory-bound" for f in findings)
        if not already_has_mem:
            findings.append(
                (
                    "_mem",
                    PATTERNS["_mem"][1],
                    "medium",
                    f"mem_throughput={mem_throughput:.1f}% >> sm_throughput={sm_throughput:.1f}%",
                )
            )

    return findings
