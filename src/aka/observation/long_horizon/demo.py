"""Sample workspace records for an explicitly selected dashboard preview."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .reader import PHASES


def create_demo_workspace(root: Path) -> Path:
    """Create sample files using the same schema consumed by the real observer."""
    def now():
        return datetime.now(timezone.utc).isoformat()

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def tokens(total, measurement="exact"):
        return {"input_tokens": int(total * .8), "output_tokens": total - int(total * .8),
                "cache_read_tokens": int(total * .4), "cache_write_tokens": 0,
                "total_tokens": total, "measurement": measurement}

    def memory(path, n, score, *, accepted=True, baseline=False):
        write(path / "memory" / f"v{n}.json", {
            "version": f"v{n}", "timestamp": now(),
            "correctness": {"status": "PASS"},
            "quality_gate": {"result": "PASS" if accepted or baseline else "FAIL"},
            "performance": {"performance_objective": "shape_speedup_arithmetic_mean",
                            "performance_score": score, "latency_us_geomean": 10 / score,
                            "latency_us_by_shape": {"sample-a": 10 / score, "sample-b": 10 / score},
                            "measurement_status": "complete", "expected_shape_count": 2,
                            "measurement_subject": "candidate" if accepted else "incumbent"}})

    def telemetry(path, episode, n, total, partial=False):
        names = (*PHASES, "orchestration", "unattributed")
        fractions = (.10, .08, .12, .35, .12, .16, .05, .01)
        amounts = [int(total * fraction) for fraction in fractions]
        amounts.append(total - sum(amounts))
        phase_usage = dict(zip(names, amounts))
        reasons = ["same_session_resume_usage_semantics_unqualified"] if partial else []
        write(path / ".atrex_long_horizon" / "episodes" / f"e{episode:04d}" / "telemetry.summary.json", {
            "schema_version": "atrex_long_horizon_episode_telemetry_v1", "episode": episode,
            "version": f"v{n}", "control_tokens": total, "measurement": "partial" if partial else "exact",
            "reason_codes": reasons,
            "phase_tokens": {"terminal_usage": tokens(total),
                             "phases": {phase: {"usage": tokens(phase_usage[phase])} for phase in PHASES},
                             "orchestration": tokens(phase_usage["orchestration"]),
                             "unattributed": tokens(phase_usage["unattributed"]),
                             "measurement": "partial" if partial else "exact", "reason_codes": reasons}})

    cuda = root / "kernel_opt_demo_attention_cuda"
    write(cuda / "framework_baseline.json", {"version": "v1", "framework": "Cuda", "platform": "H20"})
    memory(cuda, 0, 1, baseline=True); memory(cuda, 1, 1.05, baseline=True)
    sample_rows = (
        ("fast", "candidate_ready", True, 120000, 1.12, "Vectorize contiguous memory loads"),
        ("fast", "candidate_ready", False, 160000, 1.12, "Larger tiles did not improve latency"),
        ("full", "candidate_ready", True, 190000, 1.31, "Fuse normalization with the reduction"),
        ("full", "pivot", False, 200000, 1.31, "Switch from shared-memory staging to register reuse"),
        ("full", "candidate_ready", True, 180000, 1.49, "Reduce synchronization between warps"),
        ("full", "candidate_ready", False, 141000, 1.49, "Extra unrolling increased register pressure"),
        ("full", "candidate_ready", True, 190000, 1.54, "Specialize the vectorized tail path"),
        ("goal", "blocked", False, 99000, 1.54, "Target requires a different launch strategy"),
    )
    attempts = []
    for episode, (mode, status, accepted, total, score, summary) in enumerate(sample_rows, 1):
        n = episode + 1
        verification = ({"gate": "PASS" if accepted else "FAIL", "candidate_latency_us": 10 / score,
                         "incumbent_latency_us": 10 / (sample_rows[episode - 2][4] if episode > 1 else 1.05),
                         "error": None} if status == "candidate_ready" else None)
        attempt = {"episode": episode, "version": n, "mode": mode, "status": status,
                   "accepted": accepted, "tokens": total, "summary": summary, "verification": verification}
        attempts.append(attempt)
        directory = cuda / ".atrex_long_horizon" / "episodes" / f"e{episode:04d}"
        write(directory / "attempt.json", attempt)
        write(directory / "episode_runtime/journal.json", {
            "episode": episode, "memory_version": n, "outcome": {"summary": summary},
            "experiments": [{"name": summary, "result": "Passed correctness checks" if accepted else "Candidate preserved for analysis"}]})
        telemetry(cuda, episode, n, total, partial=episode == 8)
        memory(cuda, n, score, accepted=accepted)
    write(cuda / ".atrex_long_horizon/state.json", {
        "episodes": 8, "accepted": 4, "rejected": 2, "pivoted": 1, "blocked": 1,
        "interrupted": 0, "protocol_failures": 0, "tokens": sum(a["tokens"] for a in attempts),
        "usage_receipts": {f"demo-invocation-{a['episode']}": a["tokens"] for a in attempts}, "attempts": attempts})
    current = [{"name": f"Pipeline experiment {n}", "result": "Measured one launch configuration"} for n in range(1, 5)]
    current.append({"name": "Overlap memory loads with computation", "result": "Candidate ready for the next evaluator pass"})
    worktree = root / "demo-active-worktree"
    write(worktree / ".atrex_long_horizon/journal.json", {"episode": 9, "memory_version": 10, "experiments": current})
    write(cuda / ".atrex_long_horizon/active_episode.json", {
        "episode": 9, "memory_version": 10, "mode": "full", "phase": "exploring", "worktree": str(worktree)})
    write(cuda / "memory/live.json", {"episode": 9, "version": "v10", "phase": "exploring",
                                     "experiment_count": len(current), "latest_experiment": current[-1], "updated_at": now()})

    triton = root / "kernel_opt_demo_attention_triton"
    write(triton / "framework_baseline.json", {"version": "v1", "framework": "Triton", "platform": "H20"})
    memory(triton, 0, 1, baseline=True); memory(triton, 1, 1.08, baseline=True)
    attempt = {"episode": 1, "version": 2, "mode": "fast", "status": "blocked", "accepted": False,
               "tokens": 0, "summary": "Awaiting a backend with observable token usage"}
    write(triton / ".atrex_long_horizon/episodes/e0001/attempt.json", attempt)
    write(triton / ".atrex_long_horizon/episodes/e0001/telemetry.summary.json", {
        "episode": 1, "version": "v2", "control_tokens": 0,
        "reason_codes": ["backend_usage_unavailable"], "measurement": "unavailable",
        "phase_tokens": {"terminal_usage": {"total_tokens": None, "measurement": "unavailable"},
                         "measurement": "unavailable"}})
    write(triton / ".atrex_long_horizon/state.json", {"episodes": 1, "accepted": 0, "rejected": 0,
                                                     "blocked": 1, "tokens": 0, "attempts": [attempt]})
    write(triton / "memory/live.json", {"episode": 1, "version": "v2", "phase": "recorded", "updated_at": now()})
    return root
