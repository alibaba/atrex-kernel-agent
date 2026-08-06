from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CAMPAIGN_MODES = ("standard", "teacher-distill")


@dataclass(frozen=True)
class TeacherDistillRequest:
    campaign_mode: str
    name: str
    op_dir: Path
    kernel_demo: Path
    atrex_bench_root: Path | None
    platform: str
    architecture: str
    framework: str
    teacher_solution: Path
    private_root: Path
    workspace_root: Path
    geomean_ratio: float
    shape_ratio: float
    stall_before_episode: int
    partial_restarts: int
    optimization_mode: str
    framework_baseline: str
    convert_after: int
    workload_bucketing: bool
    layer: bool
    notes: str
    max_iters: int
    token_budget: int
    max_stall: int
    iter_timeout: int
    setup_timeout: int
    salvage_timeout: int
    framework_baseline_timeout: int
    sandbox_hardware: str
    sandbox_profile: str
    sandbox_url: str
    sandbox_timeout: int
    agent_cli: str


def _option_present(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in argv)


def preflight_teacher_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> None:
    """Validate mode combinations before architecture probes or setup side effects."""
    teacher_options = (
        "--teacher-solution",
        "--teacher-geomean-ratio",
        "--teacher-shape-ratio",
        "--teacher-stall-before-episode",
        "--teacher-partial-restarts",
        "--teacher-private-root",
    )
    if args.campaign_mode == "standard":
        used = [option for option in teacher_options if _option_present(raw_argv, option)]
        if used:
            parser.error("Teacher options require --campaign-mode teacher-distill: " + ", ".join(used))
        return

    if not args.framework:
        parser.error("teacher-distill requires an explicit --framework")
    if not args.teacher_solution:
        parser.error("teacher-distill requires --teacher-solution DIR")
    teacher_path = Path(args.teacher_solution).expanduser()
    if not teacher_path.is_dir():
        parser.error("--teacher-solution must be an existing directory")
    if _option_present(raw_argv, "--optimization-mode") and args.optimization_mode != "production":
        parser.error("teacher-distill rejects --optimization-mode leaderboard")
    if args.layer:
        parser.error("teacher-distill does not support --layer in the first release")
    if not args.no_workload_bucketing:
        parser.error("teacher-distill requires --no-workload-bucketing in the first release")
    if args.framework_baseline == "never":
        parser.error("teacher-distill requires a framework baseline; --framework-baseline never is invalid")
    for name in ("teacher_geomean_ratio", "teacher_shape_ratio"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 1.0:
            parser.error("--%s must be finite and >= 1.0" % name.replace("_", "-"))
    if args.teacher_stall_before_episode < 1:
        parser.error("--teacher-stall-before-episode must be at least 1")
    if args.teacher_partial_restarts < 0:
        parser.error("--teacher-partial-restarts must be non-negative")

    # These implications are part of the mode contract. Explicit conflicting
    # values were rejected above rather than silently overridden.
    args.optimization_mode = "production"
    args.framework_baseline = "always"
    args.convert_after = 0


def build_request(
    args: argparse.Namespace,
    op: dict[str, Any],
    architecture: str,
    sandbox_hardware: str,
) -> TeacherDistillRequest:
    workspace_root = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd().resolve()
    private_root = (
        Path(args.teacher_private_root).expanduser().resolve()
        if args.teacher_private_root
        else workspace_root / ".atrex_teacher_campaigns"
    )
    atrex_root = op.get("atrex_bench_root") or ""
    return TeacherDistillRequest(
        campaign_mode="teacher-distill",
        name=str(op["name"]),
        op_dir=Path(op["op_dir"]).resolve(),
        kernel_demo=Path(op["reference"]).resolve(),
        atrex_bench_root=Path(atrex_root).resolve() if atrex_root else None,
        platform=args.platform,
        architecture=architecture,
        framework=args.framework,
        teacher_solution=Path(args.teacher_solution).expanduser().resolve(),
        private_root=private_root,
        workspace_root=workspace_root,
        geomean_ratio=float(args.teacher_geomean_ratio),
        shape_ratio=float(args.teacher_shape_ratio),
        stall_before_episode=int(args.teacher_stall_before_episode),
        partial_restarts=int(args.teacher_partial_restarts),
        optimization_mode="production",
        framework_baseline="always",
        convert_after=0,
        workload_bucketing=False,
        layer=False,
        notes=args.notes,
        max_iters=args.max_iters,
        token_budget=args.token_budget,
        max_stall=args.max_stall or 5,
        iter_timeout=args.iter_timeout,
        setup_timeout=args.setup_timeout,
        salvage_timeout=args.salvage_timeout,
        framework_baseline_timeout=args.framework_baseline_timeout,
        sandbox_hardware=sandbox_hardware,
        sandbox_profile=args.sandbox_profile,
        sandbox_url=args.sandbox_url,
        sandbox_timeout=args.sandbox_timeout,
        agent_cli=args.agent_cli,
    )


def run_teacher_distill(request: TeacherDistillRequest) -> int:
    """Lazy entry point so standard campaigns do not import Teacher orchestration."""
    try:
        from .campaign import TeacherDistillCampaign
    except ImportError as exc:
        raise RuntimeError(
            "teacher-distill orchestration is not installed yet; complete the campaign implementation"
        ) from exc
    return TeacherDistillCampaign(request).run_cli()
