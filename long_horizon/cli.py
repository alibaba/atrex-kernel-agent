from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import main_adapter
from .campaign import LongHorizonCampaign
from .verifier import GatewayABBAValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone long-horizon episode supervisor for Atrex Kernel Agent."
    )
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--sandbox-hardware", default="")
    parser.add_argument("--sandbox-profile", choices=("pre", "prod"), default="")
    parser.add_argument("--sandbox-url", default="")
    parser.add_argument(
        "--sandbox-timeout", type=int, default=main_adapter.DEFAULT_SANDBOX_TIMEOUT
    )
    parser.add_argument("--agent-cli", choices=main_adapter.AGENT_CLI_CHOICES, default="claude")
    parser.add_argument(
        "--optimization-mode",
        choices=main_adapter.OPTIMIZATION_MODE_CHOICES,
        default="leaderboard",
    )
    parser.add_argument("--notes", default="none")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--session-timeout", type=int, default=18_000)
    parser.add_argument("--setup-timeout", type=int, default=7200)
    parser.add_argument("--handoff-resumes", type=int, default=2)
    parser.add_argument("--max-stall", type=int, default=0)
    parser.add_argument("--verify-repeats", type=int, default=2)
    parser.add_argument("--verify-run-timeout", type=int, default=120)
    parser.add_argument("--min-improvement-pct", type=float, default=0.0)
    parser.add_argument(
        "--framework-baseline",
        choices=main_adapter.FRAMEWORK_BASELINE_MODES,
        default="auto",
    )
    parser.add_argument("--framework-baseline-timeout", type=int, default=10_800)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sandbox_url and args.sandbox_profile:
        parser.error("--sandbox-url and --sandbox-profile are mutually exclusive")
    if not 1 <= args.sandbox_timeout <= main_adapter.MAX_SANDBOX_TIMEOUT:
        parser.error(
            f"--sandbox-timeout must be in 1..{main_adapter.MAX_SANDBOX_TIMEOUT}"
        )
    for name in (
        "max_episodes", "session_timeout", "setup_timeout", "verify_repeats",
        "verify_run_timeout", "framework_baseline_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.handoff_resumes < 0 or args.max_stall < 0 or args.token_budget < 0:
        parser.error("resume/stall/token budgets must be non-negative")
    if args.min_improvement_pct < 0.0:
        parser.error("--min-improvement-pct must be non-negative")
    if shutil.which(args.agent_cli) is None:
        parser.error(f"agent executable not found: {args.agent_cli}")
    if args.workspace:
        Path(args.workspace).mkdir(parents=True, exist_ok=True)
    main_adapter.ensure_submodules()
    operator = main_adapter.resolve_operator(args.op_dir)
    hardware = args.sandbox_hardware or args.platform
    arch = args.arch or main_adapter.detect_arch(
        hardware, args.sandbox_profile, args.sandbox_url
    )
    suffix = main_adapter.framework_workspace_suffix(
        args.framework, args.platform, args.optimization_mode
    )
    base_campaign = main_adapter.Campaign(
        name=operator.name,
        kernel_demo=operator.reference,
        platform=args.platform,
        framework=args.framework,
        notes=args.notes,
        arch=arch,
        sandbox_hardware=hardware,
        sandbox_profile=args.sandbox_profile,
        sandbox_url=args.sandbox_url,
        sandbox_timeout=args.sandbox_timeout,
        atrex_bench_root=operator.atrex_bench_root,
        agent_cli=args.agent_cli,
        optimization_mode=args.optimization_mode,
        work_dir=args.workspace,
        workspace_suffix=suffix,
        setup_timeout=args.setup_timeout,
        framework_baseline=args.framework_baseline,
        framework_baseline_timeout=args.framework_baseline_timeout,
    )
    verifier = GatewayABBAValidator(
        hardware=hardware,
        profile=args.sandbox_profile,
        url=args.sandbox_url,
        timeout=args.sandbox_timeout,
        repeats=args.verify_repeats,
        per_run_timeout=args.verify_run_timeout,
        min_improvement_pct=args.min_improvement_pct,
    )
    campaign = LongHorizonCampaign(
        base_campaign=base_campaign,
        max_episodes=args.max_episodes,
        token_budget=args.token_budget,
        session_timeout=args.session_timeout,
        handoff_resumes=args.handoff_resumes,
        max_stall=args.max_stall,
        verifier=verifier,
    )
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
