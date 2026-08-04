from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from orchestrator import optimize as base

from .campaign import LongHorizonCampaign
from .session import LongSessionRunner
from .verifier import GatewayABBAValidator


@dataclass(frozen=True)
class LongHorizonOptions:
    handoff_resumes: int = 2
    verify_repeats: int = 2
    verify_run_timeout: int = 120
    min_improvement_pct: float = 0.0

    def child_args(self) -> list[str]:
        return [
            "--handoff-resumes", str(self.handoff_resumes),
            "--verify-repeats", str(self.verify_repeats),
            "--verify-run-timeout", str(self.verify_run_timeout),
            "--min-improvement-pct", str(self.min_improvement_pct),
        ]


def build_long_horizon_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("long-horizon episode options")
    group.add_argument(
        "--handoff-resumes",
        type=int,
        default=2,
        help="Same-session Claude resume attempts when the terminal handoff is incomplete.",
    )
    group.add_argument(
        "--verify-repeats",
        type=int,
        default=2,
        help="Incumbent/candidate ABBA repeat pairs in one gateway allocation.",
    )
    group.add_argument(
        "--verify-run-timeout",
        type=int,
        default=120,
        help="Timeout for each evaluator run inside the ABBA allocation.",
    )
    group.add_argument(
        "--min-improvement-pct",
        type=float,
        default=0.0,
        help="Minimum strict candidate improvement required for squash promotion.",
    )
    return parser


def _extract_options(argv: list[str]) -> tuple[LongHorizonOptions, list[str]]:
    parser = build_long_horizon_parser()
    values, remaining = parser.parse_known_args(argv)
    if values.handoff_resumes < 0:
        parser.error("--handoff-resumes must be non-negative")
    if values.verify_repeats <= 0:
        parser.error("--verify-repeats must be positive")
    if values.verify_run_timeout <= 0:
        parser.error("--verify-run-timeout must be positive")
    if values.min_improvement_pct < 0.0:
        parser.error("--min-improvement-pct must be non-negative")
    return LongHorizonOptions(**vars(values)), remaining


def _verifier(campaign: base.Campaign, options: LongHorizonOptions) -> GatewayABBAValidator:
    return GatewayABBAValidator(
        hardware=campaign.sandbox_hardware,
        profile=campaign.sandbox_profile,
        url=campaign.sandbox_url,
        timeout=campaign.sandbox_timeout,
        repeats=options.verify_repeats,
        per_run_timeout=options.verify_run_timeout,
        min_improvement_pct=options.min_improvement_pct,
    )


def _run_campaign(campaign: base.Campaign, options: LongHorizonOptions) -> str:
    long_campaign = LongHorizonCampaign(
        base_campaign=campaign,
        max_version=campaign.max_iters,
        token_budget=campaign.token_budget,
        session_timeout=campaign.iter_timeout,
        handoff_resumes=options.handoff_resumes,
        max_stall=campaign.max_stall,
        verifier=_verifier(campaign, options),
        session_runner=LongSessionRunner(agent_cli=campaign.agent_cli),
    )
    reason = long_campaign.run()
    return campaign._finish(reason)


def _boundary_campaign(layer: base.LayerCampaign, boundary: dict) -> base.Campaign:
    name = f"{layer.name}__{boundary['name']}"
    return base.Campaign(
        name=name,
        kernel_demo=str(layer._boundary_ws(boundary["name"]) / "kernel.py"),
        platform=layer.platform,
        framework=layer.framework,
        notes=layer.notes,
        arch=layer.arch,
        work_dir=layer.work_dir,
        workspace_suffix=layer.workspace_suffix,
        max_iters=layer.max_iters,
        token_budget=0,
        iter_timeout=layer.iter_timeout,
        setup_timeout=layer.setup_timeout,
        max_stall=0,
        convert_after=0,
        sandbox_hardware=layer.sandbox_hardware,
        sandbox_profile=layer.sandbox_profile,
        sandbox_url=layer.sandbox_url,
        sandbox_timeout=layer.sandbox_timeout,
        agent_cli=layer.agent_cli,
        optimization_mode=layer.optimization_mode,
        framework_baseline="never",
    )


def _run_layer_schedule(
    layer: base.LayerCampaign,
    boundaries: list[dict],
    options: LongHorizonOptions,
) -> str | None:
    """Keep main's ROI scheduler; replace only its fresh one-cycle session."""
    while True:
        for boundary in boundaries:
            workspace = layer._boundary_ws(boundary["name"])
            base.mask_half_memory(workspace, base.latest_version(workspace))
        spent = layer._total_versions(boundaries)
        if spent >= layer.max_iters:
            return "budget: max-iters (Σ versions)"
        if layer.budget_exhausted():
            return "budget: token-budget"
        if layer._all_plateaued(boundaries):
            return "all boundaries plateaued"

        ranked = sorted(boundaries, key=layer._priority, reverse=True)
        target = ranked[0]
        if layer._priority(target) <= 0.0:
            return "all boundaries at/above ceiling"

        campaign = _boundary_campaign(layer, target)
        current_version = base.latest_version(campaign.workspace)
        print(
            f"[layer] long-horizon round {spent + 1}/{layer.max_iters} -> "
            f"{target['name']} v{current_version + 1} "
            f"(priority={layer._priority(target):.4g})",
            flush=True,
        )
        state_path = campaign.workspace / ".atrex_long_horizon" / "state.json"
        before_tokens = 0
        if state_path.is_file():
            try:
                before_tokens = int(json.loads(state_path.read_text()).get("tokens", 0))
            except (OSError, ValueError, TypeError):
                before_tokens = 0
        LongHorizonCampaign(
            base_campaign=campaign,
            max_version=current_version + 1,
            episode_limit=1,
            token_budget=0,
            session_timeout=layer.iter_timeout,
            handoff_resumes=options.handoff_resumes,
            verifier=_verifier(campaign, options),
            session_runner=LongSessionRunner(agent_cli=layer.agent_cli),
        ).run()
        after_tokens = before_tokens
        if state_path.is_file():
            try:
                after_tokens = int(json.loads(state_path.read_text()).get("tokens", 0))
            except (OSError, ValueError, TypeError):
                pass
        layer.tokens_spent += max(0, after_tokens - before_tokens)


@contextmanager
def _install_main_integration(options: LongHorizonOptions) -> Iterator[None]:
    original_campaign_run = base.Campaign.run
    original_layer_schedule = base.LayerCampaign.schedule
    original_dispatch = base.dispatch_framework_campaigns

    def campaign_run(campaign: base.Campaign) -> str:
        return _run_campaign(campaign, options)

    def layer_schedule(layer: base.LayerCampaign, boundaries: list[dict]) -> str | None:
        return _run_layer_schedule(layer, boundaries, options)

    def dispatch(argv, frameworks, workspace_base, arch, platform, optimization_mode="leaderboard"):
        # Main's dispatcher is retained in full. Point only its child entry path at
        # this wrapper and forward the long-horizon-only flags it does not parse.
        original_file = base.__file__
        try:
            base.__file__ = str(Path(__file__).with_name("__main__.py"))
            return original_dispatch(
                [*argv, *options.child_args()],
                frameworks,
                workspace_base,
                arch,
                platform,
                optimization_mode,
            )
        finally:
            base.__file__ = original_file

    base.Campaign.run = campaign_run
    base.LayerCampaign.schedule = layer_schedule
    base.dispatch_framework_campaigns = dispatch
    try:
        yield
    finally:
        base.Campaign.run = original_campaign_run
        base.LayerCampaign.schedule = original_layer_schedule
        base.dispatch_framework_campaigns = original_dispatch


def _print_long_help() -> None:
    print("\nLong-horizon additions (all other options are inherited from optimize.py):")
    build_long_horizon_parser().print_help()
    print(
        "\nLong-horizon semantics: --max-iters caps canonical memory versions, "
        "--iter-timeout bounds one complete episode, and --token-budget/--max-stall "
        "retain their main meanings."
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    options, main_argv = _extract_options(raw_argv)
    wants_help = "-h" in main_argv or "--help" in main_argv
    try:
        with _install_main_integration(options):
            return base.main(main_argv)
    except SystemExit as exc:
        if wants_help and exc.code == 0:
            _print_long_help()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
