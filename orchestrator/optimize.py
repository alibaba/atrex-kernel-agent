#!/usr/bin/env python3
"""Long-horizon episode orchestrator for atrex-kernel-agent.

Owns the OUTER optimization loop so termination no longer depends on the model's
in-session judgment (the old Stage-6 "is README's Stop Conditions met?" self-call).

Each optimization version is a long-horizon engineering episode in an isolated Git worktree.
The coding agent may profile, research, edit, validate, benchmark, repair, and checkpoint as many
times as needed before publishing one terminal handoff. The supervisor independently verifies a
candidate against the incumbent with a same-allocation ABBA schedule and squash-promotes only a
strict, correctness-passing improvement.

Termination policy
------------------
- Outer loop (this file):  HARD budget break = max versions/episodes OR token budget,
  plus a mechanical target short-circuit (peak utilization >= --target-util on a
  committed, correctness-PASS version). No convergence judge.
- Inner loop (one episode): a persistent engineering direction with structured journal and
  bounded handoff recovery, isolated from the incumbent branch.

Episode reasoning stays in the self-contained prompt under ``prompts/``;
this file only does mechanism: spawn, time-bound, token-account, read state, decide stop.

SOL and Atrex-Bench operators run the same loop over their complete workload set.

Usage
-----
    # single operator with an explicit framework:
    python orchestrator/optimize.py \
        --op-dir /path/to/operator \
        --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework CuteDSL \
        --agent-cli qodercli \
        --max-iters 20 --token-budget 8000000 --target-util 90

    # omit --framework to launch one independent campaign per supported framework:
    #   NVIDIA -> Triton, CuteDSL, Cuda
    #   AMD    -> Triton, FlyDSL
    #   other  -> Triton
    python orchestrator/optimize.py \
        --op-dir /path/to/op --platform H20 --workspace /path/to/runs

    # production: exact framework, independently reviewed external dependencies:
    python orchestrator/optimize.py \
        --op-dir /path/to/op --platform H20 --framework Triton \
        --optimization-mode production

"""
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Direct script execution and package imports must share one module identity. The episode engine
# imports ``orchestrator.optimize`` lazily; expose the repository root and bind this live module
# into the namespace package so ``python orchestrator/optimize.py`` cannot load a second copy.
if __name__ == "__main__":
    _repo_import_root = str(Path(__file__).resolve().parent.parent)
    if _repo_import_root not in sys.path:
        sys.path.insert(0, _repo_import_root)
    import orchestrator as _orchestrator_package

    sys.modules["orchestrator.optimize"] = sys.modules[__name__]
    setattr(_orchestrator_package, "optimize", sys.modules[__name__])

try:
    from . import agent_runtime as _agent_runtime
    from .campaign import Campaign
    from .constants import (
        AGENT_CLI_CHOICES,
        DEFAULT_CONVERT_AFTER,
        DEFAULT_HANDOFF_RESUMES,
        DEFAULT_SANDBOX_TIMEOUT,
        DEFAULT_VERIFY_REPEATS,
        DEFAULT_VERIFY_RUN_TIMEOUT,
        FRAMEWORK_BASELINE_MODES,
        FRAMEWORK_BASELINE_TIMEOUT_S,
        MAX_SANDBOX_TIMEOUT,
    )
    from .hardware import (
        _workspace_slug,
        framework_workspace_suffix,
        supported_frameworks,
    )
    from .optimization_policy import OPTIMIZATION_MODE_CHOICES
    from .session_io import detect_arch, ensure_submodules
    from .operator_layout import find_atrex_bench_root, is_sol_op
    from .workspace_state import latest_version, preserve_interrupted_tracked_changes
except ImportError:  # direct script execution: python orchestrator/optimize.py
    from orchestrator import agent_runtime as _agent_runtime  # type: ignore[no-redef]
    from orchestrator.campaign import Campaign  # type: ignore[no-redef]
    from orchestrator.constants import (  # type: ignore[no-redef]
        AGENT_CLI_CHOICES,
        DEFAULT_CONVERT_AFTER,
        DEFAULT_HANDOFF_RESUMES,
        DEFAULT_SANDBOX_TIMEOUT,
        DEFAULT_VERIFY_REPEATS,
        DEFAULT_VERIFY_RUN_TIMEOUT,
        FRAMEWORK_BASELINE_MODES,
        FRAMEWORK_BASELINE_TIMEOUT_S,
        MAX_SANDBOX_TIMEOUT,
    )
    from orchestrator.hardware import (  # type: ignore[no-redef]
        _workspace_slug,
        framework_workspace_suffix,
        supported_frameworks,
    )
    from orchestrator.optimization_policy import (  # type: ignore[no-redef]
        OPTIMIZATION_MODE_CHOICES,
    )
    from orchestrator.session_io import (  # type: ignore[no-redef]
        detect_arch,
        ensure_submodules,
    )
    from orchestrator.operator_layout import (  # type: ignore[no-redef]
        find_atrex_bench_root,
        is_sol_op,
    )
    from orchestrator.workspace_state import (  # type: ignore[no-redef]
        latest_version,
        preserve_interrupted_tracked_changes,
    )


def _without_cli_options(argv: list[str], option_names: tuple[str, ...]) -> list[str]:
    """Remove value-taking options from argv before adding canonical child values."""
    cleaned: list[str] = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg in option_names:
            skip_value = True
            continue
        if any(arg.startswith(name + "=") for name in option_names):
            continue
        cleaned.append(arg)
    return cleaned


def dispatch_framework_campaigns(
    argv: list[str],
    frameworks: tuple[str, ...],
    workspace_base: Path,
    arch: str,
    platform: str,
    optimization_mode: str = "leaderboard",
) -> int:
    """Launch explicit-framework optimizer children concurrently and wait for all.

    Each child receives a flat framework/hardware suffix, so its eventual
    workspace is ``<workspace_base>/kernel_opt_<op>_<framework>_<platform>``
    (with ``_production`` appended in production mode). Budgets remain per campaign; a failed
    framework does not cancel the other independent campaigns.
    """
    common_argv = _without_cli_options(
        argv, ("--framework", "--workspace", "--arch", "--workspace-suffix")
    )
    children: list[tuple[str, str, subprocess.Popen[str]]] = []
    workspace_base.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, int]] = []

    def stop_children() -> None:
        # Framework children and every coding-agent session deliberately use
        # separate process groups.  Capture the whole descendant group set
        # before signaling: once a framework coordinator exits, its Claude
        # session is reparented to PID 1 and can no longer be discovered from
        # the dispatcher, which previously left orphan optimizers behind.
        process_groups: set[int] = set()
        for _, _, proc in children:
            if proc.poll() is not None:
                continue
            try:
                process_groups.add(os.getpgid(proc.pid))
            except ProcessLookupError:
                continue
            for pid, _argv in _agent_runtime.descendant_process_commands(proc.pid):
                try:
                    process_groups.add(os.getpgid(pid))
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        for pgid in process_groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for _, _, proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }

    def interrupt_dispatch(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for handled_signal in handled_signals:
        signal.signal(handled_signal, interrupt_dispatch)
    try:
        for framework in frameworks:
            workspace_suffix = framework_workspace_suffix(
                framework, platform, optimization_mode
            )
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *common_argv,
                "--framework", framework,
                "--workspace", str(workspace_base),
                "--workspace-suffix", workspace_suffix,
            ]
            if arch:
                cmd += ["--arch", arch]
            proc = subprocess.Popen(cmd, start_new_session=True, text=True)
            children.append((framework, workspace_suffix, proc))
            print(
                f"[orchestrator] dispatched framework={framework} pid={proc.pid} "
                f"workspace_suffix={workspace_suffix} work_root={workspace_base}",
                flush=True,
            )

        # All children have already been spawned, so sequential waits do not
        # serialize their optimization work.
        for framework, _, proc in children:
            returncode = proc.wait()
            print(
                f"[orchestrator] framework={framework} finished exit={returncode}",
                flush=True,
            )
            if returncode != 0:
                failed.append((framework, returncode))
    except KeyboardInterrupt:
        print("[orchestrator] interrupt: stopping framework campaigns", file=sys.stderr, flush=True)
        stop_children()
        return 130
    except BaseException:
        stop_children()
        raise
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)

    if failed:
        summary = ", ".join(f"{name}={code}" for name, code in failed)
        print(f"[orchestrator] framework campaign failures: {summary}", file=sys.stderr, flush=True)
        return 1
    return 0


def _resolve_op(op_dir: str) -> dict:
    """Derive everything op-specific from the atrex-bench native op dir, so the CLI needs only
    --op-dir (+ the non-deducible --platform).
    """
    d = Path(op_dir).resolve()
    if not d.is_dir():
        raise SystemExit(f"--op-dir not found: {d}")
    ref = d / "reference.py"
    if not ref.is_file():
        raise SystemExit(f"--op-dir has no reference.py: {d}")
    atrex_bench_root = ""
    if not is_sol_op(d) and (d / "shapes.json").is_file():
        native_root = find_atrex_bench_root(d)
        if native_root is None:
            raise SystemExit(
                "native Atrex-Bench operator requires its canonical scripts/run_eval.py and "
                f"src/atrex_bench runtime in an ancestor directory: {d}"
            )
        atrex_bench_root = str(native_root)
    return {
        "name": d.name,
        "reference": str(ref),
        "op_dir": str(d),
        "atrex_bench_root": atrex_bench_root,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Long-horizon episode orchestrator for atrex-kernel-agent.")
    ap.add_argument("--op-dir", required=True,
                    help="The operator dir (SOL definition.json / workload.jsonl / reference.py, or "
                         "atrex-bench shapes.json / roofline.json / metadata.json / input.py / reference.py). "
                         "EVERYTHING op-specific is read from here — the workspace "
                         "name (dir basename), kernel to optimize (reference.py), and the full shape set. "
                         "Never hardcoded.")
    ap.add_argument("--platform", required=True, help="Target hardware, e.g. B200 / H20 / MI308X "
                                                      "(cannot be deduced from the op dir).")
    ap.add_argument(
        "--sandbox-hardware", default="",
        help="agate GPU scheduler token used for all tests/profiles, e.g. REMOTE_GPU. "
             "Default: --platform; set explicitly when the gateway uses a different alias.",
    )
    ap.add_argument(
        "--sandbox-profile", choices=("pre", "prod"), default="",
        help="Optional agate endpoint profile. Default: preserve AGATE_URL/config resolution.",
    )
    ap.add_argument(
        "--sandbox-url", default="",
        help="Explicit agate endpoint URL for all tests/profiles, e.g. "
             "http://127.0.0.1:8000 for `atrex-gateway serve --local`. "
             "Mutually exclusive with --sandbox-profile.",
    )
    ap.add_argument(
        "--sandbox-timeout", type=int, default=DEFAULT_SANDBOX_TIMEOUT,
        help=(
            "Per sandbox test/profile command execution timeout in seconds "
            "(1..600; queue wait is budgeted separately)."
        ),
    )
    ap.add_argument(
        "--agent-cli", choices=AGENT_CLI_CHOICES, default="claude",
        help="Coding CLI used for optimization episodes: claude, qodercli, codex, or pi "
             "(default: claude).",
    )
    ap.add_argument(
        "--discussion",
        action="store_true",
        help=(
            "Enable iterative Humanize discussion while generating each optimization plan. "
            "By default the same coding agent generates the plan itself without Humanize, "
            "Codex consultation, or a planning subagent."
        ),
    )
    ap.add_argument(
        "--optimization-mode",
        choices=OPTIMIZATION_MODE_CHOICES,
        default="leaderboard",
        help="leaderboard preserves the permissive current CLAUDE.md flow; production mechanically "
             "enforces each campaign's framework and delegates ambiguous third-party dependency "
             "provenance to an independent fail-closed Agent review.",
    )
    ap.add_argument(
        "--framework", default="",
        help="Target DSL, e.g. Triton / CuteDSL / Cuda / FlyDSL. When omitted, launch all "
             "frameworks supported by the detected hardware in parallel: NVIDIA uses "
             "Triton/CuteDSL/Cuda, AMD uses Triton/FlyDSL, and unknown hardware uses Triton. "
             "Each auto-dispatched production child is bound to its assigned framework.",
    )
    ap.add_argument("--notes", default="none", help="Extra constraints / known bottlenecks.")
    ap.add_argument("--max-iters", type=int, default=20,
                    help="Hard cap on canonical optimization versions/episodes.")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="Hard token cap across all episode turns (0 = no cap; max-iters still bounds it).")
    ap.add_argument("--target-util", type=float, default=90.0,
                    help="Peak-utilization %% short-circuit (default stop condition).")
    ap.add_argument("--iter-timeout", type=int, default=5400,
                    help="Wall-clock budget and worst-case terminal-handoff cadence for one "
                         "complete long-horizon episode (s).")
    ap.add_argument("--setup-timeout", type=int, default=7200, help="Baseline session timeout (s).")
    ap.add_argument("--handoff-resumes", type=int, default=DEFAULT_HANDOFF_RESUMES,
                    help="Same-session recovery turns after an incomplete episode handoff (default: 2).")
    ap.add_argument("--verify-repeats", type=int, default=DEFAULT_VERIFY_REPEATS,
                    help="Incumbent/candidate ABBA repeat pairs in one gateway allocation (default: 2).")
    ap.add_argument("--verify-run-timeout", type=int, default=DEFAULT_VERIFY_RUN_TIMEOUT,
                    help="Evaluator budget for each ABBA run (default: 120s; the remote driver "
                         "allows up to 60s cold-start/finalization grace within the unchanged "
                         "whole-allocation timeout).")
    ap.add_argument("--min-improvement-pct", type=float, default=0.0,
                    help="Minimum strict ABBA improvement required for promotion (default: >0%%).")
    ap.add_argument("--framework-baseline", choices=FRAMEWORK_BASELINE_MODES, default="auto",
                    help="Run one dedicated session after V0 setup that "
                         "replaces the V0 PyTorch wrapper with the first self-contained framework "
                         "kernel, recorded as v1 (so optimization episodes start at v2). Production "
                         "auto = production mode only; always = leaderboard too; "
                         "never = optimize directly from the V0 kernel.")
    ap.add_argument("--framework-baseline-timeout", type=int, default=FRAMEWORK_BASELINE_TIMEOUT_S,
                    help="Framework baseline session timeout (s).")
    ap.add_argument("--max-stall", type=int, default=0,
                    help="Optional: stop after N consecutive unpromoted episodes (0 = disabled).")
    ap.add_argument("--convert-after", type=int, default=DEFAULT_CONVERT_AFTER,
                    help="Triton only: after N consecutive stalled episodes, require conversion "
                         "to Gluon. Failed conversions retry immediately until one passes correctness "
                         "and performance parity, then optimization continues in Gluon. Default: 3; "
                         "0 disables conversion.")
    ap.add_argument("--arch", default="",
                    help="Override the real runtime GPU arch, e.g. sm_103 or gfx942. Default: auto-detect "
                         "via torch (get_device_capability / gcnArchName) — use this if auto-detect fails.")
    ap.add_argument("--workspace", default="",
                    help="Working directory for the optimization campaign. A flat "
                         "kernel_opt_<name>_<framework>_<platform>/ workspace is created under this "
                         "directory. Default: current working directory.")
    ap.add_argument("--workspace-suffix", default="", help=argparse.SUPPRESS)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = ap.parse_args(raw_argv)
    if args.workspace_suffix and args.workspace_suffix != _workspace_slug(args.workspace_suffix):
        ap.error("--workspace-suffix must be a normalized lowercase alphanumeric/underscore suffix")
    if not 1 <= args.sandbox_timeout <= MAX_SANDBOX_TIMEOUT:
        ap.error(
            "--sandbox-timeout must be in the gateway-supported range "
            f"1..{MAX_SANDBOX_TIMEOUT}"
        )
    if args.convert_after < 0:
        ap.error("--convert-after must be non-negative")
    if args.handoff_resumes < 0:
        ap.error("--handoff-resumes must be non-negative")
    if args.verify_repeats <= 0:
        ap.error("--verify-repeats must be positive")
    if args.verify_run_timeout <= 0:
        ap.error("--verify-run-timeout must be positive")
    if not 0.0 <= args.min_improvement_pct < 100.0:
        ap.error("--min-improvement-pct must be in [0, 100)")
    if args.verify_run_timeout * args.verify_repeats * 2 + 30 > args.sandbox_timeout:
        ap.error(
            "ABBA verification must fit one sandbox allocation: "
            "2 * --verify-repeats * --verify-run-timeout + 30 <= --sandbox-timeout"
        )
    if args.framework_baseline_timeout <= 0:
        ap.error("--framework-baseline-timeout must be positive")
    if args.sandbox_url and args.sandbox_profile:
        ap.error("--sandbox-url and --sandbox-profile are mutually exclusive")
    if shutil.which(args.agent_cli) is None:
        ap.error(f"--agent-cli executable not found on PATH: {args.agent_cli}")
    if args.agent_cli == "codex":
        codex_settings = (
            os.environ.get("ATREX_CODEX_SESSION_SETTINGS")
            or os.environ.get("ATREX_SESSION_SETTINGS")
            or ""
        )
        try:
            _agent_runtime.codex_settings_args(codex_settings)
        except ValueError as exc:
            ap.error(str(exc))
    if args.agent_cli == "pi":
        pi_settings = (
            os.environ.get("ATREX_PI_SESSION_SETTINGS")
            or os.environ.get("ATREX_SESSION_SETTINGS")
            or ""
        )
        try:
            _agent_runtime.pi_settings_args(pi_settings)
        except ValueError as exc:
            ap.error(str(exc))
    if args.agent_cli == "qodercli" and args.token_budget > 0:
        print(
            "[orchestrator] WARNING: qodercli token-budget enforcement depends on token usage "
            "reported in stream-json; some Qoder models report zero, so --max-iters remains "
            "the authoritative hard bound in that configuration.",
            file=sys.stderr,
            flush=True,
        )
    sandbox_hardware = args.sandbox_hardware or args.platform
    if args.workspace:
        Path(args.workspace).mkdir(parents=True, exist_ok=True)

    arch = args.arch or detect_arch(
        sandbox_hardware, args.sandbox_profile, args.sandbox_url
    )
    op = _resolve_op(args.op_dir)
    ensure_submodules()
    frameworks = (args.framework,) if args.framework else supported_frameworks(args.platform, arch)
    print(f"[orchestrator] op={op['name']} agent_cli={args.agent_cli} "
          f"optimization_mode={args.optimization_mode} platform={args.platform} "
          f"discussion={args.discussion} "
          f"sandbox_hardware={sandbox_hardware} "
          f"sandbox_endpoint={args.sandbox_url or args.sandbox_profile or 'agate-config'} "
          f"frameworks={','.join(frameworks)} "
          "runtime_arch="
          f"{arch or 'UNKNOWN (detect failed)'} "
          f"(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)

    if not args.framework:
        base = Path(args.workspace).resolve() if args.workspace else Path.cwd()
        return dispatch_framework_campaigns(
            raw_argv,
            frameworks,
            base,
            arch,
            args.platform,
            args.optimization_mode,
        )

    workspace_suffix = args.workspace_suffix or framework_workspace_suffix(
        args.framework, args.platform, args.optimization_mode
    )

    campaign = Campaign(
        name=op["name"], kernel_demo=op["reference"], platform=args.platform,
        framework=args.framework, notes=args.notes, arch=arch,
        sandbox_hardware=sandbox_hardware, sandbox_profile=args.sandbox_profile,
        sandbox_url=args.sandbox_url,
        sandbox_timeout=args.sandbox_timeout,
        atrex_bench_root=op.get("atrex_bench_root", ""),
        agent_cli=args.agent_cli,
        discussion=args.discussion,
        optimization_mode=args.optimization_mode,
        work_dir=args.workspace,
        workspace_suffix=workspace_suffix,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
        framework_baseline=args.framework_baseline,
        framework_baseline_timeout=args.framework_baseline_timeout,
        handoff_resumes=args.handoff_resumes,
        verify_repeats=args.verify_repeats,
        verify_run_timeout=args.verify_run_timeout,
        min_improvement_pct=args.min_improvement_pct,
        convert_after=args.convert_after,
    )
    if latest_version(campaign.workspace) < 0:
        campaign.setup_baseline()
    else:
        print(
            f"[orchestrator] resuming workspace at v{latest_version(campaign.workspace)}",
            flush=True,
        )
        preserve_interrupted_tracked_changes(campaign.workspace, "resume workspace")
        campaign._link_runtime()
    campaign.ensure_framework_baseline()
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
