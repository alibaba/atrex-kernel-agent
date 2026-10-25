"""Translate the historical optimization CLI before entering Bootstrap."""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from aka.legacy.workflow.constants import AGENT_CLI_CHOICES, DEFAULT_CONVERT_AFTER, DEFAULT_FAST_EPISODES, DEFAULT_FAST_TRIALS, DEFAULT_HANDOFF_RESUMES, DEFAULT_VERIFY_REPEATS, DEFAULT_VERIFY_RUN_TIMEOUT, FRAMEWORK_BASELINE_MODES, FRAMEWORK_BASELINE_TIMEOUT_S, GATEWAY_CANDIDATE_RECOVERY_MODES
from aka.legacy.execution.recovery import DEFAULT_SSH_HEALTH_COMMAND
from aka.legacy.task.hardware import _workspace_slug
from aka.legacy.workflow.policy import OPTIMIZATION_MODE_CHOICES
from aka.legacy.agent.adapter import codex_settings_args, pi_settings_args
from aka.legacy.workflow.invocation import WorkflowLaunch, encode_launch
from .cli_problem import problem_request
from .cli_request import campaign_request


def parse_arguments(argv):
    ap = argparse.ArgumentParser(
        description="Long-horizon episode orchestrator for atrex-kernel-agent."
    )
    ap.add_argument("--conversion-budget-action", choices=("fail", "retain_verified"), default="fail")
    ap.add_argument("--resume-framework-candidate", action="store_true",
                    help="Validate a retained framework candidate before generating another implementation")
    ap.add_argument("--verify-shape-batch-size", dest="shape_batch_size", type=int, default=4,
                    help="Maximum workload shapes in one verification allocation")
    ap.add_argument(
        "--op-dir",
        required=True,
        help=(
            "The operator dir (SOL definition.json/workload.jsonl/reference.py, or native "
            "Atrex-Bench reference.py/input.py/shapes.json with optional agent_problem.json). "
            "Production always optimizes from a generalized public contract: a provided "
            "agent_problem.json is used directly, otherwise AKA derives one from the detailed "
            "shapes before hiding them from optimization sessions."
        ),
    )
    ap.add_argument(
        "--platform",
        required=True,
        help="Target hardware, e.g. B200 / H20 / MI308X "
        "(cannot be deduced from the op dir).",
    )
    ap.add_argument(
        "--sandbox-hardware",
        default="",
        help="Remote GPU hardware token used for all tests/profiles, e.g. REMOTE_GPU. "
        "Default: --platform; set explicitly when the executor uses a different alias.",
    )
    ap.add_argument(
        "--sandbox-profile",
        choices=("pre", "prod"),
        default="",
        help="Optional agate endpoint profile. Default: preserve AGATE_URL/config resolution.",
    )
    ap.add_argument(
        "--sandbox-url",
        default="",
        help="Explicit agate endpoint URL for all tests/profiles, e.g. "
        "http://127.0.0.1:8000 for `atrex-gateway serve --local`. "
        "Mutually exclusive with --sandbox-profile.",
    )
    ap.add_argument(
        "--sandbox-ssh",
        default=None,
        metavar="[USER@]HOST",
        help=(
            "Run GPU work in a mandatory Bubblewrap namespace reached through "
            "OpenSSH. Reuses ~/.ssh/config and is mutually exclusive with gateway "
            "endpoint options."
        ),
    )
    ap.add_argument(
        "--sandbox-ssh-init",
        default=os.environ.get("ATREX_SANDBOX_SSH_INIT", ""),
        metavar="COMMAND",
        help="Remote runtime activation command run before every SSH job and probe.",
    )
    ap.add_argument(
        "--sandbox-ssh-runtime-bind",
        action="append",
        default=None,
        metavar="REMOTE_PATH[=SANDBOX_PATH]",
        help=(
            "Read-only runtime directory exposed inside the mandatory SSH Bubblewrap "
            "sandbox (repeatable)."
        ),
    )
    ap.add_argument(
        "--sandbox-ssh-gpu",
        default=None,
        metavar="INDEX",
        help=(
            "Physical NVIDIA GPU index assigned to SSH execution (required with "
            "--sandbox-ssh; default: ATREX_SANDBOX_SSH_GPU)."
        ),
    )
    ap.add_argument(
        "--sandbox-health-command",
        default=os.environ.get(
            "ATREX_SANDBOX_HEALTH_COMMAND", ""
        ),
        metavar="COMMAND",
        help="GPU smoke command used by SSH failure classification and recovery polling.",
    )
    ap.add_argument(
        "--environment-poll-interval",
        type=int,
        default=os.environ.get("ATREX_ENVIRONMENT_POLL_INTERVAL", "60"),
        metavar="SECONDS",
        help="Seconds between blocked SSH environment recovery probes (default: 60).",
    )
    ap.add_argument(
        "--sandbox-timeout",
        type=int,
        default=None,
        help=(
            "Override the derived sandbox allocation timeout in seconds. By default it is "
            "derived from Eval candidate/performance and ABBA run limits; queue wait is "
            "budgeted separately."
        ),
    )
    ap.add_argument(
        "--agent-cli",
        choices=AGENT_CLI_CHOICES,
        default="claude",
        help="Coding CLI used for optimization episodes: claude, qodercli, codex, or pi "
        "(default: claude).",
    )
    ap.add_argument(
        "--long-reviewer-session",
        choices=("codex", "qoder", "claude"),
        default="",
        metavar="REVIEWER",
        help="Reuse one reviewer session across episodes (implemented for codex and qoder).",
    )
    for stage, stage_label in (
        ("v1", "V1 framework-baseline"),
        ("fast-episode", "fast episodes"),
        ("full-episode", "full episodes"),
    ):
        for reviewer in ("codex", "qoder"):
            ap.add_argument(
                f"--{stage}-ask-{reviewer}",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    f"Configure ask-{reviewer} for {stage_label} "
                    "(default: off)."
                ),
            )
    ap.add_argument(
        "--optimization-mode",
        choices=OPTIMIZATION_MODE_CHOICES,
        default="leaderboard",
        help="leaderboard preserves the permissive current CLAUDE.md flow; production keeps "
        "deterministic structure/state gates and requires an independent fail-closed full-candidate "
        "framework, compute-provenance, dependency, loader, and manifest review.",
    )
    ap.add_argument(
        "--framework",
        default="",
        help="Target DSL, e.g. Triton / CuteDSL / Cuda / FlyDSL / TileLang. When omitted, "
        "launch all frameworks supported by the detected hardware in parallel: NVIDIA uses "
        "Triton/CuteDSL/Cuda/TileLang, AMD uses Triton/FlyDSL/TileLang, PPU uses "
        "Triton/Cuda/TileLang, and unknown hardware uses Triton/TileLang. "
        "Each auto-dispatched production child is bound to its assigned framework.",
    )
    ap.add_argument(
        "--notes", default="none", help="Extra constraints / known bottlenecks."
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=20,
        help="Hard cap on canonical optimization versions/episodes.",
    )
    ap.add_argument(
        "--fast-episodes",
        type=int,
        default=DEFAULT_FAST_EPISODES,
        help=(
            "Use the lightweight fast path for the first N optimization episodes after "
            "baseline (default: 2; 0 disables)."
        ),
    )
    ap.add_argument(
        "--fast-trials",
        type=int,
        default=DEFAULT_FAST_TRIALS,
        help=(
            "Number of reviewed plan->implement->evaluator trials in each fast episode "
            "(default: 5)."
        ),
    )
    ap.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="Hard token cap across all episode turns (0 = no cap; max-iters still bounds it).",
    )
    ap.add_argument(
        "--target-util",
        type=float,
        default=90.0,
        help="Peak-utilization %% short-circuit (default stop condition).",
    )
    ap.add_argument(
        "--setup-timeout", type=int, default=7200, help="Baseline session timeout (s)."
    )
    ap.add_argument(
        "--handoff-resumes",
        type=int,
        default=DEFAULT_HANDOFF_RESUMES,
        help="Same-session recovery turns after an incomplete episode handoff (default: 2).",
    )
    ap.add_argument(
        "--production-review-timeout", type=int, default=600,
        help="Independent production dependency review timeout in seconds (default: 600).",
    )
    ap.add_argument(
        "--review-trace-mode",
        choices=("off", "errors", "all"),
        default=os.environ.get("ATREX_REVIEW_TRACE_MODE", "errors"),
        help=(
            "Reviewer observation retention: off keeps receipts only, errors retains "
            "failed/timed-out sessions, and all retains every reviewer session "
            "(default: errors; env: ATREX_REVIEW_TRACE_MODE)."
        ),
    )
    ap.add_argument(
        "--verify-repeats",
        type=int,
        default=DEFAULT_VERIFY_REPEATS,
        help="Incumbent/candidate ABBA repeat pairs in one gateway allocation (default: 2).",
    )
    ap.add_argument(
        "--verify-run-timeout",
        type=int,
        default=DEFAULT_VERIFY_RUN_TIMEOUT,
        help="Evaluator budget for each ABBA run (default: 120s; the remote driver "
        "allows up to 60s cold-start/finalization grace within the unchanged "
        "whole-allocation timeout).",
    )
    ap.add_argument(
        "--min-improvement-pct",
        type=float,
        default=0.0,
        help="Minimum strict ABBA improvement required for promotion (default: >0%%).",
    )
    ap.add_argument(
        "--framework-baseline",
        choices=FRAMEWORK_BASELINE_MODES,
        default="auto",
        help="Run one dedicated session after V0 setup that "
        "replaces the V0 PyTorch wrapper with the first self-contained framework "
        "kernel, recorded as v1 (so optimization episodes start at v2). Production "
        "auto = production mode only; always = leaderboard too; "
        "never = optimize directly from the V0 kernel.",
    )
    ap.add_argument(
        "--framework-baseline-timeout",
        type=int,
        default=FRAMEWORK_BASELINE_TIMEOUT_S,
        help="Framework baseline session timeout (s).",
    )
    ap.add_argument(
        "--gateway-candidate-recovery",
        choices=GATEWAY_CANDIDATE_RECOVERY_MODES,
        default="disabled",
        help=(
            "Recovery policy for a typed gateway rejection of only the candidate. "
            "disabled preserves the legacy fail-closed behavior (default); typed-once "
            "allows the existing single framework-baseline repair turn."
        ),
    )
    ap.add_argument(
        "--max-stall",
        type=int,
        default=0,
        help="Optional: stop after N consecutive unpromoted episodes (0 = disabled). "
        "After 50 completed episodes, more than 3 stalls instead enable goal mode; "
        "goal mode takes precedence over this stop condition.",
    )
    ap.add_argument(
        "--convert-after",
        type=int,
        default=DEFAULT_CONVERT_AFTER,
        help="Triton only: after N consecutive stalled episodes, require conversion "
        "to Gluon. Failed conversions retry immediately until one passes correctness "
        "and performance parity, then optimization continues in Gluon. Default: 3; "
        "0 disables conversion.",
    )
    ap.add_argument(
        "--arch",
        default="",
        help="Override the real runtime GPU arch, e.g. sm_103 or gfx942. Default: auto-detect "
        "via torch (get_device_capability / gcnArchName) — use this if auto-detect fails.",
    )
    ap.add_argument(
        "--workspace",
        default="",
        help="Working directory for the optimization campaign. A flat "
        "kernel_opt_<name>_<framework>_<platform>/ workspace is created under this "
        "directory. Default: current working directory.",
    )
    ap.add_argument("--workspace-suffix", default="", help=argparse.SUPPRESS)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = ap.parse_args(raw_argv)
    if args.workspace_suffix and args.workspace_suffix != _workspace_slug(
        args.workspace_suffix
    ):
        ap.error(
            "--workspace-suffix must be a normalized lowercase alphanumeric/underscore suffix"
        )
    if args.sandbox_timeout is not None and args.sandbox_timeout < 1:
        ap.error("--sandbox-timeout must be positive")
    if args.convert_after < 0:
        ap.error("--convert-after must be non-negative")
    if args.fast_episodes < 0:
        ap.error("--fast-episodes must be non-negative")
    if args.fast_trials <= 0:
        ap.error("--fast-trials must be positive")
    if args.handoff_resumes < 0:
        ap.error("--handoff-resumes must be non-negative")
    if args.verify_repeats <= 0:
        ap.error("--verify-repeats must be positive")
    if args.shape_batch_size <= 0:
        ap.error("--verify-shape-batch-size must be positive")
    if args.verify_run_timeout <= 0:
        ap.error("--verify-run-timeout must be positive")
    if not 0.0 <= args.min_improvement_pct < 100.0:
        ap.error("--min-improvement-pct must be in [0, 100)")
    if (args.sandbox_timeout is not None
            and args.verify_run_timeout * args.verify_repeats * 2 + 30 > args.sandbox_timeout):
        ap.error(
            "ABBA verification must fit one sandbox allocation: "
            "2 * --verify-repeats * --verify-run-timeout + 30 <= --sandbox-timeout"
        )
    if args.framework_baseline_timeout <= 0:
        ap.error("--framework-baseline-timeout must be positive")
    if args.sandbox_ssh is None:
        args.sandbox_ssh = (
            ""
            if args.sandbox_url or args.sandbox_profile
            else os.environ.get("ATREX_SANDBOX_SSH", "")
        )
    if sum(
        bool(value)
        for value in (args.sandbox_ssh, args.sandbox_url, args.sandbox_profile)
    ) > 1:
        ap.error(
            "--sandbox-ssh, --sandbox-url, and --sandbox-profile are mutually exclusive"
        )
    if args.sandbox_ssh and (
        args.sandbox_ssh.startswith("-")
        or any(character.isspace() for character in args.sandbox_ssh)
    ):
        ap.error("--sandbox-ssh must be a non-option OpenSSH host or [user@]host")
    if args.sandbox_ssh and not args.sandbox_health_command and "--sandbox-health-command" not in (argv or []):
        args.sandbox_health_command = DEFAULT_SSH_HEALTH_COMMAND
    if args.sandbox_ssh and not args.sandbox_health_command.strip():
        ap.error("--sandbox-health-command must not be empty with --sandbox-ssh")
    if args.sandbox_ssh:
        raw_ssh_gpu = (
            args.sandbox_ssh_gpu
            if args.sandbox_ssh_gpu is not None
            else os.environ.get("ATREX_SANDBOX_SSH_GPU", "")
        )
        if not re.fullmatch(r"[0-9]+", raw_ssh_gpu):
            ap.error(
                "--sandbox-ssh-gpu must be a physical NVIDIA index; "
                "MIG/UUID selectors are not yet supported"
            )
        args.sandbox_ssh_gpu = int(raw_ssh_gpu)
        if args.sandbox_ssh_gpu > 31:
            ap.error("--sandbox-ssh-gpu must be in the range 0..31")
        if not args.framework:
            ap.error(
                "--sandbox-ssh requires an explicit --framework so one assigned GPU "
                "cannot be shared by auto-dispatched campaigns"
            )
        if args.sandbox_ssh_runtime_bind is None:
            raw_runtime_binds = os.environ.get(
                "ATREX_SANDBOX_SSH_RUNTIME_BINDS", "[]"
            )
            try:
                args.sandbox_ssh_runtime_bind = json.loads(raw_runtime_binds)
            except json.JSONDecodeError as exc:
                ap.error(f"ATREX_SANDBOX_SSH_RUNTIME_BINDS is invalid JSON: {exc}")
            if not isinstance(args.sandbox_ssh_runtime_bind, list) or not all(
                isinstance(item, str) for item in args.sandbox_ssh_runtime_bind
            ):
                ap.error(
                    "ATREX_SANDBOX_SSH_RUNTIME_BINDS must be a JSON array of strings"
                )
    elif args.sandbox_ssh_runtime_bind:
        ap.error("--sandbox-ssh-runtime-bind requires --sandbox-ssh")
    elif args.sandbox_ssh_gpu is not None:
        ap.error("--sandbox-ssh-gpu requires --sandbox-ssh")
    if args.environment_poll_interval <= 0:
        ap.error("--environment-poll-interval must be positive")
    if args.agent_cli == "codex":
        codex_settings = (
            os.environ.get("ATREX_CODEX_SESSION_SETTINGS")
            or os.environ.get("ATREX_SESSION_SETTINGS")
            or ""
        )
        try:
            codex_settings_args(codex_settings)
        except ValueError as exc:
            ap.error(str(exc))
    if args.agent_cli == "pi":
        pi_settings = (
            os.environ.get("ATREX_PI_SESSION_SETTINGS")
            or os.environ.get("ATREX_SESSION_SETTINGS")
            or ""
        )
        try:
            pi_settings_args(pi_settings)
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
    if args.workspace:
        # Campaign commands run from nested Git worktrees. Keeping a relative base here
        # makes evaluator paths get resolved a second time against the campaign workspace.
        args.workspace = str(Path(args.workspace).expanduser().resolve())


    return args, raw_argv


def adapt_argv(argv):
    args, raw_argv = parse_arguments(argv)
    sandbox_hardware = args.sandbox_hardware or args.platform
    source = problem_request(args.op_dir, args.optimization_mode,
        workspace=args.workspace,
        suffix=args.workspace_suffix or f"{args.framework}_{args.platform}")
    values = vars(args).copy()
    from aka.legacy.execution.configuration import configure_legacy_arguments
    configure_legacy_arguments(values)
    values.pop("op_dir")
    values["work_dir"] = values.pop("workspace")
    values.update(name=source.directory.name, problem=source,
                  sandbox_hardware=sandbox_hardware)
    request = campaign_request(values)
    explicit = {value.split("=",1)[0] for value in raw_argv if value.startswith("--")}
    workflow_options = {field.name for field in __import__("dataclasses").fields(request.settings)
                        if "--" + field.name.replace("_", "-") in explicit}
    runtime_options = {field.name for field in __import__("dataclasses").fields(request.runtime)
                       if "--" + field.name.replace("_", "-") in explicit}
    if "--verify-shape-batch-size" in explicit:
        runtime_options.add("shape_batch_size")
    return encode_launch(WorkflowLaunch(request, Path(args.workspace or Path.cwd()),
        args.workspace_suffix, tuple(raw_argv),
        provided_settings=tuple(sorted(workflow_options)), provided_runtime=tuple(sorted(runtime_options))))
