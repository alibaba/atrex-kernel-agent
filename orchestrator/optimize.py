#!/usr/bin/env python3
"""Clean-session orchestrator for atrex-kernel-agent.

Owns the OUTER optimization loop so termination no longer depends on the model's
in-session judgment (the old Stage-6 "is README's Stop Conditions met?" self-call).

Each iteration is a **fresh coding-agent session** (`claude` by default, or `qodercli` / `codex`
via `--agent-cli`) over the *same* git workspace. State crosses the session boundary only
through disk — exactly the artifacts atrex already maintains: `memory/v<N>.json`, `plans/`,
`profiles/`, and git. HEAD is always the best kernel (a regressing iteration reverts and is
never committed).

Termination policy
------------------
- Outer loop (this file):  HARD budget break = max iterations OR token budget,
  plus a mechanical target short-circuit (peak utilization >= --target-util on a
  committed, correctness-PASS iteration). No plateau ladder, no convergence judge.
- Inner loop (one session): exactly one profile->edit->validate->bench cycle, bounded
  by a hang-backstop timeout (SIGKILL of the process group). See prompts/iteration.md.

Per-iteration reasoning stays in markdown (the gpu-kernel-* skills + prompts/*.md);
this file only does mechanism: spawn, time-bound, token-account, read state, decide stop.

For SOL and atrex-bench operators, a workload-inspection stage first partitions
workload.jsonl or shapes.json into disjoint optimization buckets. Each bucket runs the
same loop below concurrently in its own Git workspace. A serialized aggregation gate
maintains the full-workload kernel.py in the main workspace and accepts it only after
independent full correctness and performance validation.

Usage
-----
    # single operator with an explicit framework:
    python orchestrator/optimize.py \
        --name mla_decode --kernel-demo /path/to/demo.py \
        --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework CuteDSL \
        --agent-cli qodercli \
        --max-iters 20 --token-budget 8000000 --target-util 90

    # omit --framework to launch one independent campaign per supported framework:
    #   NVIDIA -> Triton, CuteDSL, Cuda
    #   AMD    -> Triton, FlyDSL
    #   other  -> Triton
    python orchestrator/optimize.py \
        --op-dir /path/to/op --platform H20 --workspace /path/to/runs

    # production: exact framework, no third-party kernel/operator dependencies:
    python orchestrator/optimize.py \
        --op-dir /path/to/op --platform H20 --framework Triton \
        --optimization-mode production

    # whole LLM layer (optional decomposition overlay):
    #   decompose -> N per-boundary workspaces (each a standard single-op campaign) ->
    #   shared --max-iters budget scheduled by live ROI (no boundary dropped) -> recombine.
    #   Σ (per-boundary optimization versions) == --max-iters.
    python orchestrator/optimize.py --layer \
        --name decoder_layer --kernel-demo /path/to/layer.py \
        --platform H20 --framework CuteDSL --max-iters 40
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from .aggregate_dispatch import embed_bucket_sources
    from .optimization_policy import (
        OPTIMIZATION_MODE_CHOICES,
        install_workspace_policy,
        optimization_mode_directive,
        production_kernel_violations,
        reject_production_commit,
        source_uses_gluon,
    )
except ImportError:  # direct script execution: python orchestrator/optimize.py
    from aggregate_dispatch import embed_bucket_sources  # type: ignore[no-redef]
    from optimization_policy import (  # type: ignore[no-redef]
        OPTIMIZATION_MODE_CHOICES,
        install_workspace_policy,
        optimization_mode_directive,
        production_kernel_violations,
        reject_production_commit,
        source_uses_gluon,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = REPO_ROOT / "reference" / "sol_seed.py"
ATREX_BENCH_HARNESS = REPO_ROOT / "reference" / "atrex_bench_test_kernel.py"
SANDBOX_TOOL = REPO_ROOT / "tools" / "sandbox.py"
SESSION_SHELL_GUARD = REPO_ROOT / "tools" / "session_shell_guard.sh"
HUMANIZE_DIR = REPO_ROOT / "3rdparty" / "humanize"
CONVERT_PERF_TOL = 0.05   # triton->gluon is a direct translation: gluon must be within +5% of triton
CONVERT_MIN_TOKENS = 200_000  # below this, flag a shallow attempt in the retry diagnostics
DEFAULT_CONVERT_AFTER = 3     # mandatory Triton->Gluon escalation after three consecutive stalls
MEMORY_MASK_INTERVAL = 100    # periodically drop half of active optimization history
SALVAGE_TIMEOUT_S = 1200      # post-mortem session budget for an iteration killed mid-flight
INTERRUPTED_CATEGORY = "interrupted_session"
FRAMEWORK_BASELINE_FILE = "framework_baseline.json"
FRAMEWORK_BASELINE_VERSION = 1     # the framework baseline always occupies v1, retries overwrite it
FRAMEWORK_BASELINE_TIMEOUT_S = 10800
FRAMEWORK_BASELINE_MODES = ("auto", "always", "never")
FRAMEWORK_BASELINE_CATEGORY = "framework_baseline"
IMMUTABLE_BASELINE_PATHS = (
    "test_kernel.py", "reference.py", "input.py", "shapes.json", "metadata.json",
    "roofline.json", "workload.jsonl", "definition.json", "valid.py", "memory/v0.json",
)
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
AGENT_CLI_CHOICES = ("claude", "qodercli", "codex")
NVIDIA_FRAMEWORKS = ("Triton", "CuteDSL", "Cuda")
AMD_FRAMEWORKS = ("Triton", "FlyDSL")
DEFAULT_FRAMEWORKS = ("Triton",)
WORKLOAD_BUCKETS_FILE = "workload_buckets.json"
AGGREGATION_STATE_FILE = "aggregation_state.json"
DISPATCH_SIGNATURES_FILE = "dispatch_signatures.json"
DISPATCH_VISIBILITY_POLICY = "host_no_sync_structural_v1"
AGGREGATE_DISPATCH_FILE = "aggregate_dispatch.json"
AGGREGATE_KERNELS_DIR = "aggregate_kernels"
AGGREGATE_DISPATCH_SCHEMA_VERSION = 2
AGGREGATE_SOURCE_LAYOUT = "embedded_single_file"
DISPATCH_SIGNATURE_RESULT_PREFIX = "[dispatch-signatures] RESULT_JSON="
BUCKETS_DIR = "workload_buckets"
AGGREGATE_VALIDATION_TIMEOUT = 600   # public dev gateway execution limit
AGGREGATE_QUEUE_WAIT_GRACE = 14_400  # single-worker localhost queues are independent of execution time
INITIAL_AGGREGATION_MIN_ITERATIONS = 10
DEFAULT_SANDBOX_TIMEOUT = 600
MAX_SANDBOX_TIMEOUT = 600
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEPENDENCY_GUARD_POLL_SECONDS = 0.25
DEFAULT_PROTECTED_GATEWAY_SCREEN = "atrex-local-gateway"
DEFAULT_PROTECTED_GATEWAY_STATE_NAME = "atrex-local-gateway"


def _protected_gateway_identity(
    environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve shared gateway protection targets without embedding host paths."""
    values = os.environ if environment is None else environment
    screen = values.get(
        "ATREX_PROTECTED_GATEWAY_SCREEN", DEFAULT_PROTECTED_GATEWAY_SCREEN
    )
    state_dir = values.get("ATREX_PROTECTED_GATEWAY_STATE_DIR")
    if not state_dir:
        cache_home = values.get("XDG_CACHE_HOME")
        cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        state_dir = str(cache_root / DEFAULT_PROTECTED_GATEWAY_STATE_NAME)
    return screen, state_dir


def _python_import_roots(code: str, *, _depth: int = 0) -> set[str]:
    """Return real imported top-level modules without matching strings/comments."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        # Invalid ``python -c`` input cannot execute an import.  Do not turn a
        # syntax error or a research-note string into a policy violation.
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and node.args:
            target: str | None = None
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                target = "import"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                target = "import"
            if target and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                if isinstance(module, str) and module:
                    roots.add(module.split(".", 1)[0])
            if (
                _depth < 2
                and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval"}
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                roots.update(_python_import_roots(node.args[0].value, _depth=_depth + 1))
    return roots


def _status_is(value: object, expected: str) -> bool:
    """Accept a status even when a CLI accidentally stored it as a JSON-quoted string."""
    current = value
    for _ in range(2):
        if current == expected:
            return True
        if not isinstance(current, str):
            return False
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            return current.strip() == expected
        if decoded == current:
            return False
        current = decoded
    return current == expected


def _hardware_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def hardware_vendor(platform: str, arch: str = "") -> str:
    """Return ``nvidia``, ``amd``, or ``unknown`` for framework dispatch.

    Runtime architecture is authoritative because gateway device names can be
    desensitized. Platform-name matching is only a fallback for dry runs or an
    unavailable runtime probe.
    """
    runtime_arch = arch.strip().lower()
    if re.fullmatch(r"sm_?\d+", runtime_arch):
        return "nvidia"
    if re.fullmatch(r"gfx[0-9a-f]+", runtime_arch):
        return "amd"

    token = _hardware_token(platform)
    if re.match(r"^(?:AMD|MI\d|RADEON|INSTINCT)", token):
        return "amd"
    if re.match(
        r"^(?:NVIDIA|CUDA|GEFORCE|RTX|QUADRO|TESLA|DGX|GB\d|[BHALTVP]\d|PRO\d)",
        token,
    ):
        return "nvidia"
    return "unknown"


def supported_frameworks(platform: str, arch: str = "") -> tuple[str, ...]:
    """Framework campaigns to launch when ``--framework`` is omitted."""
    vendor = hardware_vendor(platform, arch)
    if vendor == "nvidia":
        return NVIDIA_FRAMEWORKS
    if vendor == "amd":
        return AMD_FRAMEWORKS
    return DEFAULT_FRAMEWORKS


def _workspace_slug(value: str) -> str:
    """Stable flat-workspace suffix component."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("workspace suffix value has no usable directory characters")
    return slug


def framework_workspace_suffix(
    framework: str, platform: str, optimization_mode: str = "leaderboard"
) -> str:
    """Flat suffix for one framework/hardware/policy campaign.

    Leaderboard keeps its historical path for resume compatibility. Production
    uses a distinct path so its immutable policy and Git history can coexist
    with a prior leaderboard campaign under the same workspace root.
    """
    suffix = f"{_workspace_slug(framework)}_{_workspace_slug(platform)}"
    if optimization_mode == "production":
        suffix += "_production"
    return suffix


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
    (or the equivalent layer paths). Budgets remain per campaign; a failed
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
            for pid, _argv in _descendant_process_commands(proc.pid):
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


def is_sol_op(op_dir: Path) -> bool:
    """A SOL-ExecBench op dir carries definition.json + workload.jsonl next to reference.py."""
    return (op_dir / "definition.json").is_file() and (op_dir / "workload.jsonl").is_file()


def find_atrex_bench_root(op_dir: Path) -> Optional[Path]:
    """Return the canonical Atrex-Bench checkout owning a native shapes op."""
    for candidate in (op_dir, *op_dir.parents):
        if (
            (candidate / "scripts" / "run_eval.py").is_file()
            and (candidate / "src" / "atrex_bench").is_dir()
        ):
            return candidate
    return None


def is_bucketable_op(op_dir: Path) -> bool:
    """Whether the op exposes an enumerable SOL workload or atrex shape set."""
    return is_sol_op(op_dir) or (op_dir / "shapes.json").is_file()


def _is_triton_family(framework: str) -> bool:
    """Triton and Gluon are one framework family — Gluon is the lower-level escalation of Triton."""
    return framework.strip().lower() in ("triton", "gluon", "triton/gluon")


def kernel_is_gluon(workspace: Path) -> bool:
    """True once the working-tree kernel.py has a real Gluon import."""
    k = workspace / "kernel.py"
    return k.exists() and source_uses_gluon(
        k.read_text(encoding="utf-8", errors="ignore")
    )


def head_kernel_is_gluon(workspace: Path) -> bool:
    """True when the COMMITTED HEAD kernel.py is Gluon. Authoritative accept signal for a convert
    session — more reliable than memory's git_commit_hash, which a session may leave unset even after
    committing."""
    try:
        out = subprocess.run(["git", "show", "HEAD:kernel.py"], cwd=str(workspace),
                             capture_output=True, text=True)
    except OSError:
        return False
    return out.returncode == 0 and source_uses_gluon(out.stdout)


def should_convert_to_gluon(
    framework: str,
    stall: int,
    convert_after: int,
    *,
    head_is_gluon: bool,
) -> bool:
    """Whether the campaign is latched into mandatory Triton->Gluon conversion.

    Once the threshold is reached this remains true after a failed conversion because
    the stall counter is deliberately not reset.  Only a committed Gluon HEAD releases
    the latch and returns the campaign to ordinary optimization.
    """
    return (
        convert_after > 0
        and _is_triton_family(framework)
        and not head_is_gluon
        and stall >= convert_after
    )


# ── thin IO ─────────────────────────────────────────────────────────────────


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""


def _render(template_path: Path, **kw: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    mode_policy = kw.pop("MODE_POLICY", "")
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


def _tokens_from_stream(stdout: str) -> int:
    """Sum core token usage from a coding CLI's JSONL stdout.

    Claude/Qoder emit a cumulative ``type=result`` event. Codex ``exec --json`` emits a
    cumulative ``type=turn.completed`` event. Prefer either terminal event and fall back to
    summing per-message usage. Codex's ``cached_input_tokens``,
    ``cache_write_input_tokens``, and ``reasoning_output_tokens`` are diagnostic subsets of
    input/output and therefore must not be added again. Never raises — budget accounting
    degrades to max-iters if the stream is unparseable.
    """
    def _usage_tokens(u: dict) -> int:
        if not isinstance(u, dict):
            return 0
        return int(
            (u.get("input_tokens") or u.get("inputTokens") or 0)
            + (u.get("output_tokens") or u.get("outputTokens") or 0)
            + (u.get("cache_creation_input_tokens") or u.get("cacheCreationInputTokens") or 0)
            + (u.get("cache_read_input_tokens") or u.get("cacheReadInputTokens") or 0)
        )

    def _model_usage_tokens(model_usage: dict) -> int:
        if not isinstance(model_usage, dict):
            return 0
        return sum(_usage_tokens(usage) for usage in model_usage.values())

    result_total = None
    summed = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(evt, dict):
            continue
        if evt.get("type") in ("result", "turn.completed"):
            usage_total = _usage_tokens(evt.get("usage"))
            model_total = _model_usage_tokens(evt.get("modelUsage"))
            result_total = usage_total or model_total
        usage = evt.get("usage")
        if usage is None and isinstance(evt.get("message"), dict):
            usage = evt["message"].get("usage")
        if isinstance(usage, dict):
            summed += _usage_tokens(usage)
    return result_total if result_total else summed


def _dependency_process_violation(argv: list[str]) -> Optional[str]:
    """Describe a forbidden dependency build or host GPU action, if any.

    Optimizer workspaces are intentionally immutable with respect to third-party
    packages.  Coding sessions must also route kernel imports, evaluators,
    profilers, and CUDA compilation through ``tools/sandbox.py``.  Gateway
    workers are not descendants of the coding session, so rejecting compiler
    descendants here does not block legitimate remote/local-gateway JIT.
    """
    if not argv:
        return None

    def command_segments(process_argv: list[str]) -> list[list[str]]:
        tokens = process_argv
        executable = Path(process_argv[0]).name.lower()
        if executable in {"bash", "sh", "dash", "zsh", "ksh"}:
            command_index = next(
                (
                    index + 1
                    for index, value in enumerate(process_argv[:-1])
                    if value.startswith("-") and "c" in value[1:]
                ),
                -1,
            )
            if command_index >= 0:
                try:
                    lexer = shlex.shlex(
                        process_argv[command_index], posix=True, punctuation_chars=";&|"
                    )
                    lexer.whitespace_split = True
                    tokens = list(lexer)
                except ValueError:
                    tokens = process_argv
        segments: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if token and all(character in ";&|" for character in token):
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        # Claude's Bash tool commonly wraps the actual command in ``eval``.
        # Expand that payload so a direct host command cannot hide behind the
        # shell snapshot preamble.
        expanded = list(segments)
        for segment in segments:
            tokens = unwrap(segment)
            if tokens and Path(tokens[0]).name.lower() == "eval" and len(tokens) > 1:
                expanded.extend(command_segments(["sh", "-c", " ".join(tokens[1:])]))
        return expanded

    def unwrap(segment: list[str]) -> list[str]:
        result = list(segment)
        while result and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", result[0]):
            result.pop(0)
        if result and Path(result[0]).name.lower() in {"env", "command"}:
            result.pop(0)
            while result and (result[0].startswith("-") or "=" in result[0]):
                result.pop(0)
        if result and Path(result[0]).name.lower() == "timeout":
            result.pop(0)
            while result and result[0].startswith("-"):
                result.pop(0)
            if result:
                result.pop(0)
        return result

    def is_installer(segment: list[str]) -> bool:
        tokens = unwrap(segment)
        if not tokens:
            return False
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        if re.fullmatch(r"pip[0-9.]*", executable):
            return len(lowered) > 1 and lowered[1] in {"install", "wheel"}
        if executable == "uv":
            return lowered[1:3] in (["pip", "install"], ["pip", "sync"], ["pip", "compile"])
        if executable in {"conda", "mamba", "micromamba"}:
            return len(lowered) > 1 and lowered[1] in {"install", "create"}
        if re.fullmatch(r"python[0-9.]*", executable):
            if len(lowered) > 3 and lowered[1:3] == ["-m", "pip"]:
                return lowered[3] in {"install", "wheel"}
            if len(lowered) > 2 and lowered[1:3] == ["-m", "build"]:
                return True
            for index, token in enumerate(lowered[:-1]):
                if Path(token).name == "setup.py" and lowered[index + 1] in {
                    "install", "build", "build_ext", "bdist_wheel",
                }:
                    return True
            # tools/sandbox.py ... -- pip install must be rejected too: the
            # immutable dependency rule applies on both sides of the gateway.
            if "--" in lowered:
                boundary = lowered.index("--")
                return is_installer(tokens[boundary + 1 :])
        if Path(executable).name == "setup.py":
            return len(lowered) > 1 and lowered[1] in {
                "install", "build", "build_ext", "bdist_wheel",
            }
        return False

    segments = command_segments(argv)

    def shared_gateway_mutation(segment: list[str]) -> bool:
        """Reject coding-session lifecycle/state changes to shared localhost infra."""
        tokens = unwrap(segment)
        if not tokens:
            return False
        executable = Path(tokens[0]).name.lower()
        lowered = [token.lower() for token in tokens]
        protected_screen, protected_state = _protected_gateway_identity()
        protected_screen = protected_screen.lower()
        protected_state = protected_state.lower()

        if executable == "screen" and any(
            token == protected_screen or token.endswith("." + protected_screen)
            for token in lowered[1:]
        ):
            return True
        if executable in {"rm", "rmdir", "unlink", "shred", "truncate", "mv"} and any(
            token == protected_state
            or token.startswith(protected_state + "/")
            or token == protected_state + ".log"
            for token in lowered[1:]
        ):
            return True
        if re.fullmatch(r"python[0-9.]*", executable):
            if any(Path(token).name == "local_gateway.py" for token in tokens[1:3]):
                return "serve" in lowered[1:]
            if "-c" in tokens:
                code_index = tokens.index("-c") + 1
                code = tokens[code_index].lower() if code_index < len(tokens) else ""
                if protected_state in code and re.search(
                    r"(?:rmtree|unlink|remove|rename|replace|sqlite3)", code
                ):
                    return True
        if executable in {"pkill", "killall"} and any(
            "local_gateway" in token or token == protected_screen
            for token in lowered[1:]
        ):
            return True
        if executable in {"curl", "wget"} and any(
            "/v1/jobs/" in token and "/cancel" in token for token in lowered[1:]
        ):
            return True
        return False

    if any(shared_gateway_mutation(segment) for segment in segments):
        return "shared localhost gateway lifecycle/state mutation"

    if any(is_installer(segment) for segment in segments):
        return "third-party package installation/build command"

    def direct_host_gpu_action(segment: list[str]) -> Optional[str]:
        tokens = unwrap(segment)
        if not tokens:
            return None
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        info_only = (
            any(token in {"--help", "-h", "--version"} for token in lowered[1:])
            or (executable == "nvcc" and "-V" in tokens[1:])
        )
        if executable in {"nvcc", "cicc", "ptxas", "fatbinary", "ninja"} and not info_only:
            return "CUDA/JIT build tool executed directly on the host"
        if executable in {"ncu", "rocprof", "rocprofv3", "compute-sanitizer"}:
            return "GPU profiler executed directly on the host"
        if re.fullmatch(r"python[0-9.]*", executable):
            if len(tokens) > 1 and Path(tokens[1]).name == "sandbox.py":
                return None
            if len(tokens) > 1 and Path(tokens[1]).name in {
                "kernel.py", "test_kernel.py", "profile_driver.py",
            }:
                return "kernel/evaluator executed directly on the host"
            if "-c" in tokens:
                code_index = tokens.index("-c") + 1
                code = tokens[code_index] if code_index < len(tokens) else ""
                imports = _python_import_roots(code)
                if "kernel" in imports:
                    return "kernel imported directly on the host"
                if imports & {"flashinfer", "flash_attn", "xformers", "vllm"}:
                    return "JIT-capable third-party GPU package imported directly on the host"
        if executable in {"bash", "sh", "dash", "zsh", "ksh"} and any(
            Path(token).name in {"profile_nvidia.sh", "profile_kernel.sh"}
            for token in tokens[1:]
        ):
            return "GPU profiler wrapper executed directly on the host"
        return None

    for segment in segments:
        reason = direct_host_gpu_action(segment)
        if reason is not None:
            return reason

    command = " ".join(argv).lower()
    package_build_tree = re.search(
        r"(?:^|[\s=])[^\s]*(?:pip-install-|pip-build-|pip-modern-metadata-)[^\s]*",
        command,
    )
    build_tools = {"cicc", "nvcc", "ninja", "cmake", "make", "gcc", "g++", "clang", "clang++"}
    if package_build_tree and any(
        unwrap(segment) and Path(unwrap(segment)[0]).name.lower() in build_tools
        for segment in segments
    ):
        return "compiler/build tool running in a package-manager temporary tree"
    return None


def _descendant_process_commands(root_pid: int) -> list[tuple[int, list[str]]]:
    """Return live descendants and argv using Linux procfs, tolerating races.

    Children created by ``subprocess.Popen`` inside a Python worker thread are
    listed under that thread's ``task/<tid>/children``, not necessarily under
    the thread-group leader. Inspect every task so bucket sessions launched by
    ``ThreadPoolExecutor`` are included in shutdown and policy-guard snapshots.
    """
    pending = [root_pid]
    seen = {root_pid}
    descendants: list[tuple[int, list[str]]] = []
    while pending:
        parent = pending.pop()
        task_dir = Path(f"/proc/{parent}/task")
        try:
            thread_dirs = list(task_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children: set[int] = set()
        for thread_dir in thread_dirs:
            try:
                children.update(
                    int(value)
                    for value in (thread_dir / "children").read_text().split()
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        for pid in children:
            if pid in seen:
                continue
            seen.add(pid)
            pending.append(pid)
            try:
                raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            descendants.append((pid, argv))
    return descendants


def _descendant_process_groups(root_pid: int) -> set[int]:
    """Capture every process group in a coding session's live process tree."""
    process_groups: set[int] = set()
    for pid in [root_pid, *[pid for pid, _argv in _descendant_process_commands(root_pid)]]:
        try:
            process_groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    return process_groups


def _signal_process_groups(process_groups: set[int], sig: signal.Signals) -> None:
    for process_group in process_groups:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass


def _dependency_guard(
    proc: subprocess.Popen[str], stop: threading.Event, violations: list[str]
) -> None:
    """Kill a coding session as soon as it starts a forbidden dependency job."""
    while not stop.wait(DEPENDENCY_GUARD_POLL_SECONDS):
        if proc.poll() is not None:
            return
        for pid, argv in _descendant_process_commands(proc.pid):
            reason = _dependency_process_violation(argv)
            if reason is None:
                continue
            rendered = " ".join(argv)
            violations.append(f"pid={pid}: {reason}: {rendered[:1000]}")
            process_groups = _descendant_process_groups(proc.pid)
            _signal_process_groups(process_groups, signal.SIGTERM)
            deadline = time.monotonic() + 1.0
            while proc.poll() is None and time.monotonic() < deadline:
                if stop.wait(0.05):
                    return
            # A Bash tool can call setsid and outlive the Claude process group.
            # Always signal the captured child groups even when Claude already
            # exited after SIGTERM.
            _signal_process_groups(process_groups, signal.SIGKILL)
            return


def _run_bounded(cmd: list[str], cwd: Path, timeout: int, env: Optional[dict] = None) -> tuple[str, str, int, bool]:
    """Run cmd in its own group with timeout and dependency-build enforcement."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group -> killpg reaps grandchildren
        env=env,
    )
    guard_stop = threading.Event()
    dependency_violations: list[str] = []
    guard = threading.Thread(
        target=_dependency_guard,
        args=(proc, guard_stop, dependency_violations),
        name=f"dependency-guard-{proc.pid}",
        daemon=True,
    )
    guard.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process_groups = _descendant_process_groups(proc.pid)
        _signal_process_groups(process_groups, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    except BaseException:
        # The coding CLI owns a separate process group. If an explicit or
        # auto-dispatched optimizer is interrupted, reap that entire group so
        # Qoder/Claude and their tool subprocesses cannot become orphaned.
        process_groups = _descendant_process_groups(proc.pid)
        _signal_process_groups(process_groups, signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_process_groups(process_groups, signal.SIGKILL)
            proc.communicate()
        raise
    finally:
        guard_stop.set()
        guard.join(timeout=1)
    returncode = proc.returncode
    if dependency_violations:
        policy_message = (
            "[orchestrator] dependency policy violation; terminated coding session:\n"
            + "\n".join(dependency_violations)
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + policy_message + "\n"
        if returncode == 0:
            returncode = 126
    return stdout or "", stderr or "", returncode, timed_out


def _session_env(agent_cli: str) -> dict:
    """Build the environment for a nested coding-agent session.

    Claude-specific auth normalization is deliberately not applied to Qoder CLI. When a Bearer
    auth token is available for Claude (ANTHROPIC_AUTH_TOKEN — e.g. an Anthropic-compatible
    gateway), drop ANTHROPIC_API_KEY so Claude authenticates via the token instead of sending
    x-api-key, which such gateways reject with 401.
    """
    env = os.environ.copy()
    # An optimizer launched via an absolute environment-Python path does not
    # implicitly activate that environment for Claude/Qoder Bash tools. Make
    # the orchestrator's own interpreter/toolchain the session default while
    # retaining the rest of the caller PATH.
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [
        part for part in env.get("PATH", "").split(os.pathsep)
        if part and part != python_bin
    ]
    env["PATH"] = os.pathsep.join([python_bin, *path_parts])
    # Defense in depth for python -m pip and short-lived installer processes that
    # could race the process watchdog.  Sessions still may not install packages;
    # binary-only ensures an attempted pip command cannot fall back to source.
    env["PIP_ONLY_BINARY"] = ":all:"
    env["PIP_INDEX_URL"] = PYPI_MIRROR
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # Bash commands launched by a coding CLI source this guard before executing
    # the tool payload. It prevents fast shared-gateway mutations before the
    # procfs watchdog could observe them.
    env["BASH_ENV"] = str(SESSION_SHELL_GUARD)
    protected_screen, protected_state = _protected_gateway_identity(env)
    env["ATREX_PROTECTED_GATEWAY_SCREEN"] = protected_screen
    env["ATREX_PROTECTED_GATEWAY_STATE_DIR"] = protected_state
    if agent_cli == "claude" and env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def _toml_config_value(value: object) -> str:
    """Encode the JSON-compatible subset accepted by ``codex exec -c key=value``."""
    if value is None or isinstance(value, dict):
        raise ValueError("Codex config values must be strings, numbers, booleans, or arrays")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Codex floating-point config values must be finite")
    if isinstance(value, list):
        if any(item is None or isinstance(item, (dict, list)) for item in value):
            raise ValueError("Codex config arrays may contain only scalar values")
        if any(isinstance(item, float) and not math.isfinite(item) for item in value):
            raise ValueError("Codex floating-point config values must be finite")
    if isinstance(value, (str, int, float, list)):
        # JSON strings/scalars/scalar arrays are also valid TOML values.
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"unsupported Codex config value type: {type(value).__name__}")


def _codex_settings_args(raw: str) -> list[str]:
    """Translate ATREX_CODEX_SESSION_SETTINGS into repeatable Codex ``-c`` flags.

    Accepted forms are a JSON object (the convenient form) or a JSON array of literal
    ``key=value`` strings (for values that need Codex-specific TOML syntax).
    """
    if not raw:
        return []
    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "ATREX_CODEX_SESSION_SETTINGS must be a JSON object or an array of key=value strings"
        ) from exc

    pairs: list[str] = []
    if isinstance(settings, dict):
        for key, value in settings.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
                raise ValueError(f"invalid Codex config key: {key!r}")
            pairs.append(f"{key}={_toml_config_value(value)}")
    elif isinstance(settings, list):
        for item in settings:
            if not isinstance(item, str) or "=" not in item or item.startswith("="):
                raise ValueError(
                    "ATREX_CODEX_SESSION_SETTINGS array entries must be key=value strings"
                )
            key = item.split("=", 1)[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
                raise ValueError(f"invalid Codex config key: {key!r}")
            pairs.append(item)
    else:
        raise ValueError(
            "ATREX_CODEX_SESSION_SETTINGS must be a JSON object or an array of key=value strings"
        )

    args: list[str] = []
    for pair in pairs:
        args += ["-c", pair]
    return args


def _session_command(
    agent_cli: str,
    prompt: str,
    session_id: str,
    reasoning_effort: str = "max",
) -> list[str]:
    """Return a non-interactive, fresh-session command for the selected coding CLI."""
    if reasoning_effort not in {"low", "medium", "high", "max"}:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort!r}")
    if agent_cli == "claude":
        cmd = [
            "claude", "--print", "--verbose",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--session-id", session_id,
            "--effort", reasoning_effort,
        ]
        provider_settings = "ATREX_CLAUDE_SESSION_SETTINGS"
    elif agent_cli == "qodercli":
        cmd = [
            "qodercli", "--print",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--session-id", session_id,
            "--no-session-persistence",
            "--reasoning-effort", reasoning_effort,
        ]
        provider_settings = "ATREX_QODER_SESSION_SETTINGS"
    elif agent_cli == "codex":
        # --ephemeral is the Codex equivalent of a fresh, non-persistent session.  The
        # dangerous bypass is intentional and symmetric with the existing Claude/Qoder
        # automation flags: the coding agent must edit the local optimization workspace and
        # invoke tools/sandbox.py non-interactively. GPU execution is still forced across the
        # separately enforced atrex gateway boundary by the injected prompt.
        cmd = [
            "codex", "exec", "--json", "--ephemeral", "--color", "never",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
        ]
        provider_settings = "ATREX_CODEX_SESSION_SETTINGS"
    else:
        raise ValueError(f"unsupported agent CLI: {agent_cli!r}")

    # Provider-specific settings win. ATREX_SESSION_SETTINGS remains the backward-compatible
    # generic fallback and is interpreted by whichever CLI was selected. Claude/Qoder expect
    # their native --settings value; Codex expects the documented JSON object/array format.
    session_settings = os.environ.get(provider_settings) or os.environ.get("ATREX_SESSION_SETTINGS")
    if agent_cli == "codex":
        cmd += _codex_settings_args(session_settings or "")
    elif session_settings:
        cmd += ["--settings", session_settings]
    # Claude uses the upstream Humanize plugin. Qoder generates plans directly and must not load
    # Humanize: its headless Skill tool cannot invoke the plugin's model-disabled gen-plan flow.
    # Codex discovers the hydrated Humanize skill from the workspace-local .agents/skills tree.
    if (
        agent_cli == "claude"
        and (HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md").exists()
    ):
        cmd += ["--plugin-dir", str(HUMANIZE_DIR)]
    cmd.append(prompt)
    return cmd


def _agent_auth_hint(agent_cli: str) -> str:
    if agent_cli == "qodercli":
        return "run `qodercli status` and `qodercli --print \"test\"` to diagnose"
    if agent_cli == "codex":
        return "run `codex login status` and `codex exec --ephemeral \"reply ok\"` to diagnose"
    return "run `claude auth status` and `claude --print \"test\"` to diagnose"


def _find_jq() -> Optional[str]:
    found = shutil.which("jq")
    if found:
        return found
    adjacent = Path(sys.executable).resolve().parent / "jq"
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return str(adjacent)
    return None


def ensure_jq() -> str:
    """Install jq with an available package manager when the runtime lacks it."""
    found = _find_jq()
    if found:
        return found

    privileged_prefix: list[str] | None
    if getattr(os, "geteuid", lambda: 1)() == 0:
        privileged_prefix = []
    elif shutil.which("sudo"):
        privileged_prefix = ["sudo"]
    else:
        privileged_prefix = None

    installers: list[tuple[str, list[str], dict[str, str] | None]] = []
    system_commands = (
        ("apt-get", ["apt-get", "install", "-y", "jq"]),
        ("dnf", ["dnf", "install", "-y", "jq"]),
        ("yum", ["yum", "install", "-y", "jq"]),
        ("apk", ["apk", "add", "jq"]),
        ("zypper", ["zypper", "--non-interactive", "install", "jq"]),
    )
    if privileged_prefix is not None:
        for manager, command in system_commands:
            if shutil.which(manager):
                installers.append((manager, [*privileged_prefix, *command], None))
    if shutil.which("brew"):
        installers.append(("brew", ["brew", "install", "jq"], None))
    if shutil.which("conda"):
        conda_env = os.environ.copy()
        conda_env["CONDA_SOLVER"] = "classic"
        installers.append(
            (
                "conda",
                ["conda", "install", "-y", "-c", "conda-forge", "jq"],
                conda_env,
            )
        )

    failures: list[str] = []
    for manager, command, environment in installers:
        print(f"[orchestrator] jq not found; installing with {manager}", flush=True)
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{manager}: {exc}")
            continue
        found = _find_jq()
        if completed.returncode == 0 and found:
            jq_dir = str(Path(found).resolve().parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if jq_dir not in path_parts:
                os.environ["PATH"] = os.pathsep.join([jq_dir, *path_parts])
            print(f"[orchestrator] jq installed with {manager}", flush=True)
            return found
        output_lines = (completed.stdout or "").strip().splitlines()
        detail = output_lines[-1] if output_lines else f"exit {completed.returncode}"
        failures.append(f"{manager}: {detail}")

    detail = "; ".join(failures) if failures else "no supported package manager found"
    raise RuntimeError(f"jq is required and automatic installation failed: {detail}")


def ensure_submodules() -> None:
    """Initialize submodules and host tools required by the optimization pipeline.

    Covers: gpu-wiki/3rdparty (KernelWiki), 3rdparty/ncu-report-skill, 3rdparty/humanize.
    Skips reference-projects (large, optional — only needed for L2 search).
    Idempotent: already-initialized submodules are untouched.
    """
    needed = [
        ("gpu-wiki/3rdparty/", REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki" / "README.md"),
        ("3rdparty/ncu-report-skill", REPO_ROOT / "3rdparty" / "ncu-report-skill" / "SKILL.md"),
        ("3rdparty/humanize", HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md"),
    ]
    to_init = [path for path, marker in needed if not marker.exists()]
    if to_init:
        print(f"[orchestrator] initializing submodules: {to_init}", flush=True)
        cmd = ["git", "submodule", "update", "--init", "--depth", "1", "--"] + to_init
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        # verify
        for path, marker in needed:
            if not marker.exists():
                raise RuntimeError(
                    f"submodule init failed for {path} — {marker} not found. "
                    "Run `git submodule update --init` manually."
                )
        print("[orchestrator] all submodules ready", flush=True)
    ensure_jq()


def run_session(
    workspace: Path,
    prompt: str,
    timeout: int,
    agent_cli: str = "claude",
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    reasoning_effort: str = "max",
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    session_id = str(uuid.uuid4())
    cmd = _session_command(agent_cli, prompt, session_id, reasoning_effort)
    env = _session_env(agent_cli)
    env["IS_SANDBOX"] = "1"
    if sandbox_hardware:
        env["ATREX_SANDBOX_GPU"] = sandbox_hardware
    if sandbox_url:
        env["ATREX_SANDBOX_URL"] = sandbox_url
        env.pop("ATREX_SANDBOX_PROFILE", None)
    elif sandbox_profile:
        env["ATREX_SANDBOX_PROFILE"] = sandbox_profile
        env.pop("ATREX_SANDBOX_URL", None)
    env["ATREX_SANDBOX_TIMEOUT"] = str(sandbox_timeout)
    stdout, stderr, exit_status, timed_out = _run_bounded(cmd, cwd=workspace, timeout=timeout, env=env)
    return SessionResult(
        exit_status=exit_status,
        timed_out=timed_out,
        tokens=_tokens_from_stream(stdout),
        stdout_tail=stdout[-2000:],
        stderr_tail=stderr[-2000:],
        session_id=session_id,
    )


def sandbox_directive(hardware: str, profile: str = "", url: str = "") -> str:
    """Mandatory execution boundary injected into every optimization session."""
    if url:
        endpoint = f" using gateway URL `{url}`"
    elif profile:
        endpoint = f" using gateway profile `{profile}`"
    else:
        endpoint = " using agate's configured gateway"
    return (
        "## GPU sandbox execution (mandatory)\n\n"
        f"- Target gateway hardware: **{hardware}**{endpoint}. All GPU execution must cross this "
        "gateway boundary, including when the endpoint is localhost; source edits, optimizer state, and "
        "Git operations remain in the workspace.\n"
        "- Run every correctness or performance test through the gateway sandbox's `run` interface. "
        "Always pass `--kind run --no-memory`; "
        "read the emitted `[test_kernel] RESULT_JSON=...` line, then update `memory/v<N>.json` locally.\n"
        "  ```bash\n"
        "  python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v<N> --no-memory\n"
        "  python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v<N> --multi-seed 5 --no-memory\n"
        "  ```\n"
        "  The harness must benchmark only the base seed. Additional `--multi-seed` runs are "
        "correctness-only (no warmup/timing/reference benchmark repetition), so the full robustness "
        "check stays within the gateway's 600-second execution limit without reducing shape or seed "
        "coverage. Follow the declared evaluator route: an orchestrator-installed Atrex-Bench adapter "
        "must never be edited; only a derived legacy boundary may create its harness before V0. After "
        "V0 every route's harness remains immutable.\n"
        "- Sandbox uploads are allowlist-only. `test_kernel.py`, dispatch-signature collection, standard "
        "profile wrappers, and direct `import kernel` checks select their required inputs automatically. "
        "For any nonstandard command or dynamically opened local file, declare each dependency before `--` "
        "with repeatable `--input <relative-file-or-directory>`. Never use a generic `python -c print(...)` "
        "job to infer evaluator health; it does not exercise the evaluator payload or code path.\n"
        "- Run NVIDIA/AMD profiling through the typed `profile` interface. The structured response is "
        "saved as `profiles/v<N>/gateway_profile.json`; the wrapper command after `--` is used only as "
        "a dev fallback when the endpoint cannot represent the workload:\n"
        "  ```bash\n"
        "  python tools/sandbox.py --kind profile --profile-level sol --sync profiles/v<N> -- bash tools/profile_nvidia.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N>\n"
        "  python tools/sandbox.py --kind profile --profile-level sol --sync profiles/v<N> -- bash tools/profile_kernel.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N>\n"
        "  ```\n"
        "  Use `--profile-level deep --kernel-regex '^<exact kernel name>$'` for a focused typed profile. "
        "If source-line correlation or another typed-profile gap is specifically required, the sandbox may "
        "use the supplied wrapper through `dev`; do not choose dev merely for convenience.\n"
        "- Never run `test_kernel.py`, GPU timers, `ncu`, `rocprofv3`, or the profile wrappers outside "
        "the gateway interface. Never upload or create optimizer `memory/` as worker state; memory updates, "
        "plans, edits, and git operations stay local.\n"
        "- Never delete or move Git-tracked workspace files or directories, including `memory/`, "
        "`roofline.json`, helpers, historical plans, and profiles. Do not remove local state to shrink a "
        "sandbox payload; the sandbox wrapper owns input filtering.\n"
        "- The campaign dependency environment is immutable. Never run `pip`, `python -m pip`, `uv pip`, "
        "`conda`, `setup.py`, or another package installer/build command on the host or through the gateway. "
        "Use only preinstalled dependencies. If an import is unavailable, record the blocker or choose an "
        "implementation that uses available tooling; do not install or locally compile a third-party library. "
        "Do not import or execute JIT-capable GPU package code directly on the host: even a preinstalled "
        "package such as `flashinfer`, `flash_attn`/`flash-attn`, `xformers`, or `vllm` can invoke `ninja`, `ptxas`, "
        "or `nvcc` on first use. Static source inspection is allowed; route any import/API probe/benchmark "
        "that may initialize GPU code through `tools/sandbox.py`. "
        "The orchestrator terminates a session that violates this rule.\n"
        "- The gateway is shared infrastructure owned by the orchestrator/monitor, not by a coding session. "
        "Never start, stop, restart, signal, or replace its service or `screen` session; never delete or edit "
        "its configured state directory, job database/log, or cancel/modify gateway jobs directly. "
        "If the endpoint is unavailable, "
        "record an infrastructure failure and exit so the orchestrator can retry. Do not attempt to repair it.\n"
    )


def _sandbox_command(
    workspace: Path,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
    command: list[str],
    *,
    sync: tuple[str, ...] = (),
    wall_timeout: Optional[int] = None,
    dispatch_signatures: bool = False,
    gateway_kind: str = "auto",
) -> subprocess.CompletedProcess[str]:
    """Run one command through tools/sandbox.py and capture its user-visible output."""
    cmd = [
        sys.executable, str(SANDBOX_TOOL),
        "--kind", gateway_kind,
        "--hardware", hardware,
        "--workspace", str(workspace),
        "--timeout", str(timeout),
    ]
    if url:
        cmd += ["--url", url]
    elif profile:
        cmd += ["--gateway-profile", profile]
    if dispatch_signatures:
        cmd.append("--dispatch-signatures")
    if sync:
        for path in sync:
            cmd += ["--sync", path]
    else:
        cmd.append("--no-sync")
    cmd += ["--", *command]
    return subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        # Gateway execution timeout starts only after a worker claims the job.
        # The local wait must additionally tolerate time spent in a shared queue.
        timeout=wall_timeout if wall_timeout is not None else timeout + 240,
    )


def _test_result_from_stdout(stdout: str) -> dict:
    """Read the structured result emitted by reference/test_kernel.py."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(TEST_RESULT_PREFIX):
            result = json.loads(line[len(TEST_RESULT_PREFIX):])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")


def _record_local_test_result(workspace: Path, version: str, result: dict) -> Path:
    """Merge a remote --no-memory test result into local optimizer memory."""
    mem_dir = workspace / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / f"{version}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("version", version)
    data.setdefault("masked", False)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    perf = data.setdefault("performance", {})
    perf["latency_us"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_geomean"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_arith_mean"] = result.get("latency_us_arith_mean", 0.0)
    perf["latency_us_by_shape"] = result.get("latency_us_by_shape", {})
    perf["speedup_vs_ref_geomean"] = result.get("speedup_vs_ref_geomean", 0.0)
    all_pass = bool(result.get("all_pass"))
    corr = data.setdefault("correctness", {})
    corr["status"] = "PASS" if all_pass else "FAIL"
    corr["max_abs_err"] = result.get("max_abs_err", 0.0)
    corr["max_rel_err"] = result.get("max_rel_err", 0.0)
    gate = data.setdefault("quality_gate", {})
    gate["result"] = "PASS" if all_pass else "FAIL"
    failures = result.get("failures") or []
    gate["failure_reason"] = None if all_pass else "; ".join(map(str, failures))[:2000]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ── workspace / memory readers ────────────────────────────────────────────────


def latest_version(workspace: Path) -> int:
    """Largest STRICT integer version present as memory/v<N>.json.

    Note: int('71_512') == 71512 in Python (PEP 515 underscores-in-numeric-literals).
    Experimental files named like v71_512.json (an experiment branch for the v71 round) must
    NOT count as v71512 — they're scratch variants of v71, not real iterations. We therefore
    accept only stems matching ^v\\d+\\.json$ exactly.
    """
    import re
    _STRICT = re.compile(r"^v(\d+)\.json$")
    mem = workspace / "memory"
    if not mem.exists():
        return -1
    vs = []
    for p in mem.glob("v*.json"):
        m = _STRICT.match(p.name)
        if m is not None:
            vs.append(int(m.group(1)))
    return max(vs) if vs else -1


def read_memory(workspace: Path, n: int) -> Optional[dict]:
    path = workspace / "memory" / f"v{n}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def memory_record_is_empty(memory: Optional[dict]) -> bool:
    """Whether a record carries no findings at all.

    `memory_manager create` writes an all-null template, so a session killed mid-round usually
    leaves a file with no verdict, no measurement, and no lessons. Treating that as a real record
    would silently skip recovery and hand the next session a blank round.
    """
    if not memory:
        return True
    gate = (memory.get("quality_gate") or {}).get("result")
    status = (memory.get("correctness") or {}).get("status")
    latency = (memory.get("performance") or {}).get("latency_us")
    category = (memory.get("optimization") or {}).get("action_category")
    lessons = memory.get("pitfalls_and_fixes") or memory.get("search_log") or memory.get("open_directions")
    return not (gate or status or isinstance(latency, (int, float)) or category or lessons)


def session_transcript_path(agent_cli: str, session_id: str) -> Optional[Path]:
    """Locate a finished session's transcript so a post-mortem session can read what it tried.

    Resolved by globbing for the session id rather than reconstructing the CLI's project-slug
    encoding, which is an internal convention. Codex sessions are ephemeral and leave none.
    """
    if not session_id:
        return None
    roots = {"claude": ".claude", "qodercli": ".qoder"}
    root = roots.get(agent_cli)
    if root is None:
        return None
    for candidate in sorted((Path.home() / root / "projects").glob(f"*/{session_id}.jsonl")):
        return candidate
    return None


def record_interrupted_iteration(
    workspace: Path,
    n: int,
    *,
    kind: str,
    exit_status: int,
    timed_out: bool,
    timeout_s: int,
    stderr_tail: str = "",
) -> Path:
    """Mechanical fallback record for an iteration killed before it wrote its own memory.

    Every version must leave a record: `latest_version` and the next session's history read are
    both driven by `memory/v*.json`, so a hole makes the following sessions re-derive state from
    scratch and re-run already-refuted experiments. Merges into a partial record instead of
    overwriting it, and stays uncommitted so it survives the safety-net `git reset --hard`.
    """
    memory_path = workspace / "memory" / f"v{n}.json"
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        memory = {}
    if not isinstance(memory, dict):
        memory = {}

    if timed_out:
        cause = f"{kind} session killed by the {timeout_s}s hang backstop"
        status = "TIMEOUT_FAIL"
    else:
        cause = f"{kind} session crashed with exit={exit_status}"
        status = "FAIL"
    tail = stderr_tail.strip()[:500]
    failure_reason = f"{cause}; no result was recorded by the session" + (f". stderr: {tail}" if tail else "")

    memory["version"] = f"v{n}"
    memory["masked"] = False
    memory.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    memory["git_commit_hash"] = None
    memory["quality_gate"] = {"result": "FAIL", "failure_reason": failure_reason}
    correctness = memory.setdefault("correctness", {})
    if isinstance(correctness, dict):
        correctness["status"] = status
    else:
        memory["correctness"] = {"status": status}
    optimization = memory.setdefault("optimization", {})
    if not isinstance(optimization, dict):
        optimization = {}
        memory["optimization"] = optimization
    optimization["action_category"] = INTERRUPTED_CATEGORY
    optimization["action_description"] = (
        f"{cause}; whatever it was attempting is unvalidated and its conclusions are untrustworthy"
    )
    pitfalls = memory.setdefault("pitfalls_and_fixes", [])
    if not isinstance(pitfalls, list):
        pitfalls = []
        memory["pitfalls_and_fixes"] = pitfalls
    pitfalls.append({
        "error_type": "infra",
        "error_message": failure_reason,
        "lesson": (
            "This round was cut off by infrastructure, not by evidence. Do not treat its direction "
            "as a refuted dead-end, and do not assume any measurement from it is valid."
        ),
    })

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return memory_path


def mask_half_memory(workspace: Path, iteration: int) -> list[str]:
    """Randomly mask half of active optimization memory every 100 iterations.

    The baseline and the latest record stay visible so the next iteration retains its
    correctness anchor and can read v<PREV>. The latest recorded committed win is also
    preserved when it can be identified from git.
    """
    if iteration <= 0 or iteration % MEMORY_MASK_INTERVAL != 0:
        return []

    active: list[tuple[int, Path, dict]] = []
    for n in range(1, latest_version(workspace) + 1):
        data = read_memory(workspace, n)
        if data is not None and not data.get("masked", False):
            active.append((n, workspace / "memory" / f"v{n}.json", data))
    if not active:
        return []

    latest_n, latest_path, latest_data = active[-1]
    pitfalls = latest_data.get("pitfalls_and_fixes")
    if isinstance(pitfalls, list) and any(
        item.get("error_type") == "periodic_memory_mask" and item.get("iteration") == iteration
        for item in pitfalls if isinstance(item, dict)
    ):
        return []

    protected = {latest_n}
    committed_wins = [
        n for n, _, data in active
        if commit_changed_kernel(workspace, data.get("git_commit_hash"))
    ]
    if committed_wins:
        protected.add(committed_wins[-1])

    eligible = [(n, path, data) for n, path, data in active if n not in protected]
    mask_count = min(len(active) // 2, len(eligible))
    selected = random.sample(eligible, mask_count)
    masked_versions = [f"v{n}" for n, _, _ in selected]
    for _, path, data in selected:
        data["masked"] = True
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not isinstance(pitfalls, list):
        pitfalls = []
        latest_data["pitfalls_and_fixes"] = pitfalls
    pitfalls.append({
        "error_type": "periodic_memory_mask",
        "iteration": iteration,
        "error_message": f"periodic memory refresh masked {', '.join(masked_versions)}",
        "lesson": "Use the remaining history as hints and reconsider previously discarded directions.",
    })
    latest_path.write_text(
        json.dumps(latest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[orchestrator] iteration {iteration}: randomly masked {mask_count}/{len(active)} "
        f"active memory records (kept v0, v{latest_n}, and latest committed win)",
        flush=True,
    )
    return masked_versions


# ── git is the SINGLE source of truth for a "committed win" ───────────────────
# A real win is a commit that CHANGES kernel.py. A dead-end "record" commit leaves kernel.py
# identical to its parent. Everything (stall counter, target-met, convert incumbent) keys off
# this git fact, NOT off the LLM-filled git_commit_hash / quality_gate in memory (which can drift
# from what actually got committed). One primitive, reused everywhere: commit_changed_kernel().

STALL_STATE_FILE = ".orchestrator_state.json"


def git_head(workspace: Path) -> str:
    """Current HEAD sha, or '' if not a repo / no commits yet."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workspace),
                           capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def git_path_blob(workspace: Path, ref: str, path: str) -> str:
    """Committed blob id of one path at one ref, or '' when it is absent there."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{ref}:{path}"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_kernel_blob(workspace: Path) -> str:
    """Committed kernel.py blob id, stable across metadata-only commits."""
    return git_path_blob(workspace, "HEAD", "kernel.py")


def git_worktree_blob(workspace: Path, path: str) -> str:
    """Blob id of the on-disk file, or '' when it is missing."""
    if not (workspace / path).is_file():
        return ""
    try:
        result = subprocess.run(
            ["git", "hash-object", "--", path],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def head_kernel_is_initial_baseline(workspace: Path) -> bool:
    """Whether committed HEAD still uses the repository's original V0 kernel.

    Production V0 is intentionally allowed to be a PyTorch reference wrapper.
    Later failed-policy/dead-end records may advance ``memory/vN.json`` and Git
    metadata without changing ``kernel.py``.  Resume must recognize that state
    by kernel provenance, not by assuming ``latest_version > 0`` means an
    optimized kernel was accepted.

    Once the framework-baseline stage lands v1 this returns False, because the
    accepted framework kernel is a real kernel change.  The framework baseline
    is tracked by ``framework_baseline.json``, not by this predicate.
    """
    roots = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    root_commits = roots.stdout.split() if roots.returncode == 0 else []
    if len(root_commits) != 1:
        return False
    baseline_blob = subprocess.run(
        ["git", "rev-parse", f"{root_commits[0]}:kernel.py"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if baseline_blob.returncode != 0:
        return False
    return bool(
        baseline_blob.stdout.strip()
        and baseline_blob.stdout.strip() == git_kernel_blob(workspace)
    )


def read_framework_baseline(workspace: Path) -> Optional[dict]:
    """Read the framework-baseline marker from committed HEAD, never from the worktree.

    An interrupted session can leave an unstaged marker behind; trusting it would pin bucket
    baselines to a kernel that was never validated.
    """
    show = subprocess.run(
        ["git", "show", f"HEAD:{FRAMEWORK_BASELINE_FILE}"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        return None
    try:
        marker = json.loads(show.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{workspace / FRAMEWORK_BASELINE_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"{workspace / FRAMEWORK_BASELINE_FILE} must contain a JSON object")
    return marker


def resolve_framework_baseline_commit(workspace: Path) -> tuple[str, int]:
    """Return the pinned (commit, version) of the framework baseline, or ("", 0) when unpinned.

    Verification is fail-closed and never consults HEAD's tree, so the pin survives HEAD
    advancing to an aggregated dispatcher kernel. A broken marker is an error rather than a
    silent fallback to the root commit: falling back would mix PyTorch and framework
    provenance across buckets, which is exactly what pinning exists to prevent.
    """
    marker = read_framework_baseline(workspace)
    if marker is None:
        return "", 0
    commit = _normalize_commit_hash(marker.get("commit"))
    if not commit:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has no usable commit hash")
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} points at missing commit {commit[:12]}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"{FRAMEWORK_BASELINE_FILE} commit {commit[:12]} is not an ancestor of HEAD"
        )
    blob = subprocess.run(
        ["git", "rev-parse", f"{commit}:kernel.py"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    recorded_blob = str(marker.get("kernel_blob") or "").strip()
    if blob.returncode != 0 or not recorded_blob or blob.stdout.strip() != recorded_blob:
        raise RuntimeError(
            f"{FRAMEWORK_BASELINE_FILE} kernel blob does not match commit {commit[:12]}"
        )
    version_text = str(marker.get("version") or "")
    match = re.fullmatch(r"v(\d+)", version_text)
    if match is None:
        raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has an unusable version {version_text!r}")
    return commit, int(match.group(1))


def head_kernel_is_framework_baseline(workspace: Path) -> bool:
    """Whether the committed HEAD kernel is still exactly the pinned framework baseline."""
    marker = read_framework_baseline(workspace)
    if marker is None:
        return False
    recorded_blob = str(marker.get("kernel_blob") or "").strip()
    return bool(recorded_blob and recorded_blob == git_kernel_blob(workspace))


def preserve_interrupted_tracked_changes(workspace: Path, context: str) -> str:
    """Stash tracked edits left by an interrupted optimizer session.

    Bucket aggregation is defined in terms of committed Git HEADs.  A killed
    coding session can nevertheless leave a newer, half-written ``kernel.py``
    in the worktree.  Aggregating while that file is present would silently
    mix an unvalidated candidate into the full kernel.  Preserve the tracked
    edits in a recoverable stash and resume from the committed state; untracked
    plans/profiles remain available as research artifacts for the next round.

    Return the created stash commit, or an empty string when the worktree had
    no tracked changes.
    """
    if not git_head(workspace):
        return ""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        return ""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = f"orchestrator interrupted-worktree recovery ({context}) {timestamp}"
    subprocess.run(
        ["git", "stash", "push", "--quiet", "--message", message],
        cwd=str(workspace),
        check=True,
    )
    stash = subprocess.run(
        ["git", "rev-parse", "refs/stash"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"[orchestrator] preserved interrupted tracked edits in "
        f"{workspace} as stash {stash[:8]} ({context})",
        flush=True,
    )
    return stash


def git_untracked_paths(workspace: Path) -> set[str]:
    """Return non-ignored untracked files as repository-relative paths."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=True,
    )
    return {path for path in result.stdout.split("\0") if path}


def preserve_session_created_untracked(
    workspace: Path, before: set[str], context: str
) -> str:
    """Stash only untracked files created by one restricted coding session.

    Aggregate sessions are allowed to edit ``kernel.py`` only.  Preserve any
    extra files they create in Git's stash, while leaving pre-existing
    untracked research artifacts untouched.  This keeps the aggregate
    workspace reproducible without destructively discarding agent output.
    """
    created = sorted(git_untracked_paths(workspace) - before)
    if not created:
        return ""
    for relative in created:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe untracked path reported by Git: {relative!r}")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message = f"orchestrator restricted-session artifacts ({context}) {timestamp}"
    subprocess.run(
        [
            "git",
            "stash",
            "push",
            "--quiet",
            "--include-untracked",
            "--message",
            message,
            "--",
            *created,
        ],
        cwd=str(workspace),
        check=True,
    )
    remaining = git_untracked_paths(workspace)
    if any(path in remaining for path in created):
        raise RuntimeError("Git stash did not preserve every session-created artifact")
    stash = subprocess.run(
        ["git", "rev-parse", "refs/stash"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"[orchestrator] preserved {len(created)} restricted-session artifact(s) "
        f"from {workspace} in stash {stash[:8]} ({context})",
        flush=True,
    )
    return stash


def _normalize_commit_hash(ref: object) -> str:
    """Return a safe git commit hash from an LLM-authored memory value.

    A short SHA containing only decimal digits can be serialized as a JSON
    number (for example ``7847485``).  Preserve that valid value by converting
    integers back to strings, while rejecting booleans, floats, and arbitrary
    git revisions/options.  Memory records are expected to contain commit
    hashes, not general revision expressions.
    """
    if isinstance(ref, bool):
        return ""
    if isinstance(ref, int):
        ref = str(ref)
    if not isinstance(ref, str):
        return ""
    ref = ref.strip()
    return ref if re.fullmatch(r"[0-9a-fA-F]{4,64}", ref) else ""


def commit_changed_kernel(workspace: Path, ref: object) -> bool:
    """True iff commit `ref` changed kernel.py vs its parent (i.e. a real win, not a dead-end
    record commit). The one git primitive the win/stall/incumbent logic all share.

    ``git_commit_hash`` is written by agents, so normalize it before passing it
    to subprocess.  In particular, all-decimal short SHAs may round-trip
    through JSON as integers.
    """
    commit_hash = _normalize_commit_hash(ref)
    if not commit_hash:
        return False
    try:
        r = subprocess.run(["git", "show", "--numstat", "--format=", commit_hash, "--", "kernel.py"],
                           cwd=str(workspace), capture_output=True, text=True)
    except (OSError, TypeError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def kernel_won(workspace: Path, pre_head: str) -> bool:
    """True iff the session produced a real win: kernel.py differs between pre_head and HEAD.
    (A transition check across the session — the session may make several commits.)"""
    if not pre_head:
        return False
    try:
        r = subprocess.run(["git", "diff", "--quiet", pre_head, "HEAD", "--", "kernel.py"],
                           cwd=str(workspace), capture_output=True)
    except OSError:
        return False
    return r.returncode == 1  # 0 = identical, 1 = differs


def read_stall(workspace: Path) -> Optional[int]:
    """Persisted live stall counter, or None when absent (caller reconstructs)."""
    p = workspace / STALL_STATE_FILE
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text(encoding="utf-8")).get("stall")
    except (OSError, ValueError):
        return None
    return int(v) if isinstance(v, int) else None


def write_stall(workspace: Path, stall: int) -> None:
    """Persist the live stall counter so a restart resumes the exact value (survives git reset —
    the file is gitignored). This is the single source of truth for the stall->convert cooldown."""
    try:
        (workspace / STALL_STATE_FILE).write_text(
            json.dumps({"stall": int(stall)}, indent=2), encoding="utf-8")
    except OSError:
        pass


def reconstruct_stall(workspace: Path) -> int:
    """Best-effort rebuild of the live stall counter from git when no persisted state exists yet
    (e.g. a workspace from before state was tracked). Counts trailing commits from HEAD that did
    NOT change kernel.py; a win (kernel.py change) stops the count. read_stall() is authoritative —
    this only bootstraps it, so it does not attempt to replay convert-issued resets."""
    try:
        r = subprocess.run(["git", "rev-list", "HEAD"], cwd=str(workspace),
                           capture_output=True, text=True)
    except OSError:
        return 0
    if r.returncode != 0:
        return 0
    trailing = 0
    for h in r.stdout.split():
        if commit_changed_kernel(workspace, h):   # a win -> stop counting
            break
        trailing += 1
    return trailing


def peak_util(mem: Optional[dict]) -> float:
    """Max of tflops / bandwidth peak utilization (%), 0 if unknown."""
    if not mem:
        return 0.0
    perf = mem.get("performance") or {}
    vals = [perf.get("tflops_peak_utilization_pct"), perf.get("bandwidth_peak_utilization_pct")]
    return max([float(v) for v in vals if isinstance(v, (int, float))] or [0.0])


def best_validated_latency_us(workspace: Path, *, from_version: Optional[int] = None) -> Optional[float]:
    """Best correctness-passing measured latency in a workspace.

    Versions before a pinned framework baseline are excluded by default: production cannot ship
    the PyTorch V0 wrapper, so treating its (typically much faster, library-backed) latency as the
    incumbent would reject every candidate a from-scratch DSL kernel can produce.
    """
    if from_version is None:
        _pinned_commit, pinned_version = resolve_framework_baseline_commit(workspace)
        from_version = pinned_version
    best: Optional[float] = None
    for version in range(from_version, latest_version(workspace) + 1):
        memory = read_memory(workspace, version)
        if not memory:
            continue
        gate = (memory.get("quality_gate") or {}).get("result")
        correctness = (memory.get("correctness") or {}).get("status")
        if gate != "PASS" and correctness != "PASS":
            continue
        latency = (memory.get("performance") or {}).get("latency_us")
        if isinstance(latency, (int, float)):
            best = float(latency) if best is None else min(best, float(latency))
    return best


def incumbent_latency(workspace: Path, upto_n: int) -> Optional[float]:
    """Best committed geomean latency (performance.latency_us) over versions [0, upto_n): the min
    among versions whose recorded commit git confirms was a real win (commit_changed_kernel). Git
    is the arbiter, so a reverted dead-end is excluded even if it recorded a hash. Used only for a
    convert session's performance-parity check — the revert TARGET is the pre-convert HEAD, which is
    always the incumbent (wins commit, dead-ends don't touch kernel.py)."""
    best: Optional[float] = None
    for i in range(0, upto_n):
        m = read_memory(workspace, i)
        h = m.get("git_commit_hash") if m else None
        if not commit_changed_kernel(workspace, h):
            continue
        lat = (m.get("performance") or {}).get("latency_us")
        if isinstance(lat, (int, float)) and (best is None or lat < best):
            best = float(lat)
    return best


def detect_arch(
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
) -> str:
    """Return the real runtime GPU architecture token (vendor-neutral), or '' if undetectable.

    NVIDIA/CUDA -> 'sm_<cap>' (e.g. 'sm_103'); AMD/ROCm -> the gfx arch (e.g. 'gfx942').
    Uses torch (get_device_capability / gcnArchName) — the AUTHORITATIVE source, which stays
    correct even when the GPU name / vendor SMI is DESENSITIZED (e.g. a target GPU reporting a
    generic compatibility alias).
    """
    code = (
        "import torch\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "if getattr(torch.version,'hip',None):\n"
        "    print(getattr(p,'gcnArchName','').split(':')[0])\n"
        "else:\n"
        "    c=torch.cuda.get_device_capability(0); print('sm_%d%d'%(c[0],c[1]))\n"
    )
    if sandbox_hardware:
        try:
            with tempfile.TemporaryDirectory(prefix="atrex-arch-") as temp_dir:
                result = _sandbox_command(
                    Path(temp_dir), sandbox_hardware, sandbox_profile, sandbox_url, 120,
                    ["python", "-c", code],
                )
            if result.returncode == 0:
                for line in reversed(result.stdout.splitlines()):
                    value = line.strip()
                    if re.fullmatch(r"sm_\d+|gfx[0-9a-fA-F]+", value):
                        return value
            print(
                f"[orchestrator] WARNING: sandbox arch detection failed on {sandbox_hardware}: "
                f"{result.stderr[-1000:]}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[orchestrator] WARNING: sandbox arch detection failed: {exc}",
                  file=sys.stderr, flush=True)
        return ""

    for py in ("python", "python3", sys.executable):
        try:
            out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
            s = out.stdout.strip()
            if s:
                return s
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def hardware_directive(platform: str, arch: str) -> str:
    """Authoritative, vendor-neutral hardware-identity block injected into every session.

    Guards against desensitized boxes: the agent must target the real architecture from the
    runtime API, not the (possibly faked) device name. Deliberately does NOT prescribe any
    vendor's feature set — the agent maps the detected arch to its own codegen choices, so this
    works on NVIDIA (Hopper/Blackwell/...) and AMD (CDNA/...) alike.
    """
    real = f"**{arch}**" if arch else "whatever the runtime GPU API reports"
    return (
        "## Hardware ground truth (authoritative — read before choosing an algorithm)\n\n"
        f"- Intended target hardware: **{platform}**. Real runtime GPU architecture: {real} — from the "
        "runtime API (`torch.cuda.get_device_capability()` on CUDA; the device gfx arch on ROCm). This is "
        "the ONLY source to trust for the architecture.\n"
        "- **The GPU *name* and vendor SMI (`nvidia-smi` / `rocm-smi`) on this box may be DESENSITIZED / "
        "FAKED** — they can report an older or entirely different GPU than the real silicon. Do NOT infer "
        "the architecture, vendor, or feature set from the device name; if it disagrees with the runtime "
        "API, the runtime API wins.\n"
        "- Design *and* build for the real architecture above: select the code paths, instructions, and "
        "build/target flags your DSL/compiler exposes for THAT architecture and generation. Do NOT fall "
        "back to an older-arch portable path because of the device name, and do NOT assume a different "
        "vendor or generation than the detected one.\n"
    )



def _install_codex_humanize_skills(skills_dir: Path) -> None:
    """Install a workspace-local, hydrated Humanize subset for Codex.

    Humanize's upstream Codex installer also changes user-global hooks and configuration. The
    orchestrator must not mutate global state, so this mirrors only the skill/runtime hydration
    into the campaign's repository-scoped ``.agents/skills`` directory.
    """
    source_skills = HUMANIZE_DIR / "skills"
    if not (source_skills / "humanize-gen-plan" / "SKILL.md").is_file():
        return

    skill_names = (
        "humanize",
        "humanize-gen-plan",
        "humanize-refine-plan",
        "humanize-rlcr",
    )
    runtime_root = skills_dir / "humanize"
    for skill_name in skill_names:
        source = source_skills / skill_name / "SKILL.md"
        if not source.is_file():
            continue
        destination_dir = skills_dir / skill_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8").replace(
            "{{HUMANIZE_RUNTIME_ROOT}}", str(runtime_root)
        )
        # These Claude plugin frontmatter keys are stripped by Humanize's own Codex installer.
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith((
                "user-invocable:",
                "disable-model-invocation:",
                "hide-from-slash-command-tool:",
            ))
        ) + "\n"
        destination = destination_dir / "SKILL.md"
        if not destination.exists() or destination.read_text(encoding="utf-8") != text:
            destination.write_text(text, encoding="utf-8")

    for component in ("scripts", "hooks", "prompt-template", "templates", "config", "agents"):
        source = HUMANIZE_DIR / component
        destination = runtime_root / component
        if source.exists() and not destination.exists():
            os.symlink(source, destination)


def _agent_runtime_directive(agent_cli: str) -> str:
    if agent_cli == "codex":
        return (
            "- `.agents/skills/` — repository-local Codex skills, including "
            "`gpu-kernel-baseline`, `ncu-report-skill`, `KernelWiki`, and "
            "`humanize-gen-plan`. Invoke a named skill with Codex's `$skill-name` syntax."
        )
    return (
        "- `.claude/skills/ncu-report-skill/` — NVIDIA profiling skill.\n"
        "- `.claude/skills/KernelWiki/` — kernel optimization knowledge base."
    )


def _baseline_driver_directive(agent_cli: str) -> str:
    if agent_cli == "codex":
        return (
            "Use the `$gpu-kernel-baseline` skill and complete its baseline workflow in this "
            "session. If Codex collaboration/sub-agent tools are available, delegate that bounded "
            "implementation task and wait for it; otherwise execute the skill directly yourself"
        )
    if agent_cli == "qodercli":
        return (
            "Complete the baseline workflow directly in this Qoder session. Do not launch an "
            "Agent/subagent: nested workload-bucket repositories can otherwise be resolved as the "
            "aggregate parent workspace. Treat the current working directory as the only writable "
            "workspace and use relative paths for every campaign file"
        )
    return (
        "Launch the `gpu-kernel-baseline` subagent (by name). You may spawn it in the "
        "background, but **you MUST wait for it to complete before you exit**"
    )


def _plan_generator_directive(agent_cli: str, version: int) -> str:
    draft = f"plans/v{version}_draft.md"
    plan = f"plans/v{version}_plan.md"
    if agent_cli == "codex":
        return (
            f"Invoke the `$humanize-gen-plan` skill with `{draft}` as input and `{plan}` as "
            "output. Use direct/no-discussion mode for this single-action optimization plan. "
            "The skill is repository-local under `.agents/skills/`; do not look for a slash "
            "command or Claude plugin."
        )
    if agent_cli == "qodercli":
        return (
            f"Read `{draft}` and generate `{plan}` yourself in this Qoder session. Do not invoke "
            "Humanize, a slash command, the Skill tool, or a planning subagent. Write a complete "
            "one-shot plan containing the evidence-to-action chain, exactly one optimization "
            "category, concrete file changes, correctness/performance validation steps, and "
            "measurable acceptance criteria. Preserve the draft's Search Log and constraints."
        )
    return (
        "```text\n"
        f"/humanize:gen-plan --input {draft} --output {plan} --direct\n"
        "```"
    )


def link_runtime(workspace: Path, atrex_bench_root: Optional[Path] = None) -> None:
    """Make the skill's `tools/`, `reference/`, `skills/`, `reference-projects/`, `gpu-wiki/` resolvable from cwd=workspace.

    The gpu-kernel-* skills reference these by relative path; sessions run with cwd=workspace,
    so symlink them in (absolute targets, so the workspace can live anywhere). Idempotent.

    Also installs the same skills and agent definitions into ``.claude/`` and ``.qoder/``, and
    repository-local Codex skills into ``.agents/skills/``.

    Claude loads Humanize via ``--plugin-dir`` after the orchestrator provisions ``jq``. Qoder
    owns plan generation directly and does not load Humanize. Codex receives a repository-scoped,
    hydrated Humanize skill without changing global user state.
    """
    for sub in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    if atrex_bench_root is not None:
        evaluator = atrex_bench_root / "scripts" / "run_eval.py"
        package = atrex_bench_root / "src" / "atrex_bench"
        if not evaluator.is_file() or not package.is_dir():
            raise FileNotFoundError(
                f"invalid Atrex-Bench runtime root (missing run_eval.py/src): {atrex_bench_root}"
            )
        runtime_link = workspace / "atrex-bench"
        if runtime_link.is_symlink():
            if runtime_link.resolve() != atrex_bench_root.resolve():
                raise RuntimeError(
                    f"workspace Atrex-Bench runtime points at {runtime_link.resolve()}, "
                    f"expected {atrex_bench_root.resolve()}"
                )
        elif runtime_link.exists():
            raise RuntimeError(
                f"workspace path blocks the Atrex-Bench runtime link: {runtime_link}"
            )
        else:
            os.symlink(atrex_bench_root.resolve(), runtime_link)
    # Claude and Qoder use parallel project-local discovery roots. Keep their contents identical
    # so selecting a different --agent-cli does not change the available optimization knowledge.
    ncu_src = REPO_ROOT / "3rdparty" / "ncu-report-skill"
    kw_src = REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki"
    agents_src = REPO_ROOT / "agents"
    for runtime_dir_name in (".claude", ".qoder"):
        runtime_dir = workspace / runtime_dir_name
        runtime_skills_dir = runtime_dir / "skills"
        runtime_agents_dir = runtime_dir / "agents"
        runtime_skills_dir.mkdir(parents=True, exist_ok=True)
        for src, name in ((ncu_src, "ncu-report-skill"), (kw_src, "KernelWiki")):
            dst = runtime_skills_dir / name
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
        # The prompts reference these agent definitions by name (gpu-kernel-baseline,
        # gpu-kernel-convert, gpu-kernel-profiler, gpu-kernel-research, kernel-optimize).
        if agents_src.exists() and not runtime_agents_dir.exists():
            os.symlink(agents_src, runtime_agents_dir)

    # Codex discovers repository-scoped skills from .agents/skills. Keep these local to the
    # campaign so choosing --agent-cli codex neither requires nor mutates the user's global
    # CODEX_HOME. The project-native optimization skills can remain symlinks; Humanize needs a
    # hydrated SKILL.md, so it is materialized by the helper above.
    codex_skills_dir = workspace / ".agents" / "skills"
    codex_skills_dir.mkdir(parents=True, exist_ok=True)
    project_skills = REPO_ROOT / "skills"
    if project_skills.is_dir():
        for source in project_skills.iterdir():
            if not (source / "SKILL.md").is_file():
                continue
            destination = codex_skills_dir / source.name
            if not destination.exists():
                os.symlink(source, destination)
    for source, name in ((ncu_src, "ncu-report-skill"), (kw_src, "KernelWiki")):
        destination = codex_skills_dir / name
        if source.exists() and not destination.exists():
            os.symlink(source, destination)
    _install_codex_humanize_skills(codex_skills_dir)
    gi = workspace / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    add = ""
    if "/tools" not in existing:
        add += "\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n"
    if "/.claude" not in existing:
        add += "/.claude\n"
    if "/.qoder" not in existing:
        add += "/.qoder\n"
    if "/.agents" not in existing:
        add += "/.agents\n"
    if atrex_bench_root is not None and "/atrex-bench" not in existing:
        add += "/atrex-bench\n"
    if "/" + STALL_STATE_FILE not in existing:
        add += ("\n# orchestrator live stall counter (rebuilt on restart; never committed)\n"
                "/" + STALL_STATE_FILE + "\n")
    if add:
        with gi.open("a", encoding="utf-8") as fh:
            fh.write(add)


# ── campaign ──────────────────────────────────────────────────────────────────


@dataclass
class Campaign:
    name: str
    kernel_demo: str
    platform: str
    framework: str
    notes: str = "none"
    arch: str = ""                 # real runtime GPU arch e.g. "sm_103" / "gfx942"; auto-detected
    work_dir: str = ""             # explicit working directory; "" = Path.cwd() (backward compat)
    workspace_suffix: str = ""     # internal auto-dispatch suffix, e.g. triton_h20
    max_iters: int = 20
    token_budget: int = 0          # 0 = no token cap (max-iters still bounds the run)
    target_util: float = 90.0
    iter_timeout: int = 5400       # 90 min hang-backstop per optimization session
    setup_timeout: int = 7200      # 120 min for the baseline session
    salvage_timeout: int = SALVAGE_TIMEOUT_S  # 0 = skip the post-mortem agent, record mechanically
    max_stall: int = 0             # 0 = disabled (budget-only); >0 = stop after N no-commit iters
    convert_after: int = DEFAULT_CONVERT_AFTER  # triton-only: mandatory Gluon conversion threshold
    sandbox_hardware: str = ""     # agate scheduler token, e.g. REMOTE_GPU (may differ from platform)
    sandbox_profile: str = ""      # pre/prod; empty preserves normal agate URL resolution
    sandbox_url: str = ""          # explicit endpoint, e.g. http://127.0.0.1:8000
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    atrex_bench_root: str = ""      # native shapes route: canonical checkout owning run_eval.py
    agent_cli: str = "claude"       # clean-session coding backend: claude, qodercli, or codex
    optimization_mode: str = "leaderboard"  # permissive contest flow or strict production gate
    framework_baseline: str = "auto"        # auto = production only; always | never override it
    framework_baseline_timeout: int = FRAMEWORK_BASELINE_TIMEOUT_S
    on_improvement: Optional[Callable[["Campaign", int, dict], None]] = field(
        default=None, repr=False, compare=False
    )
    on_iteration: Optional[
        Callable[["Campaign", int, Optional[dict], bool], None]
    ] = field(default=None, repr=False, compare=False)
    tokens_spent: int = field(default=0, init=False)

    @property
    def campaign_name(self) -> str:
        suffix = f"_{self.workspace_suffix}" if self.workspace_suffix else ""
        return f"{self.name}{suffix}"

    @property
    def workspace(self) -> Path:
        base = Path(self.work_dir) if self.work_dir else Path.cwd()
        return base / f"kernel_opt_{self.campaign_name}"

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(f"[orchestrator] {label}: exit={res.exit_status} timed_out={res.timed_out} "
              f"tokens={res.tokens} cum_tokens={self.tokens_spent}", flush=True)
        if res.exit_status != 0 or res.timed_out:
            print(f"[orchestrator] stderr tail:\n{res.stderr_tail}", file=sys.stderr, flush=True)

    def _salvage_memory(self, n: int, res: SessionResult, kind: str) -> bool:
        """Run a short record-only session that reconstructs the killed round into memory/v<N>.json."""
        if self.salvage_timeout <= 0:
            return False
        if res.timed_out:
            kill_reason = f"it hit the {self.iter_timeout}s hang backstop and was killed"
        else:
            kill_reason = (
                f"it exited with status {res.exit_status} "
                "(typically an API failure such as a provider rate limit, or a policy termination)"
            )
        transcript = session_transcript_path(self.agent_cli, res.session_id)
        prompt = _render(
            PROMPTS_DIR / "salvage.md",
            WORKSPACE=str(self.workspace),
            N=n,
            PREV=n - 1,
            KIND=kind,
            KILL_REASON=kill_reason,
            TRANSCRIPT=str(transcript) if transcript else "(no transcript was retained for this session)",
            STDOUT_TAIL=res.stdout_tail or "(empty)",
            MODE_POLICY=self._mode_directive(),
        )
        head_before = git_head(self.workspace)
        salvage_res = run_session(
            self.workspace, prompt, timeout=self.salvage_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
            reasoning_effort="high",
        )
        self._account(salvage_res, f"salvage v{n}")
        # A post-mortem session must not move HEAD. Undo with --soft: an unwanted commit disappears
        # while the interrupted round's uncommitted kernel edits stay in the worktree for the next session.
        if head_before and git_head(self.workspace) != head_before:
            subprocess.run(
                ["git", "reset", "--soft", head_before],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(
                f"[orchestrator] salvage v{n} committed against policy; HEAD restored to {head_before[:8]}",
                file=sys.stderr,
                flush=True,
            )
        return not memory_record_is_empty(read_memory(self.workspace, n))

    def _ensure_iteration_memory(self, n: int, res: SessionResult, kind: str) -> None:
        """Guarantee every round leaves a memory record, even when its session never finished one."""
        if not memory_record_is_empty(read_memory(self.workspace, n)):
            return
        salvaged = self._salvage_memory(n, res, kind)
        if memory_record_is_empty(read_memory(self.workspace, n)):
            record_interrupted_iteration(
                self.workspace, n,
                kind=kind,
                exit_status=res.exit_status,
                timed_out=res.timed_out,
                timeout_s=self.iter_timeout,
                stderr_tail=res.stderr_tail,
            )
            salvaged = False
        print(
            f"[orchestrator] v{n} memory: "
            f"{'reconstructed by salvage session' if salvaged else 'mechanical interrupted-round record'}",
            flush=True,
        )

    def _link_runtime(self) -> None:
        native_root = Path(self.atrex_bench_root) if self.atrex_bench_root else None
        link_runtime(self.workspace, native_root)
        install_workspace_policy(self.workspace, self.optimization_mode, self.framework)

    def _evaluator_directive(self) -> str:
        if self.atrex_bench_root:
            return (
                "## Evaluation route: Atrex-Bench native\n\n"
                "This workspace's `test_kernel.py` is an orchestrator-installed immutable adapter. "
                "It invokes the canonical `atrex-bench/scripts/run_eval.py` against `kernel.py` and "
                "the workspace `reference.py`/`input.py`/`shapes.json`/`metadata.json`, then emits "
                "the optimizer's `RESULT_JSON` transport line from the official `eval_result.json`. "
                "Do not edit or replace this adapter and do not implement a custom correctness or "
                "timing harness. `--multi-seed N` maps to N additional Atrex-Bench correctness "
                "cases while performance remains one official run per shape."
            )
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            return (
                "## Evaluation route: SOL-ExecBench\n\n"
                "Keep using the immutable SOL `test_kernel.py`, which invokes `sol-execbench` over "
                "the complete `workload.jsonl`. Do not substitute the Atrex-Bench native evaluator."
            )
        return (
            "## Evaluation route: derived legacy boundary\n\n"
            "This derived boundary is not a complete Atrex-Bench operator directory. Preserve its "
            "committed full-shape `test_kernel.py` methodology and do not replace it after V0."
        )

    def _install_native_evaluator(self) -> None:
        """Seed the immutable adapter used only by native Atrex-Bench shape campaigns."""
        if not self.atrex_bench_root:
            return
        if not ATREX_BENCH_HARNESS.is_file():
            raise FileNotFoundError(f"missing {ATREX_BENCH_HARNESS}")
        shutil.copy2(ATREX_BENCH_HARNESS, self.workspace / "test_kernel.py")

    def _sandbox_directive(self) -> str:
        return sandbox_directive(
            self.sandbox_hardware, self.sandbox_profile, self.sandbox_url
        )

    def _mode_directive(self) -> str:
        return optimization_mode_directive(self.optimization_mode, self.framework)

    def setup_baseline(self) -> None:
        # SOL-ExecBench op: seed a correct, directly-submittable V0 mechanically
        # (no baseline session) — sol_seed.py copies the ground-truth files, writes
        # the DPS wrapper kernel.py + solution.json; this method benches V0 in the sandbox.
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            self._setup_baseline_sol(op_dir)
            return
        if not WORKSPACE_INIT.exists():
            raise FileNotFoundError(f"missing {WORKSPACE_INIT}")
        # workspace_init.sh builds the workspace as $(pwd)/kernel_opt_<name>,
        # so cwd must be the work_dir (or the process cwd when --workspace is absent).
        subprocess.run(["bash", str(WORKSPACE_INIT), self.campaign_name, self.kernel_demo],
                       cwd=str(self.workspace.parent), check=True)
        # atrex-bench operators keep their immutable harness inputs beside
        # reference.py.  workspace_init.sh only copies the reference itself;
        # materialize the remaining ground truth before the baseline session.
        for name in ("reference.py", "input.py", "shapes.json", "roofline.json",
                     "metadata.json", "valid.py"):
            source = op_dir / name
            if source.is_file():
                shutil.copy2(source, self.workspace / name)
        self._link_runtime()
        self._install_native_evaluator()
        prompt = _render(
            PROMPTS_DIR / "setup.md",
            WORKSPACE=str(self.workspace), PLATFORM=self.platform,
            FRAMEWORK=self.framework, KERNEL_DEMO=self.kernel_demo,
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            BASELINE_DRIVER=_baseline_driver_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            EVALUATOR=self._evaluator_directive(),
            MODE_POLICY=self._mode_directive(),
        )
        res = run_session(
            self.workspace, prompt, timeout=self.setup_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
            reasoning_effort="high",
        )
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0) — "
                "this is likely an API key / authentication issue. "
                f"{_agent_auth_hint(self.agent_cli)}."
            )
        baseline_memory = read_memory(self.workspace, 0)
        baseline_problem = "missing memory/v0.json" if baseline_memory is None else ""
        if baseline_memory is not None and not git_head(self.workspace):
            baseline_problem = "memory/v0.json exists but the workspace has no Git HEAD"
        if baseline_problem:
            print(
                f"[orchestrator] WARNING: incomplete setup ({baseline_problem}); "
                "starting one clean recovery session",
                file=sys.stderr,
                flush=True,
            )
            recovery_prompt = (
                self._mode_directive()
                + "\n\n# Recover incomplete V0 setup\n\n"
                + f"Workspace: `{self.workspace}`\n\n"
                + "A previous non-interactive setup session stopped before producing the required "
                  f"baseline ({baseline_problem}). Continue from the files already present and finish V0 "
                  "autonomously. "
                  "Do not ask the user for confirmation or permission. Inspect the current workspace, "
                  "implement `kernel.py`, preserve the evaluator route described below, run the complete "
                  "workspace workload through the mandatory sandbox with `--no-memory`, parse its "
                  "`[test_kernel] RESULT_JSON=...`, write local `memory/v0.json` and `baseline_report.md`, "
                  "then Git commit `V0: baseline kernel`. Do not enter optimization iterations.\n\n"
                + self._evaluator_directive()
                + "\n\n"
                + self._sandbox_directive()
            )
            recovery = run_session(
                self.workspace,
                recovery_prompt,
                timeout=self.setup_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
                reasoning_effort="high",
            )
            self._account(recovery, "setup recovery")
            if recovery.exit_status != 0 and recovery.tokens == 0:
                raise RuntimeError(
                    f"setup recovery failed immediately (exit={recovery.exit_status}, tokens=0) — "
                    f"{_agent_auth_hint(self.agent_cli)}."
                )
            recovered_memory = read_memory(self.workspace, 0)
            recovery_problem = "missing memory/v0.json" if recovered_memory is None else ""
            if recovered_memory is not None and not git_head(self.workspace):
                recovery_problem = "memory/v0.json exists but the workspace still has no Git HEAD"
            if recovery_problem:
                detail = recovery.stderr_tail or recovery.stdout_tail
                raise RuntimeError(
                    f"setup recovery left an incomplete baseline ({recovery_problem})"
                    + (f": {detail}" if detail else "")
                )

    def _setup_baseline_sol(self, op_dir: Path) -> None:
        if not SOL_SEED.exists():
            raise FileNotFoundError(f"missing {SOL_SEED}")
        cmd = [sys.executable, str(SOL_SEED),
               "--op-dir", str(op_dir), "--name", self.campaign_name,
               "--workspace", str(self.workspace),
               "--framework", self.framework, "--platform", self.platform,
               # The local step only materializes sources and git state.  GPU
               # correctness/performance is run below in the remote sandbox.
               "--no-bench"]
        subprocess.run(cmd, check=True)
        self._link_runtime()
        test = _sandbox_command(
            self.workspace,
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_timeout,
            ["python", "test_kernel.py", "--version", "v0", "--no-memory"],
            gateway_kind="run",
        )
        if test.stdout:
            print(test.stdout, end="" if test.stdout.endswith("\n") else "\n", flush=True)
        if test.stderr:
            print(test.stderr, end="" if test.stderr.endswith("\n") else "\n",
                  file=sys.stderr, flush=True)
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"sandbox V0 baseline produced no usable result: {exc}") from exc
        memory_path = _record_local_test_result(self.workspace, "v0", result)
        if test.returncode != 0 or not result.get("all_pass"):
            raise RuntimeError("sandbox V0 baseline failed correctness/performance validation")

        # sol_seed committed the source-only baseline. Fold the locally-owned
        # memory record into that commit without ever sending memory to the pod.
        mem = json.loads(memory_path.read_text(encoding="utf-8"))
        mem["git_commit_hash"] = git_head(self.workspace)
        mem.setdefault("optimization", {})["action_category"] = "baseline"
        memory_path.write_text(json.dumps(mem, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "memory/v0.json", "CLAUDE.md", ".gitignore"],
                       cwd=str(self.workspace), check=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], cwd=str(self.workspace),
                       check=True, stdout=subprocess.DEVNULL)

    def ensure_framework_baseline(self) -> None:
        """Land the campaign's first real framework kernel as v1, exactly once.

        V0 is a PyTorch reference wrapper. Without this stage every workload bucket repeats the
        same framework bring-up from scratch, which is where whole days get lost on toolchain
        quirks. Buckets derive from the commit pinned here, so the bring-up cost is paid once.

        Idempotent and resume-safe: a pinned baseline is never rewritten, and a campaign that has
        already progressed past V0 without a pin is left exactly as it is. Not wired into
        LayerCampaign, which drives its own sessions.
        """
        action, reason = self._framework_baseline_decision()
        if action == "skip":
            if reason:
                print(f"[orchestrator] framework baseline skipped: {reason}", flush=True)
            return
        print(f"[orchestrator] framework baseline: {action} ({reason})", flush=True)
        if action == "pin":
            root_commit = self._single_root_commit()
            self._pin_framework_baseline(root_commit, version=0)
            return

        n = FRAMEWORK_BASELINE_VERSION
        root_commit = self._single_root_commit()
        v0_blob = git_path_blob(self.workspace, root_commit, "kernel.py")
        pre_head = git_head(self.workspace)

        if action == "run":
            self._link_runtime()
            res = run_session(
                self.workspace, self._framework_baseline_prompt(n),
                timeout=self.framework_baseline_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
                reasoning_effort="high",
            )
            self._account(res, f"framework baseline v{n}")
            if res.exit_status != 0 and res.tokens == 0:
                raise RuntimeError(
                    "framework baseline session produced no output "
                    f"(likely API key / auth issue — {_agent_auth_hint(self.agent_cli)})"
                )
            self._warn_restored_baseline_paths(root_commit)
            problem = self._framework_baseline_problem(v0_blob, root_commit)
            if problem:
                self._recover_framework_baseline(problem, v0_blob, root_commit, pre_head)
                self._warn_restored_baseline_paths(root_commit)
                problem = self._framework_baseline_problem(v0_blob, root_commit)
        else:  # adopt: our own interrupted run already committed the kernel
            self._warn_restored_baseline_paths(root_commit)
            problem = self._framework_baseline_problem(v0_blob, root_commit)
        result: Optional[dict] = None
        if not problem:
            result, problem = self._validate_framework_baseline(n)
        if problem:
            self._record_framework_baseline_failure(problem)
            if pre_head:
                subprocess.run(["git", "reset", "--hard", pre_head], cwd=str(self.workspace),
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raise RuntimeError(f"framework baseline v{n} rejected: {problem}")

        commit = self._commit_framework_baseline(n, result or {})
        latency = ((read_memory(self.workspace, n) or {}).get("performance") or {}).get("latency_us")
        print(
            f"[orchestrator] framework baseline v{n} accepted: {self.framework} "
            f"@ {commit[:8]} ({latency} us geomean)",
            flush=True,
        )

    def _framework_baseline_decision(self) -> tuple[str, str]:
        """Resolve what the stage should do: skip | pin | run | adopt, with the reason."""
        if self.framework_baseline == "never":
            return "skip", ""
        if latest_version(self.workspace) < 0 or not git_head(self.workspace):
            raise RuntimeError("framework baseline requires a committed V0 baseline first")

        pinned_commit, pinned_version = resolve_framework_baseline_commit(self.workspace)
        if pinned_commit:
            return "skip", f"already pinned at {pinned_commit[:8]} (v{pinned_version})"

        violations = production_kernel_violations(self.workspace, self.framework)
        progressed = not head_kernel_is_initial_baseline(self.workspace)
        if not violations and not progressed:
            return "pin", "the V0 kernel is already a compliant framework implementation"
        if any(v.startswith("unsupported production framework") for v in violations):
            return "skip", "; ".join(violations)
        if self.framework_baseline == "auto" and self.optimization_mode != "production":
            return "skip", "leaderboard mode keeps the permissive V0 (use --framework-baseline always)"
        if not progressed:
            return "run", f"V0 is not a self-contained {self.framework} kernel"
        if (
            latest_version(self.workspace) == FRAMEWORK_BASELINE_VERSION
            and not violations
            and not (self.workspace / AGGREGATE_DISPATCH_FILE).is_file()
        ):
            return "adopt", "an interrupted framework baseline is already committed"
        return "skip", (
            "HEAD has progressed beyond V0 without a framework-baseline pin; "
            "leaving this campaign on its existing baseline"
        )

    def _single_root_commit(self) -> str:
        roots = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=str(self.workspace), capture_output=True, text=True,
        )
        root_commits = roots.stdout.split() if roots.returncode == 0 else []
        if len(root_commits) != 1:
            raise RuntimeError("framework baseline requires exactly one V0 root commit")
        return root_commits[0]

    def _framework_baseline_prompt(self, n: int) -> str:
        return _render(
            PROMPTS_DIR / "framework_baseline.md",
            WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
            PLATFORM=self.platform, FRAMEWORK=self.framework,
            ARCH=self.arch or "the runtime GPU arch",
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            EVALUATOR=self._evaluator_directive(),
            MODE_POLICY=self._mode_directive(),
        )

    def _restore_immutable_baseline_paths(self, root_commit: str) -> list[str]:
        """Put back any ground-truth file the session edited, and report what was restored.

        A session that "fixes" the harness or memory/v0.json is a compliance problem, but a
        mechanically repairable one — discarding its kernel over it would throw away hours of
        work for nothing. Acceptance is decided by the kernel itself.
        """
        restored: list[str] = []
        for path in IMMUTABLE_BASELINE_PATHS:
            original = git_path_blob(self.workspace, root_commit, path)
            if not original or original == git_worktree_blob(self.workspace, path):
                continue
            checkout = subprocess.run(
                ["git", "checkout", root_commit, "--", path],
                cwd=str(self.workspace), check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if checkout.returncode == 0:
                restored.append(path)
        return restored

    def _framework_baseline_problem(self, v0_blob: str, root_commit: str) -> str:
        """Static acceptance checks on the candidate about to be validated and committed.

        Everything is judged from the worktree: that is what the gateway uploads, and it lets a
        session that wrote the kernel but never committed it still be accepted.
        """
        candidate_blob = git_worktree_blob(self.workspace, "kernel.py")
        if not candidate_blob or candidate_blob == v0_blob:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        violations = production_kernel_violations(self.workspace, self.framework)
        if violations:
            return (
                f"the candidate is not a self-contained {self.framework} implementation: "
                + "; ".join(violations)
            )
        if self.framework.lower() in {"triton", "gluon"} and kernel_is_gluon(self.workspace):
            # A Gluon v1 would permanently disarm the orchestrator's mandatory Triton->Gluon latch.
            return "the framework baseline must be plain Triton; Gluon is a later orchestrator escalation"
        mutated = [
            path for path in IMMUTABLE_BASELINE_PATHS
            if git_path_blob(self.workspace, root_commit, path)
            and git_path_blob(self.workspace, root_commit, path)
            != git_worktree_blob(self.workspace, path)
        ]
        if mutated:
            return "the session modified immutable ground truth: " + ", ".join(mutated)
        return ""

    def _validate_framework_baseline(self, n: int) -> tuple[Optional[dict], str]:
        """Re-validate the candidate through the gateway: single seed, then five seeds."""
        stages = (
            ("single-seed", ["python", "test_kernel.py", "--version", f"v{n}", "--no-memory"]),
            ("multi-seed", ["python", "test_kernel.py", "--version", f"v{n}",
                            "--multi-seed", "5", "--no-memory"]),
        )
        result: Optional[dict] = None
        for stage_name, command in stages:
            try:
                test = _sandbox_command(
                    self.workspace,
                    self.sandbox_hardware,
                    self.sandbox_profile,
                    self.sandbox_url,
                    self.sandbox_timeout,
                    command,
                    gateway_kind="run",
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return None, f"{stage_name} validation failed to run: {exc}"
            if test.stdout:
                print(test.stdout, end="" if test.stdout.endswith("\n") else "\n", flush=True)
            if test.stderr:
                print(test.stderr, end="" if test.stderr.endswith("\n") else "\n",
                      file=sys.stderr, flush=True)
            if test.returncode != 0:
                return None, f"{stage_name} validation command failed (exit={test.returncode})"
            try:
                result = _test_result_from_stdout(test.stdout)
            except (RuntimeError, json.JSONDecodeError) as exc:
                return None, f"{stage_name} validation produced no usable result: {exc}"
            if not result.get("all_pass"):
                return None, f"{stage_name} correctness validation failed"

        assert result is not None
        latency = result.get("latency_us_geomean")
        if not isinstance(latency, (int, float)) or latency <= 0:
            return None, "validation reported no usable latency_us_geomean"
        # Bucket baselines are derived from this map, so a missing or re-keyed shape here would
        # otherwise surface hours later as a hard failure in bucket seeding.
        baseline_shapes = set(
            ((read_memory(self.workspace, 0) or {}).get("performance") or {}).get(
                "latency_us_by_shape", {}
            )
        )
        measured_shapes = set(result.get("latency_us_by_shape") or {})
        if baseline_shapes and measured_shapes != baseline_shapes:
            return None, (
                "latency_us_by_shape does not cover the same workloads as v0 "
                f"(missing {sorted(baseline_shapes - measured_shapes)}, "
                f"unexpected {sorted(measured_shapes - baseline_shapes)})"
            )
        return result, ""

    def _warn_restored_baseline_paths(self, root_commit: str) -> None:
        restored = self._restore_immutable_baseline_paths(root_commit)
        if restored:
            print(
                "[orchestrator] framework baseline session edited immutable ground truth; "
                f"restored from V0: {', '.join(restored)}",
                file=sys.stderr,
                flush=True,
            )

    def _recover_framework_baseline(
        self, problem: str, v0_blob: str, root_commit: str, pre_head: str
    ) -> None:
        """Run one clean recovery session for a rejected candidate."""
        print(
            f"[orchestrator] WARNING: framework baseline rejected ({problem}); "
            "starting one clean recovery session",
            file=sys.stderr,
            flush=True,
        )
        if pre_head and git_head(self.workspace) != pre_head:
            # Undo the session's commits, keep its files: the recovery session needs to read them.
            subprocess.run(["git", "reset", "--soft", pre_head], cwd=str(self.workspace),
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        recovery_prompt = (
            self._mode_directive()
            + "\n\n# Recover a rejected framework baseline\n\n"
            + f"Workspace: `{self.workspace}`\n\n"
            + "A previous non-interactive session tried to replace the V0 PyTorch wrapper with a "
            + f"self-contained **{self.framework}** kernel and was rejected: "
            + f"**{problem}**\n\n"
            + "Continue from the files already present and finish the job autonomously. Do not ask "
            + "for confirmation. Keep the algorithm you already have where it is sound, fix the "
            + "stated problem, validate correctness through the sandbox with `--multi-seed 5`, "
            + f"write `memory/v{FRAMEWORK_BASELINE_VERSION}.json`, and commit `kernel.py`. Never "
            + "modify `test_kernel.py`, `reference.py`, `input.py`, `shapes.json`, `memory/v0.json`, "
            + f"or create `{FRAMEWORK_BASELINE_FILE}`. Do not enter optimization iterations.\n\n"
            + self._evaluator_directive()
            + "\n\n"
            + self._sandbox_directive()
        )
        recovery = run_session(
            self.workspace, recovery_prompt, timeout=self.framework_baseline_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
            reasoning_effort="high",
        )
        self._account(recovery, f"framework baseline recovery v{FRAMEWORK_BASELINE_VERSION}")

    def _record_framework_baseline_failure(self, problem: str) -> None:
        """Persist why the framework baseline was rejected, uncommitted so a reset cannot lose it."""
        n = FRAMEWORK_BASELINE_VERSION
        memory_path = self.workspace / "memory" / f"v{n}.json"
        try:
            memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            memory = {}
        if not isinstance(memory, dict):
            memory = {}
        memory["version"] = f"v{n}"
        memory["masked"] = False
        memory["git_commit_hash"] = None
        memory["quality_gate"] = {"result": "FAIL", "failure_reason": problem}
        memory["correctness"] = {"status": "FAIL"}
        memory["optimization"] = {
            "action_category": FRAMEWORK_BASELINE_CATEGORY,
            "action_description": f"rejected {self.framework} baseline attempt",
        }
        pitfalls = memory.setdefault("pitfalls_and_fixes", [])
        if not isinstance(pitfalls, list):
            pitfalls = []
            memory["pitfalls_and_fixes"] = pitfalls
        pitfalls.append({
            "error_type": "production_policy" if "self-contained" in problem else "correctness",
            "error_message": problem,
            "lesson": f"the next attempt must land a compliant, correctness-passing {self.framework} kernel",
        })
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _commit_framework_baseline(self, n: int, result: dict) -> str:
        """Commit the accepted kernel (C1) and then pin it in a metadata-only commit (C2)."""
        staged = [
            path for path in ("kernel.py", "solution.json", "CLAUDE.md", "README.md",
                              f"memory/v{n}.json")
            if (self.workspace / path).exists()
        ]
        staged += [
            str(path.relative_to(self.workspace))
            for path in sorted(self.workspace.glob(f"plans/v{n}_*.md"))
        ]
        subprocess.run(["git", "add", *staged], cwd=str(self.workspace), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(self.workspace),
                          check=False).returncode != 0:
            subprocess.run(
                ["git", "commit", "-m",
                 f"v{n}: framework baseline ({self.framework}) replacing the V0 PyTorch wrapper"],
                cwd=str(self.workspace), check=True, stdout=subprocess.DEVNULL,
            )
        kernel_commit = subprocess.run(
            ["git", "rev-list", "-1", "HEAD", "--", "kernel.py"],
            cwd=str(self.workspace), capture_output=True, text=True, check=True,
        ).stdout.strip()
        if git_kernel_blob(self.workspace) != git_worktree_blob(self.workspace, "kernel.py"):
            raise RuntimeError(
                "framework baseline kernel.py differs between the worktree and the commit"
            )

        _record_local_test_result(self.workspace, f"v{n}", result)
        memory_path = self.workspace / "memory" / f"v{n}.json"
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = FRAMEWORK_BASELINE_CATEGORY
        optimization["action_description"] = (
            f"first self-contained {self.framework} implementation of the whole operator"
        )
        memory["git_commit_hash"] = kernel_commit
        memory[FRAMEWORK_BASELINE_CATEGORY] = {
            "framework": self.framework,
            "validated_stages": ["single-seed", "multi-seed-5"],
        }
        memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._pin_framework_baseline(kernel_commit, version=n)
        return kernel_commit

    def _pin_framework_baseline(self, commit: str, *, version: int) -> None:
        """Write and commit the marker that bucket seeding resolves.

        Deliberately a separate commit rather than an amend: amending would rewrite the very
        commit whose sha the marker records, leaving a dangling pointer. This commit does not
        touch kernel.py, so it never registers as an optimization win.
        """
        marker = {
            "schema_version": 1,
            "version": f"v{version}",
            "framework": self.framework,
            "platform": self.platform,
            "arch": self.arch,
            "commit": commit,
            "kernel_blob": git_path_blob(self.workspace, commit, "kernel.py"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.workspace / FRAMEWORK_BASELINE_FILE).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths = [FRAMEWORK_BASELINE_FILE]
        if (self.workspace / "memory" / f"v{version}.json").exists():
            paths.append(f"memory/v{version}.json")
        subprocess.run(["git", "add", *paths], cwd=str(self.workspace), check=True,
                       stdout=subprocess.DEVNULL)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(self.workspace),
                          check=False).returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", f"v{version}: pin framework baseline {commit[:8]}"],
                cwd=str(self.workspace), check=True, stdout=subprocess.DEVNULL,
            )
        # The metadata commit must not read as a stalled optimization round on the next resume.
        write_stall(self.workspace, 0)

    def _record_failed_convert(self, n: int, reason: str) -> None:
        """Persist a failed/reverted triton->gluon conversion as memory/v<N>.json so the NEXT convert
        attempt reads it and avoids repeating the same lowering. Survives the safety-net git reset
        (which would otherwise destroy a committed record)."""
        mem_dir = self.workspace / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / f"v{n}.json").write_text(json.dumps({
            "version": f"v{n}", "masked": False,
            "optimization": {"action_category": "triton_to_gluon_conversion",
                             "action_description": "reverted defective conversion"},
            "correctness": {"status": "FAIL"},
            "quality_gate": {"result": "FAIL", "failure_reason": reason},
            "pitfalls_and_fixes": [{"error_type": "performance", "error_message": reason,
                                    "lesson": "this triton->gluon lowering was rejected; try a different "
                                              "approach (check async/TMA copy, layouts, accumulator residency) next attempt"}],
            "git_commit_hash": None,
        }, indent=2), encoding="utf-8")

    def budget_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_spent >= self.token_budget

    def _notify_improvement(
        self, n: int, mem: Optional[dict], previous_latency: Optional[float]
    ) -> None:
        """Notify the outer coordinator after a mechanically accepted bucket win.

        A kernel-changing commit is necessary but not sufficient: conversion commits
        may intentionally allow small parity regressions, and malformed memory must not
        trigger an aggregate rebuild.  The callback therefore fires only for a
        correctness-passing, strictly faster measured version.
        """
        if self.on_improvement is None or not mem:
            return
        gate_pass = _status_is((mem.get("quality_gate") or {}).get("result"), "PASS")
        latency = (mem.get("performance") or {}).get("latency_us")
        if not gate_pass or not isinstance(latency, (int, float)):
            return
        if previous_latency is not None and float(latency) >= previous_latency:
            return
        try:
            self.on_improvement(self, n, mem)
        except Exception as exc:
            # Aggregation is an independent quality gate.  A failed aggregate
            # attempt must not discard the valid improvement in this bucket.
            print(
                f"[orchestrator] WARNING: improvement callback for v{n} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _notify_iteration(self, n: int, mem: Optional[dict], won: bool) -> None:
        """Notify a workload coordinator that one bucket round has completed.

        The initial aggregation barrier depends on every bucket reaching ten
        completed versioned rounds, even when the tenth round itself is not a
        win.  Callback failures must not discard the bucket's optimization
        result, matching the improvement callback's isolation semantics.
        """
        if self.on_iteration is None:
            return
        try:
            self.on_iteration(self, n, mem, won)
        except Exception as exc:
            print(
                f"[orchestrator] WARNING: iteration callback for v{n} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def run(self) -> str:
        if latest_version(self.workspace) < 0:
            self.setup_baseline()
        else:
            if not git_head(self.workspace):
                raise RuntimeError(
                    "cannot resume: memory/v0.json exists but the workspace has no Git HEAD; "
                    "recover and validate the V0 baseline before starting an optimization iteration"
                )
            print(f"[orchestrator] resuming: latest = v{latest_version(self.workspace)}", flush=True)
            preserve_interrupted_tracked_changes(
                self.workspace, f"resume {self.campaign_name}"
            )
            self._link_runtime()  # ensure runtime symlinks exist for iteration sessions

        self.ensure_framework_baseline()

        if self.optimization_mode == "production" and latest_version(self.workspace) > 0:
            violations = production_kernel_violations(self.workspace, self.framework)
            if violations and not head_kernel_is_initial_baseline(self.workspace):
                raise RuntimeError(
                    "cannot resume a non-compliant production HEAD: " + "; ".join(violations)
                )
            if violations:
                print(
                    "[orchestrator] production resume: HEAD kernel is still the "
                    "original V0 baseline; continuing until a framework-compliant "
                    "candidate is accepted",
                    flush=True,
                )

        stall = read_stall(self.workspace)   # persisted live counter (single source of truth)
        if stall is None:
            stall = reconstruct_stall(self.workspace)  # bootstrap from git when no state file yet
            write_stall(self.workspace, stall)
        if stall > 0:
            print(f"[orchestrator] stall counter restored: {stall} rounds without progress", flush=True)
        infra_fails = 0  # consecutive sessions that crashed with 0 tokens (auth/infra issue)
        n = latest_version(self.workspace)  # 0 after baseline
        mask_half_memory(self.workspace, n)  # also covers resuming an unmasked v100/v200/...
        while True:
            conversion_pending = should_convert_to_gluon(
                self.framework,
                stall,
                self.convert_after,
                head_is_gluon=head_kernel_is_gluon(self.workspace),
            )
            if n >= self.max_iters:
                if conversion_pending:
                    raise RuntimeError(
                        "mandatory Triton->Gluon conversion did not succeed before max-iters"
                    )
                return self._finish("budget: max-iters")
            if self.budget_exhausted():
                if conversion_pending:
                    raise RuntimeError(
                        "mandatory Triton->Gluon conversion did not succeed before token-budget"
                    )
                return self._finish("budget: token-budget")

            n += 1
            # Triton→Gluon escalation is a latch, not a periodic best-effort attempt. Once
            # `convert_after` consecutive stalls are reached, every following session is a
            # convert-only retry until a correctness- and parity-passing Gluon HEAD is committed.
            do_convert = conversion_pending
            if do_convert:
                print(f"[orchestrator] triton stalled {stall} iters -> triton->gluon convert session v{n}", flush=True)
                prompt = _render(PROMPTS_DIR / "convert.md",
                                 WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                                 PLATFORM=self.platform, ARCH=self.arch or "the runtime GPU arch",
                                 NOTES=self.notes,
                                 HARDWARE=hardware_directive(self.platform, self.arch),
                                 SANDBOX=self._sandbox_directive(),
                                 EVALUATOR=self._evaluator_directive(),
                                 MODE_POLICY=self._mode_directive())
            else:
                prompt = _render(PROMPTS_DIR / "iteration.md",
                                 WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                                 PLATFORM=self.platform, NOTES=self.notes,
                                 AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
                                 PLAN_GENERATOR=_plan_generator_directive(self.agent_cli, n),
                                 HARDWARE=hardware_directive(self.platform, self.arch),
                                 SANDBOX=self._sandbox_directive(),
                                 EVALUATOR=self._evaluator_directive(),
                                 MODE_POLICY=self._mode_directive())
            previous_latency = incumbent_latency(self.workspace, n)
            pre_head_was_gluon = head_kernel_is_gluon(self.workspace)
            pre_head = git_head(self.workspace)  # win = a commit that changes kernel.py vs this
            res = run_session(
                self.workspace, prompt, timeout=self.iter_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
            )
            self._account(res, f"{'convert' if do_convert else 'iter'} v{n}")

            # Robust infra-failure handling: distinguish crash vs timeout, retry up to 15
            # consecutive failures with progressive backoff before giving up. A 2-fail
            # cutoff was too aggressive — transient API rate-limits and short network blips
            # regularly produced back-to-back non-zero exits with low tokens, killing the
            # whole campaign for a recoverable hiccup. Backoff avoids hammering a
            # rate-limited endpoint.
            if res.exit_status != 0 or res.timed_out:
                infra_fails += 1
                if res.timed_out:
                    backoff = min(60 * infra_fails, 300)
                    notice = f"timeout #{infra_fails}"
                else:
                    backoff = min(30 * infra_fails, 180)
                    notice = f"infra fail #{infra_fails} (exit={res.exit_status})"
                if infra_fails < 15:
                    # Back off before retrying to avoid hammering a rate-limited API
                    print(f"[orchestrator] {notice}, backing off {backoff}s", flush=True)
                    time.sleep(backoff)
                # Recover this round's findings once the endpoint has cooled: a killed session wrote
                # no memory, and an empty version forces the next session to re-derive everything.
                self._ensure_iteration_memory(n, res, "convert" if do_convert else "iter")
                if infra_fails >= 15:
                    if do_convert:
                        raise RuntimeError(
                            "mandatory Triton->Gluon conversion could not complete after "
                            f"{infra_fails} consecutive failed sessions"
                        )
                    if res.timed_out:
                        return self._finish(f"infra: {infra_fails} consecutive timeouts")
                    return self._finish(
                        f"infra: {infra_fails} consecutive sessions crashed (exit={res.exit_status}) "
                        f"(likely API key / auth issue — {_agent_auth_hint(self.agent_cli)})"
                    )
            else:
                infra_fails = 0

            mem = read_memory(self.workspace, n)
            won = kernel_won(self.workspace, pre_head)  # git-native "committed a kernel.py win" — reused below
            if won and self.optimization_mode == "production":
                violations = production_kernel_violations(
                    self.workspace,
                    self.framework,
                    require_gluon=pre_head_was_gluon,
                )
                if violations:
                    reject_production_commit(self.workspace, n, pre_head, violations)
                    print(
                        "[orchestrator] production policy rejected v"
                        f"{n}: {'; '.join(violations)}; reverted to {pre_head[:8]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    mem = read_memory(self.workspace, n)
                    won = False
            if do_convert:
                # A direct triton->gluon translation must preserve BOTH correctness and performance.
                # Accept only a committed gluon kernel whose geomean is within +CONVERT_PERF_TOL of the
                # incumbent triton HEAD. Otherwise reject, keep triton, record WHY, and immediately
                # retry conversion in the next clean session. Ordinary Triton optimization stays
                # disabled until conversion succeeds.
                # Accept only when the COMMITTED HEAD kernel is gluon, correctness PASSed, and geomean is
                # within +CONVERT_PERF_TOL of the incumbent triton HEAD. Detect the committed gluon via git
                # HEAD (not memory's git_commit_hash, which a session may leave unset even after committing).
                conv_lat = (mem.get("performance") or {}).get("latency_us") if mem else None
                gate_pass = bool(mem) and (mem.get("quality_gate") or {}).get("result") == "PASS"
                head_gluon = head_kernel_is_gluon(self.workspace)
                prev_best = incumbent_latency(self.workspace, n)
                parity_ok = (prev_best is None or (isinstance(conv_lat, (int, float))
                             and conv_lat <= prev_best * (1.0 + CONVERT_PERF_TOL)))
                if head_gluon and gate_pass and isinstance(conv_lat, (int, float)) and parity_ok:
                    stall = 0            # converted (correctness + <=5% perf parity) -> fresh Gluon phase
                    write_stall(self.workspace, stall)
                    print("[orchestrator] converted triton->gluon (perf parity ok); optimizing gluon", flush=True)
                    self._notify_improvement(n, mem, previous_latency)
                    self._notify_iteration(n, mem, True)
                    if peak_util(mem) >= self.target_util:
                        mask_half_memory(self.workspace, n)
                        return self._finish(
                            f"success: peak_util {peak_util(mem):.1f}% >= "
                            f"{self.target_util:.0f}%"
                        )
                    mask_half_memory(self.workspace, n)
                    continue
                # Revert every rejected kernel-changing conversion commit, including a session that
                # incorrectly committed another Triton kernel.  The next clean attempt must always
                # start from the same accepted Triton incumbent until a Gluon candidate passes.
                if won and pre_head:
                    subprocess.run(["git", "reset", "--hard", pre_head], cwd=str(self.workspace),
                                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if not head_gluon:
                        reason = "convert session committed a non-Gluon kernel"
                    elif isinstance(conv_lat, (int, float)) and prev_best and not parity_ok:
                        reason = (
                            f"regressed {conv_lat / prev_best - 1:+.1%} vs triton "
                            f"(> {CONVERT_PERF_TOL:.0%})"
                        )
                    else:
                        reason = "correctness gate not PASS"
                    self._record_failed_convert(n, reason)
                    print(f"[orchestrator] convert rejected ({reason}); reverted to triton HEAD {pre_head[:8]}", flush=True)
                else:
                    # A session may leave an uncommitted Gluon edit behind. Restore the committed
                    # Triton/record HEAD before retrying so every attempt starts from auditable state.
                    if kernel_is_gluon(self.workspace) and not head_gluon:
                        subprocess.run(
                            ["git", "reset", "--hard", "HEAD"],
                            cwd=str(self.workspace),
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    if read_memory(self.workspace, n) is None:
                        self._record_failed_convert(n, "convert session produced no committed gluon kernel")
                    print(
                        "[orchestrator] convert produced no committed gluon kernel; "
                        "mandatory conversion remains latched",
                        flush=True,
                    )
                if res.tokens < CONVERT_MIN_TOKENS:
                    print(
                        f"[orchestrator] convert v{n} ended shallowly ({res.tokens} tokens, no gluon); "
                        "retrying conversion next session",
                        flush=True,
                    )
                # Preserve the reached threshold. This makes the next loop iteration another
                # conversion attempt rather than returning to Triton optimization.
                stall = max(stall, self.convert_after)
                write_stall(self.workspace, stall)
                self._notify_iteration(n, read_memory(self.workspace, n), False)
                mask_half_memory(self.workspace, n)
                continue

            if won:                        # reuse the git-native win computed above
                self._notify_improvement(n, mem, previous_latency)
                self._notify_iteration(n, mem, True)
                if peak_util(mem) >= self.target_util:
                    mask_half_memory(self.workspace, n)
                    return self._finish(
                        f"success: peak_util {peak_util(mem):.1f}% >= {self.target_util:.0f}%"
                    )
                stall = 0
                write_stall(self.workspace, stall)
            else:
                stall += 1
                write_stall(self.workspace, stall)
                self._notify_iteration(n, mem, False)
                conversion_now_required = should_convert_to_gluon(
                    self.framework,
                    stall,
                    self.convert_after,
                    head_is_gluon=head_kernel_is_gluon(self.workspace),
                )
                if self.max_stall > 0 and stall >= self.max_stall and not conversion_now_required:
                    mask_half_memory(self.workspace, n)
                    return self._finish(f"stall: {stall} iterations with no commit")
            mask_half_memory(self.workspace, n)

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "memory_manager.py"),
                 "summary", "--workspace", str(self.workspace)],
                check=False,
            )
        except OSError:
            pass
        # Production output is fail-closed: do not package a PyTorch baseline,
        # alternate DSL, or third-party-backed kernel as a production candidate.
        if self.optimization_mode == "production":
            violations = production_kernel_violations(self.workspace, self.framework)
            if violations:
                raise RuntimeError(
                    "no production-compliant final kernel: " + "; ".join(violations)
                )
        # SOL op: emit the self-contained, validated submission (SOL's output format).
        if (self.workspace / "definition.json").exists() and (self.workspace / "solution.json").exists():
            try:
                subprocess.run(
                    [sys.executable, str(REPO_ROOT / "reference" / "sol_finalize.py"),
                     "--workspace", str(self.workspace)],
                    check=False,
                )
            except OSError:
                pass
        return reason


# ── workload-bucket campaign ──────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkloadBucket:
    """One inspector-produced, disjoint subset of workload.jsonl."""

    name: str
    workload_indices: tuple[int, ...]
    rationale: str = ""


@dataclass(frozen=True)
class WorkloadSource:
    """Ordered workload view over SOL JSONL or atrex-bench shapes.json."""

    kind: str
    filename: str
    ids: tuple[str, ...]
    entries: tuple[dict, ...]
    raw_lines: tuple[str, ...] = ()


def _read_workloads(path: Path) -> tuple[list[str], list[dict]]:
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    workloads: list[dict] = []
    for index, line in enumerate(raw_lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid workload.jsonl line {index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"workload.jsonl line {index + 1} is not a JSON object")
        workloads.append(value)
    if not workloads:
        raise ValueError("workload.jsonl contains no workloads")
    return raw_lines, workloads


def _shape_id_sort_key(shape_id: str) -> tuple[int, object]:
    return (0, int(shape_id)) if re.fullmatch(r"\d+", shape_id) else (1, shape_id)


def _read_workload_source(op_dir: Path) -> WorkloadSource:
    workload_path = op_dir / "workload.jsonl"
    if workload_path.is_file():
        lines, workloads = _read_workloads(workload_path)
        return WorkloadSource(
            kind="sol",
            filename="workload.jsonl",
            ids=tuple(str(index) for index in range(len(workloads))),
            entries=tuple(workloads),
            raw_lines=tuple(lines),
        )

    shapes_path = op_dir / "shapes.json"
    if not shapes_path.is_file():
        raise ValueError(f"operator has neither workload.jsonl nor shapes.json: {op_dir}")
    try:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid shapes.json: {exc}") from exc
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes.json must contain a non-empty object keyed by shape id")
    ids = tuple(sorted((str(key) for key in shapes), key=_shape_id_sort_key))
    entries: list[dict] = []
    for shape_id in ids:
        value = shapes.get(shape_id)
        if not isinstance(value, dict):
            raise ValueError(f"shapes.json entry {shape_id!r} is not an object")
        entries.append(value)
    return WorkloadSource(
        kind="shapes",
        filename="shapes.json",
        ids=ids,
        entries=tuple(entries),
    )


def validate_workload_buckets(
    manifest: object,
    workload_count: int,
    max_buckets: int,
    *,
    require_visibility_policy: bool = False,
) -> list[WorkloadBucket]:
    """Validate the inspector contract and return normalized buckets.

    Exact, disjoint coverage is deliberately enforced here rather than trusted
    to the coding agent.  A missing workload would make a bucket kernel look
    faster without ever proving it correct; an overlap would bias optimization
    resources and make aggregation provenance ambiguous.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("buckets"), list):
        raise ValueError(f"{WORKLOAD_BUCKETS_FILE} must contain a top-level buckets list")
    if isinstance(manifest.get("schema_version"), bool) or manifest.get("schema_version") != 1:
        raise ValueError(f"{WORKLOAD_BUCKETS_FILE} schema_version must be 1")
    visibility_policy = manifest.get("dispatch_visibility_policy")
    if visibility_policy not in (None, DISPATCH_VISIBILITY_POLICY):
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} dispatch visibility policy is unsupported: "
            f"{visibility_policy!r}"
        )
    if require_visibility_policy and visibility_policy != DISPATCH_VISIBILITY_POLICY:
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} must declare dispatch_visibility_policy="
            f"{DISPATCH_VISIBILITY_POLICY!r}"
        )
    if (
        isinstance(manifest.get("workload_count"), bool)
        or manifest.get("workload_count") != workload_count
    ):
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} workload_count does not match workload.jsonl"
        )
    raw_buckets = manifest["buckets"]
    if not raw_buckets:
        raise ValueError("workload inspector returned no buckets")
    if len(raw_buckets) > max_buckets:
        raise ValueError(
            f"workload inspector returned {len(raw_buckets)} buckets; maximum is {max_buckets}"
        )

    buckets: list[WorkloadBucket] = []
    seen_names: set[str] = set()
    owners: dict[int, str] = {}
    for position, raw in enumerate(raw_buckets):
        if not isinstance(raw, dict):
            raise ValueError(f"bucket {position} is not an object")
        raw_name = raw.get("name")
        if not isinstance(raw_name, str):
            raise ValueError(f"bucket {position} has no string name")
        name = _workspace_slug(raw_name)
        if name in seen_names:
            raise ValueError(f"duplicate bucket name: {name}")
        seen_names.add(name)
        indices = raw.get("workload_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"bucket {name} has no workload_indices")
        normalized: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"bucket {name} contains a non-integer workload index")
            if index < 0 or index >= workload_count:
                raise ValueError(f"bucket {name} workload index {index} is out of range")
            if index in owners:
                raise ValueError(
                    f"workload index {index} appears in both {owners[index]} and {name}"
                )
            owners[index] = name
            normalized.append(index)
        rationale = raw.get("rationale")
        buckets.append(
            WorkloadBucket(
                name=name,
                workload_indices=tuple(sorted(normalized)),
                rationale=rationale if isinstance(rationale, str) else "",
            )
        )

    missing = sorted(set(range(workload_count)) - set(owners))
    if missing:
        raise ValueError(f"workload inspector omitted workload indices: {missing}")
    return buckets


def _normalized_bucket_manifest(buckets: list[WorkloadBucket], workload_count: int) -> dict:
    return {
        "schema_version": 1,
        "dispatch_visibility_policy": DISPATCH_VISIBILITY_POLICY,
        "workload_count": workload_count,
        "buckets": [
            {
                "name": bucket.name,
                "workload_indices": list(bucket.workload_indices),
                "rationale": bucket.rationale,
            }
            for bucket in buckets
        ],
    }


def _materialize_bucket_op(
    source_op: Path,
    destination: Path,
    bucket: WorkloadBucket,
    workload_source: WorkloadSource,
) -> None:
    """Create a derived op dir whose evaluator sees exactly one bucket."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_op.iterdir():
        if source.name in (".git", "__pycache__", workload_source.filename):
            continue
        target = destination / source.name
        # sol_seed consumes top-level SOL ground truth.  Do not recursively
        # copy arbitrary op subdirectories: --workspace is allowed to live
        # under the op dir, and copying it here would recurse into the newly
        # created aggregate/bucket repositories.
        if source.is_file():
            shutil.copy2(source, target)
    if workload_source.kind == "sol":
        selected = [workload_source.raw_lines[index] for index in bucket.workload_indices]
        (destination / "workload.jsonl").write_text(
            "\n".join(selected) + "\n", encoding="utf-8"
        )
        return

    selected_ids = [workload_source.ids[index] for index in bucket.workload_indices]
    selected_shapes = {
        shape_id: workload_source.entries[index]
        for index, shape_id in zip(bucket.workload_indices, selected_ids)
    }
    (destination / "shapes.json").write_text(
        json.dumps(selected_shapes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    roofline_path = source_op / "roofline.json"
    if roofline_path.is_file():
        try:
            roofline = json.loads(roofline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid roofline.json: {exc}") from exc
        if isinstance(roofline, dict) and isinstance(roofline.get("shapes"), dict):
            all_roofline_shapes = roofline["shapes"]
            roofline["shapes"] = {
                shape_id: all_roofline_shapes[shape_id]
                for shape_id in selected_ids
                if shape_id in all_roofline_shapes
            }
            (destination / "roofline.json").write_text(
                json.dumps(roofline, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def _bucket_baseline_memory(
    aggregate_memory: dict,
    bucket: WorkloadBucket,
    workload_source: WorkloadSource,
    *,
    aggregate_workspace: Path,
    aggregate_baseline_commit: str,
    aggregate_baseline_version: int = 0,
) -> dict:
    """Derive one bucket's V0 record from the aggregate baseline measurement.

    The aggregate evaluator already measured every workload independently and
    persisted those measurements in ``latency_us_by_shape``.  Re-running the
    same baseline once per bucket wastes both coding-agent and GPU time.  A
    bucket baseline is therefore the exact subset of those measurements, with
    scalar latency aggregates recomputed over only that subset.

    The source is the aggregate framework baseline when the campaign pinned one, otherwise
    the original V0 record.
    """
    performance = aggregate_memory.get("performance") or {}
    per_workload = performance.get("latency_us_by_shape")
    if not isinstance(per_workload, dict):
        raise RuntimeError(
            f"aggregate memory/v{aggregate_baseline_version}.json has no "
            "performance.latency_us_by_shape; cannot derive bucket baselines without re-running it"
        )

    selected: dict[str, float] = {}
    selected_ids: list[str] = []
    ordered_keys = list(per_workload)
    for index in bucket.workload_indices:
        entry = workload_source.entries[index]
        candidates = [workload_source.ids[index]]
        for field_name in ("uuid", "id"):
            value = entry.get(field_name)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                candidates.append(str(value))
        key = next((candidate for candidate in candidates if candidate in per_workload), None)
        # Older SOL records can be keyed by evaluator UUID while the source view
        # exposes only ordinal ids.  Equal cardinality makes the evaluator's
        # stable insertion order an unambiguous final fallback.
        if key is None and len(ordered_keys) == len(workload_source.entries):
            key = ordered_keys[index]
        if key is None:
            raise RuntimeError(
                f"aggregate V0 has no per-workload latency for bucket {bucket.name!r} "
                f"workload index {index}"
            )
        latency = per_workload.get(key)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) <= 0
        ):
            raise RuntimeError(
                f"aggregate V0 latency for workload {key!r} is not a positive finite number"
            )
        selected[str(key)] = float(latency)
        selected_ids.append(str(key))

    latencies = list(selected.values())
    geomean = (
        latencies[0]
        if len(latencies) == 1
        else math.exp(sum(math.log(value) for value in latencies) / len(latencies))
    )
    arithmetic_mean = sum(latencies) / len(latencies)
    # JSON round-tripping gives a deep copy while also guaranteeing the derived
    # record remains JSON serializable like every other memory entry.
    derived = json.loads(json.dumps(aggregate_memory))
    derived["version"] = "v0"
    derived["masked"] = False
    derived["timestamp"] = datetime.now(timezone.utc).isoformat()
    # A bucket baseline is neither the campaign's framework baseline nor an aggregate candidate.
    derived.pop(FRAMEWORK_BASELINE_CATEGORY, None)
    derived.pop("aggregation", None)
    derived_performance = derived.setdefault("performance", {})
    derived_performance["latency_us"] = geomean
    derived_performance["latency_us_geomean"] = geomean
    derived_performance["latency_us_arith_mean"] = arithmetic_mean
    derived_performance["latency_us_by_shape"] = selected
    derived_performance["derived_from_aggregate_v0"] = True
    derived_performance["derived_from_aggregate_version"] = f"v{aggregate_baseline_version}"
    derived_performance["aggregate_v0_latency_us"] = performance.get("latency_us")
    derived_performance["aggregate_baseline_latency_us"] = performance.get("latency_us")
    # Throughput/utilization scalars in aggregate V0 were reduced over the full
    # workload set.  They are not mathematically valid for a subset and could
    # incorrectly trip the bucket's peak-utilization stop condition.
    for aggregate_only_metric in (
        "tflops",
        "bandwidth_gbps",
        "tflops_peak_utilization_pct",
        "bandwidth_peak_utilization_pct",
        "speedup_vs_ref_geomean",
    ):
        derived_performance.pop(aggregate_only_metric, None)
    source_label = (
        f"framework baseline v{aggregate_baseline_version}"
        if aggregate_baseline_version > 0
        else "V0"
    )
    derived_performance["measurement_note"] = (
        f"Mechanically selected aggregate {source_label} per-workload timings for bucket "
        f"{bucket.name}; no separate baseline evaluation was run."
    )
    optimization = derived.setdefault("optimization", {})
    optimization["action_category"] = "baseline"
    optimization["action_description"] = (
        f"V0 mechanically derived from aggregate {source_label} for bucket {bucket.name}; "
        f"selected workloads: {', '.join(selected_ids)}. No separate baseline session "
        "or GPU evaluation was run."
    )
    derived["baseline_derivation"] = {
        "source": (
            "aggregate_framework_baseline" if aggregate_baseline_version > 0 else "aggregate_v0"
        ),
        "aggregate_workspace": str(aggregate_workspace),
        "aggregate_v0_commit": aggregate_baseline_commit,
        "aggregate_baseline_commit": aggregate_baseline_commit,
        "aggregate_baseline_version": f"v{aggregate_baseline_version}",
        "bucket": bucket.name,
        "workload_indices": list(bucket.workload_indices),
        "workload_ids": selected_ids,
    }
    derived["git_commit_hash"] = None
    return derived


def _freeze_dispatch_signature(value: object) -> object:
    """Convert JSON arrays into hashable tuples while preserving scalars."""
    if isinstance(value, list):
        return tuple(_freeze_dispatch_signature(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_dispatch_signature(value[key])) for key in sorted(value)
        )
    return value


def _validate_dispatch_value_signature(value: object, path: str) -> None:
    """Validate one no-sync, production-visible value signature.

    Tensor signatures intentionally contain metadata only.  A cached or agent-edited
    signature cannot add tensor values, summaries, pointers, or other evaluator-only
    information and later smuggle it into the generated production dispatcher.
    """
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ValueError(f"{path} is not a tagged dispatch value signature")
    tag = value[0]
    if tag == "tensor":
        if len(value) != 6:
            raise ValueError(
                f"{path} tensor signature must contain only shape/stride/dtype/layout/requires_grad"
            )
        shape, stride, dtype, layout, requires_grad = value[1:]
        if (
            not isinstance(shape, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        ):
            raise ValueError(f"{path} tensor shape is invalid")
        if (
            not isinstance(stride, list)
            or len(stride) != len(shape)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in stride)
        ):
            raise ValueError(f"{path} tensor stride is invalid")
        if not isinstance(dtype, str) or not isinstance(layout, str):
            raise ValueError(f"{path} tensor dtype/layout is invalid")
        if not isinstance(requires_grad, bool):
            raise ValueError(f"{path} tensor requires_grad flag is invalid")
        return
    if tag == "none":
        if len(value) != 1:
            raise ValueError(f"{path} none signature has extra data")
        return
    if tag == "bool":
        if len(value) != 2 or not isinstance(value[1], bool):
            raise ValueError(f"{path} bool signature is invalid")
        return
    if tag == "int":
        if len(value) != 2 or isinstance(value[1], bool) or not isinstance(value[1], int):
            raise ValueError(f"{path} int signature is invalid")
        return
    if tag == "float":
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"{path} float signature is invalid")
        rendered = value[1]
        if rendered not in {"nan", "inf", "-inf"}:
            try:
                float.fromhex(rendered)
            except ValueError as exc:
                raise ValueError(f"{path} float signature is invalid") from exc
        return
    if tag in {"str", "torch.dtype"}:
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"{path} {tag} signature is invalid")
        return
    if tag in {"tuple", "list"}:
        if len(value) != 2 or not isinstance(value[1], list):
            raise ValueError(f"{path} {tag} signature is invalid")
        for index, item in enumerate(value[1]):
            _validate_dispatch_value_signature(item, f"{path}[{index}]")
        return
    if tag == "dict":
        if len(value) != 2 or not isinstance(value[1], list):
            raise ValueError(f"{path} dict signature is invalid")
        keys: list[str] = []
        for index, item in enumerate(value[1]):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError(f"{path} dict entry {index} is invalid")
            keys.append(item[0])
            _validate_dispatch_value_signature(item[1], f"{path}.{item[0]}")
        if keys != sorted(set(keys)):
            raise ValueError(f"{path} dict keys must be unique and sorted")
        return
    raise ValueError(f"{path} uses unsupported dispatch signature tag {tag!r}")


def _validate_invocation_signature(value: object, path: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[0] != "invocation"
        or not isinstance(value[1], list)
        or not isinstance(value[2], list)
    ):
        raise ValueError(f"{path} is not a valid invocation signature")
    for index, item in enumerate(value[1]):
        _validate_dispatch_value_signature(item, f"{path}.args[{index}]")
    names: list[str] = []
    for index, item in enumerate(value[2]):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ValueError(f"{path}.kwargs[{index}] is invalid")
        names.append(item[0])
        _validate_dispatch_value_signature(item[1], f"{path}.kwargs.{item[0]}")
    if names != sorted(set(names)):
        raise ValueError(f"{path} keyword names must be unique and sorted")


def validate_dispatch_signatures(
    payload: object, workload_source: WorkloadSource
) -> list[dict]:
    """Validate the sandbox collector result against immutable workload order."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{DISPATCH_SIGNATURES_FILE} schema_version must be 1")
    if payload.get("kind") != workload_source.kind:
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} kind does not match {workload_source.kind}"
        )
    if payload.get("workload_source") != workload_source.filename:
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} workload source does not match "
            f"{workload_source.filename}"
        )
    visibility_policy = payload.get("visibility_policy")
    if visibility_policy not in (None, DISPATCH_VISIBILITY_POLICY):
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} visibility policy is unsupported: "
            f"{visibility_policy!r}"
        )
    records = payload.get("workloads")
    if not isinstance(records, list) or len(records) != len(workload_source.entries):
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} must contain exactly "
            f"{len(workload_source.entries)} workloads"
        )
    normalized: list[dict] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"dispatch signature record {index} is not an object")
        if record.get("index") != index or record.get("id") != workload_source.ids[index]:
            raise ValueError(
                f"dispatch signature record {index} does not match workload id "
                f"{workload_source.ids[index]!r}"
            )
        if not isinstance(record.get("init"), list) or not isinstance(
            record.get("call"), list
        ):
            raise ValueError(f"dispatch signature record {index} is missing init/call")
        _validate_invocation_signature(record["init"], f"workloads[{index}].init")
        _validate_invocation_signature(record["call"], f"workloads[{index}].call")
        normalized.append(
            {
                "index": index,
                "id": workload_source.ids[index],
                "init": record["init"],
                "call": record["call"],
            }
        )
    return normalized


def validate_dispatch_bucket_compatibility(
    buckets: list[WorkloadBucket], signature_records: list[dict]
) -> None:
    """Reject partitions that cannot be distinguished from runtime arguments."""
    owner = {
        index: bucket.name
        for bucket in buckets
        for index in bucket.workload_indices
    }
    signature_owners: dict[object, tuple[str, list[int]]] = {}
    for record in signature_records:
        index = int(record["index"])
        key = (
            _freeze_dispatch_signature(record["init"]),
            _freeze_dispatch_signature(record["call"]),
        )
        bucket_name = owner[index]
        previous = signature_owners.get(key)
        if previous is None:
            signature_owners[key] = (bucket_name, [index])
            continue
        previous_bucket, indices = previous
        if previous_bucket != bucket_name:
            raise ValueError(
                "workloads with identical runtime signatures cannot be split across "
                f"buckets: indices {indices + [index]} map to "
                f"{previous_bucket!r} and {bucket_name!r}"
            )
        indices.append(index)


def _git_head_file(workspace: Path, relative_path: str) -> str:
    """Read a file from a bucket's committed HEAD, never from dirty worktree state."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read committed {relative_path} from {workspace}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _git_blob_file(workspace: Path, blob_hash: object) -> str:
    """Read an exact previously recorded bucket kernel blob."""
    value = str(blob_hash or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", value):
        raise RuntimeError(f"invalid recorded kernel blob: {blob_hash!r}")
    result = subprocess.run(
        ["git", "cat-file", "blob", value],
        cwd=str(workspace),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read recorded kernel blob {value} from {workspace}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _generated_dispatch_runtime() -> str:
    """Runtime signature code shared semantically with collect_dispatch_signatures.py."""
    return '''
def _value_signature(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(int(item) for item in value.shape),
            tuple(int(item) for item in value.stride()),
            str(value.dtype),
            str(value.layout),
            bool(value.requires_grad),
        )
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if math.isnan(value):
            rendered = "nan"
        elif math.isinf(value):
            rendered = "inf" if value > 0 else "-inf"
        else:
            rendered = value.hex()
        return ("float", rendered)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("torch.dtype", str(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("dispatch dictionaries must have string keys")
        return (
            "dict",
            tuple((key, _value_signature(value[key])) for key in sorted(value)),
        )
    raise TypeError(
        "unsupported dispatch input type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _invocation_signature(args, kwargs):
    return (
        "invocation",
        tuple(_value_signature(value) for value in args),
        tuple((key, _value_signature(kwargs[key])) for key in sorted(kwargs)),
    )


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for key in sorted(value):
            found = _first_tensor(value[key])
            if found is not None:
                return found
    return None
'''.strip()


def build_deterministic_dispatcher(
    *,
    kind: str,
    signature_records: list[dict],
    bucket_by_index: dict[int, str],
    module_records: dict[str, dict],
    module_sources: dict[str, str],
) -> str:
    """Generate one self-contained kernel.py with statically embedded buckets."""
    bucket_names = sorted(module_records)
    if set(module_sources) != set(bucket_names):
        raise ValueError("embedded aggregate source set does not match module records")
    bucket_indices = {name: index for index, name in enumerate(bucket_names)}
    signature_map: dict[object, int] = {}
    for record in signature_records:
        key = (
            _freeze_dispatch_signature(record["init"]),
            _freeze_dispatch_signature(record["call"]),
        )
        signature_map[key] = bucket_indices[bucket_by_index[int(record["index"])]]

    entry_name = "Model" if kind == "shapes" else "run" if kind == "sol" else ""
    if not entry_name:
        raise ValueError(f"unsupported deterministic dispatch kind: {kind}")
    embedded = embed_bucket_sources(module_sources, entry_name=entry_name)
    blob_map = {
        name: str(module_records[name].get("kernel_blob") or "")
        for name in bucket_names
    }
    header = (
        "# Generated by orchestrator/optimize.py. Do not edit.\n"
        "# Self-contained deterministic dispatcher over independently validated bucket kernels.\n"
        f"from __future__ import {', '.join(embedded.future_features)}\n\n"
        "import math\n"
        "import torch\n\n"
        + "\n\n".join(embedded.blocks)
        + "\n\n"
        + _generated_dispatch_runtime()
        + "\n\n"
        + f"_BUCKET_KERNEL_BLOBS = {blob_map!r}\n"
        + f"_SIGNATURE_TO_BUCKET = {signature_map!r}\n"
        + "def _select_bucket(init_signature, args, kwargs):\n"
        + "    signature = (init_signature, _invocation_signature(args, kwargs))\n"
        + "    index = _SIGNATURE_TO_BUCKET.get(signature)\n"
        + "    if index is None:\n"
        + "        raise RuntimeError(f'no deterministic workload bucket for signature: {signature!r}')\n"
        + "    return index\n\n"
    )
    if kind == "shapes":
        models = ", ".join(embedded.entry_symbols)
        return header + f"_BUCKET_MODELS = ({models},)\n\n" + '''
class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dispatch_init_args = args
        self._dispatch_init_kwargs = kwargs
        self._dispatch_init_signature = _invocation_signature(args, kwargs)
        self._dispatch_models = torch.nn.ModuleDict()

    def forward(self, *args, **kwargs):
        index = _select_bucket(self._dispatch_init_signature, args, kwargs)
        key = str(index)
        if key not in self._dispatch_models:
            candidate = _BUCKET_MODELS[index](
                *self._dispatch_init_args, **self._dispatch_init_kwargs
            )
            if not isinstance(candidate, torch.nn.Module):
                raise TypeError("bucket Model must inherit torch.nn.Module")
            anchor = _first_tensor((args, kwargs))
            if anchor is not None:
                candidate = candidate.to(anchor.device)
            candidate.train(self.training)
            self._dispatch_models[key] = candidate
        return self._dispatch_models[key](*args, **kwargs)
'''.lstrip()
    if kind == "sol":
        empty_init = (
            "invocation",
            (),
            (),
        )
        runners = ", ".join(embedded.entry_symbols)
        return header + f"_BUCKET_RUNNERS = ({runners},)\n\n" + f'''
def run(*args, **kwargs):
    index = _select_bucket({empty_init!r}, args, kwargs)
    return _BUCKET_RUNNERS[index](*args, **kwargs)
'''.lstrip()
    raise AssertionError("unreachable")


@dataclass
class WorkloadBucketCoordinator:
    """Inspect, optimize buckets concurrently, and maintain the full-workload kernel."""

    aggregate_campaign: Campaign
    op_dir: Path
    max_buckets: int = 8
    aggregate_min_improvement: float = 0.0
    _aggregate_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _bucket_campaigns: dict[str, Campaign] = field(default_factory=dict, init=False, repr=False)

    @property
    def workspace(self) -> Path:
        return self.aggregate_campaign.workspace

    @property
    def manifest_path(self) -> Path:
        return self.workspace / WORKLOAD_BUCKETS_FILE

    @property
    def state_path(self) -> Path:
        return self.workspace / AGGREGATION_STATE_FILE

    def _account(self, result: SessionResult, label: str) -> None:
        self.aggregate_campaign._account(result, label)

    def _ensure_main_workspace(self) -> None:
        if latest_version(self.workspace) < 0:
            self.aggregate_campaign.setup_baseline()
        else:
            print(
                f"[workload-coordinator] resuming aggregate workspace at "
                f"v{latest_version(self.workspace)}",
                flush=True,
            )
            preserve_interrupted_tracked_changes(
                self.workspace, "resume aggregate workspace"
            )
            self.aggregate_campaign._link_runtime()
        gitignore = self.workspace / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        ignore_line = f"/{BUCKETS_DIR}/"
        if ignore_line not in existing.splitlines():
            with gitignore.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n# workload inspector derived inputs and independent bucket git repos\n"
                    f"{ignore_line}\n"
                )
        self._commit_main("workload coordinator: ignore derived bucket workspaces", ".gitignore")
        # Buckets inherit this kernel, so the framework bring-up happens once here rather than
        # once per bucket. Ordered after the ignore rule: the session must never stage the
        # nested bucket repositories of a resumed campaign.
        self.aggregate_campaign.ensure_framework_baseline()

    def _commit_main(self, message: str, *paths: str) -> None:
        subprocess.run(["git", "add", "--", *paths], cwd=str(self.workspace), check=True)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(self.workspace)
        )
        if changed.returncode == 1:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        elif changed.returncode != 0:
            raise RuntimeError("could not inspect staged aggregate workspace changes")

    def _ensure_dispatch_signatures(
        self, workload_source: WorkloadSource
    ) -> list[dict]:
        """Collect and cache evaluator-faithful structural workload signatures."""
        signature_path = self.workspace / DISPATCH_SIGNATURES_FILE
        if signature_path.is_file():
            try:
                payload = json.loads(signature_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid {signature_path}: {exc}") from exc
            records = validate_dispatch_signatures(payload, workload_source)
            if payload.get("visibility_policy") != DISPATCH_VISIBILITY_POLICY:
                signature_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "visibility_policy": DISPATCH_VISIBILITY_POLICY,
                            "kind": workload_source.kind,
                            "workload_source": workload_source.filename,
                            "workloads": records,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._commit_main(
                    "workload inspector: certify no-sync dispatch signatures",
                    DISPATCH_SIGNATURES_FILE,
                )
            return records

        timeout = max(
            self.aggregate_campaign.sandbox_timeout,
            AGGREGATE_VALIDATION_TIMEOUT,
        )
        try:
            result = _sandbox_command(
                self.workspace,
                self.aggregate_campaign.sandbox_hardware,
                self.aggregate_campaign.sandbox_profile,
                self.aggregate_campaign.sandbox_url,
                timeout,
                [
                    "python",
                    "tools/collect_dispatch_signatures.py",
                    "--workspace",
                    ".",
                ],
                wall_timeout=timeout + AGGREGATE_QUEUE_WAIT_GRACE,
                dispatch_signatures=True,
                gateway_kind="dev",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"dispatch signature collection failed to run: {exc}") from exc
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"dispatch signature collection failed (exit={result.returncode})"
            )
        payload: object = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(DISPATCH_SIGNATURE_RESULT_PREFIX):
                try:
                    payload = json.loads(line[len(DISPATCH_SIGNATURE_RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"dispatch signature collector emitted invalid JSON: {exc}"
                    ) from exc
                break
        records = validate_dispatch_signatures(payload, workload_source)
        persisted = {
            "schema_version": 1,
            "visibility_policy": DISPATCH_VISIBILITY_POLICY,
            "kind": workload_source.kind,
            "workload_source": workload_source.filename,
            "workloads": records,
        }
        signature_path.write_text(
            json.dumps(persisted, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._commit_main(
            "workload inspector: record deterministic dispatch signatures",
            DISPATCH_SIGNATURES_FILE,
        )
        print(
            f"[workload-inspector] recorded {len(records)} deterministic runtime "
            f"signatures in {signature_path}",
            flush=True,
        )
        return records

    def inspect_workloads(self) -> list[WorkloadBucket]:
        workload_source = _read_workload_source(self.op_dir)
        workload_count = len(workload_source.entries)
        signature_records = self._ensure_dispatch_signatures(workload_source)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            buckets = validate_workload_buckets(manifest, workload_count, self.max_buckets)
            validate_dispatch_bucket_compatibility(buckets, signature_records)
            if manifest.get("dispatch_visibility_policy") != DISPATCH_VISIBILITY_POLICY:
                # A legacy partition is still mechanically safe when every exact
                # production-visible signature has one owner. Remove rationales that may
                # have cited evaluator-only values and certify the partition under the
                # current no-sync policy without renaming its existing bucket workspaces.
                buckets = [
                    WorkloadBucket(
                        name=bucket.name,
                        workload_indices=bucket.workload_indices,
                        rationale=(
                            "Legacy partition revalidated as a union of exact "
                            f"{DISPATCH_VISIBILITY_POLICY} signature classes; no tensor "
                            "contents or workload-source values are dispatch inputs."
                        ),
                    )
                    for bucket in buckets
                ]
                self.manifest_path.write_text(
                    json.dumps(
                        {
                            **_normalized_bucket_manifest(buckets, workload_count),
                            "workload_source": workload_source.filename,
                            "workload_ids": list(workload_source.ids),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._commit_main(
                    "workload inspector: certify production-visible bucket partition",
                    WORKLOAD_BUCKETS_FILE,
                )
            print(
                f"[workload-inspector] reusing {len(buckets)} validated buckets from "
                f"{self.manifest_path}",
                flush=True,
            )
            return buckets

        # The LLM inspector runs in a data-minimized workspace. It receives only the
        # signatures the generated production dispatcher can recompute without a device
        # synchronization; reference.py, input.py, shapes.json/workload.jsonl, tensor
        # contents, and evaluator metadata are not present.
        with tempfile.TemporaryDirectory(prefix="atrex-dispatch-inspector-") as temp_dir:
            inspector_workspace = Path(temp_dir)
            inspector_signature_path = inspector_workspace / DISPATCH_SIGNATURES_FILE
            inspector_signature_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "visibility_policy": DISPATCH_VISIBILITY_POLICY,
                        "kind": workload_source.kind,
                        "workload_source": workload_source.filename,
                        "workloads": signature_records,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=str(inspector_workspace),
                check=True,
            )
            prompt = _render(
                PROMPTS_DIR / "inspect_workloads.md",
                WORKSPACE=str(inspector_workspace),
                MAX_BUCKETS=self.max_buckets,
                WORKLOAD_COUNT=workload_count,
                WORKLOAD_FILE=workload_source.filename,
                WORKLOAD_KIND=workload_source.kind,
                VISIBILITY_POLICY=DISPATCH_VISIBILITY_POLICY,
                PLATFORM=self.aggregate_campaign.platform,
                FRAMEWORK=self.aggregate_campaign.framework,
            )
            result = run_session(
                inspector_workspace,
                prompt,
                timeout=self.aggregate_campaign.setup_timeout,
                agent_cli=self.aggregate_campaign.agent_cli,
                sandbox_hardware=self.aggregate_campaign.sandbox_hardware,
                sandbox_profile=self.aggregate_campaign.sandbox_profile,
                sandbox_url=self.aggregate_campaign.sandbox_url,
                sandbox_timeout=self.aggregate_campaign.sandbox_timeout,
            )
            inspector_manifest_path = inspector_workspace / WORKLOAD_BUCKETS_FILE
            manifest_text = (
                inspector_manifest_path.read_text(encoding="utf-8")
                if inspector_manifest_path.exists()
                else ""
            )
        self._account(result, "workload inspector")
        if result.exit_status != 0 or result.timed_out:
            raise RuntimeError(
                f"workload inspector failed (exit={result.exit_status}, timed_out={result.timed_out})"
            )
        if not manifest_text:
            raise RuntimeError(f"workload inspector did not produce {WORKLOAD_BUCKETS_FILE}")
        try:
            manifest = json.loads(manifest_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"workload inspector produced invalid JSON: {exc}") from exc
        buckets = validate_workload_buckets(
            manifest,
            workload_count,
            self.max_buckets,
            require_visibility_policy=True,
        )
        validate_dispatch_bucket_compatibility(buckets, signature_records)
        self.manifest_path.write_text(
            json.dumps(
                {
                    **_normalized_bucket_manifest(buckets, workload_count),
                    "workload_source": workload_source.filename,
                    "workload_ids": list(workload_source.ids),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self._commit_main(
            f"workload inspector: partition into {len(buckets)} buckets",
            WORKLOAD_BUCKETS_FILE,
            ".gitignore",
        )
        print(
            f"[workload-inspector] validated {len(buckets)} disjoint buckets covering "
            f"{workload_count} workloads from {workload_source.filename}",
            flush=True,
        )
        return buckets

    def _load_state(self) -> dict:
        state: dict
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    state.setdefault("schema_version", 1)
                    state.setdefault("buckets", {})
                else:
                    state = {"schema_version": 1, "buckets": {}}
            except (OSError, json.JSONDecodeError):
                state = {"schema_version": 1, "buckets": {}}
        else:
            state = {"schema_version": 1, "buckets": {}}

        state.setdefault("pending_improvements", {})
        if "bootstrap" not in state:
            # Compatibility for campaigns that accepted aggregate kernels
            # before the ten-round bootstrap barrier existed.  Those campaigns
            # have already completed their first aggregate and must continue in
            # incremental mode after a restart instead of being gated again.
            legacy_accepted = any(
                bool((read_memory(self.workspace, version) or {}).get("aggregation"))
                for version in range(1, latest_version(self.workspace) + 1)
            )
            state["bootstrap"] = {
                "status": "ACCEPTED" if legacy_accepted else "PENDING",
                "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                "reason": (
                    "migrated existing accepted aggregate"
                    if legacy_accepted
                    else "waiting for ten rounds and one improvement from every bucket"
                ),
            }
        return state

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _bootstrap_accepted(state: dict) -> bool:
        return (state.get("bootstrap") or {}).get("status") == "ACCEPTED"

    def _remember_pending_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
    ) -> dict:
        """Persist the newest pre-bootstrap win for one bucket.

        This records provenance only; it never edits or validates the aggregate
        kernel.  Keeping the pending set in the main Git repository makes the
        ten-round barrier restart-safe.
        """
        state = self._load_state()
        pending = state.setdefault("pending_improvements", {})
        entry = {
            "bucket_version": f"v{version}",
            "bucket_latency_us": (memory.get("performance") or {}).get("latency_us"),
            "bucket_head": git_head(campaign.workspace),
            "bucket_kernel_blob": git_kernel_blob(campaign.workspace),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        previous = pending.get(bucket.name) or {}
        if (
            previous.get("bucket_version") == entry["bucket_version"]
            and previous.get("bucket_kernel_blob") == entry["bucket_kernel_blob"]
        ):
            return state
        pending[bucket.name] = entry
        bootstrap = state.setdefault("bootstrap", {})
        bootstrap.update(
            {
                "status": "PENDING",
                "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                "reason": "waiting for ten rounds and one improvement from every bucket",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write_state(state)
        self._commit_main(
            f"aggregate: defer {bucket.name} v{version} until initial all-bucket barrier",
            AGGREGATION_STATE_FILE,
        )
        print(
            f"[aggregate] deferred {bucket.name} v{version}: initial aggregation waits "
            f"for every bucket to reach v{INITIAL_AGGREGATION_MIN_ITERATIONS} and improve",
            flush=True,
        )
        return state

    def _pending_bootstrap_sources(
        self, state: dict
    ) -> Optional[list[tuple[WorkloadBucket, Campaign, int, dict]]]:
        """Resolve a complete, current all-bucket source set from persisted wins."""
        pending = state.get("pending_improvements") or {}
        if set(pending) < set(self._bucket_campaigns):
            return None
        if any(
            latest_version(campaign.workspace) < INITIAL_AGGREGATION_MIN_ITERATIONS
            for campaign in self._bucket_campaigns.values()
        ):
            return None

        manifest = {
            bucket.name: bucket
            for bucket in validate_workload_buckets(
                json.loads(self.manifest_path.read_text(encoding="utf-8")),
                len(_read_workload_source(self.op_dir).entries),
                self.max_buckets,
            )
        }
        sources: list[tuple[WorkloadBucket, Campaign, int, dict]] = []
        for name in sorted(self._bucket_campaigns):
            entry = pending.get(name) or {}
            version_text = str(entry.get("bucket_version") or "")
            match = re.fullmatch(r"v(\d+)", version_text)
            if match is None or name not in manifest:
                return None
            campaign = self._bucket_campaigns[name]
            if head_kernel_is_initial_baseline(campaign.workspace):
                return None
            version = int(match.group(1))
            memory = read_memory(campaign.workspace, version)
            if not memory or not _status_is(
                (memory.get("quality_gate") or {}).get("result"), "PASS"
            ):
                return None
            sources.append((manifest[name], campaign, version, memory))
        return sources

    def _maybe_initial_aggregation_locked(self) -> bool:
        state = self._load_state()
        if self._bootstrap_accepted(state):
            return False
        sources = self._pending_bootstrap_sources(state)
        if not sources:
            return False
        fingerprint = {
            bucket.name: git_kernel_blob(campaign.workspace)
            for bucket, campaign, _version, _memory in sources
        }
        bootstrap = state.get("bootstrap") or {}
        if (
            bootstrap.get("status") == "REJECTED"
            and bootstrap.get("source_kernel_blobs") == fingerprint
            and bootstrap.get("dispatch_schema_version")
            == AGGREGATE_DISPATCH_SCHEMA_VERSION
            and bootstrap.get("source_layout") == AGGREGATE_SOURCE_LAYOUT
        ):
            return False
        print(
            "[aggregate] initial barrier satisfied: every bucket reached "
            f"v{INITIAL_AGGREGATION_MIN_ITERATIONS} and has a committed improvement; "
            "rebuilding from all bucket HEADs",
            flush=True,
        )
        bucket, campaign, version, memory = sources[-1]
        return self.aggregate_improvement(
            bucket,
            campaign,
            version,
            memory,
            initial_sources=sources,
        )

    def record_bucket_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
    ) -> bool:
        """Apply the two-phase aggregation policy to a mechanically accepted win."""
        with self._aggregate_lock:
            state = self._load_state()
            if self._bootstrap_accepted(state):
                return self.aggregate_improvement(bucket, campaign, version, memory)
            self._remember_pending_improvement(bucket, campaign, version, memory)
            return self._maybe_initial_aggregation_locked()

    def bucket_iteration_completed(
        self,
        _campaign: Campaign,
        version: int,
        _memory: Optional[dict],
        _won: bool,
    ) -> None:
        """Open the initial barrier when the slowest bucket completes round ten."""
        if version < INITIAL_AGGREGATION_MIN_ITERATIONS:
            return
        with self._aggregate_lock:
            self._maybe_initial_aggregation_locked()

    def _current_all_bucket_sources(
        self,
        override: tuple[WorkloadBucket, Campaign, int, dict],
    ) -> Optional[list[tuple[WorkloadBucket, Campaign, int, dict]]]:
        """Resolve current committed winners for one-time legacy dispatcher migration."""
        workload_source = _read_workload_source(self.op_dir)
        manifest = {
            bucket.name: bucket
            for bucket in validate_workload_buckets(
                json.loads(self.manifest_path.read_text(encoding="utf-8")),
                len(workload_source.entries),
                self.max_buckets,
            )
        }
        override_bucket, override_campaign, override_version, override_memory = override
        resolved: list[tuple[WorkloadBucket, Campaign, int, dict]] = []
        for name in sorted(manifest):
            if name == override_bucket.name:
                resolved.append(
                    (
                        override_bucket,
                        override_campaign,
                        override_version,
                        override_memory,
                    )
                )
                continue
            campaign = self._bucket_campaigns.get(name)
            if campaign is None or head_kernel_is_initial_baseline(campaign.workspace):
                return None
            best_before: Optional[float] = None
            latest_win: Optional[tuple[int, dict]] = None
            for candidate_version in range(0, latest_version(campaign.workspace) + 1):
                candidate_memory = read_memory(campaign.workspace, candidate_version)
                if not candidate_memory or not _status_is(
                    (candidate_memory.get("quality_gate") or {}).get("result"), "PASS"
                ):
                    continue
                latency = (candidate_memory.get("performance") or {}).get("latency_us")
                if not isinstance(latency, (int, float)):
                    continue
                if (
                    candidate_version > 0
                    and (best_before is None or float(latency) < best_before)
                ):
                    latest_win = (candidate_version, candidate_memory)
                best_before = (
                    float(latency)
                    if best_before is None
                    else min(best_before, float(latency))
                )
            if latest_win is None:
                return None
            candidate_version, candidate_memory = latest_win
            resolved.append(
                (manifest[name], campaign, candidate_version, candidate_memory)
            )
        return resolved

    def _dispatch_inputs(self) -> tuple[WorkloadSource, list[WorkloadBucket], list[dict]]:
        workload_source = _read_workload_source(self.op_dir)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        buckets = validate_workload_buckets(
            manifest, len(workload_source.entries), self.max_buckets
        )
        signature_payload = json.loads(
            (self.workspace / DISPATCH_SIGNATURES_FILE).read_text(encoding="utf-8")
        )
        signatures = validate_dispatch_signatures(signature_payload, workload_source)
        validate_dispatch_bucket_compatibility(buckets, signatures)
        return workload_source, buckets, signatures

    def _materialize_deterministic_dispatch(
        self,
        sources: list[tuple[WorkloadBucket, Campaign, int, dict]],
        *,
        initial: bool,
    ) -> None:
        """Embed committed bucket kernels into one exact static kernel.py."""
        workload_source, buckets, signatures = self._dispatch_inputs()
        expected_names = {bucket.name for bucket in buckets}
        module_dir = self.workspace / AGGREGATE_KERNELS_DIR
        dispatch_path = self.workspace / AGGREGATE_DISPATCH_FILE
        existing_records: dict[str, dict] = {}
        if not initial:
            if not dispatch_path.is_file():
                raise RuntimeError(
                    "incremental deterministic aggregation requires an accepted dispatcher"
                )
            try:
                existing_manifest = json.loads(dispatch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid accepted dispatcher manifest: {exc}") from exc
            if existing_manifest.get("mode") != "deterministic_dispatch":
                raise RuntimeError("accepted aggregate is not a deterministic dispatcher")
            if existing_manifest.get("schema_version") not in {1, 2}:
                raise RuntimeError("accepted dispatcher has an unsupported schema")
            raw_modules = existing_manifest.get("modules")
            if not isinstance(raw_modules, dict):
                raise RuntimeError("accepted dispatcher manifest has no modules")
            existing_records = {
                str(name): dict(record)
                for name, record in raw_modules.items()
                if isinstance(record, dict)
            }

        module_records = existing_records
        module_sources: dict[str, str] = {}
        for source_bucket, source_campaign, source_version, _source_memory in sources:
            if source_bucket.name not in expected_names:
                raise RuntimeError(f"unknown aggregate bucket: {source_bucket.name}")
            source = _git_head_file(source_campaign.workspace, "kernel.py")
            module_sources[source_bucket.name] = source
            module_records[source_bucket.name] = {
                "embedded": True,
                "bucket_version": f"v{source_version}",
                "bucket_head": git_head(source_campaign.workspace),
                "kernel_blob": git_kernel_blob(source_campaign.workspace),
            }

        if set(module_records) != expected_names:
            missing = sorted(expected_names - set(module_records))
            extra = sorted(set(module_records) - expected_names)
            raise RuntimeError(
                f"deterministic dispatcher source mismatch: missing={missing}, extra={extra}"
            )

        # Incremental schema-v2 aggregates recover unchanged source from the exact
        # recorded Git blob. Schema-v1 aggregates read their tracked module once
        # and are migrated to the single-file format by this regeneration.
        for name, record in module_records.items():
            if name in module_sources:
                continue
            legacy_path = str(record.get("path") or "")
            legacy_source = self.workspace / legacy_path if legacy_path else None
            if legacy_source is not None and legacy_source.is_file():
                module_sources[name] = legacy_source.read_text(encoding="utf-8")
            else:
                campaign = self._bucket_campaigns.get(name)
                if campaign is None:
                    raise RuntimeError(f"no bucket workspace available for aggregate source: {name}")
                module_sources[name] = _git_blob_file(
                    campaign.workspace, record.get("kernel_blob")
                )
            record.pop("path", None)
            record["embedded"] = True

        bucket_by_index = {
            index: bucket.name
            for bucket in buckets
            for index in bucket.workload_indices
        }
        dispatcher = build_deterministic_dispatcher(
            kind=workload_source.kind,
            signature_records=signatures,
            bucket_by_index=bucket_by_index,
            module_records=module_records,
            module_sources=module_sources,
        )
        if module_dir.exists():
            shutil.rmtree(module_dir)
        (self.workspace / "kernel.py").write_text(dispatcher, encoding="utf-8")
        dispatch_manifest = {
            "schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
            "mode": "deterministic_dispatch",
            "source_layout": AGGREGATE_SOURCE_LAYOUT,
            "dispatch_visibility_policy": DISPATCH_VISIBILITY_POLICY,
            "kind": workload_source.kind,
            "workload_source": workload_source.filename,
            "signature_source": DISPATCH_SIGNATURES_FILE,
            "modules": {name: module_records[name] for name in sorted(module_records)},
        }
        dispatch_path.write_text(
            json.dumps(dispatch_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _restore_aggregate_incumbent(self, pre_head: str) -> None:
        """Reset tracked candidates and remove only generated untracked paths."""
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(self.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                AGGREGATE_KERNELS_DIR,
                AGGREGATE_DISPATCH_FILE,
            ],
            cwd=str(self.workspace),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for relative in untracked:
            path = (self.workspace / relative).resolve()
            if self.workspace.resolve() not in path.parents:
                raise RuntimeError(f"refusing to remove aggregate path outside workspace: {path}")
            if path.is_file() or path.is_symlink():
                path.unlink()
        module_dir = self.workspace / AGGREGATE_KERNELS_DIR
        if module_dir.is_dir() and not any(module_dir.iterdir()):
            module_dir.rmdir()

    def _record_aggregation_state(
        self,
        *,
        bucket: WorkloadBucket,
        bucket_head: str,
        bucket_kernel_blob: str,
        bucket_version: int,
        bucket_latency: object,
        status: str,
        reason: str,
        aggregate_latency: object = None,
        state: Optional[dict] = None,
    ) -> dict:
        if state is None:
            state = self._load_state()
        state["buckets"][bucket.name] = {
            "last_seen_head": bucket_head,
            "last_seen_kernel_blob": bucket_kernel_blob,
            "bucket_version": f"v{bucket_version}",
            "bucket_latency_us": bucket_latency,
            "status": status,
            "reason": reason,
            "aggregate_latency_us": aggregate_latency,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return state

    def _refresh_aggregate_solution_metadata(self) -> None:
        """Union framework metadata from bucket solutions into the main solution."""
        solution_path = self.workspace / "solution.json"
        if not solution_path.exists():
            return
        try:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        spec = solution.setdefault("spec", {})
        languages = list(spec.get("languages") or [])
        dependencies = list(spec.get("dependencies") or [])
        for campaign in self._bucket_campaigns.values():
            bucket_solution_path = campaign.workspace / "solution.json"
            if not bucket_solution_path.exists():
                continue
            try:
                bucket_solution = json.loads(
                    bucket_solution_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            bucket_spec = bucket_solution.get("spec") or {}
            for language in bucket_spec.get("languages") or []:
                if language not in languages:
                    languages.append(language)
            for dependency in bucket_spec.get("dependencies") or []:
                if dependency not in dependencies:
                    dependencies.append(dependency)
        spec["languages"] = languages
        spec["dependencies"] = dependencies
        dispatch_path = self.workspace / AGGREGATE_DISPATCH_FILE
        if dispatch_path.is_file():
            solution["sources"] = [{"path": "kernel.py"}]
        solution["description"] = (
            "Self-contained deterministic full-workload dispatcher over independently "
            "optimized buckets"
        )
        solution_path.write_text(
            json.dumps(solution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def aggregate_improvement(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        version: int,
        memory: dict,
        *,
        initial_sources: Optional[
            list[tuple[WorkloadBucket, Campaign, int, dict]]
        ] = None,
    ) -> bool:
        """Serialize aggregate edits while bucket optimization threads keep running.

        ``initial_sources`` is supplied exactly for the first accepted aggregate:
        it forces one candidate to be rebuilt from every bucket HEAD after the
        ten-round warmup.  Later calls carry one changed bucket and preserve the
        established aggregate paths.
        """
        with self._aggregate_lock:
            if initial_sources is None and not (
                self.workspace / AGGREGATE_DISPATCH_FILE
            ).is_file():
                migration_sources = self._current_all_bucket_sources(
                    (bucket, campaign, version, memory)
                )
                if migration_sources is not None:
                    initial_sources = migration_sources
                    print(
                        "[aggregate] migrating legacy semantic aggregate to deterministic dispatch",
                        flush=True,
                    )
            is_initial = initial_sources is not None
            sources = initial_sources or [(bucket, campaign, version, memory)]
            bucket_head = git_head(campaign.workspace)
            bucket_kernel_blob = git_kernel_blob(campaign.workspace)
            state = self._load_state()
            previous = (state.get("buckets") or {}).get(bucket.name) or {}
            if (
                not is_initial
                and bucket_kernel_blob
                and previous.get("last_seen_kernel_blob") == bucket_kernel_blob
            ):
                return False
            # Backward-compatible resume for state written before blob ids were
            # recorded. Metadata-only bucket commits must not trigger a rebuild.
            if (
                not is_initial
                and
                not previous.get("last_seen_kernel_blob")
                and bucket_head
                and previous.get("last_seen_head") == bucket_head
            ):
                return False

            bucket_latency = (memory.get("performance") or {}).get("latency_us")
            incumbent_latency_us = best_validated_latency_us(self.workspace)
            aggregate_version = latest_version(self.workspace) + 1
            source_kernel_blobs = {
                source_bucket.name: git_kernel_blob(source_campaign.workspace)
                for source_bucket, source_campaign, _source_version, _source_memory in sources
            }
            if is_initial:
                source_names = ", ".join(source_bucket.name for source_bucket, *_ in sources)
                print(
                    f"[aggregate] initial deterministic dispatch from: {source_names}",
                    flush=True,
                )
            else:
                print(
                    f"[aggregate] bucket={bucket.name} v{version} improved to "
                    f"{bucket_latency} us; updating deterministic dispatch immediately",
                    flush=True,
                )
            pre_head = git_head(self.workspace)
            if not pre_head:
                raise RuntimeError("aggregate workspace has no git incumbent")
            try:
                self._materialize_deterministic_dispatch(
                    sources,
                    initial=is_initial,
                )
                self._refresh_aggregate_solution_metadata()
            except Exception as exc:
                try:
                    self._restore_aggregate_incumbent(pre_head)
                except Exception as cleanup_exc:
                    exc = RuntimeError(f"{exc}; rollback failed: {cleanup_exc}")
                reason = f"deterministic dispatch generation failed: {exc}"
                state = self._record_aggregation_state(
                    bucket=bucket,
                    bucket_head=bucket_head,
                    bucket_kernel_blob=bucket_kernel_blob,
                    bucket_version=version,
                    bucket_latency=bucket_latency,
                    status="REJECTED",
                    reason=reason,
                )
                if is_initial:
                    state["bootstrap"] = {
                        "status": "REJECTED",
                        "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                        "source_kernel_blobs": source_kernel_blobs,
                        "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                        "source_layout": AGGREGATE_SOURCE_LAYOUT,
                        "reason": reason,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                self._write_state(state)
                self._commit_main(
                    (
                        "aggregate: reject initial all-bucket bootstrap"
                        if is_initial
                        else f"aggregate: reject {bucket.name} v{version}"
                    ),
                    AGGREGATION_STATE_FILE,
                )
                print(
                    f"[aggregate] rejected "
                    f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                    f"{reason}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            kernel_path = self.workspace / "kernel.py"
            candidate = kernel_path.read_text(encoding="utf-8") if kernel_path.exists() else ""

            rejection = ""
            test_result: Optional[dict] = None
            if not candidate:
                rejection = "deterministic aggregation produced no kernel.py"
            else:
                changed = subprocess.run(
                    [
                        "git",
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                        "--",
                        "kernel.py",
                        AGGREGATE_KERNELS_DIR,
                        AGGREGATE_DISPATCH_FILE,
                        "solution.json",
                    ],
                    cwd=str(self.workspace),
                    capture_output=True,
                    text=True,
                )
                if changed.returncode != 0:
                    rejection = "could not compare aggregate kernel candidate"
                elif not changed.stdout.strip():
                    rejection = "deterministic aggregation did not change aggregate sources"

            if not rejection and self.aggregate_campaign.optimization_mode == "production":
                violations = production_kernel_violations(
                    self.workspace, self.aggregate_campaign.framework
                )
                if violations:
                    rejection = "production policy: " + "; ".join(violations)

            if not rejection:
                validation_timeout = max(
                    self.aggregate_campaign.sandbox_timeout,
                    AGGREGATE_VALIDATION_TIMEOUT,
                )
                validation_stages = [
                    (
                        "single-seed",
                        [
                            "python", "test_kernel.py", "--version",
                            f"v{aggregate_version}", "--no-memory",
                        ],
                    ),
                    (
                        "multi-seed",
                        [
                            "python", "test_kernel.py", "--version",
                            f"v{aggregate_version}", "--multi-seed", "5", "--no-memory",
                        ],
                    ),
                ]
                for stage_name, validation_command in validation_stages:
                    try:
                        test = _sandbox_command(
                            self.workspace,
                            self.aggregate_campaign.sandbox_hardware,
                            self.aggregate_campaign.sandbox_profile,
                            self.aggregate_campaign.sandbox_url,
                            validation_timeout,
                            validation_command,
                            wall_timeout=(
                                validation_timeout + AGGREGATE_QUEUE_WAIT_GRACE
                            ),
                            gateway_kind="run",
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        rejection = (
                            f"full-workload {stage_name} validation failed to run: {exc}"
                        )
                        break
                    if test.stdout:
                        print(
                            test.stdout,
                            end="" if test.stdout.endswith("\n") else "\n",
                            flush=True,
                        )
                    if test.stderr:
                        print(
                            test.stderr,
                            end="" if test.stderr.endswith("\n") else "\n",
                            file=sys.stderr,
                            flush=True,
                        )
                    if test.returncode != 0:
                        rejection = (
                            f"full-workload {stage_name} validation command failed "
                            f"(exit={test.returncode})"
                        )
                        break
                    try:
                        stage_result = _test_result_from_stdout(test.stdout)
                    except (RuntimeError, json.JSONDecodeError) as exc:
                        rejection = (
                            f"full-workload {stage_name} validation produced no usable "
                            f"result: {exc}"
                        )
                        break
                    test_result = stage_result
                    if not stage_result.get("all_pass"):
                        rejection = (
                            f"full-workload {stage_name} correctness validation failed"
                        )
                        break

            aggregate_latency = (
                test_result.get("latency_us_geomean") if test_result is not None else None
            )
            if not rejection and not isinstance(aggregate_latency, (int, float)):
                rejection = "full-workload validation returned no latency"
            if not rejection and incumbent_latency_us is not None:
                required = incumbent_latency_us * (1.0 - self.aggregate_min_improvement)
                if float(aggregate_latency) >= required:
                    rejection = (
                        f"full-workload latency {float(aggregate_latency):.6g} us did not beat "
                        f"incumbent {incumbent_latency_us:.6g} us"
                    )

            if rejection:
                self._restore_aggregate_incumbent(pre_head)
                state = self._record_aggregation_state(
                    bucket=bucket,
                    bucket_head=bucket_head,
                    bucket_kernel_blob=bucket_kernel_blob,
                    bucket_version=version,
                    bucket_latency=bucket_latency,
                    status="REJECTED",
                    reason=rejection,
                    aggregate_latency=aggregate_latency,
                )
                if is_initial:
                    state["bootstrap"] = {
                        "status": "REJECTED",
                        "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                        "source_kernel_blobs": source_kernel_blobs,
                        "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                        "source_layout": AGGREGATE_SOURCE_LAYOUT,
                        "reason": rejection,
                        "aggregate_latency_us": aggregate_latency,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                self._write_state(state)
                self._commit_main(
                    (
                        "aggregate: reject initial all-bucket bootstrap"
                        if is_initial
                        else f"aggregate: reject {bucket.name} v{version}"
                    ),
                    AGGREGATION_STATE_FILE,
                )
                print(
                    f"[aggregate] rejected "
                    f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                    f"{rejection}",
                    flush=True,
                )
                return False

            assert test_result is not None
            memory_path = _record_local_test_result(
                self.workspace, f"v{aggregate_version}", test_result
            )
            aggregate_memory = json.loads(memory_path.read_text(encoding="utf-8"))
            aggregate_memory["aggregation"] = {
                "mode": (
                    "deterministic_dispatch_bootstrap"
                    if is_initial
                    else "deterministic_dispatch_incremental"
                ),
                "source_bucket": bucket.name,
                "source_bucket_version": f"v{version}",
                "source_bucket_head": bucket_head,
                "source_bucket_kernel_blob": bucket_kernel_blob,
                "previous_aggregate_head": pre_head,
            }
            if is_initial:
                aggregate_memory["aggregation"].update(
                    {
                        "sources": [
                            {
                                "bucket": source_bucket.name,
                                "version": f"v{source_version}",
                                "head": git_head(source_campaign.workspace),
                                "kernel_blob": git_kernel_blob(source_campaign.workspace),
                            }
                            for source_bucket, source_campaign, source_version, _source_memory
                            in sources
                        ],
                    }
                )
            aggregate_memory["git_commit_hash"] = None
            memory_path.write_text(
                json.dumps(aggregate_memory, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            state = self._record_aggregation_state(
                bucket=bucket,
                bucket_head=bucket_head,
                bucket_kernel_blob=bucket_kernel_blob,
                bucket_version=version,
                bucket_latency=bucket_latency,
                status="ACCEPTED",
                reason=(
                    "deterministic dispatch passed full-workload correctness and improved geomean"
                ),
                aggregate_latency=aggregate_latency,
            )
            if is_initial:
                state["bootstrap"] = {
                    "status": "ACCEPTED",
                    "minimum_iterations": INITIAL_AGGREGATION_MIN_ITERATIONS,
                    "source_kernel_blobs": source_kernel_blobs,
                    "dispatch_mode": "deterministic_dispatch",
                    "dispatch_schema_version": AGGREGATE_DISPATCH_SCHEMA_VERSION,
                    "source_layout": AGGREGATE_SOURCE_LAYOUT,
                    "reason": (
                        "deterministic dispatch passed full-workload correctness and improved geomean"
                    ),
                    "aggregate_latency_us": aggregate_latency,
                    "aggregate_version": f"v{aggregate_version}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            self._write_state(state)
            commit_paths = [
                "kernel.py",
                AGGREGATE_DISPATCH_FILE,
                str(memory_path.relative_to(self.workspace)),
                AGGREGATION_STATE_FILE,
            ]
            legacy_modules = subprocess.run(
                ["git", "ls-files", "--", AGGREGATE_KERNELS_DIR],
                cwd=str(self.workspace),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if legacy_modules or (self.workspace / AGGREGATE_KERNELS_DIR).exists():
                commit_paths.append(AGGREGATE_KERNELS_DIR)
            if (self.workspace / "solution.json").exists():
                commit_paths.append("solution.json")
            self._commit_main(
                (
                    f"aggregate: accept initial all-bucket bootstrap "
                    f"({float(aggregate_latency):.3f} us)"
                    if is_initial
                    else f"aggregate: accept {bucket.name} v{version} "
                    f"({float(aggregate_latency):.3f} us)"
                ),
                *commit_paths,
            )
            print(
                f"[aggregate] accepted "
                f"{'initial all-bucket bootstrap' if is_initial else f'{bucket.name} v{version}'}: "
                "full-workload latency "
                f"{float(aggregate_latency):.6g} us, git={git_head(self.workspace)[:8]}",
                flush=True,
            )
            return True

    def _make_bucket_campaign(
        self, bucket: WorkloadBucket, workload_source: WorkloadSource
    ) -> Campaign:
        bucket_root = self.workspace / BUCKETS_DIR
        bucket_op = bucket_root / "ops" / bucket.name
        bucket_runs = bucket_root / "runs"
        # Campaign.setup_baseline() invokes workspace_init.sh with work_dir as
        # its cwd.  Materialize this shared parent before any bucket threads
        # start so every campaign sees a valid cwd.
        bucket_runs.mkdir(parents=True, exist_ok=True)
        _materialize_bucket_op(self.op_dir, bucket_op, bucket, workload_source)
        parent = self.aggregate_campaign

        def on_improvement(campaign: Campaign, version: int, memory: dict) -> None:
            self.record_bucket_improvement(bucket, campaign, version, memory)

        def on_iteration(
            campaign: Campaign, version: int, memory: Optional[dict], won: bool
        ) -> None:
            self.bucket_iteration_completed(campaign, version, memory, won)

        campaign = Campaign(
            name=bucket.name,
            kernel_demo=str(bucket_op / "reference.py"),
            platform=parent.platform,
            framework=parent.framework,
            notes=(
                f"Workload bucket {bucket.name}: {bucket.rationale or 'inspector-grouped workloads'}. "
                "Optimize every workload in this bucket; the coordinator owns full-kernel aggregation."
            ),
            arch=parent.arch,
            work_dir=str(bucket_runs),
            workspace_suffix=parent.workspace_suffix,
            max_iters=parent.max_iters,
            token_budget=parent.token_budget,
            target_util=parent.target_util,
            iter_timeout=parent.iter_timeout,
            setup_timeout=parent.setup_timeout,
            salvage_timeout=parent.salvage_timeout,
            framework_baseline="never",
            framework_baseline_timeout=parent.framework_baseline_timeout,
            max_stall=parent.max_stall,
            convert_after=parent.convert_after,
            sandbox_hardware=parent.sandbox_hardware,
            sandbox_profile=parent.sandbox_profile,
            sandbox_url=parent.sandbox_url,
            sandbox_timeout=parent.sandbox_timeout,
            atrex_bench_root=parent.atrex_bench_root,
            agent_cli=parent.agent_cli,
            optimization_mode=parent.optimization_mode,
            on_improvement=on_improvement,
            on_iteration=on_iteration,
        )
        return campaign

    def _seed_bucket_baseline_from_aggregate(
        self,
        bucket: WorkloadBucket,
        campaign: Campaign,
        workload_source: WorkloadSource,
    ) -> None:
        """Create a bucket V0 from the already-validated aggregate baseline.

        This is deliberately mechanical: no coding-agent session and no GPU
        command are involved.  The bucket receives the aggregate's pinned
        framework-baseline kernel (or its original V0 kernel when the campaign has none)
        plus only its own workload files and per-workload measurements.
        """
        if self.aggregate_campaign.framework_baseline == "never":
            baseline_commit, baseline_version = "", 0
        else:
            baseline_commit, baseline_version = resolve_framework_baseline_commit(self.workspace)
        if latest_version(campaign.workspace) >= 0:
            self._assert_bucket_baseline_provenance(bucket, campaign, baseline_commit)
            return
        aggregate_memory = read_memory(self.workspace, baseline_version)
        if aggregate_memory is None:
            raise RuntimeError(
                "cannot seed bucket baseline before aggregate "
                f"memory/v{baseline_version}.json exists"
            )
        aggregate_passed = _status_is(
            (aggregate_memory.get("quality_gate") or {}).get("result"), "PASS"
        ) or _status_is(
            (aggregate_memory.get("correctness") or {}).get("status"), "PASS"
        )
        if not aggregate_passed:
            raise RuntimeError(
                f"cannot seed bucket baseline from a non-passing aggregate v{baseline_version}"
            )

        roots = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if len(roots) != 1:
            raise RuntimeError("aggregate workspace must have exactly one V0 root commit")
        aggregate_baseline_commit = baseline_commit or roots[0]

        def baseline_file(name: str, *, required: bool = False) -> Optional[str]:
            result = subprocess.run(
                ["git", "show", f"{aggregate_baseline_commit}:{name}"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout
            if required:
                raise RuntimeError(
                    f"aggregate baseline commit {aggregate_baseline_commit[:8]} has no {name}"
                )
            return None

        workspace = campaign.workspace
        for directory in (workspace, workspace / "memory", workspace / "plans", workspace / "profiles"):
            directory.mkdir(parents=True, exist_ok=True)
        if not (workspace / ".git").exists():
            subprocess.run(
                ["git", "init"], cwd=str(workspace), check=True, stdout=subprocess.DEVNULL
            )
            subprocess.run(
                ["git", "config", "user.email", "gpu-kernel-optimizer@local"],
                cwd=str(workspace),
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "GPU Kernel Optimizer"],
                cwd=str(workspace),
                check=True,
            )

        bucket_op = Path(campaign.kernel_demo).resolve().parent
        tracked_files: set[str] = set()
        for source in bucket_op.iterdir():
            if not source.is_file():
                continue
            shutil.copy2(source, workspace / source.name)
            tracked_files.add(source.name)

        # These files define the runnable baseline and must come from the aggregate's validated
        # framework-baseline commit (or its original V0 commit when the campaign has none),
        # never from a later dispatcher HEAD.
        for name in (
            "kernel.py",
            "solution.json",
            "test_kernel.py",
            "config.json",
            ".gitignore",
            "CLAUDE.md",
        ):
            content = baseline_file(name, required=name == "kernel.py")
            if content is None and name == "test_kernel.py":
                aggregate_harness = self.workspace / name
                if aggregate_harness.is_file():
                    content = aggregate_harness.read_text(encoding="utf-8")
                else:
                    raise RuntimeError("aggregate workspace has no immutable test_kernel.py")
            if content is not None:
                (workspace / name).write_text(content, encoding="utf-8")
                tracked_files.add(name)

        source_label = (
            f"framework baseline v{baseline_version}" if baseline_version > 0 else "V0"
        )
        selected_ids = [workload_source.ids[index] for index in bucket.workload_indices]
        readme = baseline_file("README.md") or "# GPU kernel optimization\n"
        (workspace / "README.md").write_text(
            f"# Workload bucket: {bucket.name}\n\n"
            f"This workspace V0 is derived from aggregate {source_label} "
            f"`{aggregate_baseline_commit[:12]}`. "
            f"It contains workload ids {', '.join(selected_ids)} and was not re-evaluated.\n\n"
            + readme,
            encoding="utf-8",
        )
        tracked_files.add("README.md")
        aggregate_report = baseline_file("baseline_report.md") or ""
        (workspace / "baseline_report.md").write_text(
            f"# Derived bucket V0 baseline — {bucket.name}\n\n"
            f"This baseline reuses the aggregate {source_label} kernel, correctness result, and the "
            "per-workload timings selected below. No separate baseline agent or GPU run was used.\n\n"
            f"- Aggregate baseline: {source_label} at `{aggregate_baseline_commit}`\n"
            f"- Workload indices: {', '.join(map(str, bucket.workload_indices))}\n"
            f"- Workload ids: {', '.join(selected_ids)}\n\n"
            + aggregate_report,
            encoding="utf-8",
        )
        tracked_files.add("baseline_report.md")

        derived_memory = _bucket_baseline_memory(
            aggregate_memory,
            bucket,
            workload_source,
            aggregate_workspace=self.workspace,
            aggregate_baseline_commit=aggregate_baseline_commit,
            aggregate_baseline_version=baseline_version,
        )
        memory_path = workspace / "memory" / "v0.json"
        memory_path.write_text(
            json.dumps(derived_memory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tracked_files.add("memory/v0.json")

        campaign._link_runtime()
        tracked_files.update(name for name in (".gitignore", "CLAUDE.md") if (workspace / name).is_file())
        # A previously interrupted setup may have staged arbitrary files.  Keep
        # the worktree intact, but rebuild the index so this mechanical commit
        # contains only the deterministic V0 files listed above.
        if git_head(workspace):
            subprocess.run(
                ["git", "reset"],
                cwd=str(workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            subprocess.run(["git", "read-tree", "--empty"], cwd=str(workspace), check=True)
        subprocess.run(
            ["git", "add", "--", *sorted(tracked_files)],
            cwd=str(workspace),
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"V0: derive bucket baseline from aggregate {source_label}"],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        first_commit = git_head(workspace)
        derived_memory["git_commit_hash"] = first_commit
        memory_path.write_text(
            json.dumps(derived_memory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "memory/v0.json"], cwd=str(workspace), check=True)
        subprocess.run(
            ["git", "commit", "--amend", "--no-edit"],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(
            f"[workload-coordinator] derived {bucket.name} V0 from aggregate {source_label} "
            f"for {len(bucket.workload_indices)} workloads (no agent/GPU rerun)",
            flush=True,
        )

    def _assert_bucket_baseline_provenance(
        self, bucket: WorkloadBucket, campaign: Campaign, baseline_commit: str
    ) -> None:
        """Refuse to mix baselines: an already-seeded bucket must match the pinned baseline.

        Continuing silently would leave the aggregate advertising a framework baseline while some
        bucket sources still descend from the PyTorch V0 — invisible from the outside and exactly
        the provenance confusion the pin exists to prevent. Re-seeding is not an option either: it
        would destroy that bucket's optimization history.
        """
        if not baseline_commit:
            return
        derivation = (read_memory(campaign.workspace, 0) or {}).get("baseline_derivation") or {}
        seeded_from = _normalize_commit_hash(
            derivation.get("aggregate_baseline_commit") or derivation.get("aggregate_v0_commit")
        )
        if not seeded_from or seeded_from == baseline_commit:
            return
        message = (
            f"bucket {bucket.name} was seeded from aggregate commit {seeded_from[:12]} but the "
            f"aggregate now pins framework baseline {baseline_commit[:12]}; re-seeding would "
            "destroy that bucket's history. Continue with --framework-baseline never to keep the "
            "existing bucket baselines, or start a fresh --workspace."
        )
        if self.aggregate_campaign.optimization_mode == "production":
            raise RuntimeError(message)
        print(f"[workload-coordinator] WARNING: {message}", file=sys.stderr, flush=True)

    def _reconcile_resumed_bucket(
        self, bucket: WorkloadBucket, campaign: Campaign
    ) -> None:
        """Aggregate a committed win that survived a coordinator interruption."""
        current_blob = git_kernel_blob(campaign.workspace)
        if not current_blob:
            return
        state = self._load_state()
        bootstrap_pending = not self._bootstrap_accepted(state)
        previous = (state.get("buckets") or {}).get(bucket.name) or {}
        previous_reason = str(previous.get("reason") or "")
        retry_incomplete_validation = (
            previous.get("status") == "REJECTED"
            and "sandbox test output has no structured RESULT_JSON line" in previous_reason
        )
        if (
            not bootstrap_pending
            and
            previous.get("last_seen_kernel_blob") == current_blob
            and not retry_incomplete_validation
        ):
            return
        if bootstrap_pending and head_kernel_is_initial_baseline(campaign.workspace):
            return

        best_before: Optional[float] = None
        improvement: Optional[tuple[int, dict]] = None
        for version in range(0, latest_version(campaign.workspace) + 1):
            memory = read_memory(campaign.workspace, version)
            if not memory or not _status_is(
                (memory.get("quality_gate") or {}).get("result"), "PASS"
            ):
                continue
            latency = (memory.get("performance") or {}).get("latency_us")
            if not isinstance(latency, (int, float)):
                continue
            if version > 0 and (best_before is None or float(latency) < best_before):
                improvement = (version, memory)
            best_before = (
                float(latency) if best_before is None else min(best_before, float(latency))
            )
        if improvement is not None:
            version, memory = improvement
            print(
                f"[workload-coordinator] recovering unaggregated {bucket.name} v{version}",
                flush=True,
            )
            if bootstrap_pending:
                self.record_bucket_improvement(bucket, campaign, version, memory)
            else:
                self.aggregate_improvement(bucket, campaign, version, memory)

    def run(self) -> str:
        self._ensure_main_workspace()
        buckets = self.inspect_workloads()
        workload_source = _read_workload_source(self.op_dir)
        if sum(len(bucket.workload_indices) for bucket in buckets) != len(workload_source.entries):
            raise RuntimeError(
                f"validated workload buckets no longer match {workload_source.filename}"
            )
        self._bucket_campaigns = {
            bucket.name: self._make_bucket_campaign(bucket, workload_source)
            for bucket in buckets
        }
        for bucket in buckets:
            self._seed_bucket_baseline_from_aggregate(
                bucket, self._bucket_campaigns[bucket.name], workload_source
            )
        # Reconciliation prompts read bucket files from the worktree, while
        # aggregation provenance is the committed HEAD.  Clean interrupted
        # tracked edits before either reconciling or launching new iterations
        # so those two views cannot disagree.
        for bucket in buckets:
            campaign = self._bucket_campaigns[bucket.name]
            if latest_version(campaign.workspace) >= 0:
                preserve_interrupted_tracked_changes(
                    campaign.workspace, f"resume bucket {bucket.name}"
                )
        for bucket in buckets:
            self._reconcile_resumed_bucket(bucket, self._bucket_campaigns[bucket.name])
        print(
            f"[workload-coordinator] launching {len(buckets)} bucket optimization lines in parallel; "
            f"aggregate workspace={self.workspace}",
            flush=True,
        )
        reasons: dict[str, str] = {}
        failures: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(buckets), thread_name_prefix="workload-bucket"
        ) as executor:
            future_to_name = {
                executor.submit(self._bucket_campaigns[bucket.name].run): bucket.name
                for bucket in buckets
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    reasons[name] = future.result()
                except Exception as exc:
                    failures[name] = str(exc)
                    print(
                        f"[workload-coordinator] bucket {name} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        if failures:
            summary = "; ".join(f"{name}: {error}" for name, error in sorted(failures.items()))
            raise RuntimeError(f"bucket optimization failures: {summary}")
        summary = ", ".join(f"{name}={reason}" for name, reason in sorted(reasons.items()))
        print(
            f"\n[workload-coordinator] all bucket lines finished; aggregate HEAD="
            f"{git_head(self.workspace)[:8]} ({summary})",
            flush=True,
        )
        if self.aggregate_campaign.optimization_mode == "production":
            violations = production_kernel_violations(
                self.workspace, self.aggregate_campaign.framework
            )
            if violations:
                raise RuntimeError(
                    "no production-compliant aggregate kernel: " + "; ".join(violations)
                )
        if (self.workspace / "solution.json").exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "reference" / "sol_finalize.py"),
                        "--workspace",
                        str(self.workspace),
                    ],
                    check=False,
                )
            except OSError:
                pass
        return summary


# ── layer campaign (optional decomposition overlay) ─────────────────────────────

# Default expected achievable %SOL per op class — the ROI ceiling ONLY (never a stop gate).
# Overridden per-boundary by boundaries.json "ceiling"; see agents/gpu-kernel-decompose.md §5.
DEFAULT_CEILING = {
    "gemm": 0.85, "moe_gemm": 0.85,
    "attention": 0.72,
    "norm": 0.85, "elementwise": 0.85, "reduce": 0.85,
    "sort": 0.70, "scatter": 0.70,
}


def best_latency_us(workspace: Path) -> Optional[float]:
    """Best (min) recorded latency across all versions of a boundary workspace, or None."""
    lv = latest_version(workspace)
    best = None
    for n in range(0, lv + 1):
        mem = read_memory(workspace, n)
        if not mem:
            continue
        lat = (mem.get("performance") or {}).get("latency_us")
        if isinstance(lat, (int, float)):
            best = lat if best is None else min(best, float(lat))
    return best


def best_perf_by_shape(workspace: Path) -> Optional[dict]:
    """Per-shape best (min) latency_us across all versions, keyed by integer sid.

    Reads ``performance.latency_us_by_shape`` from each memory/v<n>.json. Returns
    None when no version records per-shape latencies (caller falls back to the
    scalar path). SOL and latency MUST be aggregated over the same shape set, so
    the sids here match those in the workspace roofline.json (see sol_ms_by_shape).
    """
    lv = latest_version(workspace)
    best: dict[str, float] = {}
    for n in range(0, lv + 1):
        mem = read_memory(workspace, n)
        if not mem:
            continue
        per = (mem.get("performance") or {}).get("latency_us_by_shape")
        if not isinstance(per, dict):
            continue
        for sid, lat in per.items():
            if isinstance(lat, (int, float)):
                best[sid] = min(best.get(sid, float("inf")), float(lat))
    return best or None


def shape_sol_ms(entry: dict) -> Optional[float]:
    """SOL (ms) for one roofline.json shape entry. A campaign targets ONE platform, so we do
    NOT match a platform key — just take the SOL value however it's stored:
      - flat:  entry["sol_time_ms"] = 0.123
      - nested: entry["SOL_time_ms"] = {<anything>: 0.123}  -> take the value (any key)
    This deliberately ignores the platform label so "B200" / "NVIDIA B200" / "NVIDIA B200
    (SM100)" all just work — there is no key to get wrong.
    """
    if not isinstance(entry, dict):
        return None
    flat = entry.get("sol_time_ms")
    if isinstance(flat, (int, float)):
        return float(flat)
    block = entry.get("SOL_time_ms")
    if isinstance(block, (int, float)):
        return float(block)
    if isinstance(block, dict):
        vals = [v for v in block.values() if isinstance(v, (int, float))]
        if vals:
            return float(vals[0])
    return None


def sol_ms_by_shape(workspace: Path) -> Optional[dict]:
    """Per-shape SOL (ms) for a boundary, read from the workspace's ``roofline.json``
    (``shapes[sid]`` -> SOL via shape_sol_ms). Keyed by the integer sid shared with
    shapes.json and the memory latency_us_by_shape. None if roofline.json is absent.
    """
    rp = workspace / "roofline.json"
    if not rp.exists():
        return None
    try:
        shapes = (json.loads(rp.read_text(encoding="utf-8")).get("shapes") or {})
    except (OSError, json.JSONDecodeError):
        return None
    out = {sid: shape_sol_ms(entry) for sid, entry in shapes.items()}
    out = {sid: v for sid, v in out.items() if v is not None}
    return out or None


def plateau_rounds(workspace: Path, eps: float = 0.05) -> int:
    """Trailing count of optimization versions (v1..) that did NOT reduce best latency by >= eps.

    LAYER MODE ONLY: drives per-boundary priority decay and the all-boundaries-plateaued short-
    circuit. Distinct from the single-op stall->convert counter (see kernel_won / read_stall),
    which keys off committed kernel.py changes, not latency deltas. A reverted / no-latency
    version counts as non-progress — a boundary is never dropped, its priority just shrinks.
    """
    lv = latest_version(workspace)
    if lv < 1:
        return 0
    best = None
    progressed: list[bool] = []
    for n in range(0, lv + 1):
        mem = read_memory(workspace, n)
        lat = (mem.get("performance") or {}).get("latency_us") if mem else None
        lat = float(lat) if isinstance(lat, (int, float)) else None
        if n == 0:
            best = lat
            continue
        made = bool(lat is not None and best is not None and lat < best * (1.0 - eps))
        if lat is not None and (best is None or lat < best):
            best = lat
        progressed.append(made)
    trailing = 0
    for made in reversed(progressed):
        if made:
            break
        trailing += 1
    return trailing


@dataclass
class LayerCampaign:
    """Whole-LLM-layer campaign: decompose -> N per-boundary workspaces -> shared-budget
    scheduler -> recombine. Each boundary is a standard single-operator campaign; this class
    only adds the decomposition, the live-ROI scheduler, and the recombine. The single-op path
    (Campaign) is untouched.
    """
    name: str
    layer_demo: str
    platform: str
    framework: str
    notes: str = "none"
    arch: str = ""
    work_dir: str = ""             # explicit working directory; "" = Path.cwd() (backward compat)
    workspace_suffix: str = ""     # internal auto-dispatch suffix, e.g. triton_h20
    roofline_py: str = ""
    op_dir: str = ""               # atrex-bench native op dir (shapes.json / roofline.json /
                                   # metadata.json / input.py / reference.py) — the full shape
                                   # set + SOL + anchor source. Passed in; never hardcoded.
    max_iters: int = 20            # SHARED across boundaries: sum of per-boundary versions
    token_budget: int = 0
    plateau_k: int = 3             # all boundaries plateau_rounds >= k -> layer short-circuit
    plateau_eps: float = 0.05
    iter_timeout: int = 5400
    setup_timeout: int = 7200
    decompose_timeout: int = 5400
    recombine_timeout: int = 5400
    sandbox_hardware: str = ""
    sandbox_profile: str = ""
    sandbox_url: str = ""
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    agent_cli: str = "claude"
    optimization_mode: str = "leaderboard"
    tokens_spent: int = field(default=0, init=False)

    @property
    def campaign_name(self) -> str:
        suffix = f"_{self.workspace_suffix}" if self.workspace_suffix else ""
        return f"{self.name}{suffix}"

    @property
    def layer_dir(self) -> Path:
        base = Path(self.work_dir) if self.work_dir else Path.cwd()
        return base / f"layer_{self.campaign_name}"

    def _boundary_ws(self, bname: str) -> Path:
        base = Path(self.work_dir) if self.work_dir else Path.cwd()
        boundary_name = f"{self.name}__{bname}"
        if self.workspace_suffix:
            boundary_name += f"_{self.workspace_suffix}"
        return base / f"kernel_opt_{boundary_name}"

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(f"[layer] {label}: exit={res.exit_status} timed_out={res.timed_out} "
              f"tokens={res.tokens} cum_tokens={self.tokens_spent}", flush=True)
        if res.exit_status != 0 or res.timed_out:
            print(f"[layer] stderr tail:\n{res.stderr_tail}", file=sys.stderr, flush=True)

    def budget_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_spent >= self.token_budget

    def _sandbox_directive(self) -> str:
        return sandbox_directive(
            self.sandbox_hardware, self.sandbox_profile, self.sandbox_url
        )

    def _mode_directive(self) -> str:
        return optimization_mode_directive(self.optimization_mode, self.framework)

    def _manifest_path(self) -> Path:
        return self.layer_dir / "boundaries.json"

    def _read_manifest(self) -> dict:
        return json.loads(self._manifest_path().read_text(encoding="utf-8"))

    # ── phase 1: decompose ────────────────────────────────────────────────────
    def decompose(self) -> None:
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        link_runtime(self.layer_dir)
        install_workspace_policy(self.layer_dir, self.optimization_mode, self.framework)
        prompt = _render(
            PROMPTS_DIR / "decompose.md",
            LAYER_DIR=str(self.layer_dir), LAYER_DEMO=self.layer_demo,
            PLATFORM=self.platform, ROOFLINE_PY=self.roofline_py,
            OP_DIR=self.op_dir, NOTES=self.notes,
            DECOMPOSE_DOC=str(REPO_ROOT / "agents" / "gpu-kernel-decompose.md"),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            MODE_POLICY=self._mode_directive(),
        )
        res = run_session(
            self.layer_dir, prompt, timeout=self.decompose_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
        )
        self._account(res, "decompose")
        if not self._manifest_path().exists():
            raise RuntimeError("decompose did not produce boundaries.json")

    # ── phase 2: per-boundary baseline workspaces ─────────────────────────────
    def setup_boundaries(self) -> list[dict]:
        manifest = self._read_manifest()
        boundaries = manifest.get("boundaries") or []
        if not boundaries:
            raise RuntimeError("boundaries.json lists no boundaries")
        for b in boundaries:
            ws = self._boundary_ws(b["name"])
            b["workspace"] = str(ws)
            if latest_version(ws) >= 0:
                install_workspace_policy(ws, self.optimization_mode, self.framework)
                continue  # already set up (resume)
            demo = self.layer_dir / b["kernel_demo"]
            boundary_name = f"{self.name}__{b['name']}"
            if self.workspace_suffix:
                boundary_name += f"_{self.workspace_suffix}"
            subprocess.run(["bash", str(WORKSPACE_INIT), boundary_name, str(demo)],
                           cwd=str(ws.parent), check=True)
            link_runtime(ws)
            install_workspace_policy(ws, self.optimization_mode, self.framework)
            self._write_shape_frame(ws, b)
            prompt = _render(
                PROMPTS_DIR / "setup.md",
                WORKSPACE=str(ws), PLATFORM=self.platform, FRAMEWORK=self.framework,
                KERNEL_DEMO=str(demo), NOTES=self.notes,
                AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
                BASELINE_DRIVER=_baseline_driver_directive(self.agent_cli),
                HARDWARE=hardware_directive(self.platform, self.arch),
                SANDBOX=self._sandbox_directive(),
                EVALUATOR=(
                    "## Evaluation route: derived layer boundary\n\n"
                    "This boundary is not a complete Atrex-Bench operator directory. Build its "
                    "V0 full-shape harness once, commit it, and keep it immutable afterwards."
                ),
                MODE_POLICY=self._mode_directive(),
            )
            res = run_session(
                ws, prompt, timeout=self.setup_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
                reasoning_effort="high",
            )
            self._account(res, f"baseline {b['name']}")
            if read_memory(ws, 0) is None:
                raise RuntimeError(f"baseline failed for boundary {b['name']} (no memory/v0.json)")
        # persist workspace paths back into the manifest for the recombine session
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return boundaries

    def _write_shape_frame(self, ws: Path, b: dict) -> None:
        """Materialize the boundary's atrex-bench-format op files into its workspace so the
        optimization session benches the SAME full shape set every round, keyed by integer sid:
          - shapes.json  : {"0": {"init_kwargs": null, "input_kwargs": {...}}, ...}  (layer-shared)
          - roofline.json: {"shapes": {"0": {"semantic_W_flops": {..}, "SOL_time_ms": {"<hw>": ms}}}}
        sid is the atrex-bench integer id ("0","1",...) shared across shapes.json, roofline.json,
        and the memory latency_us_by_shape — NOT a uuid hash and NOT a "BxS" string. This is the
        ground-truth bench set (immutable per campaign); the session must NOT bench a single
        hand-picked "representative" shape.
        """
        manifest = self._read_manifest()
        shapes = manifest.get("shapes")            # atrex-bench shapes.json body: {sid: {...}}
        roofline = b.get("roofline")               # {"shapes": {sid: {...SOL_time_ms...}}}
        if isinstance(shapes, dict) and shapes:
            (ws / "shapes.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
        if isinstance(roofline, dict) and roofline:
            (ws / "roofline.json").write_text(json.dumps(roofline, indent=2), encoding="utf-8")

    # ── scheduler helpers ─────────────────────────────────────────────────────
    def _priority(self, b: dict) -> float:
        """Live ROI = reachable savings toward the *single* layer SOL-score, decayed by stall.

        The whole layer is scored ONCE (official SOL-ExecBench, recombined kernel); the
        boundaries are not scored separately. Layer latency is additive over boundaries
        (T_layer = Σ_b T_b), so a boundary's reachable savings is its gradient on the one
        layer score. The official per-shape score is
            S[s] = 1 / (1 + (Tk_layer[s]-SOL_layer[s]) / (Tb_layer[s]-SOL_layer[s]))
        whose sensitivity to cutting a boundary at shape s is the per-shape weight
            w[s] = 1 / (Tb_layer[s] - SOL_layer[s])        (boundary-independent; from setup)
        so the score-consistent priority is

            priority(b) = mean_over_shapes( w[s] * max(0, Tk[b,s] - SOL[b,s]) ) * 0.5**plateau_rounds

        `w[s]` (manifest `shape_weights`) is measured once at setup by benching the optimized-
        PyTorch anchor (`solution.py`) — see setup_anchor_weights(). Without it, w[s]=1 (raw
        ms-gap ROI). The `0.5**plateau_rounds` decay is essential: when a boundary stops improving
        for a few rounds it is deprioritized so the scheduler moves on to boundaries that can
        still gain (no boundary is ever dropped — its priority just decays). SOL and latency are
        BOTH aggregated over the full shape set — never one "representative" shape (attention
        cost ∝ B·S², so a shape mismatch blows up the SOL and zeroes the boundary; that bug
        starved gqa_attention). Falls back to the scalar path when per-shape data is absent;
        a fresh boundary with no latency gets top priority.
        """
        ws = self._boundary_ws(b["name"])
        decay = 0.5 ** plateau_rounds(ws, self.plateau_eps)

        # ── per-shape path (preferred): score-weighted mean reachable ms over the shape set ──
        # SOL from the workspace roofline.json (platform-agnostic); w[s] from manifest shape_weights.
        sol_by_shape = sol_ms_by_shape(ws)
        lat_by_shape = best_perf_by_shape(ws)
        if sol_by_shape and lat_by_shape:
            common = [sid for sid in sol_by_shape if sid in lat_by_shape]
            if common:
                weights = (self._read_manifest().get("shape_weights") or {})
                gap = sum((float(weights.get(sid, 1.0))) * max(0.0, lat_by_shape[sid] / 1000.0 - sol_by_shape[sid])
                          for sid in common) / len(common)
                return gap * decay

        # ── legacy scalar fallback ──
        lat_us = best_latency_us(ws)
        if lat_us is None:
            return 1e12 * decay
        sol_ms = float(b["sol_time_ms"]) if isinstance(b.get("sol_time_ms"), (int, float)) else 0.0
        return max(0.0, lat_us / 1000.0 - sol_ms) * decay

    def _total_versions(self, boundaries: list[dict]) -> int:
        # optimization iterations spent = sum of per-boundary latest versions (v0 = baseline, not counted)
        return sum(max(0, latest_version(self._boundary_ws(b["name"]))) for b in boundaries)

    def _all_plateaued(self, boundaries: list[dict]) -> bool:
        return all(plateau_rounds(self._boundary_ws(b["name"]), self.plateau_eps) >= self.plateau_k
                   for b in boundaries)

    # ── phase 3: shared-budget scheduler ──────────────────────────────────────
    def schedule(self, boundaries: list[dict]) -> Optional[str]:
        while True:
            for b in boundaries:
                ws = self._boundary_ws(b["name"])
                mask_half_memory(ws, latest_version(ws))
            spent = self._total_versions(boundaries)
            if spent >= self.max_iters:
                return "budget: max-iters (Σ versions)"
            if self.budget_exhausted():
                return "budget: token-budget"
            if self._all_plateaued(boundaries):
                return "all boundaries plateaued"

            ranked = sorted(boundaries, key=self._priority, reverse=True)
            target = ranked[0]
            if self._priority(target) <= 0.0:
                return "all boundaries at/above ceiling"

            ws = self._boundary_ws(target["name"])
            n = latest_version(ws) + 1
            print(f"[layer] round {spent + 1}/{self.max_iters} -> {target['name']} v{n} "
                  f"(priority={self._priority(target):.4g})", flush=True)
            prompt = _render(PROMPTS_DIR / "iteration.md",
                             WORKSPACE=str(ws), N=n, PREV=n - 1,
                             PLATFORM=self.platform, NOTES=self.notes,
                             AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
                             PLAN_GENERATOR=_plan_generator_directive(self.agent_cli, n),
                             HARDWARE=hardware_directive(self.platform, self.arch),
                             SANDBOX=self._sandbox_directive(),
                             EVALUATOR=(
                                 "## Evaluation route: derived layer boundary\n\n"
                                 "Keep using this boundary's committed full-shape harness."
                             ),
                             MODE_POLICY=self._mode_directive())
            pre_head = git_head(ws)
            res = run_session(
                ws, prompt, timeout=self.iter_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
            )
            self._account(res, f"{target['name']} v{n}")

            if (
                self.optimization_mode == "production"
                and kernel_won(ws, pre_head)
            ):
                violations = production_kernel_violations(ws, self.framework)
                if violations:
                    reject_production_commit(ws, n, pre_head, violations)
                    print(
                        f"[layer] production policy rejected {target['name']} v{n}: "
                        + "; ".join(violations),
                        file=sys.stderr,
                        flush=True,
                    )

            # Guard: if the session exited without producing v<n>.json, write a
            # minimal failed-iteration record so latest_version() advances.  Without
            # this the scheduler would keep targeting the same v<N> forever (the
            # spent count never increments and plateau_rounds never sees it).
            if read_memory(ws, n) is None:
                mem_dir = ws / "memory"
                mem_dir.mkdir(parents=True, exist_ok=True)
                failed = {
                    "version": f"v{n}",
                    "correctness": {"status": "FAIL", "details": "session did not produce v{n}.json"},
                    "quality_gate": {"result": "FAIL"},
                    "git_commit_hash": None,
                    "optimization_category": "failed-iteration",
                    "notes": "orchestrator: session exited without output; recorded to advance budget",
                }
                (mem_dir / f"v{n}.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
                print(f"[layer] WARNING: {target['name']} v{n} session produced no memory — "
                      f"wrote failed record to advance budget", flush=True)
            mask_half_memory(ws, n)

    # ── phase 2b: SOL-score weights (only if a real production baseline exists) ────
    def setup_anchor_weights(self) -> None:
        """Write per-shape SOL-score weights into the manifest from the op's production
        baseline (metadata.production_performance). If that baseline is absent, no weights are
        written and the scheduler uses the unweighted raw ms-gap priority. Pure JSON transform
        (no bench); always re-run so stale weights are recomputed/cleared. Non-fatal.
        """
        if not self.op_dir:
            print("[layer] no --op-dir; priority uses raw ms-gap (unweighted)", flush=True)
            return
        cmd = [sys.executable, str(Path(__file__).parent / "anchor_bench.py"),
               "--op-dir", str(self.op_dir), "--manifest", str(self._manifest_path())]
        print(f"[layer] SOL-score weights (from production baseline, if any): {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("[layer] WARNING: anchor step failed — priority falls back to raw ms-gap (unweighted)",
                  file=sys.stderr, flush=True)

    # ── phase 4: recombine ────────────────────────────────────────────────────
    def recombine(self) -> None:
        link_runtime(self.layer_dir)
        install_workspace_policy(self.layer_dir, self.optimization_mode, self.framework)
        prompt = _render(PROMPTS_DIR / "recombine.md",
                         LAYER_DIR=str(self.layer_dir),
                         HARDWARE=hardware_directive(self.platform, self.arch),
                         SANDBOX=self._sandbox_directive(),
                         MODE_POLICY=self._mode_directive())
        res = run_session(
            self.layer_dir, prompt, timeout=self.recombine_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
        )
        self._account(res, "recombine")
        if self.optimization_mode == "production":
            violations = production_kernel_violations(self.layer_dir, self.framework)
            if violations:
                raise RuntimeError(
                    "recombined kernel violates production policy: " + "; ".join(violations)
                )

    def run(self) -> str:
        if not self._manifest_path().exists():
            self.decompose()
        boundaries = self.setup_boundaries()
        self.setup_anchor_weights()
        reason = self.schedule(boundaries)
        self.recombine()
        print(f"\n[layer] STOP — {reason}", flush=True)
        for b in boundaries:
            ws = self._boundary_ws(b["name"])
            print(f"[layer]   {b['name']}: v{latest_version(ws)} best_latency_us={best_latency_us(ws)}", flush=True)
        return reason or "done"


def _resolve_op(op_dir: str) -> dict:
    """Derive everything op-specific from the atrex-bench native op dir, so the CLI needs only
    --op-dir (+ the non-deducible --platform). Returns name / reference / roofline_py.
    """
    d = Path(op_dir).resolve()
    if not d.is_dir():
        raise SystemExit(f"--op-dir not found: {d}")
    ref = d / "reference.py"
    if not ref.is_file():
        raise SystemExit(f"--op-dir has no reference.py: {d}")
    roofline_py = ""  # atrex-bench root is an ancestor of the op dir; find scripts/roofline.py
    atrex_bench_root = ""
    for p in (d, *d.parents):
        cand = p / "scripts" / "roofline.py"
        if cand.is_file():
            roofline_py = str(cand)
            break
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
        "roofline_py": roofline_py,
        "op_dir": str(d),
        "atrex_bench_root": atrex_bench_root,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Clean-session orchestrator for atrex-kernel-agent.")
    ap.add_argument("--op-dir", required=True,
                    help="The operator dir (SOL definition.json / workload.jsonl / reference.py, or "
                         "atrex-bench shapes.json / roofline.json / metadata.json / input.py / reference.py). "
                         "EVERYTHING op-specific is read from here — the workspace "
                         "name (dir basename), the kernel/layer to optimize (reference.py), the full shape set, "
                         "per-shape SOL, and the priority anchor (metadata.production_performance). Never hardcoded.")
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
        help="Coding CLI used for clean optimization sessions: claude, qodercli, or codex "
             "(default: claude).",
    )
    ap.add_argument(
        "--optimization-mode",
        choices=OPTIMIZATION_MODE_CHOICES,
        default="leaderboard",
        help="leaderboard preserves the permissive current CLAUDE.md flow; production forbids "
             "third-party kernel/operator dependencies and mechanically enforces each campaign's framework.",
    )
    ap.add_argument(
        "--framework", default="",
        help="Target DSL, e.g. Triton / CuteDSL / Cuda / FlyDSL. When omitted, launch all "
             "frameworks supported by the detected hardware in parallel: NVIDIA uses "
             "Triton/CuteDSL/Cuda, AMD uses Triton/FlyDSL, and unknown hardware uses Triton. "
             "Each auto-dispatched production child is bound to its assigned framework.",
    )
    ap.add_argument("--layer", action="store_true",
                    help="Decomposition overlay: treat the op's reference as a composite of more than one fused "
                         "op (a whole LLM layer, or e.g. rope+attention / attention+moe), carve it into "
                         "fused-operator boundaries (per agents/gpu-kernel-decompose.md), optimize each in its "
                         "own workspace under one shared --max-iters budget, then recombine. Default off "
                         "(single-op SOL path uses workload bucketing instead).")
    ap.add_argument("--notes", default="none", help="Extra constraints / known bottlenecks.")
    ap.add_argument("--max-iters", type=int, default=20, help="Hard cap on optimization iterations.")
    ap.add_argument(
        "--max-workload-buckets",
        type=int,
        default=8,
        help="Maximum number of inspector-created workload buckets (default: 8).",
    )
    ap.add_argument(
        "--aggregate-min-improvement-pct",
        type=float,
        default=0.0,
        help="Minimum full-workload geomean improvement required to accept an aggregate candidate "
             "(percent; default: any strict improvement).",
    )
    ap.add_argument(
        "--no-workload-bucketing",
        action="store_true",
        help="Disable the default workload/shape inspector and bucket coordinator and run one legacy campaign.",
    )
    ap.add_argument("--token-budget", type=int, default=0,
                    help="Hard token cap across all sessions (0 = no cap; max-iters still bounds it).")
    ap.add_argument("--target-util", type=float, default=90.0,
                    help="Peak-utilization %% short-circuit (default stop condition).")
    ap.add_argument("--iter-timeout", type=int, default=5400, help="Per-iteration hang backstop (s).")
    ap.add_argument("--setup-timeout", type=int, default=7200, help="Baseline session timeout (s).")
    ap.add_argument("--salvage-timeout", type=int, default=SALVAGE_TIMEOUT_S,
                    help="Budget (s) for the post-mortem session that records an iteration killed by "
                         "timeout or API failure (0 = write the mechanical record only).")
    ap.add_argument("--framework-baseline", choices=FRAMEWORK_BASELINE_MODES, default="auto",
                    help="Run one dedicated session between V0 setup and workload bucketing that "
                         "replaces the V0 PyTorch wrapper with the first self-contained framework "
                         "kernel, recorded as v1 (so optimization rounds start at v2) and inherited "
                         "by every workload bucket. auto = production mode only; always = "
                         "leaderboard too; never = legacy flow (buckets derive from the V0 kernel).")
    ap.add_argument("--framework-baseline-timeout", type=int, default=FRAMEWORK_BASELINE_TIMEOUT_S,
                    help="Framework baseline session timeout (s).")
    ap.add_argument("--max-stall", type=int, default=0,
                    help="Optional: stop after N consecutive no-commit iterations (0 = disabled).")
    ap.add_argument("--convert-after", type=int, default=DEFAULT_CONVERT_AFTER,
                    help="Triton only: after N consecutive stalled iterations, require conversion "
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
    if args.max_workload_buckets < 1:
        ap.error("--max-workload-buckets must be at least 1")
    if args.convert_after < 0:
        ap.error("--convert-after must be non-negative")
    if args.salvage_timeout < 0:
        ap.error("--salvage-timeout must be non-negative (0 disables the post-mortem session)")
    if args.framework_baseline_timeout <= 0:
        ap.error("--framework-baseline-timeout must be positive")
    if not 0.0 <= args.aggregate_min_improvement_pct < 100.0:
        ap.error("--aggregate-min-improvement-pct must be in [0, 100)")
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
            _codex_settings_args(codex_settings)
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

    if args.layer:
        layer = LayerCampaign(
            name=op["name"], layer_demo=op["reference"], platform=args.platform,
            framework=args.framework, notes=args.notes, arch=arch,
            sandbox_hardware=sandbox_hardware, sandbox_profile=args.sandbox_profile,
            sandbox_url=args.sandbox_url,
            sandbox_timeout=args.sandbox_timeout,
            agent_cli=args.agent_cli,
            optimization_mode=args.optimization_mode,
            work_dir=args.workspace,
            workspace_suffix=workspace_suffix,
            roofline_py=op["roofline_py"], op_dir=op["op_dir"],
            max_iters=args.max_iters, token_budget=args.token_budget,
            iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout,
        )
        layer.run()
        return 0

    campaign = Campaign(
        name=op["name"], kernel_demo=op["reference"], platform=args.platform,
        framework=args.framework, notes=args.notes, arch=arch,
        sandbox_hardware=sandbox_hardware, sandbox_profile=args.sandbox_profile,
        sandbox_url=args.sandbox_url,
        sandbox_timeout=args.sandbox_timeout,
        atrex_bench_root=op.get("atrex_bench_root", ""),
        agent_cli=args.agent_cli,
        optimization_mode=args.optimization_mode,
        work_dir=args.workspace,
        workspace_suffix=workspace_suffix,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
        salvage_timeout=args.salvage_timeout,
        framework_baseline=args.framework_baseline,
        framework_baseline_timeout=args.framework_baseline_timeout,
        convert_after=args.convert_after,
    )
    if is_bucketable_op(Path(op["op_dir"])) and not args.no_workload_bucketing:
        coordinator = WorkloadBucketCoordinator(
            aggregate_campaign=campaign,
            op_dir=Path(op["op_dir"]).resolve(),
            max_buckets=args.max_workload_buckets,
            aggregate_min_improvement=args.aggregate_min_improvement_pct / 100.0,
        )
        coordinator.run()
    else:
        if not args.no_workload_bucketing:
            print(
                "[orchestrator] workload bucketing requires workload.jsonl or shapes.json; "
                "running the legacy single campaign",
                flush=True,
            )
        campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
