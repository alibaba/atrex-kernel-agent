#!/usr/bin/env python3
"""Clean-session orchestrator for atrex-kernel-agent.

Owns the OUTER optimization loop so termination no longer depends on the model's
in-session judgment (the old Stage-6 "is README's Stop Conditions met?" self-call).

Each iteration is a **fresh coding-agent session** (`claude` by default, or `qodercli` via
`--agent-cli`) over the *same* git workspace. State crosses the session boundary only through
disk — exactly the artifacts atrex already maintains: `memory/v<N>.json`, `plans/`, `profiles/`,
and git. HEAD is always the best kernel (a regressing iteration reverts and is never committed).

Termination policy
------------------
- Outer loop (this file):  HARD budget break = max iterations OR token budget,
  plus a mechanical target short-circuit (peak utilization >= --target-util on a
  committed, correctness-PASS iteration). No plateau ladder, no convergence judge.
- Inner loop (one session): exactly one profile->edit->validate->bench cycle, bounded
  by a hang-backstop timeout (SIGKILL of the process group). See prompts/iteration.md.

Per-iteration reasoning stays in markdown (the gpu-kernel-* skills + prompts/*.md);
this file only does mechanism: spawn, time-bound, token-account, read state, decide stop.

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
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from .optimization_policy import (
        OPTIMIZATION_MODE_CHOICES,
        install_workspace_policy,
        optimization_mode_directive,
        production_kernel_violations,
        reject_production_commit,
    )
except ImportError:  # direct script execution: python orchestrator/optimize.py
    from optimization_policy import (  # type: ignore[no-redef]
        OPTIMIZATION_MODE_CHOICES,
        install_workspace_policy,
        optimization_mode_directive,
        production_kernel_violations,
        reject_production_commit,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = REPO_ROOT / "reference" / "sol_seed.py"
SANDBOX_TOOL = REPO_ROOT / "tools" / "sandbox.py"
HUMANIZE_DIR = REPO_ROOT / "3rdparty" / "humanize"
CONVERT_PERF_TOL = 0.05   # triton->gluon is a direct translation: gluon must be within +5% of triton
CONVERT_MIN_TOKENS = 200_000  # a convert session below this (and no gluon) barely ran -> "bailed"
                              # (set below a genuine-but-incomplete attempt, e.g. ~336K; the launch-and-
                              # exit bail we saw was ~85K — only that class should trip the give-up)
CONVERT_MAX_BAILS = 2         # consecutive bails -> disable escalation, continue triton-only
MEMORY_MASK_INTERVAL = 100    # periodically drop half of active optimization history
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
AGENT_CLI_CHOICES = ("claude", "qodercli")
NVIDIA_FRAMEWORKS = ("Triton", "CuteDSL", "Cuda")
AMD_FRAMEWORKS = ("Triton", "FlyDSL")
DEFAULT_FRAMEWORKS = ("Triton",)


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


def framework_workspace_suffix(framework: str, platform: str) -> str:
    """Flat suffix for an auto-dispatched framework/hardware campaign."""
    return f"{_workspace_slug(framework)}_{_workspace_slug(platform)}"


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
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_dispatch(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_dispatch)
    try:
        for framework in frameworks:
            workspace_suffix = framework_workspace_suffix(framework, platform)
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
        signal.signal(signal.SIGTERM, previous_sigterm)

    if failed:
        summary = ", ".join(f"{name}={code}" for name, code in failed)
        print(f"[orchestrator] framework campaign failures: {summary}", file=sys.stderr, flush=True)
        return 1
    return 0


def is_sol_op(op_dir: Path) -> bool:
    """A SOL-ExecBench op dir carries definition.json + workload.jsonl next to reference.py."""
    return (op_dir / "definition.json").is_file() and (op_dir / "workload.jsonl").is_file()


def _is_triton_family(framework: str) -> bool:
    """Triton and Gluon are one framework family — Gluon is the lower-level escalation of Triton."""
    return framework.strip().lower() in ("triton", "gluon", "triton/gluon")


def kernel_is_gluon(workspace: Path) -> bool:
    """True once kernel.py has been converted to Gluon (import present)."""
    k = workspace / "kernel.py"
    return k.exists() and "gluon" in k.read_text(encoding="utf-8", errors="ignore")


def head_kernel_is_gluon(workspace: Path) -> bool:
    """True when the COMMITTED HEAD kernel.py is Gluon. Authoritative accept signal for a convert
    session — more reliable than memory's git_commit_hash, which a session may leave unset even after
    committing."""
    try:
        out = subprocess.run(["git", "show", "HEAD:kernel.py"], cwd=str(workspace),
                             capture_output=True, text=True)
    except OSError:
        return False
    return out.returncode == 0 and "gluon" in out.stdout


# ── thin IO ─────────────────────────────────────────────────────────────────


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str


def _render(template_path: Path, **kw: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    mode_policy = kw.pop("MODE_POLICY", "")
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


def _tokens_from_stream(stdout: str) -> int:
    """Sum core token usage from a `--output-format stream-json` stdout.

    Prefer the terminal `{"type":"result", ...,"usage":{...}}` event (cumulative);
    fall back to summing per-message usage. Counts input+output (+cache) tokens.
    Never raises — budget accounting degrades to max-iters if the stream is unparseable.
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
        if evt.get("type") == "result":
            usage_total = _usage_tokens(evt.get("usage"))
            model_total = _model_usage_tokens(evt.get("modelUsage"))
            result_total = usage_total or model_total
        usage = evt.get("usage")
        if usage is None and isinstance(evt.get("message"), dict):
            usage = evt["message"].get("usage")
        if isinstance(usage, dict):
            summed += _usage_tokens(usage)
    return result_total if result_total else summed


def _run_bounded(cmd: list[str], cwd: Path, timeout: int, env: Optional[dict] = None) -> tuple[str, str, int, bool]:
    """Run cmd in its own process group; SIGKILL the whole tree on timeout."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group -> killpg reaps grandchildren
        env=env,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    except BaseException:
        # The coding CLI owns a separate process group. If an explicit or
        # auto-dispatched optimizer is interrupted, reap that entire group so
        # Qoder/Claude and their tool subprocesses cannot become orphaned.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        raise
    return stdout or "", stderr or "", proc.returncode, timed_out


def _session_env(agent_cli: str) -> dict:
    """Build the environment for a nested coding-agent session.

    Claude-specific auth normalization is deliberately not applied to Qoder CLI. When a Bearer
    auth token is available for Claude (ANTHROPIC_AUTH_TOKEN — e.g. an Anthropic-compatible
    gateway), drop ANTHROPIC_API_KEY so Claude authenticates via the token instead of sending
    x-api-key, which such gateways reject with 401.
    """
    env = os.environ.copy()
    if agent_cli == "claude" and env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def _session_command(agent_cli: str, prompt: str, session_id: str) -> list[str]:
    """Return a non-interactive, fresh-session command for the selected coding CLI."""
    if agent_cli == "claude":
        cmd = [
            "claude", "--print", "--verbose",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--session-id", session_id,
            "--effort", "max",
        ]
        provider_settings = "ATREX_CLAUDE_SESSION_SETTINGS"
    elif agent_cli == "qodercli":
        cmd = [
            "qodercli", "--print",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--session-id", session_id,
            "--no-session-persistence",
            "--reasoning-effort", "max",
        ]
        provider_settings = "ATREX_QODER_SESSION_SETTINGS"
    else:
        raise ValueError(f"unsupported agent CLI: {agent_cli!r}")

    # Provider-specific settings win. ATREX_SESSION_SETTINGS remains the backward-compatible
    # generic fallback and is interpreted by whichever CLI was selected.
    session_settings = os.environ.get(provider_settings) or os.environ.get("ATREX_SESSION_SETTINGS")
    if session_settings:
        cmd += ["--settings", session_settings]
    # Both supported CLIs accept Claude-compatible local plugins. humanize is loaded from the
    # source submodule rather than installed into the user's global runtime.
    if (HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md").exists():
        cmd += ["--plugin-dir", str(HUMANIZE_DIR)]
    cmd.append(prompt)
    return cmd


def _agent_auth_hint(agent_cli: str) -> str:
    if agent_cli == "qodercli":
        return "run `qodercli status` and `qodercli --print \"test\"` to diagnose"
    return "run `claude auth status` and `claude --print \"test\"` to diagnose"


def ensure_submodules() -> None:
    """Initialize all git submodules required by the optimization pipeline.

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
    if not to_init:
        return
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


def run_session(
    workspace: Path,
    prompt: str,
    timeout: int,
    agent_cli: str = "claude",
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = 600,
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    session_id = str(uuid.uuid4())
    cmd = _session_command(agent_cli, prompt, session_id)
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
        "- Run every correctness or performance test through the gateway sandbox. Always pass `--no-memory`; "
        "read the emitted `[test_kernel] RESULT_JSON=...` line, then update `memory/v<N>.json` locally.\n"
        "  ```bash\n"
        "  python tools/sandbox.py --no-sync -- python test_kernel.py --version v<N> --no-memory\n"
        "  python tools/sandbox.py --no-sync -- python test_kernel.py --version v<N> --multi-seed 5 --no-memory\n"
        "  ```\n"
        "- Run NVIDIA/AMD profiling in the sandbox as one self-contained command; `profiles/` analysis "
        "artifacts are synchronized back automatically:\n"
        "  ```bash\n"
        "  python tools/sandbox.py --sync profiles/v<N> -- bash tools/profile_nvidia.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N> --source\n"
        "  python tools/sandbox.py --sync profiles/v<N> -- bash tools/profile_kernel.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N>\n"
        "  ```\n"
        "- Never run `test_kernel.py`, GPU timers, `ncu`, `rocprofv3`, or the profile wrappers outside "
        "the gateway interface. Never upload or create optimizer `memory/` as worker state; memory updates, "
        "plans, edits, and git operations stay local.\n"
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
) -> subprocess.CompletedProcess[str]:
    """Run one command through tools/sandbox.py and capture its user-visible output."""
    cmd = [
        sys.executable, str(SANDBOX_TOOL),
        "--hardware", hardware,
        "--workspace", str(workspace),
        "--timeout", str(timeout),
    ]
    if url:
        cmd += ["--url", url]
    elif profile:
        cmd += ["--gateway-profile", profile]
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
        timeout=timeout + 240,
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



def link_runtime(workspace: Path) -> None:
    """Make the skill's `tools/`, `reference/`, `skills/`, `reference-projects/`, `gpu-wiki/` resolvable from cwd=workspace.

    The gpu-kernel-* skills reference these by relative path; sessions run with cwd=workspace,
    so symlink them in (absolute targets, so the workspace can live anywhere). Idempotent.

    Also installs the same skills and agent definitions into ``.claude/`` and ``.qoder/`` so
    either supported coding CLI can discover them.

    humanize is loaded via ``--plugin-dir`` (see ``run_session``); it is NOT installed as a
    workspace skill.
    """
    for sub in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
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
    gi = workspace / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    add = ""
    if "/tools" not in existing:
        add += "\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n"
    if "/.claude" not in existing:
        add += "/.claude\n"
    if "/.qoder" not in existing:
        add += "/.qoder\n"
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
    max_stall: int = 0             # 0 = disabled (budget-only); >0 = stop after N no-commit iters
    convert_after: int = 5         # triton-only: after N stalled iters, run ONE triton->gluon convert session (0=off)
    sandbox_hardware: str = ""     # agate scheduler token, e.g. REMOTE_GPU (may differ from platform)
    sandbox_profile: str = ""      # pre/prod; empty preserves normal agate URL resolution
    sandbox_url: str = ""          # explicit endpoint, e.g. http://127.0.0.1:8000
    sandbox_timeout: int = 600      # agate dev hard limit
    agent_cli: str = "claude"       # clean-session coding backend: claude or qodercli
    optimization_mode: str = "leaderboard"  # permissive contest flow or strict production gate
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

    def _link_runtime(self) -> None:
        link_runtime(self.workspace)
        install_workspace_policy(self.workspace, self.optimization_mode, self.framework)

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
        prompt = _render(
            PROMPTS_DIR / "setup.md",
            WORKSPACE=str(self.workspace), PLATFORM=self.platform,
            FRAMEWORK=self.framework, KERNEL_DEMO=self.kernel_demo,
            NOTES=self.notes,
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            MODE_POLICY=self._mode_directive(),
        )
        res = run_session(
            self.workspace, prompt, timeout=self.setup_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
        )
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0) — "
                "this is likely an API key / authentication issue. "
                f"{_agent_auth_hint(self.agent_cli)}."
            )
        if read_memory(self.workspace, 0) is None:
            raise RuntimeError("setup did not produce memory/v0.json (baseline failed)")

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

    def run(self) -> str:
        if latest_version(self.workspace) < 0:
            self.setup_baseline()
        else:
            print(f"[orchestrator] resuming: latest = v{latest_version(self.workspace)}", flush=True)
            self._link_runtime()  # ensure runtime symlinks exist for iteration sessions

        if self.optimization_mode == "production" and latest_version(self.workspace) > 0:
            violations = production_kernel_violations(self.workspace, self.framework)
            if violations:
                raise RuntimeError(
                    "cannot resume a non-compliant production HEAD: " + "; ".join(violations)
                )

        stall = read_stall(self.workspace)   # persisted live counter (single source of truth)
        if stall is None:
            stall = reconstruct_stall(self.workspace)  # bootstrap from git when no state file yet
            write_stall(self.workspace, stall)
        if stall > 0:
            print(f"[orchestrator] stall counter restored: {stall} rounds without progress", flush=True)
        infra_fails = 0  # consecutive sessions that crashed with 0 tokens (auth/infra issue)
        convert_bails = 0  # consecutive convert sessions that bailed early (low tokens, no gluon)
        n = latest_version(self.workspace)  # 0 after baseline
        mask_half_memory(self.workspace, n)  # also covers resuming an unmasked v100/v200/...
        while True:
            if n >= self.max_iters:
                return self._finish("budget: max-iters")
            if self.budget_exhausted():
                return self._finish("budget: token-budget")

            n += 1
            # Triton→Gluon escalation: after `convert_after` stalled triton iterations, spend ONE
            # session purely converting the kernel to Gluon (no optimization). Gluon is lower-level,
            # so the following sessions can go further. Re-fires after each `convert_after` stalled
            # rounds — the cooldown resets on every convert issued, win or lose (see below).
            do_convert = (
                self.optimization_mode == "leaderboard"
                and self.convert_after > 0
                and _is_triton_family(self.framework)
                and not kernel_is_gluon(self.workspace)
                and stall >= self.convert_after
            )
            if do_convert:
                print(f"[orchestrator] triton stalled {stall} iters -> triton->gluon convert session v{n}", flush=True)
                prompt = _render(PROMPTS_DIR / "convert.md",
                                 WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                                 PLATFORM=self.platform, ARCH=self.arch or "the runtime GPU arch",
                                 NOTES=self.notes,
                                 HARDWARE=hardware_directive(self.platform, self.arch),
                                 SANDBOX=self._sandbox_directive(),
                                 MODE_POLICY=self._mode_directive())
            else:
                prompt = _render(PROMPTS_DIR / "iteration.md",
                                 WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                                 PLATFORM=self.platform, NOTES=self.notes,
                                 HARDWARE=hardware_directive(self.platform, self.arch),
                                 SANDBOX=self._sandbox_directive(),
                                 MODE_POLICY=self._mode_directive())
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
            if res.exit_status != 0 and not res.timed_out:
                infra_fails += 1
                if infra_fails >= 15:
                    return self._finish(
                        f"infra: {infra_fails} consecutive sessions crashed (exit={res.exit_status}) "
                        f"(likely API key / auth issue — {_agent_auth_hint(self.agent_cli)})"
                    )
                # Back off before retrying to avoid hammering a rate-limited API
                import time as _time
                _backoff = min(30 * infra_fails, 180)
                print(f"[orchestrator] infra fail #{infra_fails} (exit={res.exit_status}), backing off {_backoff}s", flush=True)
                _time.sleep(_backoff)
            elif res.exit_status != 0 and res.timed_out:
                infra_fails += 1
                if infra_fails >= 15:
                    return self._finish(f"infra: {infra_fails} consecutive timeouts")
                import time as _time
                _backoff = min(60 * infra_fails, 300)
                print(f"[orchestrator] timeout #{infra_fails}, backing off {_backoff}s", flush=True)
                _time.sleep(_backoff)
            else:
                infra_fails = 0

            mem = read_memory(self.workspace, n)
            won = kernel_won(self.workspace, pre_head)  # git-native "committed a kernel.py win" — reused below
            if won and self.optimization_mode == "production":
                violations = production_kernel_violations(self.workspace, self.framework)
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
            if won and peak_util(mem) >= self.target_util:
                mask_half_memory(self.workspace, n)
                return self._finish(f"success: peak_util {peak_util(mem):.1f}% >= {self.target_util:.0f}%")

            if do_convert:
                # A direct triton->gluon translation must preserve BOTH correctness and performance.
                # Accept only a committed gluon kernel whose geomean is within +CONVERT_PERF_TOL of the
                # incumbent triton HEAD. Otherwise reject, keep triton, and record WHY — then let triton
                # run another `convert_after` stalled rounds and RETRY conversion (informed by the record).
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
                    mask_half_memory(self.workspace, n)
                    continue
                # rejected: if a bad gluon kernel got committed as HEAD, revert to the triton incumbent
                # (pre_head — the HEAD before this convert session, which is always the best triton kernel)
                if head_gluon and pre_head:
                    subprocess.run(["git", "reset", "--hard", pre_head], cwd=str(self.workspace),
                                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    reason = (f"regressed {conv_lat / prev_best - 1:+.1%} vs triton (> {CONVERT_PERF_TOL:.0%})"
                              if isinstance(conv_lat, (int, float)) and prev_best and not parity_ok
                              else "correctness gate not PASS")
                    self._record_failed_convert(n, reason)
                    print(f"[orchestrator] convert rejected ({reason}); reverted to triton HEAD {pre_head[:8]}", flush=True)
                else:
                    if read_memory(self.workspace, n) is None:
                        self._record_failed_convert(n, "convert session produced no committed gluon kernel")
                    print("[orchestrator] convert produced no committed gluon kernel; staying on triton", flush=True)
                # Classify the failed convert. A "bail" (session barely ran — low tokens AND no gluon)
                # is a model-compliance failure (e.g. launched work then exited), not a genuinely hard
                # conversion. Retrying it identically just loops forever, so after a few bails disable
                # the escalation and stay triton-only rather than burning cooldown cycles for nothing.
                # A genuine attempt (did real work: extracted TTGIR, rewrote, validated) uses many tokens
                # even when it fails parity — that resets the streak and keeps the learning-retry path.
                if res.tokens < CONVERT_MIN_TOKENS:
                    convert_bails += 1
                    print(f"[orchestrator] convert v{n} bailed early ({res.tokens} tokens, no gluon) "
                          f"[{convert_bails}/{CONVERT_MAX_BAILS}]", flush=True)
                    if convert_bails >= CONVERT_MAX_BAILS:
                        self.convert_after = 0  # disable escalation for the rest of this run
                        print("[orchestrator] convert repeatedly bailed -> disabling triton->gluon "
                              "escalation; continuing triton-only", flush=True)
                else:
                    convert_bails = 0
                # A convert was issued -> reset the cooldown regardless of outcome; conversion
                # re-fires only after another `convert_after` stalled rounds.
                stall = 0
                write_stall(self.workspace, stall)
                mask_half_memory(self.workspace, n)
                continue

            if won:                        # reuse the git-native win computed above
                stall = 0
                write_stall(self.workspace, stall)
            else:
                stall += 1
                write_stall(self.workspace, stall)
                if self.max_stall > 0 and stall >= self.max_stall:
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
    sandbox_timeout: int = 600
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
                HARDWARE=hardware_directive(self.platform, self.arch),
                SANDBOX=self._sandbox_directive(),
                MODE_POLICY=self._mode_directive(),
            )
            res = run_session(
                ws, prompt, timeout=self.setup_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
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
                             HARDWARE=hardware_directive(self.platform, self.arch),
                             SANDBOX=self._sandbox_directive(),
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
    for p in (d, *d.parents):
        cand = p / "scripts" / "roofline.py"
        if cand.is_file():
            roofline_py = str(cand)
            break
    return {"name": d.name, "reference": str(ref), "roofline_py": roofline_py, "op_dir": str(d)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Clean-session orchestrator for atrex-kernel-agent.")
    ap.add_argument("--op-dir", required=True,
                    help="The atrex-bench native op dir (shapes.json / roofline.json / metadata.json / "
                         "input.py / reference.py). EVERYTHING op-specific is read from here — the workspace "
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
        "--sandbox-timeout", type=int, default=600,
        help="Per sandbox test/profile command timeout in seconds (1..600; gateway dev limit).",
    )
    ap.add_argument(
        "--agent-cli", choices=AGENT_CLI_CHOICES, default="claude",
        help="Coding CLI used for clean optimization sessions (default: claude; alternative: qodercli).",
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
                         "(single-op path unchanged).")
    ap.add_argument("--notes", default="none", help="Extra constraints / known bottlenecks.")
    ap.add_argument("--max-iters", type=int, default=20, help="Hard cap on optimization iterations.")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="Hard token cap across all sessions (0 = no cap; max-iters still bounds it).")
    ap.add_argument("--target-util", type=float, default=90.0,
                    help="Peak-utilization %% short-circuit (default stop condition).")
    ap.add_argument("--iter-timeout", type=int, default=5400, help="Per-iteration hang backstop (s).")
    ap.add_argument("--setup-timeout", type=int, default=7200, help="Baseline session timeout (s).")
    ap.add_argument("--max-stall", type=int, default=0,
                    help="Optional: stop after N consecutive no-commit iterations (0 = disabled).")
    ap.add_argument("--convert-after", type=int, default=5,
                    help="Leaderboard Triton only: after N consecutive stalled iterations, spend ONE session "
                         "converting the kernel Triton->Gluon (no optimization), then optimize the Gluon "
                         "kernel. Always disabled in production mode. 0 = disabled.")
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
    if not 1 <= args.sandbox_timeout <= 600:
        ap.error("--sandbox-timeout must be in the gateway-supported range 1..600")
    if args.sandbox_url and args.sandbox_profile:
        ap.error("--sandbox-url and --sandbox-profile are mutually exclusive")
    if shutil.which(args.agent_cli) is None:
        ap.error(f"--agent-cli executable not found on PATH: {args.agent_cli}")
    if args.agent_cli == "qodercli" and args.token_budget > 0:
        print(
            "[orchestrator] WARNING: qodercli token-budget enforcement depends on token usage "
            "reported in stream-json; some Qoder models report zero, so --max-iters remains "
            "the authoritative hard bound in that configuration.",
            file=sys.stderr,
            flush=True,
        )
    if args.optimization_mode == "production" and args.convert_after > 0:
        print(
            "[orchestrator] production mode disables Triton->Gluon conversion so each campaign's "
            "framework remains an exact implementation constraint",
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
        return dispatch_framework_campaigns(raw_argv, frameworks, base, arch, args.platform)

    workspace_suffix = args.workspace_suffix or framework_workspace_suffix(
        args.framework, args.platform
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
        agent_cli=args.agent_cli,
        optimization_mode=args.optimization_mode,
        work_dir=args.workspace,
        workspace_suffix=workspace_suffix,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
        convert_after=(0 if args.optimization_mode == "production" else args.convert_after),
    )
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
