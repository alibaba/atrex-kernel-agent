#!/usr/bin/env python3
"""Clean-session orchestrator for atrex-kernel-agent.

Owns the OUTER optimization loop so termination no longer depends on the model's
in-session judgment (the old Stage-6 "is README's Stop Conditions met?" self-call).

Each iteration is a **fresh `claude` session** (`--print`, new `--session-id`) over the
*same* git workspace. State crosses the session boundary only through disk — exactly the
artifacts atrex already maintains: `memory/v<N>.json`, `plans/`, `profiles/`, and git.
HEAD is always the best kernel (a regressing iteration reverts and is never committed).

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
Everything op-specific is read from a native atrex-bench op dir (``--op-dir``): the workspace
name (dir basename), the reference kernel/layer, the full shape set, per-shape SOL, and the
priority anchor. Only ``--platform`` / ``--framework`` cannot be deduced and must be passed.

    # single operator (default):
    python orchestrator/optimize.py \
        --op-dir /path/to/atrex-bench/data/<set>/<op> \
        --platform H20 --framework CuteDSL \
        --max-iters 20 --token-budget 8000000 --target-util 90

    # whole LLM layer (optional decomposition overlay):
    #   decompose -> N per-boundary workspaces (each a standard single-op campaign) ->
    #   shared --max-iters budget scheduled by live ROI (no boundary dropped) -> recombine.
    #   Σ (per-boundary optimization versions) == --max-iters.
    python orchestrator/optimize.py --layer \
        --op-dir /path/to/atrex-bench/data/<set>/<layer> \
        --platform H20 --framework CuteDSL --max-iters 40

Self-contained run root (``--prefix``, one command)
---------------------------------------------------
A run is driven by a single **run root** — one directory that is BOTH the self-contained install
base AND the workspace the nested ``claude`` sessions operate in::

    python orchestrator/optimize.py --prefix /work/kernelA \\
        --op-dir <op> --platform B300 --framework CuteDSL ...

At startup ``_setup_run_root()`` natively reproduces ``install.sh --prefix <run_root>`` (message: the
flow lives in optimize.py, not a shell-out; ``install.sh`` is kept for standalone/codex/uninstall):
it copies the skill (``reference/ skills/ tools/ SKILL.md``) into ``<run_root>/.claude/skills/…``,
copies the subagents into ``<run_root>/.claude/agents`` (so the by-name Stage subagents are
discoverable), installs the hooks (from the single shared ``hooks/gpu_kernel_optimizer_hook.py``) +
``settings.json`` into ``<run_root>/.claude``, symlinks ``<run_root>/gpu-wiki`` and
``reference-projects``, copies the orchestrator, and best-effort inits the knowledge submodules +
rocprof-trace-decoder. It then **chdir's into the run root** and pins each nested session's
``CLAUDE_CONFIG_DIR`` there, so the workflow gates and subagents are always active.

The **workspace is the run root itself** — ``init_workspace`` creates ``memory/ plans/ profiles/
kernel.py CLAUDE.md .git`` + a workspace sentinel directly in it (no per-op subdirectory), and every
agent operation runs with ``cwd`` = the run root. The hook gates a directory by that sentinel.

``--prefix`` defaults to ``DEFAULT_RUN_ROOT`` (``/tmp/aka-opt``); use one ``--prefix`` per kernel to
keep campaigns isolated (the default is a single shared scratch root). ``--skip-bootstrap`` skips the
install (rerun of an already-set-up run root). Running the copied orchestrator from inside an
installed run root also works (the install step is a no-op).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# The RUN ROOT is a single directory that is BOTH the self-contained install base AND the workspace
# the nested `claude` sessions run in: `--prefix <path>` (default DEFAULT_RUN_ROOT). optimize.py
# installs into it (skill/agents/hooks/orchestrator + gpu-wiki/reference-projects symlinks) and then
# chdir's there; the workspace (memory/ plans/ profiles/ kernel.py .git + a sentinel) lives directly
# in it — no per-op subdirectory. _setup_run_root() populates RUN_ROOT / CONFIG_DIR.
DEFAULT_RUN_ROOT = "/tmp/aka-opt"
RUN_ROOT: Optional[Path] = None   # set by _setup_run_root(); the campaign workspace when non-None

# Sentinel file marking a directory as a gpu-kernel-optimizer workspace, so the hook can gate it by
# marker rather than by directory name. MUST match the hook's WORKSPACE_SENTINEL.
WORKSPACE_SENTINEL = ".gpu_kernel_optimizer_workspace"

# The config home the nested `claude` sessions are pinned to (hooks + subagents live here); after
# _setup_run_root this is <run_root>/.claude. Provisional value below (the source repo's .claude).
CONFIG_DIR = REPO_ROOT / ".claude"

# The single shared gpu-kernel-optimizer hook script (install.sh installs the SAME file). Present
# in the source tree; absent in an install.sh-produced detached base (already installed there).
_HOOK_NAME = "gpu_kernel_optimizer_hook.py"
HOOK_SRC = REPO_ROOT / "hooks" / _HOOK_NAME

# Runtime roots — where the skill's helpers (reference/, tools/, skills/) and subagents actually
# live. SOURCE_MODE = running inside the original source repo, i.e. SKILL.md + agents/ sit at the
# top level. An `install.sh --prefix <base>` layout is NOT source mode: there SKILL.md/agents were
# copied under <base>/.claude, so we read them from that installed copy instead. SOURCE_MODE keys
# off static top-level files (bootstrap never creates them), so it is stable before/after bootstrap.
SOURCE_MODE = (REPO_ROOT / "SKILL.md").is_file() and (REPO_ROOT / "agents").is_dir()

# Provisional defaults; _resolve_roots() (re)computes them, and main() calls it after bootstrap.
SKILL_ROOT = REPO_ROOT
AGENTS_ROOT = REPO_ROOT / "agents"

WORKSPACE_INIT = SKILL_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = SKILL_ROOT / "reference" / "sol_seed.py"


def is_sol_op(op_dir: Path) -> bool:
    """A SOL-ExecBench op dir carries definition.json + workload.jsonl next to reference.py."""
    return (op_dir / "definition.json").is_file() and (op_dir / "workload.jsonl").is_file()


def _resolve_roots() -> None:
    """(Re)compute SKILL_ROOT / AGENTS_ROOT from SOURCE_MODE. Reads happen at call time inside the
    campaign methods, so calling this once in main() (via _setup_run_root) is sufficient."""
    global SKILL_ROOT, AGENTS_ROOT, WORKSPACE_INIT, SOL_SEED
    if SOURCE_MODE:
        SKILL_ROOT = REPO_ROOT
        AGENTS_ROOT = REPO_ROOT / "agents"
    else:
        SKILL_ROOT = CONFIG_DIR / "skills" / "gpu-kernel-optimizer"
        AGENTS_ROOT = CONFIG_DIR / "agents"
    WORKSPACE_INIT = SKILL_ROOT / "reference" / "workspace_init.sh"
    SOL_SEED = SKILL_ROOT / "reference" / "sol_seed.py"


_resolve_roots()


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
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
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
            (u.get("input_tokens") or 0)
            + (u.get("output_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0)
            + (u.get("cache_read_input_tokens") or 0)
        )

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
        if evt.get("type") == "result" and isinstance(evt.get("usage"), dict):
            result_total = _usage_tokens(evt["usage"])
        usage = evt.get("usage")
        if usage is None and isinstance(evt.get("message"), dict):
            usage = evt["message"].get("usage")
        if isinstance(usage, dict):
            summed += _usage_tokens(usage)
    return result_total if result_total is not None else summed


def _run_bounded(cmd: list[str], cwd: Path, timeout: int, env: Optional[dict] = None) -> tuple[str, str, int, bool]:
    """Run cmd in its own process group; SIGKILL the whole tree on timeout."""
    import errno as _errno
    _EXEC_RETRIES = 3
    for attempt in range(_EXEC_RETRIES):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own process group -> killpg reaps grandchildren
                env=env,
            )
            break
        except OSError as e:
            # Exec format error (errno 8) or ENOENT (errno 2) can occur if the binary
            # is being replaced by an auto-updater at the exact moment we try to exec it.
            # Retry a few times with a short delay — the updater finishes in <1s.
            if e.errno in (_errno.ENOEXEC, _errno.ENOENT) and attempt < _EXEC_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s
                print(f"[orchestrator] exec failed ({e}); retrying in {wait}s "
                      f"(attempt {attempt + 1}/{_EXEC_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            raise
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
    return stdout or "", stderr or "", proc.returncode, timed_out


def _session_env() -> dict:
    """Env for a nested `claude` session. When a Bearer auth token is available
    (ANTHROPIC_AUTH_TOKEN — e.g. a gateway like idealab), drop ANTHROPIC_API_KEY so the
    CLI authenticates via the token instead of sending x-api-key, which such gateways reject
    with 401. On a plain api-key setup (no auth token) nothing is removed.
    """
    env = os.environ.copy()
    _drop_api_key_if_token(env)
    # Pin the nested `claude` session's config home to CONFIG_DIR (this repo/base's .claude), where
    # _setup_run_root() installed the gpu-kernel-optimizer hooks + subagents. This makes the
    # workflow gates and by-name subagents active, isolates config/hooks/history per run root, and
    # makes discovery independent of cwd. Auth still comes from ANTHROPIC_* env vars (above) and
    # CONFIG_DIR/settings.json (whose ANTHROPIC_API_KEY bootstrap drops when a token is present).
    if not env.get("CLAUDE_CONFIG_DIR"):
        env["CLAUDE_CONFIG_DIR"] = str(CONFIG_DIR)
    # Disable auto-updates in nested sessions.  The claude.exe binary (Bun-compiled) has a
    # built-in auto-updater that can replace the binary while the orchestrator is spawning
    # new sessions, causing OSError [Errno 8] Exec format error — a race between the
    # updater's unlink/link/rename cycle and subprocess.Popen's exec().  Long-running
    # campaigns (8+ hours, 15+ sessions) are especially vulnerable.
    env["DISABLE_AUTOUPDATER"] = "1"
    env["DISABLE_UPDATES"] = "1"
    env["CLAUDE_CODE_PACKAGE_MANAGER_AUTO_UPDATE"] = "false"
    return env


def run_session(workspace: Path, prompt: str, timeout: int) -> SessionResult:
    """One clean `claude` session. Fresh session-id = no memory of prior sessions.

    `--dangerously-skip-permissions`: these nested sessions are fully autonomous and headless
    (`--print`), so they must run Bash/Write/Edit/Task without an interactive permission prompt.
    In print mode Claude cannot prompt, so unlisted tools are otherwise denied — and once each
    session's CLAUDE_CONFIG_DIR is pinned to the workdir's own .claude (see _session_env), it no
    longer inherits ~/.claude's allow-list. The flag bypasses the permission *prompt* only; the
    gpu-kernel-optimizer PreToolUse HOOKS still run and can still deny (verified), so the workflow
    gates remain enforced. This mirrors the operator's own `claude` wrapper.
    """
    session_id = str(uuid.uuid4())
    cmd = [
        "claude", "--print", "--verbose",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--session-id", session_id,
        prompt,
    ]
    stdout, stderr, exit_status, timed_out = _run_bounded(cmd, cwd=workspace, timeout=timeout, env=_session_env())
    return SessionResult(
        exit_status=exit_status,
        timed_out=timed_out,
        tokens=_tokens_from_stream(stdout),
        stdout_tail=stdout[-2000:],
        stderr_tail=stderr[-2000:],
    )


# ── workspace / memory readers ────────────────────────────────────────────────


def latest_version(workspace: Path) -> int:
    mem = workspace / "memory"
    if not mem.exists():
        return -1
    vs = []
    for p in mem.glob("v*.json"):
        try:
            vs.append(int(p.stem[1:]))
        except ValueError:
            continue
    return max(vs) if vs else -1


def read_memory(workspace: Path, n: int) -> Optional[dict]:
    path = workspace / "memory" / f"v{n}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def committed(mem: Optional[dict]) -> bool:
    return bool(mem and mem.get("git_commit_hash"))


def peak_util(mem: Optional[dict]) -> float:
    """Max of tflops / bandwidth peak utilization (%), 0 if unknown."""
    if not mem:
        return 0.0
    perf = mem.get("performance") or {}
    vals = [perf.get("tflops_peak_utilization_pct"), perf.get("bandwidth_peak_utilization_pct")]
    return max([float(v) for v in vals if isinstance(v, (int, float))] or [0.0])


def target_met(mem: Optional[dict], target_util: float) -> bool:
    """Mechanical success: a committed, correctness-PASS iteration at/above target util."""
    if not committed(mem):
        return False
    if (mem.get("quality_gate") or {}).get("result") != "PASS":
        return False
    return peak_util(mem) >= target_util


def detect_arch() -> str:
    """Return the real runtime GPU architecture token (vendor-neutral), or '' if undetectable.

    NVIDIA/CUDA -> 'sm_<cap>' (e.g. 'sm_103'); AMD/ROCm -> the gfx arch (e.g. 'gfx942').
    Uses torch (get_device_capability / gcnArchName) — the AUTHORITATIVE source, which stays
    correct even when the GPU name / vendor SMI is DESENSITIZED (e.g. a B300 reporting as 'L20D').
    """
    code = (
        "import torch\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "if getattr(torch.version,'hip',None):\n"
        "    print(getattr(p,'gcnArchName','').split(':')[0])\n"
        "else:\n"
        "    c=torch.cuda.get_device_capability(0); print('sm_%d%d'%(c[0],c[1]))\n"
    )
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
    """Make the skill's `tools/`, `reference/`, `skills/` resolvable from cwd=workspace.

    The gpu-kernel-* skills reference these by relative path; sessions run with cwd=workspace,
    so symlink them in (absolute targets, so the workspace can live anywhere). gpu-wiki is
    passed by absolute path instead. Idempotent.
    """
    for sub in ("tools", "reference", "skills"):
        src, dst = SKILL_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    # (/tools /reference /skills are already excluded by the workspace .gitignore written in
    # init_workspace, so no append is needed here.)


# ── run-root install (native, absorbs install.sh's Claude-target flow) ───────────
#
# These helpers install into CONFIG_DIR (retargeted to <run_root>/.claude by _setup_run_root) — the
# Python, Claude-only equivalent of install.sh's claude target: the shared hook script + merged
# PreToolUse/PostToolUse hooks in settings.json, the subagents (so they resolve by name), the skill
# whitelist, gpu-wiki/reference-projects symlinks, the orchestrator copy, and a best-effort init of
# the knowledge submodules + rocprof-trace-decoder. Idempotent and non-fatal.

_HOOK_TAG = "gpu-kernel-optimizer-claude-hook-v1"
_HOOK_PRE_MATCHER = "Write|Edit|MultiEdit|apply_patch|file_replace|shell|Bash"
_HOOK_POST_MATCHER = "Read|read_file|Write|Edit|MultiEdit|apply_patch|file_replace|shell|Bash"
_ROCPROF_DECODER_REPO = "https://github.com/ROCm/rocprof-trace-decoder.git"
_NET_TIMEOUT = 600  # hard cap (s) on best-effort startup git so a hung network can't block a run


def _drop_api_key_if_token(env: dict) -> None:
    """If a Bearer auth token is present, drop ANTHROPIC_API_KEY so the CLI authenticates via the
    token instead of sending x-api-key (gateways like idealab reject x-api-key with 401). Safe
    no-op when either key is absent. Shared by _session_env (process env) and the settings.json
    merge (config env) so the rule stays in exactly one place."""
    if isinstance(env, dict) and env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)


def _strip_tagged(existing: Optional[list]) -> list:
    """Drop our tagged inner hook objects and any matcher entry left with no hooks — the jq
    `strip_tagged` half of install.sh merge_hooks. Idempotent."""
    out: list = []
    for entry in (existing or []):
        if not isinstance(entry, dict):
            continue
        kept = [h for h in (entry.get("hooks") or [])
                if not (isinstance(h, dict) and _HOOK_TAG in (h.get("command") or ""))]
        if kept:
            out.append({**entry, "hooks": kept})
    return out


def _merge_hook_entries(existing: Optional[list], matcher: str, command: str, status: str) -> list:
    """strip_tagged + append the fresh tagged entry (install.sh merge_hooks). Re-running never
    accumulates duplicates."""
    out = _strip_tagged(existing)
    out.append({
        "matcher": matcher,
        "hooks": [{"type": "command", "command": command, "statusMessage": status, "timeout": 10}],
    })
    return out


def _install_hooks_settings() -> bool:
    """Install the hook script into CONFIG_DIR/hooks and merge its hooks into CONFIG_DIR/settings.json
    (install.sh install_hook_script + merge_hooks, in Python). Preserves every non-`hooks` key.
    Returns True if the gates are in place. Never overwrites a settings.json it could not parse —
    that file may hold the operator's credentials."""
    hooks_dir = CONFIG_DIR / "hooks"
    hook_dst = hooks_dir / _HOOK_NAME
    if HOOK_SRC.is_file():
        hooks_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HOOK_SRC, hook_dst)  # overwrite each run so edits to the shared hook propagate
        os.chmod(hook_dst, 0o755)
    elif not hook_dst.is_file():
        print("[bootstrap] WARNING: no hook script (hooks/gpu_kernel_optimizer_hook.py) in the source "
              "tree and none already installed — nested sessions will run WITHOUT the workflow gates.",
              file=sys.stderr, flush=True)
        return False

    settings_path = CONFIG_DIR / "settings.json"
    settings: dict = {}
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Do NOT clobber a settings.json we can't read — it likely holds the operator's creds.
            print(f"[bootstrap] WARNING: {settings_path} is not readable/valid JSON ({exc}); leaving it "
                  "untouched — hooks NOT merged. Fix or remove it to enable the workflow gates.",
                  file=sys.stderr, flush=True)
            return False
        if not isinstance(loaded, dict):
            print(f"[bootstrap] WARNING: {settings_path} is not a JSON object; leaving it untouched — "
                  "hooks NOT merged.", file=sys.stderr, flush=True)
            return False
        settings = loaded

    # NOTE: we deliberately do NOT copy the source repo's env/model into the run root's settings.json.
    # Auth for the nested sessions comes from the launch shell's environment (ANTHROPIC_*), passed
    # through by _session_env(); the run root's settings.json carries ONLY the workflow-gate hooks.

    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    pre_cmd = f'python3 "{hook_dst}" pre --target claude --tag {_HOOK_TAG}'
    post_cmd = f'python3 "{hook_dst}" post --target claude --tag {_HOOK_TAG}'
    hooks["PreToolUse"] = _merge_hook_entries(hooks.get("PreToolUse"), _HOOK_PRE_MATCHER, pre_cmd,
                                              "gpu-kernel-optimizer pre-edit guard")
    hooks["PostToolUse"] = _merge_hook_entries(hooks.get("PostToolUse"), _HOOK_POST_MATCHER, post_cmd,
                                               "gpu-kernel-optimizer memory/kernel edit gate")
    # install.sh deletes any tagged Stop entry (a legacy install may have written one); mirror that
    # so an old tagged Stop gate doesn't survive the self-bootstrap.
    stop = _strip_tagged(hooks.get("Stop"))
    if stop:
        hooks["Stop"] = stop
    else:
        hooks.pop("Stop", None)
    settings["hooks"] = hooks

    # Pinning CLAUDE_CONFIG_DIR makes nested sessions read this settings.json env; normalize the auth
    # key the same way _session_env does (drop x-api-key when a Bearer token is present).
    _drop_api_key_if_token(settings.get("env"))

    # Atomic write via a UNIQUE temp (concurrent same-repo runs must not race on one fixed name) +
    # one-time backup — settings.json may hold credentials; never leave it truncated.
    backup_path = settings_path.parent / (settings_path.name + ".bak")
    if settings_path.is_file() and not backup_path.exists():
        try:
            shutil.copy2(settings_path, backup_path)
        except OSError:
            pass
    fd, tmp_name = tempfile.mkstemp(dir=str(settings_path.parent),
                                    prefix=settings_path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(settings, indent=2) + "\n")
        os.replace(tmp_name, settings_path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return True


def _sync_config_agents() -> None:
    """Copy the subagent definitions into CONFIG_DIR/agents so the pinned nested sessions can
    discover the by-name Stage subagents (gpu-kernel-baseline / -profiler / -research /
    kernel-optimize). Mirrors install.sh copy_agents; re-synced each run so edits propagate.
    Symlink-safe: replaces a symlinked/file agents path rather than rmtree-ing through it."""
    src = REPO_ROOT / "agents"
    if not src.is_dir():
        return  # detached base: agents already live under CONFIG_DIR/agents (installed by install.sh)
    dst = CONFIG_DIR / "agents"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _bootstrap_knowledge_repos() -> None:
    """Best-effort, non-fatal knowledge-repo setup (install.sh prepare_knowledge_repos +
    clone_decoder_to_tools, source-tree subset). In place, gpu-wiki / reference-projects are already
    the source dirs, so this reduces to: `git submodule update --init` each group, and clone the
    rocprof-trace-decoder if absent.

    The submodule init runs UNCONDITIONALLY (git makes it idempotent: a fast local no-op for
    submodules already at the recorded commit, and it pulls any that are still uninitialized). We do
    NOT skip a group when it merely *looks* populated — a group is often PARTIALLY initialized (e.g.
    reference-projects/DeepGEMM present but reference-projects/aiter not), and an "is anything
    populated?" skip would leave the rest un-pulled.

    We do NOT force `--depth 1`: that overrides each submodule's `.gitmodules` `shallow` setting, and a
    forced-shallow fetch can end up unable to check out the recorded commit (it fetches only the branch
    tip) — which left reference-projects/DeepGEMM (shallow=false) with just a `.git` file and no tree.
    Omitting `--depth` lets git honor per-submodule shallow flags. `--force` completes a checkout that a
    prior run left half-done (e.g. its batch was SIGKILL'd on timeout mid-checkout), so the step is
    self-healing across runs. Network hangs are bounded by a timeout; failures are non-fatal (a run
    proceeds with fewer references) but are surfaced, not swallowed, so a missing reference set is
    diagnosable."""
    if (REPO_ROOT / ".gitmodules").is_file():
        for sub in ("reference-projects/", "gpu-wiki/3rdparty/"):
            try:
                r = subprocess.run(["git", "submodule", "update", "--init", "--force", "--", sub],
                                   cwd=str(REPO_ROOT), check=False, timeout=_NET_TIMEOUT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if r.returncode != 0:
                    tail = ((r.stderr or "").strip().splitlines() or [""])[-1][:200]
                    print(f"[run-root] NOTE: `git submodule update --init -- {sub}` exited {r.returncode} "
                          f"(best-effort; some references may be missing — check git remote/network config): "
                          f"{tail}", file=sys.stderr, flush=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"[run-root] NOTE: submodule init for {sub} could not run ({exc}); continuing.",
                      file=sys.stderr, flush=True)
    decoder = REPO_ROOT / "tools" / "rocprof-trace-decoder"
    if not (decoder / ".git").exists():
        try:
            subprocess.run(["git", "clone", "--depth", "1", _ROCPROF_DECODER_REPO, str(decoder)],
                           check=False, timeout=_NET_TIMEOUT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass


_SKILL_WHITELIST = ("reference", "skills", "tools", "SKILL.md")


def _copy_skill(run_root: Path) -> None:
    """Copy the skill whitelist (reference/ skills/ tools/ SKILL.md) from the source repo into
    <run_root>/.claude/skills/gpu-kernel-optimizer/ — install.sh copy_skill, in Python. Copies (not
    symlinks) so the run root is self-contained (survives the source repo moving/removing). Rebuilds the
    skill dir each run so files removed/renamed in the source don't linger (install.sh uses rsync
    --delete); runs once per optimize.py launch, not per iteration."""
    dst = run_root / ".claude" / "skills" / "gpu-kernel-optimizer"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in _SKILL_WHITELIST:
        src = REPO_ROOT / item
        if src.is_dir():
            shutil.copytree(src, dst / item)
        elif src.is_file():
            shutil.copy2(src, dst / item)


def _symlink_gpu_wiki(run_root: Path) -> None:
    """Symlink <run_root>/gpu-wiki -> <repo>/gpu-wiki (install.sh link_gpu_wiki). Idempotent."""
    src = REPO_ROOT / "gpu-wiki"
    if not src.exists():
        return
    dst = run_root / "gpu-wiki"
    target = str(src.resolve())
    if dst.is_symlink():
        if os.readlink(dst) == target:
            return
        dst.unlink()
    elif dst.exists():
        return  # a real gpu-wiki already present; leave it
    os.symlink(target, dst)


def _symlink_reference_projects(run_root: Path) -> None:
    """Symlink each populated reference-projects submodule into <run_root>/reference-projects/
    (install.sh prepare_knowledge_repos). Idempotent; skips unpopulated entries."""
    src_dir = REPO_ROOT / "reference-projects"
    if not src_dir.is_dir():
        return
    dst_dir = run_root / "reference-projects"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for repo in src_dir.glob("*"):
        if not repo.is_dir() or not (repo / ".git").exists():
            continue
        dst = dst_dir / repo.name
        target = str(repo.resolve())
        if dst.is_symlink():
            if os.readlink(dst) == target:
                continue
            dst.unlink()
        elif dst.exists():
            continue
        os.symlink(target, dst)


def _copy_orchestrator(run_root: Path) -> None:
    """Copy the orchestrator (optimize.py + prompts/ + anchor_bench.py) into <run_root>/orchestrator
    so the run root is a self-contained rerun point (install.sh install_orchestrator). Idempotent."""
    src = REPO_ROOT / "orchestrator"
    dst = run_root / "orchestrator"
    if src.resolve() == dst.resolve():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))


# The workspace git repo lives in the run root, which ALSO holds the install artifacts (single-op:
# workspace == run root). Ignore those so `git status` / a stray `git add -A` never sweeps them in.
_WORKSPACE_GITIGNORE = (
    "__pycache__/\n"
    "*.pyc\n"
    "*.ncu-rep\n"
    "profiles/*/att/*.att\n"
    "profiles/*/att/*.out\n"
    "profiles/*/att/*.pftrace\n"
    "profiles/*/att/*.otf2\n"
    "# run-root install artifacts (not part of the kernel workspace)\n"
    "/.claude/\n"
    "/orchestrator/\n"
    "/gpu-wiki\n"
    "/reference-projects/\n"
    "/tools\n"
    "/reference\n"
    "/skills\n"
    "# gpu-kernel-optimizer runtime markers\n"
    ".gpu_kernel_optimizer_*\n"
)


def init_workspace(workspace: Path, kernel_demo: Path, label: str = "") -> None:
    """Initialize the workspace IN `workspace` (the run root itself, or a layer boundary subdir) —
    the Python port of reference/workspace_init.sh: the memory/plans/profiles tree, a local git repo,
    kernel.py (from the demo), .gitignore, CLAUDE.md, and a workspace sentinel (so the hook gates this
    directory by marker, not by directory name). Only called for a not-yet-initialized
    workspace (see setup_baseline / setup_boundaries), so it never clobbers an in-progress campaign;
    git init is guarded on a missing .git. Fails loudly if a required input is missing."""
    kernel_demo = Path(kernel_demo)
    if not kernel_demo.is_file():
        raise FileNotFoundError(f"kernel_demo file not found: {kernel_demo}")
    claude_md = SKILL_ROOT / "reference" / "CLAUDE.md"
    if not claude_md.is_file():
        raise FileNotFoundError(
            f"agent constraints file not found: {claude_md} — broken skill install? "
            "(run install.sh, or check SKILL_ROOT resolution)")
    label = label or workspace.name
    # (Cross-op reuse of a shared run root is caught reachably in _setup_run_root via the op marker;
    # init_workspace only ever runs for a not-yet-initialized workspace, so a sentinel check here
    # would be dead.)
    workspace.mkdir(parents=True, exist_ok=True)
    for sub in ("memory", "plans", "profiles"):
        (workspace / sub).mkdir(exist_ok=True)
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(workspace), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"],
                       cwd=str(workspace), check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"],
                       cwd=str(workspace), check=True)
    shutil.copy2(kernel_demo, workspace / "kernel.py")
    (workspace / ".gitignore").write_text(_WORKSPACE_GITIGNORE, encoding="utf-8")
    shutil.copy2(claude_md, workspace / "CLAUDE.md")
    (workspace / WORKSPACE_SENTINEL).write_text(label + "\n", encoding="utf-8")


# ── campaign ──────────────────────────────────────────────────────────────────

# Number of retry attempts when setup_baseline() completes but v0.json is missing
# (e.g., the session ended before its subagent finished). Each retry is a lightweight
# "completion" session that picks up where the previous one left off — it does NOT
# re-initialize the workspace (no init_workspace, no kernel.py overwrite).
_SETUP_RETRIES = 2


@dataclass
class Campaign:
    name: str
    kernel_demo: str
    platform: str
    framework: str
    notes: str = "none"
    arch: str = ""                 # real runtime GPU arch e.g. "sm_103" / "gfx942"; auto-detected
    gpu_wiki: str = ""             # abs path to gpu-wiki (default: <repo>/gpu-wiki)
    max_iters: int = 20
    token_budget: int = 0          # 0 = no token cap (max-iters still bounds the run)
    target_util: float = 90.0
    iter_timeout: int = 5400       # 90 min hang-backstop per optimization session
    setup_timeout: int = 7200      # 120 min for the baseline session
    max_stall: int = 0             # 0 = disabled (budget-only); >0 = stop after N no-commit iters
    tokens_spent: int = field(default=0, init=False)

    @property
    def workspace(self) -> Path:
        # The run root IS the workspace (set by _setup_run_root); main() always sets it. The fallback
        # (a bare invocation that skipped _setup_run_root) just uses the current directory.
        return RUN_ROOT if RUN_ROOT is not None else Path.cwd()

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(f"[orchestrator] {label}: exit={res.exit_status} timed_out={res.timed_out} "
              f"tokens={res.tokens} cum_tokens={self.tokens_spent}", flush=True)
        if res.exit_status != 0 or res.timed_out:
            print(f"[orchestrator] stderr tail:\n{res.stderr_tail}", file=sys.stderr, flush=True)
        # Log stdout tail for debugging — helps diagnose sessions that exit without
        # producing expected artifacts (subagent killed, early end_turn, etc.)
        if res.stdout_tail:
            tail_lines = res.stdout_tail.strip().splitlines()
            if tail_lines:
                summary = "; ".join(tail_lines[-3:])[:500]
                print(f"[orchestrator] {label} stdout tail: {summary}", flush=True)

    def _link_runtime(self) -> None:
        link_runtime(self.workspace)

    def setup_baseline(self) -> None:
        # SOL-ExecBench op: seed a correct, directly-submittable V0 mechanically
        # (no baseline session) — sol_seed.py copies the ground-truth files, writes
        # the DPS wrapper kernel.py + solution.json, and benches V0 via test_kernel.py.
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            self._setup_baseline_sol(op_dir)
            return
        init_workspace(self.workspace, self.kernel_demo, self.name)
        self._link_runtime()
        prompt = _render(
            PROMPTS_DIR / "setup.md",
            WORKSPACE=str(self.workspace), PLATFORM=self.platform,
            FRAMEWORK=self.framework, KERNEL_DEMO=self.kernel_demo,
            NOTES=self.notes, GPU_WIKI=self.gpu_wiki,
            HARDWARE=hardware_directive(self.platform, self.arch),
        )
        res = run_session(self.workspace, prompt, timeout=self.setup_timeout)
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0). "
                "The nested `claude` produced no tokens — often an auth/config issue. "
                "Nested sessions inherit the shell environment, so make sure ANTHROPIC_* is exported in "
                "the shell you launch optimize.py from, and try `claude --print --verbose \"test\"`.\n"
                f"--- nested claude stderr tail ---\n{res.stderr_tail or '(empty)'}\n"
                f"--- nested claude stdout tail ---\n{res.stdout_tail or '(empty)'}"
            )

        # Retry if v0.json was not produced (session ended before subagent completed, etc.)
        # The retry is a lightweight "completion" session — it does NOT re-initialize the
        # workspace (no init_workspace, no kernel.py overwrite). It reviews the current
        # state and finishes whatever the previous session left incomplete.
        for attempt in range(_SETUP_RETRIES):
            if read_memory(self.workspace, 0) is not None:
                return  # v0.json exists — success
            print(f"[orchestrator] setup did not produce memory/v0.json "
                  f"(retry {attempt + 1}/{_SETUP_RETRIES})", flush=True)
            completion_prompt = (
                "The previous setup session ended before completing the V0 baseline. "
                "Review the current workspace state and finish the remaining steps.\n\n"
                "## Review current state\n"
                "1. Read `kernel.py` — check if it has been implemented in the target framework "
                "(Triton/CuteDSL/FlyDSL) or is still the PyTorch reference.\n"
                "2. Read `test_kernel.py` — check if correctness tests exist for ALL shapes in `shapes.json`.\n"
                "3. Read `README.md` — check if hardware specs and roofline analysis are present.\n"
                "4. Check `memory/` — if v0.json exists, the baseline is already done.\n\n"
                "## Complete remaining steps\n"
                "- If kernel.py needs implementation: launch the `gpu-kernel-baseline` subagent with "
                "`run_in_background: false` (CRITICAL — must be synchronous).\n"
                "- If kernel.py is implemented but not validated: run correctness tests for ALL shapes, "
                "measure performance, write `baseline_report.md`, write `memory/v0.json` via "
                "`python tools/memory_manager.py create --workspace . --version v0` then update, "
                "and `git commit`.\n"
                "- If only the memory/commit step is missing: write v0.json and commit.\n\n"
                "## CRITICAL — Subagent Execution Rules\n"
                "When launching ANY subagent, you MUST pass `run_in_background: false`. "
                "This makes the call blocking. NEVER end your turn before the subagent returns its result. "
                "If you launched a subagent and haven't received its result, DO NOT stop.\n\n"
                f"Platform: {self.platform}, Framework: {self.framework}"
            )
            res = run_session(self.workspace, completion_prompt, timeout=self.setup_timeout)
            self._account(res, f"setup-retry-{attempt + 1}")
            if res.exit_status != 0 and res.tokens == 0:
                break  # infra failure, don't retry

        if read_memory(self.workspace, 0) is None:
            raise RuntimeError(
                f"setup did not produce memory/v0.json after {_SETUP_RETRIES + 1} attempts "
                "(baseline failed — check session logs above for details)"
            )

    def _setup_baseline_sol(self, op_dir: Path) -> None:
        if not SOL_SEED.exists():
            raise FileNotFoundError(f"missing {SOL_SEED}")
        subprocess.run(
            [sys.executable, str(SOL_SEED),
             "--op-dir", str(op_dir), "--name", self.name,
             "--workspace", str(self.workspace),
             "--framework", self.framework, "--platform", self.platform,
             "--gpu-wiki", self.gpu_wiki],
            check=True,
        )
        self._link_runtime()
        if read_memory(self.workspace, 0) is None:
            raise RuntimeError("sol_seed did not produce memory/v0.json (V0 baseline failed)")

    def budget_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_spent >= self.token_budget

    def run(self) -> str:
        if latest_version(self.workspace) < 0:
            self.setup_baseline()
        else:
            print(f"[orchestrator] resuming: latest = v{latest_version(self.workspace)}", flush=True)
            self._link_runtime()  # ensure runtime symlinks exist for iteration sessions

        stall = 0
        infra_fails = 0  # consecutive sessions that crashed with 0 tokens (auth/infra issue)
        n = latest_version(self.workspace)  # 0 after baseline
        while True:
            if n >= self.max_iters:
                return self._finish("budget: max-iters")
            if self.budget_exhausted():
                return self._finish("budget: token-budget")

            n += 1
            prompt = _render(PROMPTS_DIR / "iteration.md",
                             WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                             PLATFORM=self.platform, NOTES=self.notes,
                             HARDWARE=hardware_directive(self.platform, self.arch))
            res = run_session(self.workspace, prompt, timeout=self.iter_timeout)
            self._account(res, f"iter v{n}")

            # Early detection of auth/infra failures: exit != 0 with 0 tokens
            # means the session never even started (bad API key, network, etc.)
            if res.exit_status != 0 and res.tokens == 0:
                infra_fails += 1
                if infra_fails >= 2:
                    return self._finish(
                        f"infra: {infra_fails} consecutive sessions crashed with 0 tokens "
                        "(likely API key / auth issue — run `claude auth status`)"
                    )
            else:
                infra_fails = 0

            # Guard: if the session exited without producing v<n>.json, write a
            # minimal failed-iteration record so latest_version() advances and the
            # stall counter increments correctly. Without this the orchestrator
            # would never see progress past a session that crashed mid-flight.
            if read_memory(self.workspace, n) is None:
                mem_dir = self.workspace / "memory"
                mem_dir.mkdir(parents=True, exist_ok=True)
                failed = {
                    "version": f"v{n}",
                    "correctness": {"status": "FAIL", "details": f"session did not produce v{n}.json"},
                    "quality_gate": {"result": "FAIL"},
                    "git_commit_hash": None,
                    "optimization": {"action_category": "failed-iteration"},
                    "notes": "orchestrator: session exited without output; recorded to advance budget",
                }
                (mem_dir / f"v{n}.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
                print(f"[orchestrator] WARNING: iter v{n} session produced no memory — "
                      f"wrote failed record to advance budget", flush=True)

            mem = read_memory(self.workspace, n)
            if target_met(mem, self.target_util):
                return self._finish(f"success: peak_util {peak_util(mem):.1f}% >= {self.target_util:.0f}%")

            if committed(mem):
                stall = 0
            else:
                stall += 1
                if self.max_stall > 0 and stall >= self.max_stall:
                    return self._finish(f"stall: {stall} iterations with no commit")

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(SKILL_ROOT / "tools" / "memory_manager.py"),
                 "summary", "--workspace", str(self.workspace)],
                check=False,
            )
        except OSError:
            pass
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
    """Per-workload best (min) latency_us across all versions, keyed by workload uuid.

    Reads ``performance.latency_us_by_shape`` from each memory/v<n>.json. Returns
    None when no version records per-workload latencies (caller falls back to the
    scalar path). SOL and latency MUST be aggregated over the same workload set, so
    the uuids here match those in workload.jsonl (the SOL-ExecBench ground truth).
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
    (``shapes[sid]`` -> SOL via shape_sol_ms). Keyed by the integer sid from
    roofline.json. To match with latency data (keyed by workload uuid), use
    ``_build_uuid_to_sid()`` to bridge the two key spaces. None if roofline.json
    is absent.
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


def _build_uuid_to_sid(workspace: Path) -> dict:
    """Build a uuid->sid mapping from workload.jsonl (SOL-ExecBench ground truth).

    Each line in workload.jsonl has a ``uuid`` field. The position (0-indexed) is
    the integer sid used in roofline.json. Returns {} if workload.jsonl is absent
    or unreadable.
    """
    wl = workspace / "workload.jsonl"
    if not wl.exists():
        return {}
    mapping = {}
    try:
        for idx, line in enumerate(wl.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                entry = json.loads(line)
                uuid = entry.get("uuid")
                if uuid:
                    mapping[str(uuid)] = str(idx)
    except (OSError, json.JSONDecodeError):
        return {}
    return mapping


def stall_rounds(workspace: Path, eps: float = 0.05) -> int:
    """Trailing count of optimization versions (v1..) that did NOT reduce best latency by >= eps.

    A reverted / no-latency version counts as non-progress. Used for a *decaying* deprioritization —
    a boundary is never dropped, its priority just shrinks while it stalls.
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
    gpu_wiki: str = ""
    roofline_py: str = ""
    op_dir: str = ""               # atrex-bench native op dir (shapes.json / roofline.json /
                                   # metadata.json / input.py / reference.py) — the full shape
                                   # set + SOL + anchor source. Passed in; never hardcoded.
    max_iters: int = 20            # SHARED across boundaries: sum of per-boundary versions
    token_budget: int = 0
    plateau_k: int = 3             # all boundaries stall_rounds >= k -> layer short-circuit
    plateau_eps: float = 0.05
    iter_timeout: int = 5400
    setup_timeout: int = 7200
    decompose_timeout: int = 5400
    recombine_timeout: int = 5400
    tokens_spent: int = field(default=0, init=False)

    @property
    def layer_dir(self) -> Path:
        # The run root holds the decomposition manifest + per-boundary workspaces (subdirs).
        return RUN_ROOT if RUN_ROOT is not None else Path.cwd() / f"layer_{self.name}"

    def _boundary_ws(self, bname: str) -> Path:
        # Each boundary is its own sentinel'd workspace, under run_root/boundaries/ — a namespace that
        # can't collide with the run root's install artifacts (.claude/ orchestrator/ gpu-wiki/ tools …).
        return self.layer_dir / "boundaries" / bname

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(f"[layer] {label}: exit={res.exit_status} timed_out={res.timed_out} "
              f"tokens={res.tokens} cum_tokens={self.tokens_spent}", flush=True)
        if res.exit_status != 0 or res.timed_out:
            print(f"[layer] stderr tail:\n{res.stderr_tail}", file=sys.stderr, flush=True)

    def budget_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_spent >= self.token_budget

    def _manifest_path(self) -> Path:
        return self.layer_dir / "boundaries.json"

    def _read_manifest(self) -> dict:
        return json.loads(self._manifest_path().read_text(encoding="utf-8"))

    # ── phase 1: decompose ────────────────────────────────────────────────────
    def decompose(self) -> None:
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        prompt = _render(
            PROMPTS_DIR / "decompose.md",
            LAYER_DIR=str(self.layer_dir), LAYER_DEMO=self.layer_demo,
            PLATFORM=self.platform, ROOFLINE_PY=self.roofline_py, GPU_WIKI=self.gpu_wiki,
            OP_DIR=self.op_dir, NOTES=self.notes,
            DECOMPOSE_DOC=str(AGENTS_ROOT / "gpu-kernel-decompose.md"),
            HARDWARE=hardware_directive(self.platform, self.arch),
        )
        res = run_session(self.layer_dir, prompt, timeout=self.decompose_timeout)
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
                continue  # already set up (resume)
            demo = self.layer_dir / b["kernel_demo"]
            init_workspace(ws, demo, b["name"])
            link_runtime(ws)
            self._write_shape_frame(ws, b)
            prompt = _render(
                PROMPTS_DIR / "setup.md",
                WORKSPACE=str(ws), PLATFORM=self.platform, FRAMEWORK=self.framework,
                KERNEL_DEMO=str(demo), NOTES=self.notes, GPU_WIKI=self.gpu_wiki,
                HARDWARE=hardware_directive(self.platform, self.arch),
            )
            res = run_session(ws, prompt, timeout=self.setup_timeout)
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

            priority(b) = mean_over_shapes( w[s] * max(0, Tk[b,s] - SOL[b,s]) ) * 0.5**stall_rounds

        `w[s]` (manifest `shape_weights`) is measured once at setup by benching the optimized-
        PyTorch anchor (`solution.py`) — see setup_anchor_weights(). Without it, w[s]=1 (raw
        ms-gap ROI). The `0.5**stall_rounds` decay is essential: when a boundary stops improving
        for a few rounds it is deprioritized so the scheduler moves on to boundaries that can
        still gain (no boundary is ever dropped — its priority just decays). SOL and latency are
        BOTH aggregated over the full shape set — never one "representative" shape (attention
        cost ∝ B·S², so a shape mismatch blows up the SOL and zeroes the boundary; that bug
        starved gqa_attention). Falls back to the scalar path when per-shape data is absent;
        a fresh boundary with no latency gets top priority.
        """
        ws = self._boundary_ws(b["name"])
        decay = 0.5 ** stall_rounds(ws, self.plateau_eps)

        # ── per-shape path (preferred): score-weighted mean reachable ms over the shape set ──
        # SOL from the workspace roofline.json (keyed by integer sid); latency from
        # memory (keyed by workload uuid from test_kernel.py / sol_execbench).
        # Bridge via workload.jsonl uuid->sid mapping.
        sol_by_sid = sol_ms_by_shape(ws)
        lat_by_uuid = best_perf_by_shape(ws)
        if sol_by_sid and lat_by_uuid:
            uuid_to_sid = _build_uuid_to_sid(ws)
            # Remap roofline SOL from sid-keyed to uuid-keyed
            sol_by_uuid = {}
            for uuid, sid in uuid_to_sid.items():
                if sid in sol_by_sid:
                    sol_by_uuid[uuid] = sol_by_sid[sid]
            if sol_by_uuid:
                common = [u for u in sol_by_uuid if u in lat_by_uuid]
                if common:
                    weights = (self._read_manifest().get("shape_weights") or {})
                    gap = sum((float(weights.get(uuid_to_sid.get(u, u), 1.0)))
                              * max(0.0, lat_by_uuid[u] / 1000.0 - sol_by_uuid[u])
                              for u in common) / len(common)
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
        return all(stall_rounds(self._boundary_ws(b["name"]), self.plateau_eps) >= self.plateau_k
                   for b in boundaries)

    # ── phase 3: shared-budget scheduler ──────────────────────────────────────
    def schedule(self, boundaries: list[dict]) -> Optional[str]:
        while True:
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
                             HARDWARE=hardware_directive(self.platform, self.arch))
            res = run_session(ws, prompt, timeout=self.iter_timeout)
            self._account(res, f"{target['name']} v{n}")

            # Guard: if the session exited without producing v<n>.json, write a
            # minimal failed-iteration record so latest_version() advances.  Without
            # this the scheduler would keep targeting the same v<N> forever (the
            # spent count never increments and stall_rounds never sees it).
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
        prompt = _render(PROMPTS_DIR / "recombine.md",
                         LAYER_DIR=str(self.layer_dir),
                         HARDWARE=hardware_directive(self.platform, self.arch))
        res = run_session(self.layer_dir, prompt, timeout=self.recombine_timeout)
        self._account(res, "recombine")

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
    --op-dir (+ the non-deducible --platform / --framework). Returns name / reference / roofline_py.
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


def _setup_run_root(args: argparse.Namespace) -> None:
    """Establish the RUN ROOT (`--prefix <path>`, default DEFAULT_RUN_ROOT) as BOTH a self-contained
    install base AND the workspace, then relocate the run there.

    Native, in-place equivalent of `install.sh --prefix <run_root>` (message 1: the flow lives in
    optimize.py, not a shell-out): copy the skill/agents/orchestrator into <run_root>, symlink
    gpu-wiki/reference-projects, install the hooks + settings — then chdir there and point RUN_ROOT /
    CONFIG_DIR / SKILL_ROOT / AGENTS_ROOT / gpu-wiki at it. The workspace (memory/ kernel.py .git …)
    is later initialized directly in <run_root> by init_workspace. `install.sh` stays for
    standalone/codex/uninstall. Best-effort: an I/O hiccup degrades to a warning, not an abort."""
    global SOURCE_MODE, CONFIG_DIR, RUN_ROOT

    run_root = Path(args.prefix).expanduser().resolve() if args.prefix else Path(DEFAULT_RUN_ROOT).resolve()

    # Resolve user paths to absolute BEFORE we chdir, so relative --op-dir / --gpu-wiki still work.
    args.op_dir = str(Path(args.op_dir).expanduser().resolve())
    op_name = Path(args.op_dir).name
    default_gpu_wiki = str(REPO_ROOT / "gpu-wiki")
    gpu_wiki_overridden = args.gpu_wiki != default_gpu_wiki
    if gpu_wiki_overridden:
        args.gpu_wiki = str(Path(args.gpu_wiki).expanduser().resolve())

    # Guard: never use the source repo checkout as a run root — that would git-init over the repo and
    # dump kernel work into it (mirrors reference/workspace_init.sh's guard).
    if (run_root / "install.sh").is_file() and (run_root / "SKILL.md").is_file():
        raise SystemExit(f"refusing to use the source repo ({run_root}) as a run root — it would git-init "
                         "over the repo. Pass a dedicated --prefix directory (default /tmp/aka-opt).")

    run_root.mkdir(parents=True, exist_ok=True)

    # Guard: a run root is single-kernel. Refuse to run a DIFFERENT op in a run root that already holds
    # one — otherwise Campaign.run would silently RESUME the prior op's memory/kernel (the shared default
    # /tmp/aka-opt makes this easy to hit). This check is reachable on every run, unlike a workspace-level
    # sentinel that only exists after init_workspace.
    op_marker = run_root / ".gpu_kernel_optimizer_op"
    if op_marker.is_file():
        prev = op_marker.read_text(encoding="utf-8").strip()
        if prev and prev != args.op_dir:
            raise SystemExit(
                f"run root {run_root} already holds op '{Path(prev).name}' ({prev}); refusing to run "
                f"'{op_name}' there — that would resume/overwrite the other kernel. Use a distinct "
                f"--prefix per kernel, or clear {run_root}.")

    installable = (REPO_ROOT / "SKILL.md").is_file() and (REPO_ROOT / "agents").is_dir() and HOOK_SRC.is_file()
    already = ((run_root / ".claude" / "skills" / "gpu-kernel-optimizer").is_dir()
               and (run_root / ".claude" / "hooks" / _HOOK_NAME).is_file())

    # Warn (always, not only in the install branch) if the operator pinned their own CLAUDE_CONFIG_DIR:
    # nested sessions would use THAT, not <run_root>/.claude, so the gates here would be inactive.
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override and Path(override).resolve() != (run_root / ".claude").resolve():
        print(f"[run-root] NOTE: CLAUDE_CONFIG_DIR={override} is set; nested sessions use it, not "
              f"{run_root / '.claude'} — the workflow gates will be INACTIVE unless that dir has them. "
              "Unset CLAUDE_CONFIG_DIR to use the self-install.", file=sys.stderr, flush=True)

    if args.skip_bootstrap:
        if not already:
            raise SystemExit(
                f"--skip-bootstrap but run root {run_root} is not set up (no installed hooks/skill). "
                f"Run without --skip-bootstrap to self-install, or `install.sh --prefix {run_root}` first.")
    elif installable:
        print(f"[run-root] installing into {run_root} …", flush=True)
        CONFIG_DIR = run_root / ".claude"   # retarget the hook/agents install helpers at the run root

        def _try(step, label):
            try:
                return step()
            except OSError as exc:  # shutil.Error is an OSError subclass
                print(f"[run-root] WARNING: {label} failed ({exc}); continuing.", file=sys.stderr, flush=True)
                return None

        # Hooks are the run root's whole purpose — install them FIRST and let `ok` track ONLY that, so a
        # failure in a best-effort extra below never skips or mis-reports the gate install. Each step is
        # isolated: one failure warns and continues.
        ok = bool(_try(_install_hooks_settings, "hook install"))
        _try(lambda: _copy_skill(run_root), "skill copy")          # helpers the run reads (SKILL_ROOT)
        _try(_sync_config_agents, "subagent copy")                 # by-name subagent discovery
        _try(_bootstrap_knowledge_repos, "submodule / decoder init")
        _try(lambda: _symlink_gpu_wiki(run_root), "gpu-wiki symlink")
        _try(lambda: _symlink_reference_projects(run_root), "reference-projects symlink")
        _try(lambda: _copy_orchestrator(run_root), "orchestrator copy")
        if not ok:
            print("[run-root] WARNING: workflow-gate hooks were not installed.", file=sys.stderr, flush=True)
    elif already:
        pass  # running the copied orchestrator from an already-installed run root — nothing to do
    else:
        raise SystemExit(
            f"cannot set up run root {run_root}: this is not a source checkout (no top-level SKILL.md/"
            f"agents/hooks) and {run_root} is not already installed. Run install.sh --prefix {run_root}, "
            "or pass --skip-bootstrap if it is set up.")

    # Record the op identity so a later run of a different op in this run root is refused (above).
    try:
        op_marker.write_text(args.op_dir + "\n", encoding="utf-8")
    except OSError:
        pass

    # Relocate: the run root becomes cwd (workspace) and the installed config home.
    os.chdir(run_root)
    RUN_ROOT = run_root
    SOURCE_MODE = False
    CONFIG_DIR = run_root / ".claude"
    _resolve_roots()
    if not gpu_wiki_overridden:
        args.gpu_wiki = str(run_root / "gpu-wiki")  # symlinked above
    print(f"[run-root] {run_root}  (workspace + config={CONFIG_DIR}, skill={SKILL_ROOT}, "
          f"gpu_wiki={args.gpu_wiki})", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Clean-session orchestrator for atrex-kernel-agent.")
    ap.add_argument("--op-dir", required=True,
                    help="The atrex-bench native op dir (shapes.json / roofline.json / metadata.json / "
                         "input.py / reference.py). EVERYTHING op-specific is read from here — the workspace "
                         "name (dir basename), the kernel/layer to optimize (reference.py), the full shape set, "
                         "per-shape SOL, and the priority anchor (metadata.production_performance). Never hardcoded.")
    ap.add_argument("--platform", required=True, help="Target hardware, e.g. B200 / H20 / MI308X "
                                                      "(cannot be deduced from the op dir).")
    ap.add_argument("--framework", required=True, help="Target DSL, e.g. CuteDSL / FlyDSL "
                                                       "(cannot be deduced from the op dir).")
    ap.add_argument("--layer", action="store_true",
                    help="Decomposition overlay: treat the op's reference as a composite of more than one fused "
                         "op (a whole LLM layer, or e.g. rope+attention / attention+moe), carve it into "
                         "fused-operator boundaries (per agents/gpu-kernel-decompose.md), optimize each in its "
                         "own workspace under one shared --max-iters budget, then recombine. Default off "
                         "(single-op path unchanged).")
    ap.add_argument("--notes", default="none", help="Extra constraints / known bottlenecks.")
    ap.add_argument("--gpu-wiki", default=str(REPO_ROOT / "gpu-wiki"),
                    help="Absolute path to the gpu-wiki knowledge base (default: <repo>/gpu-wiki).")
    ap.add_argument("--max-iters", type=int, default=20, help="Hard cap on optimization iterations.")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="Hard token cap across all sessions (0 = no cap; max-iters still bounds it).")
    ap.add_argument("--target-util", type=float, default=90.0,
                    help="Peak-utilization %% short-circuit (default stop condition).")
    ap.add_argument("--iter-timeout", type=int, default=5400, help="Per-iteration hang backstop (s).")
    ap.add_argument("--setup-timeout", type=int, default=7200, help="Baseline session timeout (s).")
    ap.add_argument("--max-stall", type=int, default=0,
                    help="Optional: stop after N consecutive no-commit iterations (0 = disabled).")
    ap.add_argument("--arch", default="",
                    help="Override the real runtime GPU arch, e.g. sm_103 or gfx942. Default: auto-detect "
                         "via torch (get_device_capability / gcnArchName) — use this if auto-detect fails.")
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="Skip the self-install into the run root (reruns / CI, or you ran install.sh). The "
                         "run root MUST already be set up — otherwise the run aborts with a clear error.")
    ap.add_argument("--prefix", default="",
                    help=f"The RUN ROOT: a self-contained directory that is BOTH the install base AND the "
                         f"workspace (default {DEFAULT_RUN_ROOT}). optimize.py installs the skill/agents/"
                         "orchestrator + hooks there (native equivalent of `install.sh --prefix <path>`), "
                         "cd's into it, and initializes the workspace (memory/ kernel.py .git …) directly in "
                         "it — no per-op subdir. Use one --prefix per kernel; the default is a single shared "
                         "scratch root.")
    args = ap.parse_args(argv)

    # Establish the run root (install + relocate + workspace); the run root becomes cwd and the
    # workspace. --prefix names it (default DEFAULT_RUN_ROOT); --skip-bootstrap skips the install.
    _setup_run_root(args)

    arch = args.arch or detect_arch()
    op = _resolve_op(args.op_dir)
    print(f"[orchestrator] op={op['name']} platform={args.platform} runtime_arch="
          f"{arch or 'UNKNOWN (detect failed)'} "
          f"(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)

    if args.layer:
        layer = LayerCampaign(
            name=op["name"], layer_demo=op["reference"], platform=args.platform,
            framework=args.framework, notes=args.notes, arch=arch, gpu_wiki=args.gpu_wiki,
            roofline_py=op["roofline_py"], op_dir=op["op_dir"],
            max_iters=args.max_iters, token_budget=args.token_budget,
            iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout,
        )
        layer.run()
        return 0

    campaign = Campaign(
        name=op["name"], kernel_demo=op["reference"], platform=args.platform,
        framework=args.framework, notes=args.notes, arch=arch, gpu_wiki=args.gpu_wiki,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
    )
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
