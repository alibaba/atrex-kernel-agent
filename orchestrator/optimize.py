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
    # single operator (default, unchanged):
    python orchestrator/optimize.py \
        --name mla_decode --kernel-demo /path/to/demo.py \
        --platform H20 --framework CuteDSL \
        --max-iters 20 --token-budget 8000000 --target-util 90

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
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = REPO_ROOT / "reference" / "sol_seed.py"
HUMANIZE_DIR = REPO_ROOT / "3rdparty" / "humanize"
CONVERT_PERF_TOL = 0.05   # triton->gluon is a direct translation: gluon must be within +5% of triton


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
    return stdout or "", stderr or "", proc.returncode, timed_out


def _session_env() -> dict:
    """Env for a nested `claude` session. When a Bearer auth token is available
    (ANTHROPIC_AUTH_TOKEN — e.g. a gateway like idealab), drop ANTHROPIC_API_KEY so the
    CLI authenticates via the token instead of sending x-api-key, which such gateways reject
    with 401. On a plain api-key setup (no auth token) nothing is removed.
    """
    env = os.environ.copy()
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)
    return env


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


def run_session(workspace: Path, prompt: str, timeout: int) -> SessionResult:
    """One clean `claude` session. Fresh session-id = no memory of prior sessions."""
    session_id = str(uuid.uuid4())
    cmd = [
        "claude", "--print", "--verbose",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--session-id", session_id,
        "--effort", "max",
    ]
    # humanize is loaded via --plugin-dir pointing to the local 3rdparty submodule;
    # it is NOT installed as a skill in .claude/skills/.
    if (HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md").exists():
        cmd += ["--plugin-dir", str(HUMANIZE_DIR)]
    cmd.append(prompt)
    env = _session_env()
    env["IS_SANDBOX"] = "1"
    stdout, stderr, exit_status, timed_out = _run_bounded(cmd, cwd=workspace, timeout=timeout, env=env)
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


def commit_changed_kernel(workspace: Path, ref: str) -> bool:
    """True iff commit `ref` changed kernel.py vs its parent (i.e. a real win, not a dead-end
    record commit). The one git primitive the win/stall/incumbent logic all share."""
    if not ref:
        return False
    try:
        r = subprocess.run(["git", "show", "--numstat", "--format=", ref, "--", "kernel.py"],
                           cwd=str(workspace), capture_output=True, text=True)
    except OSError:
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
    """Make the skill's `tools/`, `reference/`, `skills/`, `reference-projects/`, `gpu-wiki/` resolvable from cwd=workspace.

    The gpu-kernel-* skills reference these by relative path; sessions run with cwd=workspace,
    so symlink them in (absolute targets, so the workspace can live anywhere). Idempotent.

    Also installs agent definitions into ``.claude/`` so inner ``claude`` sessions can discover
    subagents (gpu-kernel-baseline, gpu-kernel-profiler, etc.).

    humanize is loaded via ``--plugin-dir`` (see ``run_session``); it is NOT installed as a
    skill into ``.claude/skills/``.
    """
    for sub in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    # ── .claude/ skills ──
    claude_dir = workspace / ".claude"
    claude_skills_dir = claude_dir / "skills"
    claude_agents_dir = claude_dir / "agents"
    claude_skills_dir.mkdir(parents=True, exist_ok=True)
    # Link 3rdparty/ncu-report-skill into .claude/skills/ so `claude` sessions can use it
    ncu_src = REPO_ROOT / "3rdparty" / "ncu-report-skill"
    ncu_dst = claude_skills_dir / "ncu-report-skill"
    if ncu_src.exists() and not ncu_dst.exists():
        os.symlink(ncu_src, ncu_dst)
    # Link gpu-wiki/3rdparty/KernelWiki into .claude/skills/ for kernel knowledge access
    kw_src = REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki"
    kw_dst = claude_skills_dir / "KernelWiki"
    if kw_src.exists() and not kw_dst.exists():
        os.symlink(kw_src, kw_dst)
    # ── .claude/ agents ──
    # The prompts (setup.md, convert.md, gpu-kernel-profile-optimizer) reference agents by
    # name (gpu-kernel-baseline, gpu-kernel-convert, gpu-kernel-profiler, gpu-kernel-research,
    # kernel-optimize). Link the repo's agents/ into .claude/agents/ so inner claude sessions
    # discover them as subagent types.
    agents_src = REPO_ROOT / "agents"
    if agents_src.exists() and not claude_agents_dir.exists():
        os.symlink(agents_src, claude_agents_dir)
    gi = workspace / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    add = ""
    if "/tools" not in existing:
        add += "\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n"
    if "/.claude" not in existing:
        add += "/.claude\n"
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
    max_iters: int = 20
    token_budget: int = 0          # 0 = no token cap (max-iters still bounds the run)
    target_util: float = 90.0
    iter_timeout: int = 5400       # 90 min hang-backstop per optimization session
    setup_timeout: int = 7200      # 120 min for the baseline session
    max_stall: int = 0             # 0 = disabled (budget-only); >0 = stop after N no-commit iters
    convert_after: int = 5         # triton-only: after N stalled iters, run ONE triton->gluon convert session (0=off)
    tokens_spent: int = field(default=0, init=False)

    @property
    def workspace(self) -> Path:
        return Path.cwd() / f"kernel_opt_{self.name}"

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(f"[orchestrator] {label}: exit={res.exit_status} timed_out={res.timed_out} "
              f"tokens={res.tokens} cum_tokens={self.tokens_spent}", flush=True)
        if res.exit_status != 0 or res.timed_out:
            print(f"[orchestrator] stderr tail:\n{res.stderr_tail}", file=sys.stderr, flush=True)

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
        if not WORKSPACE_INIT.exists():
            raise FileNotFoundError(f"missing {WORKSPACE_INIT}")
        subprocess.run(["bash", str(WORKSPACE_INIT), self.name, self.kernel_demo], check=True)
        self._link_runtime()
        prompt = _render(
            PROMPTS_DIR / "setup.md",
            WORKSPACE=str(self.workspace), PLATFORM=self.platform,
            FRAMEWORK=self.framework, KERNEL_DEMO=self.kernel_demo,
            NOTES=self.notes,
            HARDWARE=hardware_directive(self.platform, self.arch),
        )
        res = run_session(self.workspace, prompt, timeout=self.setup_timeout)
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0) — "
                "this is likely an API key / authentication issue. "
                "Run `claude auth status` and `claude --print \"test\"` to diagnose."
            )
        if read_memory(self.workspace, 0) is None:
            raise RuntimeError("setup did not produce memory/v0.json (baseline failed)")

    def _setup_baseline_sol(self, op_dir: Path) -> None:
        if not SOL_SEED.exists():
            raise FileNotFoundError(f"missing {SOL_SEED}")
        subprocess.run(
            [sys.executable, str(SOL_SEED),
             "--op-dir", str(op_dir), "--name", self.name,
             "--workspace", str(self.workspace),
             "--framework", self.framework, "--platform", self.platform],
            check=True,
        )
        self._link_runtime()
        if read_memory(self.workspace, 0) is None:
            raise RuntimeError("sol_seed did not produce memory/v0.json (V0 baseline failed)")

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

        stall = read_stall(self.workspace)   # persisted live counter (single source of truth)
        if stall is None:
            stall = reconstruct_stall(self.workspace)  # bootstrap from git when no state file yet
            write_stall(self.workspace, stall)
        if stall > 0:
            print(f"[orchestrator] stall counter restored: {stall} rounds without progress", flush=True)
        infra_fails = 0  # consecutive sessions that crashed with 0 tokens (auth/infra issue)
        n = latest_version(self.workspace)  # 0 after baseline
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
                self.convert_after > 0
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
                                 HARDWARE=hardware_directive(self.platform, self.arch))
            else:
                prompt = _render(PROMPTS_DIR / "iteration.md",
                                 WORKSPACE=str(self.workspace), N=n, PREV=n - 1,
                                 PLATFORM=self.platform, NOTES=self.notes,
                                 HARDWARE=hardware_directive(self.platform, self.arch))
            pre_head = git_head(self.workspace)  # win = a commit that changes kernel.py vs this
            res = run_session(self.workspace, prompt, timeout=self.iter_timeout)
            self._account(res, f"{'convert' if do_convert else 'iter'} v{n}")

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

            mem = read_memory(self.workspace, n)
            won = kernel_won(self.workspace, pre_head)  # git-native "committed a kernel.py win" — reused below
            if won and peak_util(mem) >= self.target_util:
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
                # A convert was issued -> reset the cooldown regardless of outcome; conversion
                # re-fires only after another `convert_after` stalled rounds.
                stall = 0
                write_stall(self.workspace, stall)
                continue

            if won:                        # reuse the git-native win computed above
                stall = 0
                write_stall(self.workspace, stall)
            else:
                stall += 1
                write_stall(self.workspace, stall)
                if self.max_stall > 0 and stall >= self.max_stall:
                    return self._finish(f"stall: {stall} iterations with no commit")

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
    tokens_spent: int = field(default=0, init=False)

    @property
    def layer_dir(self) -> Path:
        return Path.cwd() / f"layer_{self.name}"

    def _boundary_ws(self, bname: str) -> Path:
        return Path.cwd() / f"kernel_opt_{self.name}__{bname}"

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
            PLATFORM=self.platform, ROOFLINE_PY=self.roofline_py,
            OP_DIR=self.op_dir, NOTES=self.notes,
            DECOMPOSE_DOC=str(REPO_ROOT / "agents" / "gpu-kernel-decompose.md"),
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
            subprocess.run(["bash", str(WORKSPACE_INIT), f"{self.name}__{b['name']}", str(demo)], check=True)
            link_runtime(ws)
            self._write_shape_frame(ws, b)
            prompt = _render(
                PROMPTS_DIR / "setup.md",
                WORKSPACE=str(ws), PLATFORM=self.platform, FRAMEWORK=self.framework,
                KERNEL_DEMO=str(demo), NOTES=self.notes,
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
                    help="Triton only: after N consecutive stalled iterations, spend ONE session converting "
                         "the kernel Triton->Gluon (no optimization), then optimize the Gluon kernel. 0 = disabled.")
    ap.add_argument("--arch", default="",
                    help="Override the real runtime GPU arch, e.g. sm_103 or gfx942. Default: auto-detect "
                         "via torch (get_device_capability / gcnArchName) — use this if auto-detect fails.")
    args = ap.parse_args(argv)

    arch = args.arch or detect_arch()
    op = _resolve_op(args.op_dir)
    ensure_submodules()
    print(f"[orchestrator] op={op['name']} platform={args.platform} runtime_arch="
          f"{arch or 'UNKNOWN (detect failed)'} "
          f"(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)

    if args.layer:
        layer = LayerCampaign(
            name=op["name"], layer_demo=op["reference"], platform=args.platform,
            framework=args.framework, notes=args.notes, arch=arch,
            roofline_py=op["roofline_py"], op_dir=op["op_dir"],
            max_iters=args.max_iters, token_budget=args.token_budget,
            iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout,
        )
        layer.run()
        return 0

    campaign = Campaign(
        name=op["name"], kernel_demo=op["reference"], platform=args.platform,
        framework=args.framework, notes=args.notes, arch=arch,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
        convert_after=args.convert_after,
    )
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
