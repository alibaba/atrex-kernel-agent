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


def _run_bounded(cmd: list[str], cwd: Path, timeout: int) -> tuple[str, str, int, bool]:
    """Run cmd in its own process group; SIGKILL the whole tree on timeout."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group -> killpg reaps grandchildren
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


def run_session(workspace: Path, prompt: str, timeout: int) -> SessionResult:
    """One clean `claude` session. Fresh session-id = no memory of prior sessions."""
    session_id = str(uuid.uuid4())
    cmd = [
        "claude", "--print", "--verbose",
        "--output-format", "stream-json",
        "--session-id", session_id,
        prompt,
    ]
    stdout, stderr, exit_status, timed_out = _run_bounded(cmd, cwd=workspace, timeout=timeout)
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
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    gi = workspace / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    if "/tools" not in existing:
        with gi.open("a", encoding="utf-8") as fh:
            fh.write("\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n")


# ── campaign ──────────────────────────────────────────────────────────────────


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
        if not WORKSPACE_INIT.exists():
            raise FileNotFoundError(f"missing {WORKSPACE_INIT}")
        subprocess.run(["bash", str(WORKSPACE_INIT), self.name, self.kernel_demo], check=True)
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
        if read_memory(self.workspace, 0) is None:
            raise RuntimeError("setup did not produce memory/v0.json (baseline failed)")

    def budget_exhausted(self) -> bool:
        return self.token_budget > 0 and self.tokens_spent >= self.token_budget

    def run(self) -> str:
        if latest_version(self.workspace) < 0:
            self.setup_baseline()
        else:
            print(f"[orchestrator] resuming: latest = v{latest_version(self.workspace)}", flush=True)
            self._link_runtime()  # ensure runtime symlinks exist for iteration sessions

        stall = 0
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
                [sys.executable, str(REPO_ROOT / "tools" / "memory_manager.py"),
                 "summary", "--workspace", str(self.workspace)],
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
            PLATFORM=self.platform, ROOFLINE_PY=self.roofline_py, GPU_WIKI=self.gpu_wiki,
            NOTES=self.notes,
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

    # ── scheduler helpers ─────────────────────────────────────────────────────
    def _priority(self, b: dict) -> float:
        """Live ROI: remaining reachable savings (ms), decayed by how long the boundary has stalled.

        priority = max(0, best_latency_ms - SOL_ms/ceiling) * 0.5**stall_rounds
        A fresh boundary with no recorded latency gets top priority so it is profiled first.
        """
        ws = self._boundary_ws(b["name"])
        lat_us = best_latency_us(ws)
        decay = 0.5 ** stall_rounds(ws, self.plateau_eps)
        if lat_us is None:
            return 1e12 * decay
        ceiling = b.get("ceiling") or DEFAULT_CEILING.get(b.get("op_type", ""), 0.85)
        sol_ms = b.get("sol_time_ms")
        floor_ms = (float(sol_ms) / ceiling) if isinstance(sol_ms, (int, float)) and ceiling > 0 else 0.0
        return max(0.0, lat_us / 1000.0 - floor_ms) * decay

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
        reason = self.schedule(boundaries)
        self.recombine()
        print(f"\n[layer] STOP — {reason}", flush=True)
        for b in boundaries:
            ws = self._boundary_ws(b["name"])
            print(f"[layer]   {b['name']}: v{latest_version(ws)} best_latency_us={best_latency_us(ws)}", flush=True)
        return reason or "done"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Clean-session orchestrator for atrex-kernel-agent.")
    ap.add_argument("--name", required=True, help="Workspace name -> ./kernel_opt_<name>/")
    ap.add_argument("--kernel-demo", required=True,
                    help="Single-op mode: path to the kernel to optimize. "
                         "Layer mode (--layer): path to the whole LLM-layer module to decompose.")
    ap.add_argument("--layer", action="store_true",
                    help="Optional decomposition overlay: treat --kernel-demo as a composite of more than one "
                         "fused op (a whole LLM layer, or e.g. rope+attention / attention+moe), carve it into "
                         "fused-operator boundaries (per agents/gpu-kernel-decompose.md), optimize each in its "
                         "own workspace under one shared --max-iters budget, then recombine. Default off "
                         "(single-op path unchanged).")
    ap.add_argument("--roofline-py", default=str(REPO_ROOT.parent / "atrex-bench" / "scripts" / "roofline.py"),
                    help="Layer mode: per-boundary SOL calculator (default: <repo>/../atrex-bench/scripts/roofline.py).")
    ap.add_argument("--platform", required=True, help="Target hardware, e.g. H20 / MI308X.")
    ap.add_argument("--framework", required=True, help="e.g. CuteDSL / FlyDSL.")
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
    args = ap.parse_args(argv)

    arch = args.arch or detect_arch()
    print(f"[orchestrator] platform={args.platform} runtime_arch="
          f"{arch or 'UNKNOWN (detect failed)'} "
          f"(device name / vendor-smi may be desensitized; trusting the runtime API)", flush=True)

    if args.layer:
        layer = LayerCampaign(
            name=args.name, layer_demo=args.kernel_demo, platform=args.platform,
            framework=args.framework, notes=args.notes, arch=arch, gpu_wiki=args.gpu_wiki,
            roofline_py=args.roofline_py, max_iters=args.max_iters, token_budget=args.token_budget,
            iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout,
        )
        layer.run()
        return 0

    campaign = Campaign(
        name=args.name, kernel_demo=args.kernel_demo, platform=args.platform,
        framework=args.framework, notes=args.notes, arch=arch, gpu_wiki=args.gpu_wiki,
        max_iters=args.max_iters, token_budget=args.token_budget, target_util=args.target_util,
        iter_timeout=args.iter_timeout, setup_timeout=args.setup_timeout, max_stall=args.max_stall,
    )
    campaign.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
