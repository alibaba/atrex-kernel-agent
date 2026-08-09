"""Hardware/framework identity: vendor detection, supported DSLs, Gluon escalation."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .constants import AMD_FRAMEWORKS, DEFAULT_FRAMEWORKS, NVIDIA_FRAMEWORKS
from .optimization_policy import source_uses_gluon


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
    the latch and returns the campaign to unconstrained optimization episodes.
    """
    return (
        convert_after > 0
        and _is_triton_family(framework)
        and not head_is_gluon
        and stall >= convert_after
    )


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
