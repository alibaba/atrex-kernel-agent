"""Hardware/framework identity: vendor detection, supported DSLs, Gluon escalation."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .constants import (
    AMD_FRAMEWORKS,
    DEFAULT_FRAMEWORKS,
    NVIDIA_FRAMEWORKS,
    PPU_FRAMEWORKS,
)
from aka.task.source.inspection import source_uses_gluon


def _hardware_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def hardware_vendor(platform: str, arch: str = "") -> str:
    """Return ``nvidia``, ``amd``, ``ppu``, or ``unknown`` for framework dispatch.

    Runtime architecture is authoritative because gateway device names can be
    desensitized. Platform-name matching is only a fallback for dry runs or an
    unavailable runtime probe.

    PPU is the exception: it reports a borrowed ``sm_89``, so the runtime arch
    cannot separate it from a genuine Ada part and the platform name is the only
    signal that can. Its name check therefore runs before the arch match.
    """
    token = _hardware_token(platform)
    if re.match(r"^(?:PPU|ZW)", token):
        return "ppu"

    runtime_arch = arch.strip().lower()
    if re.fullmatch(r"sm_?\d+", runtime_arch):
        return "nvidia"
    if re.fullmatch(r"gfx[0-9a-f]+", runtime_arch):
        return "amd"

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
    if vendor == "ppu":
        return PPU_FRAMEWORKS
    return DEFAULT_FRAMEWORKS


def _workspace_slug(value: str) -> str:
    """Stable flat-workspace suffix component."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("workspace suffix value has no usable directory characters")
    return slug


def framework_workspace_suffix(
    framework: str, platform: str, optimization_mode: str = "fast"
) -> str:
    """Flat suffix for one framework/hardware/policy campaign.

    Leaderboard keeps its historical path for resume compatibility. Production
    uses a distinct path so its immutable policy and Git history can coexist
    with a prior leaderboard campaign under the same workspace root.
    """
    suffix = f"{_workspace_slug(framework)}_{_workspace_slug(platform)}"
    if optimization_mode != "leaderboard":
        suffix += f"_{_workspace_slug(optimization_mode)}"
    return suffix


def _is_triton_family(framework: str) -> bool:
    """Triton and Gluon are one framework family — Gluon is the lower-level escalation of Triton."""
    return framework.strip().lower() in ("triton", "gluon", "triton/gluon")


def kernel_is_gluon(workspace: Path) -> bool:
    """True once any working-tree Kernel source has a real Gluon import."""
    try:
        from aka.task.source.formats import collect, policy_at
        policy = policy_at(workspace)
        files = collect(workspace, policy) if policy is not None else {
            "kernel.py": (workspace / "kernel.py").read_bytes()
        }
    except (OSError, ValueError):
        return False
    return any(source_uses_gluon(data.decode("utf-8", errors="ignore"))
               for data in files.values())


def head_kernel_is_gluon(workspace: Path) -> bool:
    """True when any source in the committed HEAD Kernel is Gluon."""
    try:
        from aka.task.source.formats import from_git, unpack
        _, files = unpack(from_git(workspace, "HEAD"))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return False
    return any(source_uses_gluon(data.decode("utf-8", errors="ignore"))
               for data in files.values())


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
    """Concise, vendor-neutral hardware identity injected into every session."""
    del platform
    return f"Hardware: **{arch or 'unknown'}**\n"
