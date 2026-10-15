"""Workflow hardware implementation and callable interface."""
from __future__ import annotations


def _is_triton_family(framework: str) -> bool:
    """Triton and Gluon are one framework family — Gluon is the lower-level escalation of Triton."""
    return framework.strip().lower() in ("triton", "gluon", "triton/gluon")


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
        "- Hardware facts must have auditable sources. State the exact true product and runtime "
        "architecture when consulting enabled knowledge tools or primary specifications; never "
        "substitute another hardware identity or guess missing limits.\n"
    )

