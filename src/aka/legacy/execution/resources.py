"""Declare only the runtime and optional profiling content owned by Execution."""
from importlib.resources import files
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution

ROOT = Path(str(files("aka.legacy.execution")))
PROFILING_CAPABILITIES = ("nvidia", "amd", "ppu", "timeline", "ppu-acu")
PROFILING_CONFIG = {"type": "array", "items": {"type": "string", "enum": list(PROFILING_CAPABILITIES)}}


def validate_profiling(profiling):
    selected = tuple(profiling)
    if len(set(selected)) != len(selected) or set(selected) - set(PROFILING_CAPABILITIES):
        raise ValueError("invalid Execution profiling capabilities")


def contribute(ctx, profiling=()):
    """Keep Task's mutable driver copy separate from frozen Execution resources."""
    validate_profiling(profiling)
    selected = tuple(profiling)
    ctx.contribute("workspace_resources", "execution-runtime", WorkspaceResourceContribution(
        "execution-runtime", ROOT / "runtime_tools", visibility="runtime"))
    ctx.contribute("workspace_resources", "execution-profile-driver", WorkspaceResourceContribution(
        "execution-profile-driver", ROOT / "profiling/profile_driver.py", visibility="runtime"))
    ctx.contribute("workspace_resources", "execution-shell-guard", WorkspaceResourceContribution(
        "execution-shell-guard", ROOT / "runtime_tools/session_shell_guard.sh", visibility="runtime"))
    if selected:
        ctx.contribute("workspace_resources", "execution-profile-guide", WorkspaceResourceContribution(
            "execution-profile-guide", ROOT / "profiling/profile_guide.md", "profiling/profile_guide.md"))
        ctx.contribute("workspace_resources", "execution-profile-common", WorkspaceResourceContribution(
            "execution-profile-common", ROOT / "profiling/common", "profiling/common"))
    if "ppu" in selected or "ppu-acu" in selected:
        ctx.contribute("workspace_resources", "execution-ppu-profile-validation", WorkspaceResourceContribution(
            "execution-ppu-profile-validation", ROOT / "skills/ppu-acu-joint-profile/scripts",
            visibility="runtime"))
        ctx.contribute("workspace_resources", "execution-ppu-profile-validator", WorkspaceResourceContribution(
            "execution-ppu-profile-validator", ROOT / "skills/ppu-acu-joint-profile/scripts/profile_report.py",
            visibility="runtime"))
    for capability in selected:
        if capability in ("nvidia", "amd", "ppu"):
            name = "execution-profile-" + capability
            ctx.contribute("workspace_resources", name, WorkspaceResourceContribution(
                name, ROOT / "profiling" / capability, "profiling/" + capability))
        else:
            skill = "autonomous-gpu-kernel-timeline" if capability == "timeline" else "ppu-acu-joint-profile"
            ctx.contribute("agent_skills", skill, AgentSkill(skill, ROOT / "skills" / skill,
                "Collect and interpret bounded profiling evidence for the selected backend."))
