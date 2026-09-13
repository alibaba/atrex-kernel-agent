"""Install required and selected Agent assets; Supervisor helpers stay private."""

from __future__ import annotations

from pathlib import Path
from long_horizon.promotion_audit import preserve_legacy_promotion_audits

from .agent_assets import (
    BACKEND_SKILL_ROOTS,
    DEFAULT_AGENT_SKILLS,
    materialize_agent_assets,
    resolve_agent_skills,
)
from .constants import REPO_ROOT
from .git_metadata import install_git_excludes


def _agent_runtime_directive(
    agent_cli: str,
    *,
    agent_skills: tuple[str, ...] = DEFAULT_AGENT_SKILLS,
    agent_reference_projects: bool = False,
) -> str:
    names = resolve_agent_skills(agent_skills)
    discovery = (
        ".agents"
        if agent_cli in {"codex", "pi"}
        else (".qoder" if agent_cli == "qodercli" else ".claude")
    )
    skills = ", ".join(f"`{name}`" for name in names)
    references = (
        "- `reference-projects/` contains read-only source references; inspect only installed files."
        if agent_reference_projects
        else "- No reference-project repository is mounted."
    )
    return (
        "- `tools/sandbox.py` is the HTTP entry for GPU, Wiki, Journal, and report operations.\n"
        "- `tools/` is writable, including its local `sandbox.py` copy. You may add or edit scripts; "
        "changes stay in this Episode's workspace and survive same-Episode recovery. "
        "GPU execution still requires Runtime requests.\n"
        f"- Mounted read-only Skills under `skills/` and `{discovery}/skills/`: {skills}. "
        "Consult them only when relevant. Skill examples do not override the current task or "
        "Runtime execution boundary.\n"
        "- Before your first GPU request, read `skills/gpu-measurement/SKILL.md` for operation "
        "selection and request examples. Before using historical records, Journal tools, or "
        "episode-report, read `skills/runtime-records/SKILL.md` for formats, ID linkage, "
        "and a worked example. Follow the current phase's finish procedure.\n" + references
    )


def _install_link(destination: Path, source: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Agent asset path contains unmanaged data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def link_runtime(
    workspace: Path,
    *,
    agent_skills: tuple[str, ...] = DEFAULT_AGENT_SKILLS,
    agent_reference_projects: bool = False,
) -> None:
    preserve_legacy_promotion_audits(workspace)
    assets = materialize_agent_assets(workspace, REPO_ROOT, agent_skills)
    for name in ("tools", "skills"):
        _install_link(workspace / name, assets / name)
    for name in ("reference", "atrex-bench", "gpu-wiki"):
        legacy = workspace / name
        if legacy.is_symlink():
            legacy.unlink()
        elif legacy.exists():
            raise RuntimeError(f"Legacy Agent asset path requires explicit cleanup: {legacy}")
    reference = workspace / "reference-projects"
    if agent_reference_projects:
        _install_link(reference, REPO_ROOT / "reference-projects")
    elif reference.is_symlink():
        reference.unlink()
    elif reference.exists():
        raise RuntimeError(f"Unselected reference projects require explicit cleanup: {reference}")
    for backend in BACKEND_SKILL_ROOTS:
        destination = workspace / backend / "skills"
        if destination.is_dir() and not destination.is_symlink():
            children = list(destination.iterdir())
            if any(not child.is_symlink() for child in children):
                raise RuntimeError(
                    f"Unmanaged Skill data requires explicit migration: {destination}"
                )
            for child in children:
                child.unlink()
            destination.rmdir()
        _install_link(destination, assets / "skills")
    install_git_excludes(workspace)
