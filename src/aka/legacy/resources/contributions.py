"""Select legacy workspace resources without performing installation."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceResourceContribution:
    runtime_links: tuple[tuple[Path, str], ...]
    backend_skills: tuple[tuple[Path, str], ...]
    project_skills: tuple[Path, ...]
    agents: Path
    ncu_skill: Path
    evaluator_root: Path | None
    ppu_enabled: bool
    legacy_mounts: tuple[tuple[str, Path], ...]


def select_workspace_resources(repo_root: Path, *, is_ppu=False, atrex_bench_root=None):
    project = repo_root / "skills"
    names = ["gen-plan", "autonomous-gpu-kernel-timeline"]
    if is_ppu:
        names.append("ppu-acu-joint-profile")
    wiki = repo_root / "gpu-wiki"
    return WorkspaceResourceContribution(
        tuple((repo_root / name, name) for name in ("tools", "reference", "skills", "reference-projects")
              if (repo_root / name).exists()),
        tuple((project / name, name) for name in names if (project / name / "SKILL.md").is_file()),
        tuple(source for source in project.iterdir() if (source / "SKILL.md").is_file()
              and (source.name != "ppu-acu-joint-profile" or is_ppu)) if project.is_dir() else (),
        repo_root / "agents", repo_root / "3rdparty" / "ncu-report-skill", atrex_bench_root, is_ppu,
        (("gpu-wiki", wiki), *((f"{backend}/skills/KernelWiki", wiki / "3rdparty/KernelWiki")
                              for backend in (".claude", ".qoder", ".agents"))))
