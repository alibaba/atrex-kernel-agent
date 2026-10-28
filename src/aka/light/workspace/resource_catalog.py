"""Resolve the published Skill layout from its single source owner.

Light retains its own instructions; profiling scripts and attachments belong
to common Execution. Install and worker export use the same virtual tree.
"""
from importlib.resources import files
from pathlib import Path, PurePosixPath

LIGHT = Path(str(files("aka.light"))).resolve()
EXECUTION = Path(str(files("aka.execution"))).resolve()
SHARED = frozenset(
    f"skills/{name}/{member}"
    for name, members in {
        "autonomous-gpu-kernel-timeline": (
            "scripts/timeline.py", "backends/cuda_backend/adapter.py",
            "backends/cuda_backend/atrex_timeline.cuh", "backends/cuda_backend/test_backend.cu",
            "backends/cutedsl_backend/adapter.py",
        ),
        "ppu-acu-joint-profile": (
            "backends/ppu_backend/adapter.py", "backends/ppu_backend/ppu_timeline.cuh",
            "references/acu_collection.md", "references/ppu0015_bottlenecks.md",
            "references/recorder.md", "references/timeline_contract.md",
            "scripts/acu_report.py", "scripts/critical_path.py", "scripts/evidence.py",
            "scripts/merge.py", "scripts/profile_report.py", "scripts/timeline.py",
        ),
    }.items()
    for member in members
)


def resource(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Invalid packaged Skill name: {name}")
    if len(path.parts) < 2 or path.parts[0] != "skills":
        raise ValueError(f"Unknown packaged Skill: {name}")
    return (EXECUTION if name in SHARED else LIGHT) / name


def resources():
    """Yield the virtual Skill tree while keeping distinct Light entry files."""
    names = set(SHARED)
    for path in (LIGHT / "skills").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            names.add(path.relative_to(LIGHT).as_posix())
    for name in sorted(names):
        yield name, resource(name)
