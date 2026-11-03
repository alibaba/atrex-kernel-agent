"""Reviewed Agent assets. Unlisted repository/submodule files are never installed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillManifest:
    root: str
    files: tuple[str, ...]


SKILL_MANIFEST = {
    "gpu-measurement": SkillManifest("skills/gpu-measurement", (
        "SKILL.md",
        "references/requests.md",
    )),
    "runtime-records": SkillManifest("skills/runtime-records", (
        "SKILL.md",
        "references/journal.md",
        "references/records.md",
    )),
    "KernelWiki": SkillManifest("skills/KernelWiki", ("SKILL.md",)),
    "autonomous-gpu-kernel-timeline": SkillManifest("skills/autonomous-gpu-kernel-timeline", (
        "SKILL.md",
        "references/cuda-backend.md",
        "references/iket-quickstart.md",
        "scripts/timeline.py",
        "backends/cuda_backend/atrex_timeline.cuh",
        "backends/cuda_backend/adapter.py",
        # Also the documented runnable allocation/launch/export example.
        "backends/cuda_backend/test_backend.cu",
        "backends/cutedsl_backend/adapter.py",
    )),
    "ppu-acu-joint-profile": SkillManifest("skills/ppu-acu-joint-profile", (
        "SKILL.md",
        "references/acu_collection.md",
        "references/ppu0015_bottlenecks.md",
        "references/recorder.md",
        "references/remote-capture.md",
        "references/timeline_contract.md",
        "scripts/acu_report.py",
        "scripts/critical_path.py",
        "scripts/evidence.py",
        "scripts/merge.py",
        "scripts/profile_report.py",
        "scripts/timeline.py",
        "backends/ppu_backend/adapter.py",
        "backends/ppu_backend/ppu_timeline.cuh",
    )),
}
