"""Reviewed Agent assets. Unlisted repository/submodule files are never installed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillManifest:
    root: str
    files: tuple[str, ...]


SKILL_MANIFEST = {
    "atrex-gpu-gateway": SkillManifest("skills/atrex-gpu-gateway", (
        'SKILL.md',
        'assets/eval-manifest/cuda-cutlass/kernel.cu',
        'assets/eval-manifest/cuda-native/kernel.cu',
        'assets/eval-manifest/cuda/binding.cpp',
        'assets/eval-manifest/cuda/entry.py',
        'assets/eval-manifest/cuda/kernel.cu',
        'assets/eval-manifest/make_manifest.py',
        'assets/eval-manifest/reference/input.py',
        'assets/eval-manifest/reference/reference.py',
        'assets/eval-manifest/reference/shapes.json',
        'assets/eval-manifest/triton/entry.py',
        'assets/eval-manifest/triton/kernel.py',
        'reference/aka-workspace.md',
        'reference/capacity-control.md',
        'reference/collectives.md',
        'reference/compile.md',
        'reference/custom-sol-problem.md',
        'reference/eval-manifest.md',
        'reference/evaluation.md',
        'reference/jobs-and-troubleshooting.md',
        'reference/operations.md',
        'reference/reference-dir.md',
        'reference/solution-json.md',
        'scripts/collectives/allgather.py',
        'scripts/collectives/allreduce.py',
        'scripts/collectives/reduce_scatter.py',
    )),
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
