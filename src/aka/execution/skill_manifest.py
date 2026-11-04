"""Reviewed upstream Gateway Skill files for the Execution-owned sync tool."""

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
}
