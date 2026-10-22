"""Resolve owned assets without keeping a second copy of shared Legacy resources.

Keys describe the *published* layout, not the package layout. Agent workspaces and
remote bundles keep their existing filenames while each source has one owner.
Only reviewed, controller-selected names belong here; this is not a file-read API.
"""
from importlib.resources import files
from pathlib import Path, PurePosixPath

# Package roots are controller-owned; canonicalize venv lib64/lib aliases here,
# before assets enter the same no-follow readers used for untrusted drafts.
LIGHT = Path(str(files("aka.light"))).resolve()
LEGACY = Path(str(files("aka.legacy"))).resolve()
SHARED = {
    "tools/ncu_classification.py": "execution/profiling/nvidia/ncu_classification.py",
    "tools/compute_utilization.py": "execution/profiling/common/compute_utilization.py",
    "tools/extract_asm.py": "execution/profiling/amd/extract_asm.py",
    "prompts/dependency_review.md": "workflow/prompts/dependency_review.md",
    "reference/test_kernel.py": "task/problem/sol_harness.py",
    "reference/v_iteration.schema.json": "workflow/resources/v_iteration.schema.json",
    "prompts/framework_baseline_exit_supervisor.md": "workflow/prompts/framework_baseline_exit_supervisor.md",
    "prompts/generalize_agent_problem.md": "task/problem/prompts/generalize_agent_problem.md",
    "tools/extract_ttgir.py": "execution/profiling/common/extract_ttgir.py",
    "tools/iteration_trace.py": "workflow/iteration_trace.py",
}
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
}.items():
    for member in members:
        SHARED[f"skills/{name}/{member}"] = f"execution/skills/{name}/{member}"
for name in ("__init__", "analyze_reports", "disasm", "envelope", "extract_stall_hotspots",
             "ncu_utils", "row_key", "source_evidence", "source_metrics", "warp_stalls"):
    SHARED[f"tools/ncu_helpers/{name}.py"] = f"execution/profiling/nvidia/ncu_helpers/{name}.py"

OWNED = {
    "tools/sandbox.py": "runtime/client/sandbox.py",
    "tools/plugin.py": "runtime/client/plugin.py",
    "tools/session_shell_guard.sh": "agent/session_shell_guard.sh",
    "tools/memory_manager.py": "workflow/memory.py",
    "reference/sol_seed.py": "workflow/sol_seed.py",
    "reference/sol_finalize.py": "workflow/sol_finalize.py",
}
TREES = {
    "tools": "evaluation/profiling",
    "skills": "workspace/resources/skills",
    "reference": "workspace/resources/reference",
    "prompts": "workflow/prompts",
}


def resource(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Invalid packaged asset name: {name}")
    if name in SHARED:
        return LEGACY / SHARED[name]
    if name in OWNED:
        return LIGHT / OWNED[name]
    tree, _, relative = name.partition("/")
    if tree not in TREES or not relative:
        raise ValueError(f"Unknown packaged asset: {name}")
    return LIGHT / TREES[tree] / relative


def resources(tree: str):
    """Yield the same virtual tool/Skill tree for install and bundle creation."""
    names = {name for name in SHARED.keys() | OWNED.keys() if name.startswith(tree + "/")}
    for path in (LIGHT / TREES[tree]).rglob("*"):
        if (path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}):
            names.add(tree + "/" + path.relative_to(LIGHT / TREES[tree]).as_posix())
    for name in sorted(names):
        yield name, resource(name)
