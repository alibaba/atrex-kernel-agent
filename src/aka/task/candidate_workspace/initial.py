"""Original initial-workspace scaffold, assets and Git metadata mechanics."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from aka.contracts.content import WorkspaceFile
from .content import install_files, selected_file_bytes
from .context import begin_context, context_changes, install_context, load_context, register_context_directories, uses_context

GITIGNORE = '__pycache__/\n*.pyc\ntraces.jsonl\n.finalize_traces.jsonl\nsubmission.json\n*.ncu-rep\nprofiles/*/att/*.att\nprofiles/*/att/*.out\nprofiles/*/att/*.pftrace\nprofiles/*/att/*.otf2\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n'
SOL_CONFIG = json.dumps(
    {"seed": 200, "warmup_runs": 10, "iterations": 50, "benchmark_reference": True},
    indent=2,
) + "\n"


def _validate_initial_files(workspace, source_commit, selected, *, retain_unrelated=False):
    paths = ("--", *selected) if retain_unrelated else ()
    if subprocess.run(["git", "diff", "--cached", "--quiet", *paths], cwd=workspace).returncode:
        raise ValueError("refusing to reseed V0 with staged changes")
    for name, expected in selected.items():
        original = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=workspace)
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=workspace)
        path = workspace / name
        if committed != original or not path.is_file() or path.read_bytes() != original:
            raise ValueError(f"refusing to reseed changed V0 source: {name}")
        if original != expected:
            raise ValueError(f"V0 selected input or source changed: {name}")


def _legacy_initialize_kernel(workspace, content, *, runtime_files):
    _require_kernel(content)
    root = Path(workspace)
    selected = selected_file_bytes(root, (*content.files, *runtime_files),
                                reserved=(".gitignore", "memory", "plans", "profiles"))
    if "CLAUDE.md" not in selected:
        raise ValueError("kernel initialization requires explicit CLAUDE.md constraints")
    if (root / "kernel.py").exists():
        if (root / "kernel.py").read_bytes() != selected["kernel.py"]:
            raise ValueError("refusing to initialize a changed existing kernel")
        if (root / ".git").exists():
            history = subprocess.check_output(["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"], cwd=root, text=True).split()
            if history:
                _validate_initial_files(root, history[0], selected)
                return 0
    for sub in ("memory", "plans", "profiles"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=root, check=True)
    install_files(root, (*content.files, *runtime_files))
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n*.ncu-rep\nprofiles/*/att/*.att\nprofiles/*/att/*.out\n"
        "profiles/*/att/*.pftrace\nprofiles/*/att/*.otf2\n", encoding="utf-8")
    print(f"Workspace initialized at: {root}")
    return 0


def _legacy_prepare_sol(ws, content, *, problem_files, harness, runtime_files):
    if harness is None:
        raise ValueError("SOL preparation requires the frozen Task harness")
    ws = Path(ws)
    contributions = (*problem_files, *content.files, harness, *runtime_files)
    selected = selected_file_bytes(ws, contributions,
                                reserved=("config.json", "README.md", ".gitignore", "memory", "plans", "profiles"))
    if "profile_driver.py" not in selected:
        raise ValueError("SOL preparation requires an explicit profile_driver.py")
    if (ws / ".git").exists():
        history = subprocess.run(["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
                                 cwd=ws, capture_output=True, text=True, check=True)
        if history.stdout.strip():
            source_commit = history.stdout.split()[0]
            _validate_initial_files(ws, source_commit, {
                **selected, "config.json": SOL_CONFIG.encode("utf-8"),
            })
            return source_commit

    for sub in ("memory", "plans", "profiles"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    # 1) ground truth, verbatim
    install_files(ws, problem_files)

    # 2) pinned eval config (matches a submission; stable across versions)
    (ws / "config.json").write_text(SOL_CONFIG, encoding="utf-8")

    # 3) V0 kernel + solution
    install_files(ws, content.files)

    # 4) explicitly selected evaluator, profiling and workflow content
    install_files(ws, (harness, *runtime_files))
    (ws / "README.md").write_text(content.readme, encoding="utf-8")
    (ws / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    commit_sol(ws, tuple(selected))
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ws, text=True).strip()



def commit_sol(ws, contribution_paths):
    # 6) Git source commit. Keep memory out of this commit so its stable SHA can
    # be recorded without an impossible self-referential amend loop.
    if not (ws / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=str(ws), check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=str(ws), check=True)
    source_paths = list(dict.fromkeys((*contribution_paths, "config.json", "README.md", ".gitignore")))
    subprocess.run(["git", "add", *source_paths], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "V0: baseline (SOL reference wrapper)"], cwd=str(ws), check=True,
                   stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ws, text=True).strip()


def _source_history(workspace):
    if not (workspace / ".git").exists():
        return []
    result = subprocess.run(["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
                            cwd=workspace, capture_output=True, text=True)
    return result.stdout.split() if not result.returncode else []


def _initialize_git(workspace):
    for name in ("memory", "plans", "profiles"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=workspace, check=True)


def _retry_context(workspace, source_commit, source_files, groups):
    selected = selected_file_bytes(workspace, source_files)
    if set(load_context(workspace)["source_paths"]) != set(selected):
        raise ValueError("V0 selected source membership changed")
    _validate_initial_files(workspace, source_commit, selected, retain_unrelated=True)
    if subprocess.run(["git", "diff", "--quiet", "--", *selected], cwd=workspace).returncode:
        raise ValueError("refusing to reseed changed V0 source modes or bytes")
    changed = context_changes(workspace, "HEAD")
    if changed:
        raise ValueError("refusing to reseed changed V0 context: " + ", ".join(changed))
    for role, files in groups:
        install_context(workspace, files, role=role)


def _require_kernel(content):
    if not any(item.path == "kernel.py" for item in content.files):
        raise ValueError("kernel initialization requires an explicit kernel.py contribution")


def _validate_uncommitted_source(workspace, source_files, selected):
    """Do not replace exploratory source left by an interrupted preparation."""
    for item in source_files:
        target = workspace / item.path
        if target.exists() and (target.read_bytes() != selected[item.path]
                or (item.source is not None and item.copy == "copy2"
                    and target.stat().st_mode & 0o777 != item.source.stat().st_mode & 0o777)):
            raise ValueError(f"refusing to initialize changed existing source: {item.path}")


def initialize_kernel(workspace, content, *, runtime_files, context_directories=()):
    workspace = Path(workspace)
    runtime_files = tuple(runtime_files)
    history = _source_history(workspace)
    if history and not uses_context(workspace):
        if context_directories:
            raise ValueError("existing tracked workspaces cannot acquire new context directory namespaces")
        return _legacy_initialize_kernel(workspace, content, runtime_files=runtime_files)
    _require_kernel(content)
    readme = (WorkspaceFile("README.md", text=content.readme),) if content.readme is not None else ()
    selected = selected_file_bytes(workspace, (*content.files, *runtime_files, *readme),
                                   reserved=(".atrex_context.json", ".gitignore", "memory", "plans", "profiles"))
    if "CLAUDE.md" not in selected:
        raise ValueError("kernel initialization requires explicit CLAUDE.md constraints")
    if history:
        register_context_directories(workspace, context_directories)
        _retry_context(workspace, history[0], content.files, (("runtime", runtime_files), ("readme", readme)))
        return 0
    _validate_uncommitted_source(workspace, content.files, selected)
    if uses_context(workspace) and set(load_context(workspace)["source_paths"]) != {item.path for item in content.files}:
        raise ValueError("V0 selected source membership changed")
    _initialize_git(workspace)
    if not uses_context(workspace):
        begin_context(workspace, (item.path for item in content.files), context_directories=context_directories)
    else:
        register_context_directories(workspace, context_directories)
    install_files(workspace, content.files)
    install_context(workspace, runtime_files, role="runtime")
    install_context(workspace, readme, role="readme")
    (workspace / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n*.ncu-rep\nprofiles/*/att/*.att\nprofiles/*/att/*.out\n"
        "profiles/*/att/*.pftrace\nprofiles/*/att/*.otf2\n", encoding="utf-8")
    return 0


def prepare_sol(ws, content, *, problem_files, harness, runtime_files, context_directories=()):
    ws = Path(ws)
    history = _source_history(ws)
    if history and not uses_context(ws):
        if context_directories:
            raise ValueError("existing tracked workspaces cannot acquire new context directory namespaces")
        return _legacy_prepare_sol(ws, content, problem_files=problem_files, harness=harness,
                                   runtime_files=runtime_files)
    if harness is None:
        raise ValueError("SOL preparation requires the frozen Task harness")
    problem_files, runtime_files = tuple(problem_files), tuple(runtime_files)
    readme = (WorkspaceFile("README.md", text=content.readme),) if content.readme is not None else ()
    config = (WorkspaceFile("config.json", text=SOL_CONFIG),)
    selected = selected_file_bytes(ws, (*content.files, *problem_files, harness, *runtime_files, *readme, *config),
                                   reserved=(".atrex_context.json", ".gitignore", "memory", "plans", "profiles"))
    if "profile_driver.py" not in selected:
        raise ValueError("SOL preparation requires an explicit profile_driver.py")
    groups = (("problem", problem_files), ("harness", (harness,)), ("runtime", runtime_files),
              ("readme", readme), ("configuration", config))
    if history:
        register_context_directories(ws, context_directories)
        _retry_context(ws, history[0], content.files, groups)
        return history[0]
    _initialize_git(ws)
    if not uses_context(ws):
        begin_context(ws, (item.path for item in content.files), context_directories=context_directories)
    else:
        register_context_directories(ws, context_directories)
    install_files(ws, content.files)
    for role, files in groups:
        install_context(ws, files, role=role)
    (ws / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    from .records import record
    files = tuple(WorkspaceFile(item.path, source=ws / item.path) for item in content.files)
    files += (WorkspaceFile(".gitignore", source=ws / ".gitignore"),)
    return record(ws, expected_revision="", files=files, message="V0: baseline (SOL reference wrapper)")
