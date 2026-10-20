"""Original initial-workspace scaffold, assets and Git metadata mechanics."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from .content import install_files, selected_file_bytes

GITIGNORE = '__pycache__/\n*.pyc\ntraces.jsonl\n.finalize_traces.jsonl\nsubmission.json\n*.ncu-rep\nprofiles/*/att/*.att\nprofiles/*/att/*.out\nprofiles/*/att/*.pftrace\nprofiles/*/att/*.otf2\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n'
SOL_CONFIG = json.dumps(
    {"seed": 200, "warmup_runs": 10, "iterations": 50, "benchmark_reference": True},
    indent=2,
) + "\n"


def _validate_initial_files(workspace, source_commit, selected):
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=workspace).returncode:
        raise ValueError("refusing to reseed V0 with staged changes")
    for name, expected in selected.items():
        original = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=workspace)
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=workspace)
        path = workspace / name
        if committed != original or not path.is_file() or path.read_bytes() != original:
            raise ValueError(f"refusing to reseed changed V0 source: {name}")
        if original != expected:
            raise ValueError(f"V0 selected input or source changed: {name}")


def initialize_kernel(workspace, content, *, runtime_files):
    if len(content.files) != 1 or content.files[0].path != "kernel.py" or content.files[0].source is None:
        raise ValueError("kernel initialization requires one kernel.py copy contribution")
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


def prepare_sol(ws, content, *, problem_files, harness, runtime_files):
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
    # 7) Record measurement metadata in a second commit. This deliberately leaves
    # kernel.py untouched, so memory can point at the immutable source commit.
    v0 = ws / "memory" / "v0.json"
    if v0.exists():
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ws), capture_output=True, text=True).stdout.strip()
        mem = json.loads(v0.read_text(encoding="utf-8"))
        mem["git_commit_hash"] = h
        mem.setdefault("optimization", {})["action_category"] = "baseline"
        v0.write_text(json.dumps(mem, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "memory/v0.json"], cwd=str(ws), check=True)
        subprocess.run(
            ["git", "commit", "-m", "V0: record baseline measurement"],
            cwd=str(ws),
            check=True,
            stdout=subprocess.DEVNULL,
        )

