"""Original initial-workspace scaffold, assets and Git metadata mechanics."""
from __future__ import annotations
import json
import os
import sys
import subprocess
from pathlib import Path
from .content import install_files

GROUND_TRUTH = ("definition.json", "reference.py", "workload.jsonl")
GITIGNORE = '__pycache__/\n*.pyc\ntraces.jsonl\n.finalize_traces.jsonl\nsubmission.json\n*.ncu-rep\nprofiles/*/att/*.att\nprofiles/*/att/*.out\nprofiles/*/att/*.pftrace\nprofiles/*/att/*.otf2\n# orchestrator runtime symlinks (not part of the workspace)\n/tools\n/reference\n/skills\n/reference-projects\n/gpu-wiki\n'


def initialize_kernel(reference_dir, entry, arguments, *, content, working_directory=None):
    if len(content.files) != 1 or content.files[0].path != "kernel.py" or content.files[0].source is None:
        raise ValueError("kernel initialization requires one kernel.py copy contribution")
    return subprocess.run(["bash", str(Path(__file__).with_name("init.sh")),
                           *(arguments[:2] + [""] * max(0, 2 - len(arguments))),
                           str(reference_dir), entry, str(content.files[0].source)],
                          cwd=working_directory,
                          env={**os.environ, "AKA_TASK_PYTHON": sys.executable}).returncode


def prepare_sol(ws, content, *, problem_files, reference_dir):
    for sub in ("memory", "plans", "profiles"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    # 1) ground truth, verbatim
    install_files(ws, problem_files)

    # 2) pinned eval config (matches a submission; stable across versions)
    (ws / "config.json").write_text(
        json.dumps({"seed": 200, "warmup_runs": 10, "iterations": 50, "benchmark_reference": True}, indent=2) + "\n",
        encoding="utf-8",
    )

    # 3) V0 kernel + solution
    install_files(ws, content.files)

    # 4) harness + constraints + docs (copied from reference/)
    (ws / "test_kernel.py").write_text((reference_dir / "test_kernel.py").read_text(encoding="utf-8"), encoding="utf-8")
    (ws / "profile_driver.py").write_text(
        (reference_dir / "profile_driver.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    claude = reference_dir / "CLAUDE.md"
    if claude.exists():
        (ws / "CLAUDE.md").write_text(claude.read_text(encoding="utf-8"), encoding="utf-8")
    (ws / "README.md").write_text(content.readme, encoding="utf-8")
    (ws / ".gitignore").write_text(GITIGNORE, encoding="utf-8")



def commit_sol(ws):
    # 6) Git source commit. Keep memory out of this commit so its stable SHA can
    # be recorded without an impossible self-referential amend loop.
    if not (ws / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "gpu-kernel-optimizer@local"], cwd=str(ws), check=True)
        subprocess.run(["git", "config", "user.name", "GPU Kernel Optimizer"], cwd=str(ws), check=True)
    source_paths = [
        *GROUND_TRUTH,
        "config.json",
        "kernel.py",
        "solution.json",
        "test_kernel.py",
        "profile_driver.py",
        "README.md",
        ".gitignore",
    ]
    if (ws / "CLAUDE.md").is_file():
        source_paths.append("CLAUDE.md")
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

