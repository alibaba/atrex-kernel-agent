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
    root = Path(working_directory or Path.cwd()) / f"kernel_opt_{arguments[0]}"
    if (root / "kernel.py").exists():
        if (root / "kernel.py").read_bytes() != content.files[0].source.read_bytes():
            raise ValueError("refusing to initialize a changed existing kernel")
        if (root / ".git").exists():
            history = subprocess.check_output(["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"], cwd=root, text=True).split()
            if history:
                if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode:
                    raise ValueError("refusing to initialize with staged changes")
                original = subprocess.check_output(["git", "show", f"{history[0]}:kernel.py"], cwd=root)
                current = subprocess.check_output(["git", "show", "HEAD:kernel.py"], cwd=root)
                if original != current or current != (root / "kernel.py").read_bytes():
                    raise ValueError("refusing to initialize a changed existing kernel history")
                return 0
    code = subprocess.run(["bash", str(Path(__file__).with_name("init.sh")),
                           *(arguments[:2] + [""] * max(0, 2 - len(arguments))),
                           str(reference_dir), entry, str(content.files[0].source)],
                          cwd=working_directory,
                          env={**os.environ, "AKA_TASK_PYTHON": sys.executable}).returncode
    if code == 0:
        root = Path(working_directory or Path.cwd()) / f"kernel_opt_{arguments[0]}"
        if (root / "kernel.py").read_bytes() != content.files[0].source.read_bytes():
            raise ValueError("installed initial kernel differs from selected source")
    return code


def prepare_sol(ws, content, *, problem_files, reference_dir):
    if (ws / ".git").exists():
        history = subprocess.run(["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
                                 cwd=ws, capture_output=True, text=True, check=True)
        if history.stdout.strip():
            source_commit = history.stdout.split()[0]
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ws).returncode:
                raise ValueError("refusing to reseed V0 with staged changes")
            for name in (*GROUND_TRUTH, "config.json", "kernel.py", "solution.json",
                         "test_kernel.py", "profile_driver.py"):
                original = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ws)
                committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ws)
                if committed != original or not (ws / name).is_file() or (ws / name).read_bytes() != original:
                    raise ValueError(f"refusing to reseed changed V0 source: {name}")
            for item in (*problem_files, *content.files):
                expected = item.source.read_bytes() if item.source is not None else item.text.encode("utf-8")
                if (ws / item.path).read_bytes() != expected:
                    raise ValueError(f"V0 selected input or source changed: {item.path}")
            return source_commit

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
    commit_sol(ws)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ws, text=True).strip()



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

