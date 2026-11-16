"""Create an Agent-owned repository once, without evaluating or reseeding it."""
from pathlib import Path
import subprocess

from .content import install_files, selected_file_bytes


def _git(workspace, *args):
    return subprocess.run(["git", *args], cwd=workspace, text=True,
                          capture_output=True, check=True).stdout.strip()


def initialize_agent(workspace: Path, content) -> str:
    workspace = Path(workspace)
    if workspace.is_symlink() or (workspace / ".git").is_symlink():
        raise ValueError("Agent workspace and Git directory must not be symlinks")
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace / ".git").exists():
        current = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
            cwd=workspace, text=True, capture_output=True)
        if current.returncode == 0:
            # Resuming never restores the seed or rewrites the Agent's branch,
            # index, uncommitted changes, ignore policy or unrelated files.
            return current.stdout.strip()
        if _git(workspace, "ls-files"):
            raise ValueError("Uninitialized Agent repository already has staged files")
    selected = selected_file_bytes(workspace, content.files, reserved=(".gitignore",))
    if not selected:
        raise ValueError("Initial Agent source must contain files")
    for name, data in selected.items():
        path = workspace / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite exploratory source: {name}")
    install_files(workspace, content.files)
    # Keep only source roots and the ignore policy eligible for ordinary git
    # add. impl/ remains optional and can be introduced later by the Agent.
    roots = sorted({Path(name).parts[0] for name in selected} | {"impl"})
    def pattern(name):
        return "".join("\\" + char if char in "\\*?[]#! " else char for char in name)
    ignores = ["/*", "!/.gitignore", *("!/" + pattern(name) for name in roots)]
    (workspace / ".gitignore").write_text("\n".join(ignores) + "\n", encoding="utf-8")
    if not (workspace / ".git").exists():
        _git(workspace, "init", "--quiet")
        _git(workspace, "config", "user.name", "GPU Kernel Optimizer")
        _git(workspace, "config", "user.email", "gpu-kernel-optimizer@local")
    _git(workspace, "add", "--", ".gitignore", *sorted(selected))
    _git(workspace, "commit", "--quiet", "-m", "Initial Kernel source")
    return _git(workspace, "rev-parse", "HEAD")
