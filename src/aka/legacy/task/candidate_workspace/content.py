"""Write explicit file contributions with the original copy and text semantics."""
import hashlib
from pathlib import PurePosixPath
import stat
import shutil
import subprocess

from aka.contracts.workspace import SourceSnapshot


def selected_file_bytes(workspace, files, *, reserved=()):
    """Validate all selected destinations and read their content before writing."""
    selected = {}
    if workspace.is_symlink():
        raise ValueError("workspace must not be a symlink")
    for item in files:
        relative = PurePosixPath(item.path)
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != item.path or "\\" in item.path
                or any(ord(char) < 32 for char in item.path)
                or relative.parts[0] == ".git"):
            raise ValueError(f"unsafe workspace path: {item.path}")
        for name in (*reserved, *selected):
            other = PurePosixPath(name)
            if relative == other or relative in other.parents or other in relative.parents:
                raise ValueError(f"conflicting workspace contribution: {item.path}")
        target = workspace / item.path
        for path in (target, *target.parents):
            if path == workspace.parent:
                break
            if path.is_symlink():
                raise ValueError(f"workspace contribution traverses a symlink: {item.path}")
            if path != target and path.exists() and not path.is_dir():
                raise ValueError(f"workspace contribution parent is not a directory: {item.path}")
        if target.exists() and not target.is_file():
            raise ValueError(f"workspace contribution is not a file: {item.path}")
        selected[item.path] = (item.source.read_bytes() if item.source is not None
                               else item.text.encode("utf-8"))
    return selected


def install_files(workspace, files, *, immutable=False):
    files = tuple(files)
    if immutable:
        selected = selected_file_bytes(workspace, files)
        for item in files:
            target = workspace / item.path
            if target.exists() and (target.read_bytes() != selected[item.path] or (
                    item.source is not None
                    and stat.S_IMODE(target.stat().st_mode) != stat.S_IMODE(item.source.stat().st_mode))):
                raise ValueError(f"immutable workspace contribution changed: {item.path}")
    paths = []
    for item in files:
        relative = PurePosixPath(item.path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe workspace path: {item.path}")
        target = workspace / item.path
        if immutable and target.exists():
            paths.append(item.path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.source is not None:
            if item.copy == "cp":
                subprocess.run(["cp", str(item.source), str(target)], check=True)
            else:
                shutil.copy2(item.source, target)
        else:
            target.write_text(item.text, encoding="utf-8")
        paths.append(item.path)
    digest = hashlib.sha256()
    for name in paths:
        content = (workspace / name).read_bytes()
        digest.update(name.encode() + b"\0" + len(content).to_bytes(8, "big") + content)
    return SourceSnapshot(tuple(paths), digest.hexdigest())
