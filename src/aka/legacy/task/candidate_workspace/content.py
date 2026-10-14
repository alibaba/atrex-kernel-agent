"""Write explicit file contributions with the original copy and text semantics."""
import hashlib
from pathlib import PurePosixPath
import shutil
import subprocess

from aka.contracts.workspace import SourceSnapshot


def install_files(workspace, files):
    paths = []
    for item in files:
        relative = PurePosixPath(item.path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe workspace path: {item.path}")
        target = workspace / item.path
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
