"""Standalone, standard-library-only bootstrap for one sealed Dev file archive."""

from __future__ import annotations

import hashlib
import shutil
import stat
import sys
import zipfile
from pathlib import Path


def validate_path(path: str) -> None:
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"OSS archive file path must be normalized and relative: {path!r}")


def unpack(archive: Path, digest: str) -> None:
    """Verify exact packed content and reject escapes before writing any files."""
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("OSS archive must be a regular file")
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != digest:
        raise ValueError("OSS archive SHA-256 does not match the sealed payload")
    root = Path.cwd().resolve()
    with zipfile.ZipFile(archive) as bundle:
        paths: set[str] = set()
        for member in bundle.infolist():
            validate_path(member.filename)
            if member.filename in paths:
                raise ValueError("OSS archive contains duplicate file paths")
            paths.add(member.filename)
            mode = member.external_attr >> 16
            if member.is_dir() or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError("OSS archive contains a non-regular file")
            target = root / member.filename
            if (
                not target.resolve().is_relative_to(root)
                or target.is_symlink()
                or target == archive.resolve()
                or target == Path(__file__).resolve()
                or any(
                    parent.is_symlink() for parent in target.parents if parent.is_relative_to(root)
                )
            ):
                raise ValueError("OSS archive file destination is unsafe")
        if any(parent.as_posix() in paths for name in paths for parent in Path(name).parents):
            raise ValueError("OSS archive file paths conflict with parent directories")
        for member in bundle.infolist():
            target = root / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    archive.unlink()


if __name__ == "__main__":
    unpack(Path(sys.argv[1]), sys.argv[2])
