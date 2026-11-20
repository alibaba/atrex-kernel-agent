"""Explicit launch-owner permission for a pinned, supplied source snapshot."""
from __future__ import annotations
import hashlib
import re
from pathlib import PurePosixPath


def validate_permission(value, files=None):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"name", "provenance", "files"}:
        raise ValueError("provided_bootstrap requires name, provenance and exact files SHA256 map")
    for key in ("name", "provenance"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("provided_bootstrap needs nonempty " + key)
    hashes = value["files"]
    if not isinstance(hashes, dict) or "kernel.py" not in hashes:
        raise ValueError("provided_bootstrap must pin the complete initial source")
    for path, digest in hashes.items():
        if (not isinstance(path, str) or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or (path != "kernel.py" and not path.startswith("impl/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid provided_bootstrap file identity")
    if files is not None:
        observed = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        if hashes != observed:
            raise ValueError("provided_bootstrap hashes do not match the complete initial source")


def directive(value):
    if not value:
        return ""
    return ("\n## Explicit supplied Bootstrap permission\n"
            f"The launch owner permits the supplied **{value['name']}** source snapshot "
            f"({value['provenance']}) and your in-candidate modifications to it. "
            "This is an explicit exception to the self-authored provenance requirement, "
            "not an instruction to rewrite this starting implementation from scratch. "
            "Keep the implementation source in kernel.py / impl/. This does not authorize "
            "calling installed prebuilt operator libraries, downloading other core implementations, "
            "or PyTorch/ATen compute fallback. The selected DSL and full correctness requirements still apply.\n")
