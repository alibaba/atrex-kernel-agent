"""Process-wide writer lease for one AKA Light Campaign.

The lease deliberately lives beside the Campaign root.  A new Campaign root
must remain empty until ``campaign.json`` is written atomically, and acquiring
the writer lease must never create Legacy state inside that root.
"""

import fcntl
import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path


def _lock_path(workspace: Path) -> Path:
    workspace = Path(workspace).expanduser().resolve()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()
    # Retain the existing external lock identity so a mixed-version launch
    # cannot acquire both the former Legacy lease and the Light lease.
    return workspace.parent / f".aka-campaign-{digest}.lock"


@contextmanager
def campaign_writer_lease(workspace: Path):
    """Reject a second writer without mutating the Campaign workspace."""

    path = _lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("AKA Light Campaign writer lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("AKA Light Campaign already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
