"""Light launch policy over the shared process registry/guardian implementation."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aka.legacy.execution import processes as shared
from aka.legacy.execution.processes import (
    HANDOFF_ID_ENV, HANDOFF_LOCK_FD_ENV, STATE_FILE_ENV, ACTIVE_MARKER,
    ProcessRecord, TerminationResult, recovery_pass_fds, session_process_snapshot,
    matching_processes, process_identity_matches, terminate_processes,
    clear_process_registry,
)


def spawn_owned_session(command, **options):
    # A file entrypoint remains usable with the restricted Agent import path.
    return shared.spawn_owned_session(command, launcher=(
        str(Path(sys.executable).resolve()), str(Path(shared.__file__).resolve()),
    ), **options)


if __name__ == "__main__":
    # Existing handoffs can still invoke the previous Light file entrypoint.
    # Preserve direct-file grandchildren even when the Agent has no PYTHONPATH.
    import os
    os.execv(sys.executable, [sys.executable, shared.__file__, *sys.argv[1:]])
