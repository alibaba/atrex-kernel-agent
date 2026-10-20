"""One process lease guards a campaign's writer and stopped snapshots."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import uuid


LOGGER = logging.getLogger(__name__)


def _lock_path(workspace):
    workspace = Path(workspace).resolve()
    return workspace.parent / ('.aka-campaign-' + hashlib.sha256(str(workspace).encode()).hexdigest() + '.lock')


@contextmanager
def campaign_lease(workspace, *, writer=False):
    path = _lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('campaign has an active writer or checkpoint operation') from exc
        body_error = None
        writer_started = False
        try:
            if writer:
                root = Path(workspace) / '.atrex_long_horizon'
                root.mkdir(parents=True, exist_ok=True)
                identity = root / 'public-identity.json'
                if not identity.exists():
                    temporary = identity.with_suffix('.tmp-' + str(os.getpid()))
                    temporary.write_text(json.dumps({'schema': 1, 'campaign_id': str(uuid.uuid4())}) + '\n')
                    temporary.replace(identity)
                (root / 'pause-requested').unlink(missing_ok=True)
                (root / 'writer-state.json').write_text(json.dumps({'state':'running'})+'\n')
                writer_started = True
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                if writer and writer_started:
                    try:
                        (root / 'writer-state.json').write_text(json.dumps({'state':'stopped'})+'\n')
                    except OSError:
                        if body_error is None:
                            raise
                        LOGGER.exception(
                            'failed to record the campaign clean stop; preserving the original failure'
                        )
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def writer_active(workspace):
    path = _lock_path(workspace)
    if not path.exists():
        return False
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
