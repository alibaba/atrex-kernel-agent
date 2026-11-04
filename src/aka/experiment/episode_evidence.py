"""Read source-bound episode receipts without exposing their local ledger layout."""
import json
from pathlib import Path
import re

from aka.contracts.experiment import RetainedEpisodeReceipt
from .measurement_files import EPISODE_EVALUATIONS_PATH


_LEDGER_PATH = Path(EPISODE_EVALUATIONS_PATH)


def _read_local_rows(path, *, allow_tree_receipts=False):
    """Retain append order and the legacy validation order of stored rows."""
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("episode evaluation ledger is not a regular file")
    for sequence, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int:
            raise ValueError("episode evaluation receipt is corrupt")
        tree_receipt = row["schema_version"] == 2 and allow_tree_receipts
        if row["schema_version"] < 3 and not tree_receipt:
            continue
        if (not isinstance(row.get("kernel_sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", row["kernel_sha256"])
                or (not tree_receipt and "solution_sha256" not in row)
                or (row.get("solution_sha256") is not None and (
                    not isinstance(row["solution_sha256"], str)
                    or not re.fullmatch("[0-9a-f]{64}", row["solution_sha256"])))
                or not isinstance(row.get("result"), dict)):
            raise ValueError("episode evaluation receipt has invalid content identity")
        yield sequence, row


class LocalExperimentEpisodeEvidenceReader:
    """Own the current local ledger, with an explicit owner-selected read location."""
    def __init__(self, *, ledger_directory=None):
        self._ledger_directory = Path(ledger_directory) if ledger_directory is not None else None

    def _matching_receipts(self, episode_workspace, source_sha256, source_identity=None):
        source_sha256 = dict(source_sha256)
        directory = self._ledger_directory or Path(episode_workspace) / _LEDGER_PATH.parent
        tree_receipts = source_identity is not None and "solution.json" not in source_sha256
        for sequence, row in _read_local_rows(directory / _LEDGER_PATH.name, allow_tree_receipts=tree_receipts):
            if row["schema_version"] == 2:
                if row["kernel_sha256"] != source_identity or row.get("solution_sha256") is not None:
                    continue
            elif row['schema_version'] >= 4:
                identities = row.get('source_sha256')
                if (not isinstance(identities, dict) or not identities or any(
                        not isinstance(name, str) or not isinstance(value, str)
                        or not re.fullmatch('[0-9a-f]{64}', value) for name, value in identities.items())):
                    raise ValueError('episode receipt has invalid source identities')
                if identities != source_sha256:
                    continue
            elif (set(source_sha256) - {'kernel.py', 'solution.json'}
                  or row['kernel_sha256'] != source_sha256.get('kernel.py')
                  or row['solution_sha256'] != source_sha256.get('solution.json')):
                continue
            yield RetainedEpisodeReceipt(
                sequence=sequence, operation=str(row.get("gateway_kind") or "run"),
                timestamp=row.get("timestamp"), source_sha256=source_sha256,
                job_id=row.get("job_id"),
                result_json=json.dumps(row["result"], ensure_ascii=True).encode("utf-8"),
            )

    def read_episode_receipts(self, episode_workspace, *, source_sha256, source_identity=None):
        return self._matching_receipts(episode_workspace, dict(source_sha256), source_identity)
