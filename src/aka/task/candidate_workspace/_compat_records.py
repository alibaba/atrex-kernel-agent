"""Historical promotion keyword adapter; active Workflow passes opaque files."""
import json
from aka.contracts.content import WorkspaceFile


def promotion_record_files(*, episode, evidence, memory_version, memory_record):
    return tuple(WorkspaceFile(path=name, text=json.dumps(value, indent=2, ensure_ascii=False) + "\n")
                 for name, value in ((f"memory/long_horizon_e{episode:04d}.json", evidence),
                                     (f"memory/v{memory_version}.json", memory_record)))
