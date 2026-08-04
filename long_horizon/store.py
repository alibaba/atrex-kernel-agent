from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from . import main_adapter
from .models import SupervisorState
from .protocol import atomic_write_json, atomic_write_text


RUNTIME_DIR = ".atrex_long_horizon"
VERIFY_DIR = "aggregate_kernels/.atrex_long_horizon_verify"


class CampaignStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / RUNTIME_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_excluded(self.workspace)

    @staticmethod
    def ensure_excluded(workspace: Path) -> None:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=str(workspace), capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"cannot locate git exclude file: {result.stderr.strip()}")
        path = Path(result.stdout.strip())
        if not path.is_absolute():
            path = workspace / path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rules = (
            f"/{RUNTIME_DIR}/",
            f"/{VERIFY_DIR}/",
            f"/{main_adapter.STALL_STATE_FILE}",
        )
        missing = [rule for rule in rules if rule not in text.splitlines()]
        if missing:
            suffix = ("" if not text or text.endswith("\n") else "\n") + "\n".join(missing) + "\n"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(suffix)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def active_path(self) -> Path:
        return self.root / "active_episode.json"

    def episode_dir(self, episode: int) -> Path:
        path = self.root / "episodes" / f"e{episode:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_state(self) -> SupervisorState:
        try:
            return SupervisorState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return SupervisorState()

    def save_state(self, state: SupervisorState) -> None:
        atomic_write_json(self.state_path, state.as_dict())

    def load_active(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def save_active(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.active_path, value)

    def clear_active(self) -> None:
        self.active_path.unlink(missing_ok=True)

    def write_brief(self, episode: int, value: str) -> Path:
        path = self.episode_dir(episode) / "BRIEF.md"
        atomic_write_text(path, value)
        return path

    def archive_attempt(self, episode: int, value: dict[str, Any]) -> Path:
        path = self.episode_dir(episode) / "attempt.json"
        atomic_write_json(path, value)
        return path
