from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.durable_state import durable_write_json

from . import main_adapter
from .models import SupervisorState
from .protocol import atomic_write_json, atomic_write_text
from .telemetry import render_episode_brief

RUNTIME_DIR = ".atrex_long_horizon"
VERIFY_DIR = "verification_artifacts/.atrex_long_horizon_verify"
LIVE_MEMORY_FILE = "memory/live.json"


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
            f"/{LIVE_MEMORY_FILE}",
            # Episode evidence is archived by the supervisor and must never
            # become part of the candidate commit.
            "/plans/",
            "/profiles/",
            "/.humanize/",
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

    @property
    def live_memory_path(self) -> Path:
        """Uncommitted, best-effort progress for the currently active episode."""
        return self.workspace / LIVE_MEMORY_FILE

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
        durable_write_json(self.state_path, state.as_dict())

    def record_usage(
        self, state: SupervisorState, receipt_id: str, cumulative_tokens: int
    ) -> None:
        """Persist usage and its replay receipt in one durable replacement."""
        previous = state.usage_receipts.get(receipt_id, 0)
        observed = max(previous, max(0, int(cumulative_tokens)))
        # Do not mutate the caller's state until the durable write succeeds.
        updated = SupervisorState.from_dict(state.as_dict())
        updated.tokens += observed - previous
        updated.usage_receipts[receipt_id] = observed
        self.save_state(updated)
        state.tokens = updated.tokens
        state.usage_receipts = updated.usage_receipts

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

    def archive_telemetry(self, episode: int, value: dict[str, Any]) -> Path:
        directory = self.episode_dir(episode)
        path = directory / "telemetry.summary.json"
        brief = render_episode_brief(value) + "\n"
        atomic_write_json(path, value)
        atomic_write_text(directory / "telemetry.brief.md", brief)
        return path
