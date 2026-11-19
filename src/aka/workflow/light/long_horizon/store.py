from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from aka.core.filesystem import durable_write_json

from aka.workflow.light.long_horizon.models import SupervisorState
from aka.workflow.light.long_horizon.protocol import atomic_write_json, atomic_write_text
from aka.workflow.light.long_horizon.telemetry import render_episode_brief

class CampaignStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        from aka.workflow.light.support.paths import supervisor_campaign_root

        # Segment state is Supervisor-private.  It must not appear in the
        # Agent repository or rely on Git ignore rules for isolation.
        self.root = supervisor_campaign_root(self.workspace) / "segments"
        self.root.mkdir(parents=True, exist_ok=True)

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
        durable_write_json(self.state_path, state.as_dict())

    def record_session(self, state: SupervisorState, *, session_id: str, backend: str) -> None:
        """Persist native identity before invocation, independently of segment completion."""
        if not session_id or not backend:
            raise ValueError("Native Session identity and backend must be non-empty")
        if state.session_id == session_id and state.session_backend == backend:
            return
        updated = SupervisorState.from_dict(state.as_dict())
        updated.session_id = session_id
        updated.session_backend = backend
        self.save_state(updated)
        state.session_id = session_id
        state.session_backend = backend

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
