from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

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

    @property
    def staged_checkpoint_dir(self) -> Path:
        return self.root / "staged_checkpoint"

    @property
    def staged_checkpoint_path(self) -> Path:
        return self.staged_checkpoint_dir / "checkpoint.json"

    def load_staged_checkpoint(self) -> tuple[dict[str, Any], bytes] | None:
        if not self.staged_checkpoint_path.is_file():
            return None
        try:
            metadata = json.loads(
                self.staged_checkpoint_path.read_text(encoding="utf-8")
            )
            kernel = base64.b64decode(metadata["kernel_b64"], validate=True)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(f"staged checkpoint cannot be read: {exc}") from exc
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
            raise RuntimeError("staged checkpoint metadata is invalid")
        initiative_id = metadata.get("initiative_id")
        stage = metadata.get("stage")
        next_stage = metadata.get("next_stage")
        if not isinstance(initiative_id, str) or not initiative_id.strip():
            raise RuntimeError("staged checkpoint initiative_id is invalid")
        if isinstance(stage, bool) or not isinstance(stage, int) or stage < 1:
            raise RuntimeError("staged checkpoint stage is invalid")
        if not isinstance(next_stage, str) or not next_stage.strip():
            raise RuntimeError("staged checkpoint next_stage is invalid")
        for field in (
            "escape_hypothesis",
            "architectural_delta",
            "final_success_criterion",
            "abort_criterion",
        ):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                raise RuntimeError(f"staged checkpoint {field} is invalid")
        digest = hashlib.sha256(kernel).hexdigest()
        if metadata.get("kernel_sha256") != digest:
            raise RuntimeError("staged checkpoint kernel digest does not match metadata")
        public_metadata = {
            key: item for key, item in metadata.items() if key != "kernel_b64"
        }
        return public_metadata, kernel

    def save_staged_checkpoint(
        self, metadata: dict[str, Any], kernel: bytes
    ) -> dict[str, Any]:
        value = {
            **metadata,
            "schema_version": 2,
            "kernel_sha256": hashlib.sha256(kernel).hexdigest(),
            "kernel_b64": base64.b64encode(kernel).decode("ascii"),
        }
        self.staged_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.staged_checkpoint_path, value)
        return {key: item for key, item in value.items() if key != "kernel_b64"}

    def clear_staged_checkpoint(self) -> None:
        self.staged_checkpoint_path.unlink(missing_ok=True)
        try:
            self.staged_checkpoint_dir.rmdir()
        except OSError:
            pass

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

    def archive_telemetry(self, episode: int, value: dict[str, Any]) -> Path:
        directory = self.episode_dir(episode)
        path = directory / "telemetry.summary.json"
        brief = render_episode_brief(value) + "\n"
        atomic_write_json(path, value)
        atomic_write_text(directory / "telemetry.brief.md", brief)
        return path
