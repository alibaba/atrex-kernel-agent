"""Read only the current CLI session from a shared, unsandboxed provider home."""

from __future__ import annotations

import json
import re
import stat
import time
from pathlib import Path

from .agent_runtime.codex_ledger import codex_thread_id_from_stream
from .agent_workspace import _regular_bytes

DISCOVERY_INTERVAL_SECONDS = 5.0


class HostSessionTranscripts:
    def __init__(self, backend: str, environment: dict[str, str], command: list[str], *,
                 max_files: int = 4096, on_limit=lambda: None):
        self.max_files = max_files
        self.on_limit = on_limit
        self.backend = backend
        home = Path(environment.get("HOME") or str(Path.home())).expanduser()
        variable, default, label, patterns = {
            "claude": ("CLAUDE_CONFIG_DIR", home / ".claude", ".claude", ("projects/**/*.jsonl",)),
            "codex": ("CODEX_HOME", home / ".codex", ".codex", ("sessions/**/*.jsonl",)),
            "qodercli": ("", home / ".qoder", ".qoder", ("projects/**/*.jsonl", "tasks/**/*.jsonl")),
            "pi": ("PI_CODING_AGENT_DIR", home / ".pi/agent", ".pi/agent", ("sessions/**/*.jsonl",)),
        }.get(backend, ("", home, "", ()))
        self.root = Path(environment.get(variable) or default).expanduser().resolve()
        self.label = "provider/native/" + label
        self.patterns = patterns
        self.session_id = ""
        for option in ("--session-id", "--resume"):
            if option in command and command.index(option) + 1 < len(command):
                self.session_id = command[command.index(option) + 1]
                break
        if backend == "codex" and "resume" in command:
            self.session_id = next((arg for arg in command if re.fullmatch(r"[a-fA-F0-9-]{32,64}", arg)), "")
        # File fingerprints and parsed identities, including negative/unfinished headers.
        self._metadata: dict[Path, tuple[tuple[int, int, int, int], tuple[str, str]]] = {}
        # Snapshot sizes, not unrelated conversations. Codex announces its thread
        # after launch; these offsets still exclude pre-invocation resume history.
        self._known_paths = tuple(self._paths())
        self._initial_sizes = {path: path.stat().st_size for path in self._known_paths}
        self._next_discovery = time.monotonic() + DISCOVERY_INTERVAL_SECONDS
        self._discovery_session_id = self.session_id

    def _discovered_paths(self, *, force: bool) -> tuple[Path, ...]:
        now = time.monotonic()
        if force or now >= self._next_discovery:
            self._known_paths = tuple(self._paths())
            retained = set(self._known_paths)
            self._metadata = {path: value for path, value in self._metadata.items() if path in retained}
            self._next_discovery = time.monotonic() + DISCOVERY_INTERVAL_SECONDS
        return self._known_paths

    def _paths(self):
        count = 0
        for pattern in self.patterns:
            for path in self.root.glob(pattern):
                count += 1
                if count > self.max_files:
                    self.on_limit()
                    return
                # Never follow a session-directory symlink into credentials or
                # another location. Selected payloads also use no-follow reads.
                components = [path, *(parent for parent in path.parents if parent.is_relative_to(self.root))]
                if any(part.is_symlink() for part in components):
                    continue
                if path.is_file():
                    yield path

    def _codex_identity(self, path: Path) -> tuple[str, str]:
        cached = self._metadata.get(path)
        if cached is not None and cached[1][0]:
            # Keep known sessions selected: TranscriptTail must detect replacement
            # or truncation, rather than silently losing their capture/usage.
            return cached[1]
        try:
            info = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            self._metadata.pop(path, None)
            return "", ""
        if not stat.S_ISREG(info.st_mode):
            self._metadata.pop(path, None)
            return "", ""
        fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if cached is not None and fingerprint == cached[0]:
            return cached[1]
        if cached is None and len(self._metadata) >= self.max_files:
            self.on_limit()
            return "", ""
        payload = _regular_bytes(path, limit=65536)
        for line in payload.splitlines()[:32]:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") != "session_meta":
                continue
            body = event.get("payload")
            if not isinstance(body, dict):
                continue
            source = body.get("source")
            subagent = source.get("subagent") if isinstance(source, dict) else None
            spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
            parent = spawn.get("parent_thread_id", "") if isinstance(spawn, dict) else ""
            identity = body.get("id", "")
            if isinstance(identity, str) and identity and isinstance(parent, str):
                self._metadata[path] = (fingerprint, (identity, parent))
                return identity, parent
        self._metadata[path] = (fingerprint, ("", ""))
        return "", ""

    def selected(self, stdout: str = "", *, force: bool = False) -> dict[str, tuple[Path, int]]:
        if self.backend == "codex":
            self.session_id = codex_thread_id_from_stream(stdout) or self.session_id
        identity = self.session_id
        if not identity or not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
            return {}
        paths = list(self._discovered_paths(force=force or identity != self._discovery_session_id))
        self._discovery_session_id = identity
        if self.backend == "codex":
            identities = {path: self._codex_identity(path) for path in paths}
            family = {identity}
            while True:
                children = {child for child, parent in identities.values() if parent in family}
                if children <= family:
                    break
                family.update(children)
            paths = [path for path, (child, _) in identities.items() if child in family]
        else:
            paths = [path for path in paths if (
                identity in path.relative_to(self.root).parts
                or re.search(r"(?:^|[_-])" + re.escape(identity) + r"$", path.stem)
            )]
        selected = {}
        for path in paths:
            name = self.label + "/" + path.relative_to(self.root).as_posix()
            selected[name] = (path, self._initial_sizes.get(path, 0))
        return selected
