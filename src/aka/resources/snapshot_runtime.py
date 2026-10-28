"""Check saved Resources argv in the current runtime without calling tools."""
import json
import os
from pathlib import Path

from .snapshot_facts import _verified_snapshot
from .snapshot_selection import _Closure
from .plugin_runtime.execution import _command_argv, resolve_command


class _CommandRoots:
    """Require real retained originals for executable and PATH dependencies."""

    def __init__(self, manifest):
        self.roots = {Path(row["source"]): row for row in manifest["roots"]}
        self.checked = set()

    def _verify(self, root):
        from .snapshot_restore import validate_original_destinations
        if validate_original_destinations({"roots": [self.roots[root]]}):
            raise ValueError("retained execution root is missing")
        self.checked.add(root)

    def guard(self, path):
        # Resolve for dependency detection only, never to change a saved argv.
        # A symlinked venv launcher still keeps its original lexical spelling.
        # Keep '..' until physical resolution: a preceding directory symlink
        # can make its real parent differ from a lexical normalization.
        lexical = Path(path).absolute()
        physical = lexical.resolve(strict=False)
        for root in self.roots:
            if any(probe.is_relative_to(root) or root.is_relative_to(probe)
                   for probe in (lexical, physical)):
                self._verify(root)

    def guard_search(self, executable, resolved):
        path = os.environ.get("PATH")
        if path is None:
            # Match shutil.which's platform default when PATH is not set.
            try:
                path = os.confstr("CS_PATH")
            except (AttributeError, ValueError):
                path = os.defpath
        for directory in path.split(os.pathsep):
            candidate = os.path.join(directory, executable)
            self.guard(candidate)
            if os.path.normcase(candidate) == os.path.normcase(resolved):
                return
        raise ValueError("current command search cannot be verified")

    def recheck(self):
        for root in tuple(self.checked):
            self._verify(root)


def _error(plugin, tool, reason):
    return ValueError(f"Resources command {plugin}/{tool}: {reason}")


def validate_runtime_binding(snapshot) -> None:
    """Verify retained data and current saved command bindings, with zero calls.

    Missing retained executables/search roots are unsupported. Actual execution,
    a different future environment and non-executable argv dependencies remain
    separate requirements of normal strict runtime validation.
    """
    try:
        manifest, reader, _absent = _verified_snapshot(snapshot)
    except Exception:
        raise ValueError("Resources retained data cannot be verified") from None
    closure = _Closure(manifest, reader)
    roots = _CommandRoots(manifest)
    for plugin in manifest["selection"]["plugins"]["plugins"]:
        root = Path(plugin["root"])
        try:
            declaration = json.loads(closure.file(root / "plugin.json"))
        except Exception:
            raise ValueError("Resources retained tool declarations cannot be verified") from None
        for name, tool in declaration.get("tools", {}).items():
            try:
                argv = _command_argv(root, tool)
                if "/" in argv[0]:
                    roots.guard((root / argv[0]).absolute())
            except Exception:
                raise _error(plugin["id"], name, "original execution root cannot be verified") from None
            try:
                current = resolve_command(root, tool)
            except Exception:
                raise _error(plugin["id"], name, "current executable is unavailable") from None
            if "/" not in argv[0]:
                try:
                    roots.guard_search(argv[0], current[0])
                except Exception:
                    raise _error(plugin["id"], name, "current search roots cannot be verified") from None
            if current != tuple(plugin["commands"][name]):
                raise _error(plugin["id"], name, "recorded argv does not match") from None
    try:
        roots.recheck()
    except Exception:
        raise ValueError("Resources execution roots changed during binding checks") from None
