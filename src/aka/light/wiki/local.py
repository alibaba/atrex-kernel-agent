"""Built-in provider for a GPU Wiki store on the Supervisor's filesystem."""
from __future__ import annotations

from pathlib import Path

from aka.contracts.wiki import WikiCall, WikiProviderError
from aka.light.runtime.plugins.files import FileBudget, tree_digest

CATALOG = Path(__file__).resolve().parents[1] / "runtime" / "catalogs" / "gpu-wiki"


class LocalWikiProvider:
    """Serve the GPU Wiki command-line tools from ``root`` (default: the selected checkout's ``gpu-wiki``)."""

    catalog_root = CATALOG

    def __init__(self, root: Path | None = None):
        self._root = root

    @property
    def root(self) -> Path:
        if self._root is None:
            from aka.light.runtime.paths import legacy_resource_root
            return (legacy_resource_root() / "gpu-wiki").resolve()
        return Path(self._root).expanduser().resolve()

    @property
    def cli_root(self) -> Path:
        return self.root / "tools"

    def revision(self) -> str:
        root = self.root
        if not (root / "tools" / "query_nl.py").is_file():
            raise WikiProviderError("wiki_configuration_invalid",
                                    "The configured local Wiki store has no tools/query_nl.py")
        try:
            return "local:" + tree_digest(root, budget=FileBudget())
        except ValueError as error:
            raise WikiProviderError("wiki_configuration_invalid", str(error)) from error

    def invoke(self, call: WikiCall):
        raise WikiProviderError(
            "wiki_operation_unsupported", f"The local GPU Wiki has no {call.operation!r} operation",
            next_action="Use the Wiki tools listed by `python3 tools/plugin.py list`.")
