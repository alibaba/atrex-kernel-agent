"""Pure initial source-tree selection; the Problem snapshot owns retention."""
from __future__ import annotations

import json
from pathlib import Path

from aka.contracts.content import SourceContent, WorkspaceFile
from .tree import POLICY, import_tree, unpack, validate_import_options


class TreeSourceProvider:
    def __init__(self, options):
        self.options = dict(validate_import_options(options))

    def prepare(self, request=None):
        if request is not None and request.initial is not None:
            if any(item.path == POLICY for item in request.initial.files):
                return SourceContent(request.initial.files)
        options = self.options
        source = import_tree(Path(options["manifest"]).expanduser(),
            Path(options["repository"]).expanduser() if "repository" in options else None,
            directory=Path(options["directory"]).expanduser() if "directory" in options else None)
        policy, files = unpack(source)
        return SourceContent(tuple(WorkspaceFile(name, text=data.decode("utf-8"))
            for name, data in files.items()) + (WorkspaceFile(POLICY,
                text=json.dumps(policy, sort_keys=True) + "\n"),))
