"""SOL initial files rendered with the original source semantics."""
from dataclasses import dataclass
from pathlib import Path
import json
from .render import _render_kernel, _solution_json, _readme
from aka.contracts.content import SourceContent, WorkspaceFile

@dataclass(frozen=True)
class SolSourceProvider:
    operator_dir: str
    name: str
    framework: str
    platform: str
    gpu_wiki: str = ""
    definition: dict | None = None

    def prepare(self, request=None):
        op = Path(self.operator_dir)
        def read(name):
            if request is not None and request.initial is not None:
                item = next(item for item in request.initial.files if item.path == name)
                return item.source.read_text(encoding="utf-8") if item.source is not None else item.text
            return (op / name).read_text(encoding="utf-8")
        definition = self.definition if self.definition is not None else json.loads(read("definition.json"))
        files = (
            WorkspaceFile("kernel.py", text=_render_kernel(definition, read("reference.py"))),
            WorkspaceFile("solution.json", text=json.dumps(_solution_json(definition, self.name, self.framework, self.platform), indent=2) + "\n"),
        )
        workloads = sum(1 for line in read("workload.jsonl").splitlines() if line.strip())
        return SourceContent(files,
                             _readme(self.name, definition, self.framework, self.platform, self.gpu_wiki, workloads))

