"""Dispatch request data through the preserved kernel and SOL renderers."""
from aka.contracts.workspace import SourceRequest


class LegacySourceProvider:
    def prepare(self, request: SourceRequest):
        if request.initial is not None:
            if request.initial.kind == "kernel":
                from aka.contracts.content import SourceContent
                return SourceContent(request.initial.files)
            if request.initial.kind == "sol":
                from .sol.source import SolSourceProvider
                return SolSourceProvider("", request.name, request.framework, request.platform,
                                         request.gpu_wiki).prepare(request)
            raise ValueError(f"unsupported initial source kind: {request.initial.kind}")
        if request.kind == "kernel":
            from .kernel import KernelSourceProvider
            return KernelSourceProvider(request.kernel_demo).prepare()
        if request.kind == "sol":
            from .sol.source import SolSourceProvider
            return SolSourceProvider(request.operator_dir, request.name, request.framework,
                                     request.platform, request.gpu_wiki, request.definition).prepare()
        raise ValueError("source kind must be kernel or sol")
