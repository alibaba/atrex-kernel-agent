"""Prepare native candidate manifest content without reading or writing a workspace."""
import json

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import CandidateManifestRequest


def prepare_candidate_manifest(request: CandidateManifestRequest) -> WorkspaceFile:
    framework = request.framework.strip().lower()
    language = {
        "cutedsl": "cutedsl",
        "cuda": "cuda",
        "flydsl": "flydsl",
        "gluon": "gluon",
        "triton": "triton",
    }.get(framework, framework)
    dependencies = {
        "cuda": ["torch", "cuda-python"],
        "triton": ["torch", "triton"],
        "gluon": ["torch", "triton", "gluon"],
        "cutedsl": ["torch", "nvidia-cutlass-dsl"],
        "flydsl": ["torch", "flydsl"],
    }.get(framework, ["torch"])
    reference_source = (
        request.reference_source.decode("utf-8", errors="replace")
        if request.reference_source is not None
        else ""
    )
    entry_symbol = "Model" if "class Model" in reference_source else "run"
    manifest = {
        "name": f"{request.campaign_name}_v1_{language}",
        "spec": {
            "languages": ["python", language],
            "dependencies": dependencies,
            "entry_point": f"kernel.py::{entry_symbol}",
        },
        "sources": [
            {
                "path": "kernel.py",
                "role": (
                    "Single self-contained candidate file. Update this description and "
                    "dependency roles to match the implemented framework loader exactly."
                ),
            }
        ],
        "dependency_roles": {
            dependency: (
                "candidate framework/runtime plumbing; update with the exact non-compute "
                "role used by kernel.py"
            )
            for dependency in dependencies
        },
    }
    return WorkspaceFile(
        "solution.json", text=json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
