from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchestrator.agent_runtime.process import ProcessAccessPolicy


@dataclass(frozen=True)
class TeacherSessionPolicy:
    """Workspace and process policy for a hidden-audited optimization session."""

    knowledge_view: Path
    teacher_solution: Path
    private_root: Path
    source_wiki: Path
    reference_projects: Path

    def __post_init__(self) -> None:
        for name in (
            "knowledge_view",
            "teacher_solution",
            "private_root",
            "source_wiki",
            "reference_projects",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())

    def link_runtime(self, workspace: Path, atrex_bench_root: Path | None = None) -> None:
        # Import lazily to keep the policy model usable without importing the
        # large orchestrator module in evidence-only tooling.
        from orchestrator import optimize

        optimize.link_runtime(
            workspace,
            atrex_bench_root,
            gpu_wiki_root=self.knowledge_view,
            include_reference_projects=False,
            include_kernel_wiki=False,
        )

    def process_access_policy(self) -> ProcessAccessPolicy:
        return ProcessAccessPolicy(
            forbidden_roots=(
                self.teacher_solution,
                self.private_root,
                self.source_wiki,
                self.reference_projects,
            ),
            network_disabled=True,
            audit_log=self.private_root / "audit" / "access-violations.jsonl",
            label="teacher-hidden-audited",
        )

    def filter_prompt(self, prompt: str) -> str:
        """Remove inherited search instructions that violate Teacher isolation."""
        forbidden = (
            "reference-projects",
            "public web",
            "web search",
            "kernelwiki",
            "gpu-wiki/3rdparty",
        )
        lines = [
            line
            for line in prompt.splitlines()
            if not any(term in line.casefold() for term in forbidden)
        ]
        filtered = "\n".join(lines).strip()
        if any(term in filtered.casefold() for term in forbidden):
            raise RuntimeError("Teacher prompt still contains an external-search instruction")
        return filtered + "\n"

    def knowledge_directive(self) -> str:
        return (
            "## Hidden-Teacher knowledge policy (mandatory)\n\n"
            "- This is a `hidden-audited` campaign, not a security sandbox.\n"
            "- Read optimization knowledge only through the workspace `gpu-wiki/`, which is a "
            "physically sanitized, architecture-scoped view.\n"
            "- `reference-projects/` is intentionally unavailable. Do not search for or reconstruct it.\n"
            "- Public web, remote Git operations, downloads, and external source lookup are forbidden.\n"
            "- Do not search outside the current workspace for the operator, Teacher, upstream source, "
            "or alternative copies of gpu-wiki.\n"
            "- A forbidden-path or network attempt terminates the session and invalidates the campaign's "
            "distillation output.\n"
        )
