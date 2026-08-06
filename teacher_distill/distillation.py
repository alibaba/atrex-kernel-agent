from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from orchestrator import optimize
from orchestrator.agent_runtime.process import ProcessAccessPolicy

from .draft_validator import (
    validate_distillation_drafts,
    validate_gap_markdown,
    validate_gap_source_fragments,
)
from .evidence import build_evidence_bundle
from .models import CampaignTerminalStatus, TeacherCampaignResult
from .state import PRIVATE_STATE_FILE, read_json_object, write_json_atomic


AgentRunner = Callable[[Path, str], None]


@dataclass(frozen=True)
class DistillationArtifacts:
    root: Path
    gap_json: Path
    audit_only: bool


def _safe_sources(solution_path: Path) -> tuple[str, ...]:
    if not solution_path.is_file():
        return ("kernel.py",)
    value = json.loads(solution_path.read_text(encoding="utf-8"))
    sources = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(sources, list):
        return ("kernel.py",)
    paths: list[str] = []
    for source in sources:
        raw = source.get("path") if isinstance(source, dict) else None
        if not isinstance(raw, str) or "\\" in raw:
            raise ValueError("invalid solution source path")
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe solution source path")
        paths.append(path.as_posix())
    return tuple(sorted(set(paths) | {"kernel.py", "solution.json"}))


def _copy_solution(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for relative in _safe_sources(source_root / "solution.json"):
        source = source_root / relative
        if not source.is_file():
            if relative == "solution.json":
                continue
            raise ValueError("solution source is missing: %s" % relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _default_runner(
    agent_cli: str,
    timeout: int,
    *,
    forbidden_roots: tuple[Path, ...],
) -> AgentRunner:
    def run(workspace: Path, prompt: str) -> None:
        result = optimize.run_session(
            workspace,
            prompt,
            timeout=timeout,
            agent_cli=agent_cli,
            reasoning_effort="high",
            access_policy=ProcessAccessPolicy(
                forbidden_roots=forbidden_roots,
                network_disabled=True,
                label="teacher-distillation-post-run",
            ),
        )
        if result.exit_status != 0 or result.timed_out:
            raise RuntimeError(
                "distillation Agent failed: exit=%d timeout=%s %s"
                % (result.exit_status, result.timed_out, result.stderr_tail)
            )

    return run


def _validate_gap_contract(path: Path) -> None:
    value = read_json_object(path, "Teacher gap analysis")
    if value.get("schema_version") != 1:
        raise RuntimeError("Teacher gap analysis has unsupported schema")
    if value.get("status") != "hypothesis" or value.get("promotion_eligible") is not False:
        raise RuntimeError("Teacher gap analysis must remain a non-promotable hypothesis")
    findings = value.get("findings")
    if not isinstance(findings, list) or any(
        not isinstance(finding, dict) or finding.get("status") != "hypothesis"
        for finding in findings
    ):
        raise RuntimeError("every Teacher gap finding must be a hypothesis")
    forbidden_assertions = (
        "verified",
        "causal",
        "proven",
        "confirmed",
        "promotion-eligible",
        "promotion eligible",
        "已验证",
        "因果",
        "已证明",
    )
    for finding in findings:
        if finding.get("promotion_eligible") not in (None, False):
            raise RuntimeError("Teacher gap findings are not promotion-eligible")
        rendered = json.dumps(finding, ensure_ascii=False, sort_keys=True).casefold()
        if any(term in rendered for term in forbidden_assertions):
            raise RuntimeError("Teacher gap finding contains a verified or causal assertion")


def generate_distillation(
    candidate_workspace: Path | str,
    private_dir: Path | str,
    result: TeacherCampaignResult,
    *,
    agent_runner: AgentRunner | None = None,
    agent_cli: str = "claude",
    timeout: int = 1800,
) -> DistillationArtifacts:
    candidate = Path(candidate_workspace).resolve()
    private = Path(private_dir).resolve()
    output = private / "distillation" / "drafts"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_json_atomic(output / "campaign_result.json", result.to_mapping())

    if result.status == CampaignTerminalStatus.TEACHER_LEAKAGE_VIOLATION:
        (output / "AUDIT_ONLY.md").write_text(
            "# Audit-only result\n\n"
            "This campaign recorded a Teacher leakage policy violation. Its optimization "
            "findings are not eligible for distillation or wiki promotion.\n",
            encoding="utf-8",
        )
        return DistillationArtifacts(output, output / "teacher_gap_analysis.json", True)

    evidence = build_evidence_bundle(candidate, private)
    shutil.copytree(evidence.root, output / "evidence", dirs_exist_ok=True)

    private_state = read_json_object(private / PRIVATE_STATE_FILE, "private campaign state")
    teacher_root = Path(private_state["bundle_path"]).resolve()
    gap_prompt = (Path(__file__).parent / "prompts" / "gap-analysis.md").read_text(
        encoding="utf-8"
    )
    distill_prompt = (Path(__file__).parent / "prompts" / "distill.md").read_text(
        encoding="utf-8"
    )

    # Neither post-run Agent executes below the campaign-private tree. The gap
    # Agent receives explicit Teacher/Candidate copies; the distillation Agent
    # receives only evidence and hypothesis-only gap artifacts. Both staging
    # workspaces are deleted before this function returns.
    with tempfile.TemporaryDirectory(prefix="atrex-teacher-distill-") as staging_dir:
        staging = Path(staging_dir).resolve()
        gap_workspace = staging / "gap"
        gap_workspace.mkdir()
        _copy_solution(teacher_root, gap_workspace / "teacher")
        _copy_solution(candidate, gap_workspace / "candidate")
        gap_runner = agent_runner or _default_runner(
            agent_cli,
            timeout,
            forbidden_roots=(
                private,
                candidate,
                teacher_root,
                optimize.REPO_ROOT / "gpu-wiki",
                optimize.REPO_ROOT / "reference-projects",
            ),
        )
        gap_runner(gap_workspace, gap_prompt)
        gap_json_source = gap_workspace / "teacher_gap_analysis.json"
        gap_markdown_source = gap_workspace / "teacher_gap_analysis.md"
        _validate_gap_contract(gap_json_source)
        validate_gap_source_fragments(gap_json_source, private)
        if not gap_markdown_source.is_file():
            raise RuntimeError("gap-analysis Agent did not create teacher_gap_analysis.md")
        validate_gap_markdown(gap_markdown_source)

        gap_json = output / "teacher_gap_analysis.json"
        shutil.copy2(gap_json_source, gap_json)
        shutil.copy2(gap_markdown_source, output / "teacher_gap_analysis.md")

        distill_workspace = staging / "drafts"
        distill_workspace.mkdir()
        shutil.copy2(output / "campaign_result.json", distill_workspace)
        shutil.copytree(evidence.root, distill_workspace / "evidence")
        shutil.copy2(gap_json_source, distill_workspace / "teacher_gap_analysis.json")
        shutil.copy2(gap_markdown_source, distill_workspace / "teacher_gap_analysis.md")
        distill_runner = agent_runner or _default_runner(
            agent_cli,
            timeout,
            forbidden_roots=(
                private,
                candidate,
                teacher_root,
                gap_workspace,
                optimize.REPO_ROOT / "gpu-wiki",
                optimize.REPO_ROOT / "reference-projects",
            ),
        )
        distill_runner(distill_workspace, distill_prompt)
        required = (
            "journey.md",
            "pitfalls.md",
            "promotion_checklist.md",
            "draft_manifest.json",
        )
        missing = [name for name in required if not (distill_workspace / name).is_file()]
        if missing:
            raise RuntimeError("distillation Agent did not create: " + ", ".join(missing))

        if result.status == CampaignTerminalStatus.SUCCESS:
            _copy_solution(candidate, distill_workspace / "reference_kernel")
        validate_distillation_drafts(distill_workspace, private)
        shutil.copytree(distill_workspace, output, dirs_exist_ok=True)

    return DistillationArtifacts(output, gap_json, False)
