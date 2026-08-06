from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from orchestrator import optimize

from .state import PRIVATE_STATE_FILE, read_json_object, write_json_atomic


_CITATION = re.compile(r"\[(E-[A-Z0-9-]+)\]")
_PERFORMANCE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:us|ms|%|x|×|gb/s|tb/s|tflops)\b",
    re.IGNORECASE,
)
_VERIFIED_CLAIM = re.compile(
    r"\b(?:improved|reduced|increased|speedup|faster|effective|verified|提升|降低|加速|有效)\b",
    re.IGNORECASE,
)
_HARDWARE_FACT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:sms?|kb|mb|gb|tb|mhz|ghz|threads?|warps?)\b",
    re.IGNORECASE,
)
_CAUSAL_CLAIM = re.compile(
    r"\b(?:because|due to|caused?|therefore|results? in|led to)\b|由于|因为|导致|因而",
    re.IGNORECASE,
)
_GAP_ASSERTION = re.compile(
    r"\b(?:verified|causal|proven|confirmed|promotion[- ]eligible)\b|已验证|因果|已证明",
    re.IGNORECASE,
)


class DraftValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DraftValidationResult:
    report_path: Path
    documents: tuple[str, ...]


def _safe_document(root: Path, relative: object) -> tuple[str, Path]:
    if not isinstance(relative, str) or "\\" in relative:
        raise DraftValidationError("draft document path must be a POSIX relative string")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DraftValidationError("unsafe draft document path: %r" % relative)
    target = root.joinpath(*path.parts)
    if not target.is_file():
        raise DraftValidationError("draft manifest document is missing: %s" % relative)
    return path.as_posix(), target


def _teacher_source_lines(private: Path) -> set[str]:
    state = read_json_object(private / PRIVATE_STATE_FILE, "private campaign state")
    teacher = Path(state["bundle_path"]).resolve()
    lines: set[str] = set()
    for path in teacher.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            normalized = line.strip()
            if len(normalized) >= 8 and (
                re.search(r"[=(){}\[\];@]", normalized)
                or normalized.startswith(("def ", "class ", "import ", "from "))
            ):
                lines.add(normalized)
    return lines


def _json_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def validate_gap_source_fragments(path: Path, private: Path) -> None:
    gap = read_json_object(path, "Teacher gap analysis")
    teacher_lines = _teacher_source_lines(private)
    for value in _json_strings(gap):
        for line in value.splitlines():
            normalized = line.strip()
            if any(
                fragment in normalized
                or (len(normalized) >= 8 and normalized in fragment)
                for fragment in teacher_lines
            ):
                raise DraftValidationError("Teacher source fragment found in gap JSON")


def _validate_gap(root: Path) -> None:
    gap = read_json_object(root / "teacher_gap_analysis.json", "Teacher gap analysis")
    if gap.get("status") != "hypothesis" or gap.get("promotion_eligible") is not False:
        raise DraftValidationError("Teacher gap analysis must remain a non-promotable hypothesis")
    findings = gap.get("findings")
    if not isinstance(findings, list) or any(
        not isinstance(finding, dict) or finding.get("status") != "hypothesis"
        for finding in findings
    ):
        raise DraftValidationError("every Teacher gap finding must remain a hypothesis")
    for finding in findings:
        if finding.get("promotion_eligible") not in (None, False):
            raise DraftValidationError("Teacher gap findings are not promotion-eligible")
        claim_text = json.dumps(finding, ensure_ascii=False, sort_keys=True)
        if _GAP_ASSERTION.search(claim_text):
            raise DraftValidationError(
                "Teacher gap finding contains a verified or causal assertion"
            )


def validate_gap_markdown(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    if _GAP_ASSERTION.search(text):
        raise DraftValidationError(
            "Teacher gap Markdown must remain hypothesis-only and non-causal"
        )


def _validate_markdown(
    relative: str,
    path: Path,
    evidence: dict[str, dict[str, Any]],
    teacher_lines: set[str],
) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    for line in text.splitlines():
        normalized = line.strip()
        if any(
            fragment in normalized
            or (len(normalized) >= 8 and normalized in fragment)
            for fragment in teacher_lines
        ):
            raise DraftValidationError("Teacher source fragment found in %s" % relative)
    citations = _CITATION.findall(text)
    unknown = sorted(set(citations) - set(evidence))
    if unknown:
        raise DraftValidationError(
            "%s cites unknown evidence IDs: %s" % (relative, ", ".join(unknown))
        )
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph_citations = _CITATION.findall(paragraph)
        if (_PERFORMANCE.search(paragraph) or _HARDWARE_FACT.search(paragraph)) and not paragraph_citations:
            raise DraftValidationError(
                "%s contains a performance or hardware fact without evidence citation" % relative
            )
        if _VERIFIED_CLAIM.search(paragraph) or _CAUSAL_CLAIM.search(paragraph):
            if not any(
                evidence[evidence_id].get("citable_as_verified")
                for evidence_id in paragraph_citations
                if evidence_id in evidence
            ):
                raise DraftValidationError(
                    "%s makes a verified claim without verified evidence" % relative
                )
    if relative.startswith("optimization_cards/"):
        lowered = text.casefold()
        for field in ("architecture:", "framework:", "scope:"):
            if field not in lowered:
                raise DraftValidationError("%s is missing %s metadata" % (relative, field[:-1]))
        if not any(evidence[item].get("citable_as_verified") for item in citations):
            raise DraftValidationError(
                "%s has no verified evidence and is not promotion-eligible" % relative
            )


def validate_distillation_drafts(
    drafts_root: Path | str,
    private_dir: Path | str,
) -> DraftValidationResult:
    root = Path(drafts_root).resolve()
    private = Path(private_dir).resolve()
    canonical_wiki = (optimize.REPO_ROOT / "gpu-wiki").resolve()
    if root == canonical_wiki or canonical_wiki in root.parents:
        raise DraftValidationError("distillation output must not be under canonical gpu-wiki")

    manifest = read_json_object(root / "draft_manifest.json", "draft manifest")
    if manifest.get("schema_version") != 1:
        raise DraftValidationError("unsupported draft manifest schema")
    evidence_level = manifest.get("evidence_level")
    if evidence_level not in {"single-campaign", "reproduced", "cross-shape", "cross-operator"}:
        raise DraftValidationError("unsupported evidence level")
    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise DraftValidationError("draft manifest must list generated documents")
    documents = [_safe_document(root, relative) for relative in raw_documents]
    declared_documents = {relative for relative, _path in documents}
    if len(declared_documents) != len(documents):
        raise DraftValidationError("draft manifest contains duplicate documents")
    allowed_auxiliary = {"teacher_gap_analysis.md", "AUDIT_ONLY.md"}
    generated_markdown = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.md")
        if "evidence" not in path.relative_to(root).parts
    }
    undeclared = sorted(generated_markdown - declared_documents - allowed_auxiliary)
    if undeclared:
        raise DraftValidationError(
            "draft manifest omits generated documents: " + ", ".join(undeclared)
        )

    evidence_manifest = read_json_object(
        root / "evidence" / "evidence_manifest.json",
        "evidence manifest",
    )
    evidence_rows = evidence_manifest.get("evidence")
    if not isinstance(evidence_rows, list):
        raise DraftValidationError("evidence manifest has no evidence list")
    evidence = {
        row["evidence_id"]: row
        for row in evidence_rows
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
    }
    if len(evidence) != len(evidence_rows):
        raise DraftValidationError("evidence manifest contains invalid or duplicate IDs")

    _validate_gap(root)
    validate_gap_source_fragments(root / "teacher_gap_analysis.json", private)
    teacher_lines = _teacher_source_lines(private)
    for relative, path in documents:
        if path.suffix.lower() == ".md":
            _validate_markdown(relative, path, evidence, teacher_lines)
    for gap_name in ("teacher_gap_analysis.md",):
        gap_path = root / gap_name
        if gap_path.is_file():
            validate_gap_markdown(gap_path)
            _validate_markdown(gap_name, gap_path, evidence, teacher_lines)

    report = {
        "schema_version": 1,
        "result": "PASS",
        "evidence_level": evidence_level,
        "documents": sorted(relative for relative, _path in documents),
    }
    report_path = root / "validation_report.json"
    write_json_atomic(report_path, report)
    return DraftValidationResult(report_path, tuple(report["documents"]))
