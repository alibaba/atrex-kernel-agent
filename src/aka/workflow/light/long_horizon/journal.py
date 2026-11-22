from __future__ import annotations


import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PPU_KERNEL_BOUND_SCHEMAS = {
    "ppu-acu-extraction/v4",
    "ppu-acu-comparison/v1",
    "ppu-envelope-measurement/v1",
    "ppu-fixed-slot-receipt/v5",
    "ppu-critical-path-report/v3",
    "ppu-joint-profile/v4",
}
PPU_ARTIFACT_SECTIONS = {
    "ppu-acu-extraction/v4": ("inputs", "bound_artifacts", "outputs"),
    "ppu-acu-comparison/v1": ("incumbent", "candidate"),
    "ppu-envelope-measurement/v1": ("inputs",),
    "ppu-calibration-measurement/v1": ("source_artifacts",),
    "ppu-fixed-slot-receipt/v5": (
        "inputs",
        "provenance",
        "coverage",
        "outputs",
    ),
    "ppu-critical-path-report/v3": ("inputs",),
    "ppu-joint-profile/v4": ("inputs", "outputs"),
}
from aka.resources.bundles import resource_path as resource

PPU_PROFILE_VALIDATOR = resource("skills/ppu-acu-joint-profile/scripts/profile_report.py")




# Reusable receipts/reports are JSON metadata, not raw captures or binaries.
MAX_PPU_REPORT_BYTES = 64 * 1024 * 1024


def _profile_json_snapshot(path: Path) -> tuple[bytearray, str]:
    digest = hashlib.sha256()
    data = bytearray()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if len(data) + len(chunk) > MAX_PPU_REPORT_BYTES:
                raise ValueError("PPU JSON report exceeds 64 MiB")
            digest.update(chunk)
            data.extend(chunk)
    return data, digest.hexdigest()


def _profile_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_nested_json(path: Path) -> object:
    # Sources and loaded binaries need hashes but are not necessarily JSON. Detect
    # JSON objects without depending on an extension or reading a binary wholesale.
    with path.open("rb") as source:
        while chunk := source.read(4096):
            prefix = chunk.lstrip()
            if prefix:
                if not prefix.startswith(b"{"):
                    return None
                break
        else:
            return None
    return json.loads(_profile_json_snapshot(path)[0])


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_profile_reference_path(
    value: object,
    *,
    owner_path: Path,
    workspace: Path,
    label: str,
    errors: list[str],
) -> None:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        errors.append(f"{label} path is missing")
        return
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (owner_path.parent / candidate).resolve()
    )
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        errors.append(f"{label} is outside the workspace or missing")


def _validate_profile_reference_document(
    document: dict[str, Any],
    *,
    owner_path: Path,
    workspace: Path,
    label: str,
    errors: list[str],
) -> None:
    schema = document.get("schema")
    if schema == "ppu-acu-collection/v2":
        for field in (
            "report",
            "producer_artifact",
            "authoritative_kernel",
            "correctness_artifact",
        ):
            _validate_profile_reference_path(
                document.get(field),
                owner_path=owner_path,
                workspace=workspace,
                label=f"{label}.{field}",
                errors=errors,
            )
        for field in ("source_artifacts", "binary_artifacts", "workload_inputs"):
            rows = document.get(field)
            if not isinstance(rows, list):
                errors.append(f"{label}.{field} must be a list")
                continue
            for index, row in enumerate(rows):
                _validate_profile_reference_path(
                    row,
                    owner_path=owner_path,
                    workspace=workspace,
                    label=f"{label}.{field}[{index}]",
                    errors=errors,
                )
    elif schema == "ppu-fixed-slot-timeline-manifest/v5":
        _validate_profile_reference_path(
            document.get("correctness_artifact"),
            owner_path=owner_path,
            workspace=workspace,
            label=f"{label}.correctness_artifact",
            errors=errors,
        )
        provenance = document.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{label}.provenance is missing")
        else:
            _validate_profile_reference_path(
                provenance.get("authoritative_kernel"),
                owner_path=owner_path,
                workspace=workspace,
                label=f"{label}.provenance.authoritative_kernel",
                errors=errors,
            )
            for field in (
                "instrumented_sources",
                "compiled_binaries",
                "workload_inputs",
            ):
                rows = provenance.get(field)
                if not isinstance(rows, list):
                    errors.append(f"{label}.provenance.{field} must be a list")
                    continue
                for index, row in enumerate(rows):
                    _validate_profile_reference_path(
                        row,
                        owner_path=owner_path,
                        workspace=workspace,
                        label=f"{label}.provenance.{field}[{index}]",
                        errors=errors,
                    )
        coverage = document.get("coverage")
        instrumented = coverage.get("instrumented_launch") if isinstance(coverage, dict) else None
        if isinstance(instrumented, dict) and instrumented.get("evidence_artifact") is not None:
            _validate_profile_reference_path(
                instrumented["evidence_artifact"],
                owner_path=owner_path,
                workspace=workspace,
                label=f"{label}.coverage.instrumented_launch.evidence_artifact",
                errors=errors,
            )
        timer_sanity = document.get("timer_sanity")
        if isinstance(timer_sanity, dict):
            _validate_profile_reference_path(
                timer_sanity.get("artifact"),
                owner_path=owner_path,
                workspace=workspace,
                label=f"{label}.timer_sanity.artifact",
                errors=errors,
            )
    elif schema == "ppu-critical-path-plan/v2":
        clean_reference = document.get("clean_reference")
        if isinstance(clean_reference, dict):
            _validate_profile_reference_path(
                clean_reference.get("artifact"),
                owner_path=owner_path,
                workspace=workspace,
                label=f"{label}.clean_reference.artifact",
                errors=errors,
            )


def _validate_profile_artifact_tree(
    value: object,
    *,
    artifact_dir: Path,
    workspace: Path,
    label: str,
    errors: list[str],
    visited: set[Path] | None = None,
) -> None:
    if visited is None:
        visited = set()
    if isinstance(value, dict):
        descriptor_fields = {"path", "sha256", "size_bytes"}
        present_fields = descriptor_fields.intersection(value)
        if present_fields:
            if present_fields != descriptor_fields:
                errors.append(f"{label} has an incomplete artifact descriptor")
                return
            logical_path = value["path"]
            if not isinstance(logical_path, str) or not logical_path:
                errors.append(f"{label}.path must be a non-empty string")
                return
            candidate = Path(logical_path)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (artifact_dir / candidate).resolve()
            )
            if not resolved.is_relative_to(workspace) or not resolved.is_file():
                errors.append(f"{label} is outside the workspace or missing")
                return
            digest = value.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                errors.append(f"{label}.sha256 must be a lowercase SHA-256 digest")
                return
            if _profile_file_hash(resolved) != digest:
                errors.append(f"{label} content hash mismatch")
            if value.get("size_bytes") != resolved.stat().st_size:
                errors.append(f"{label} size mismatch")
            if resolved not in visited:
                visited.add(resolved)
                try:
                    nested_document = _profile_nested_json(resolved)
                except (UnicodeError, json.JSONDecodeError):
                    nested_document = None
                except (OSError, ValueError) as error:
                    errors.append(f"{label} cannot read referenced JSON: {error}")
                    return
                if isinstance(nested_document, dict):
                    _validate_profile_reference_document(
                        nested_document,
                        owner_path=resolved,
                        workspace=workspace,
                        label=label,
                        errors=errors,
                    )
                    nested_schema = nested_document.get("schema")
                    for section in PPU_ARTIFACT_SECTIONS.get(nested_schema, ()):
                        nested_value = nested_document.get(section)
                        if not isinstance(nested_value, (dict, list)):
                            errors.append(f"{label}.{section} is missing")
                            continue
                        _validate_profile_artifact_tree(
                            nested_value,
                            artifact_dir=resolved.parent,
                            workspace=workspace,
                            label=f"{label}.{section}",
                            errors=errors,
                            visited=visited,
                        )
            return
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                _validate_profile_artifact_tree(
                    item,
                    artifact_dir=artifact_dir,
                    workspace=workspace,
                    label=f"{label}.{key}",
                    errors=errors,
                    visited=visited,
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_profile_artifact_tree(
                item,
                artifact_dir=artifact_dir,
                workspace=workspace,
                label=f"{label}[{index}]",
                errors=errors,
                visited=visited,
            )


def validate_accepted_ppu_evidence(
    rows: list[dict[str, Any]], workspace: Path
) -> list[str]:
    """Verify that reusable conclusions bind to decision-grade artifacts."""
    errors: list[str] = []
    workspace = workspace.resolve()
    for index, row in enumerate(rows):
        error_count_before = len(errors)
        label = f"accepted_ppu_diagnostics[{index}].evidence"
        evidence = row["evidence"]
        artifact = Path(evidence["artifact"])
        if artifact.is_absolute():
            errors.append(f"{label}.artifact must be workspace-relative")
            continue
        resolved = (workspace / artifact).resolve()
        if not resolved.is_relative_to(workspace) or not resolved.is_file():
            errors.append(f"{label}.artifact is outside the workspace or missing")
            continue
        try:
            artifact_bytes, digest = _profile_json_snapshot(resolved)
        except (OSError, ValueError) as error:
            errors.append(f"{label} is unreadable: {error}")
            continue
        if digest != evidence["sha256"]:
            errors.append(f"{label} content hash mismatch")
            continue
        try:
            document = json.loads(artifact_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"{label} is not readable JSON: {error}")
            continue
        if not isinstance(document, dict):
            errors.append(f"{label} must reference a JSON object")
            continue
        if document.get("schema") != evidence["schema"]:
            errors.append(f"{label} schema mismatch")
        if document.get("evidence_id") != evidence["evidence_id"]:
            errors.append(f"{label} evidence_id mismatch")
        if document.get("evidence_grade") != "decision":
            errors.append(f"{label} is not decision-grade")
        schema = evidence.get("schema")
        if schema in PPU_KERNEL_BOUND_SCHEMAS:
            kernel_sha256 = document.get("kernel_sha256")
            if not isinstance(kernel_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", kernel_sha256
            ):
                errors.append(f"{label} has no authoritative kernel_sha256")
            binding_payload = document.get("binding_payload")
            if not isinstance(binding_payload, dict):
                errors.append(f"{label} has no binding_payload")
            else:
                binding_id = hashlib.sha256(_canonical_json(binding_payload)).hexdigest()
                if binding_id != document.get("evidence_id"):
                    errors.append(f"{label} binding_payload hash mismatch")
                if binding_payload.get("kernel_sha256") != kernel_sha256:
                    errors.append(f"{label} binding_payload kernel hash mismatch")
            for section in PPU_ARTIFACT_SECTIONS.get(schema, ()):
                value = document.get(section)
                if not isinstance(value, (dict, list)):
                    errors.append(f"{label}.{section} is missing")
                    continue
                _validate_profile_artifact_tree(
                    value,
                    artifact_dir=resolved.parent,
                    workspace=workspace,
                    label=f"{label}.{section}",
                    errors=errors,
                )
            if len(errors) == error_count_before:
                if not PPU_PROFILE_VALIDATOR.is_file():
                    errors.append(f"{label} schema-specific validator is missing")
                else:
                    try:
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(PPU_PROFILE_VALIDATOR),
                                "validate",
                                "--artifact",
                                str(resolved),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired) as error:
                        errors.append(
                            f"{label} schema-specific validation failed: {error}"
                        )
                    else:
                        if result.returncode:
                            detail = (result.stderr or result.stdout).strip().splitlines()
                            errors.append(
                                f"{label} schema-specific validation failed: "
                                + (
                                    detail[-1]
                                    if detail
                                    else f"exit {result.returncode}"
                                )
                            )
        validation = document.get("validation")
        accepted = validation == "accepted" or (
            isinstance(validation, dict) and validation.get("status") == "accepted"
        )
        if not accepted:
            errors.append(f"{label} validation is not accepted")
    return errors
