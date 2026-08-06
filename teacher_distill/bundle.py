from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .models import TeacherProvenance, canonical_json


_REQUIRED_FILES = ("kernel.py", "solution.json", "provenance.json")
_ALLOWED_TOP_LEVEL = frozenset({*_REQUIRED_FILES, "helpers"})


@dataclass(frozen=True)
class ValidatedTeacherBundle:
    """Private validated Teacher artifact metadata.

    `root` and `provenance` must never be serialized into Candidate-visible state.
    """

    root: Path
    bundle_hash: str
    teacher_id: str
    provenance: TeacherProvenance
    entry_point: str
    source_paths: tuple[str, ...]

    def public_identity(self) -> str:
        return canonical_json({"teacher_id": self.teacher_id})


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("%s contains duplicate JSON key: %s" % (label, key))
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except UnicodeDecodeError as exc:
        raise ValueError("%s must be UTF-8 JSON" % label) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("%s is not valid JSON: %s" % (label, exc)) from exc
    if not isinstance(value, Mapping):
        raise ValueError("%s must contain a JSON object" % label)
    return value


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _safe_source_path(root: Path, raw: Any) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("solution source path must be a non-empty POSIX relative path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or raw.startswith("/") or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("unsafe solution source path: %s" % raw)
    normalized = relative.as_posix()
    target = root.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError("solution source path escapes the bundle: %s" % raw) from exc
    if not target.is_file():
        raise ValueError("solution source path is not a file: %s" % raw)
    if target.is_symlink():
        raise ValueError("solution source path must not be a symlink: %s" % raw)
    return normalized, target


def _solution_contract(
    root: Path,
    solution: Mapping[str, Any],
    expected_framework: str,
) -> tuple[str, tuple[str, ...]]:
    spec = solution.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("solution.json spec must be an object")
    entry_point = spec.get("entry_point")
    if entry_point not in {"kernel.py::run", "kernel.py::Model"}:
        raise ValueError(
            "solution.json entry_point must be kernel.py::run or kernel.py::Model"
        )

    languages = spec.get("languages")
    if not isinstance(languages, list) or any(not isinstance(item, str) for item in languages):
        raise ValueError("solution.json spec.languages must be a list of strings")
    expected = _normalized_token(expected_framework)
    if expected not in {_normalized_token(item) for item in languages}:
        raise ValueError(
            "solution.json languages do not contain the campaign framework %s" % expected_framework
        )

    sources = solution.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("solution.json sources must be a non-empty list")
    normalized_sources: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or set(source) != {"path"}:
            raise ValueError("solution source %d must contain only a path" % index)
        normalized, _target = _safe_source_path(root, source["path"])
        normalized_sources.append(normalized)
    if len(set(normalized_sources)) != len(normalized_sources):
        raise ValueError("solution source paths must be unique")
    if "kernel.py" not in normalized_sources:
        raise ValueError("solution source paths must include kernel.py")
    return entry_point, tuple(sorted(normalized_sources))


def _bundle_files(root: Path, source_paths: tuple[str, ...]) -> tuple[tuple[str, Path], ...]:
    for child in root.iterdir():
        if child.name not in _ALLOWED_TOP_LEVEL:
            raise ValueError("unsupported top-level Teacher bundle entry: %s" % child.name)
    declared = set(source_paths) | {"solution.json", "provenance.json"}
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("Teacher bundle must not contain symlink: %s" % relative)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Teacher bundle entry is not a regular file: %s" % relative)
        found[relative] = path
    undeclared = sorted(set(found) - declared)
    if undeclared:
        raise ValueError("Teacher bundle files are not declared by solution.json: %s" % ", ".join(undeclared))
    missing = sorted(declared - set(found))
    if missing:
        raise ValueError("Teacher bundle declared files are missing: %s" % ", ".join(missing))
    return tuple(sorted(found.items()))


def _hash_files(files: tuple[tuple[str, Path], ...]) -> str:
    digest = hashlib.sha256()
    for relative, path in files:
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def validate_teacher_bundle(
    bundle_path: Path | str,
    expected_framework: str,
    expected_architecture: str,
) -> ValidatedTeacherBundle:
    """Validate and fingerprint a same-framework, self-contained Teacher bundle."""
    root_input = Path(bundle_path).expanduser()
    if root_input.is_symlink():
        raise ValueError("Teacher bundle root must not be a symlink")
    root = root_input.resolve()
    if not root.is_dir():
        raise ValueError("Teacher bundle directory not found: %s" % root_input)
    for filename in _REQUIRED_FILES:
        required = root / filename
        if not required.is_file() or required.is_symlink():
            raise ValueError("Teacher bundle is missing required file: %s" % filename)

    provenance = TeacherProvenance.from_mapping(
        _load_json_object(root / "provenance.json", "provenance.json")
    )
    if _normalized_token(provenance.target.framework) != _normalized_token(expected_framework):
        raise ValueError(
            "Teacher framework %s does not match campaign framework %s"
            % (provenance.target.framework, expected_framework)
        )
    if _normalized_token(provenance.target.architecture) != _normalized_token(expected_architecture):
        raise ValueError(
            "Teacher architecture %s does not match campaign architecture %s"
            % (provenance.target.architecture, expected_architecture)
        )

    solution = _load_json_object(root / "solution.json", "solution.json")
    entry_point, source_paths = _solution_contract(root, solution, expected_framework)
    files = _bundle_files(root, source_paths)
    bundle_hash = _hash_files(files)
    return ValidatedTeacherBundle(
        root=root,
        bundle_hash=bundle_hash,
        teacher_id="teacher_" + bundle_hash[:24],
        provenance=provenance,
        entry_point=entry_point,
        source_paths=source_paths,
    )
