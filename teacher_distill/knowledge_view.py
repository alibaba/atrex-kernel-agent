from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote

from .models import TeacherProvenance, canonical_json


_MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")


@dataclass(frozen=True)
class KnowledgeView:
    root: Path
    view_hash: str
    included_count: int
    excluded_count: int


def _load_query_module(source_root: Path) -> ModuleType:
    query_path = source_root / "scripts" / "query.py"
    if not query_path.is_file():
        raise ValueError("gpu-wiki query script is missing: scripts/query.py")
    module_name = "_atrex_teacher_distill_gpu_wiki_query"
    spec = importlib.util.spec_from_file_location(module_name, query_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load gpu-wiki query script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _normalized_source(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _visible_markdown(text: str) -> str:
    return _MARKDOWN_LINK.sub(lambda match: match.group(1), text)


def _deny_reason(page: Any, provenance: TeacherProvenance) -> str | None:
    display_path = "%s/%s" % (page.area, page.rel_path)
    normalized_path = display_path.casefold().replace("\\", "/")
    for denied in provenance.knowledge_deny.paths:
        marker = denied.casefold().replace("\\", "/").strip("/")
        if marker and marker in normalized_path:
            return "explicit-path"

    denied_sources = {
        _normalized_source(value)
        for value in (*provenance.knowledge_deny.sources, provenance.source.project)
    }
    page_source = _normalized_source(page.source or "")
    path_components = {_normalized_source(part) for part in normalized_path.split("/")}
    if any(source and (source == page_source or source in path_components) for source in denied_sources):
        return "teacher-source"

    denied_terms = {
        _normalized_words(value)
        for value in (
            provenance.operator.canonical_id,
            *provenance.operator.aliases,
            *provenance.knowledge_deny.tags,
        )
        if _normalized_words(value)
    }
    page_operators = {_normalized_words(value) for value in page.operators}
    if denied_terms & page_operators:
        return "operator-metadata"
    visible = _normalized_words(
        "%s %s %s"
        % (page.rel_path, page.title, _visible_markdown(page.body or ""))
    )
    padded = " %s " % visible
    if any(" %s " % term in padded for term in denied_terms):
        return "operator-identity"
    return None


def _page_metadata(page: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for output, attribute in (
        ("architectures", "architectures"),
        ("vendors", "vendors"),
        ("dsls", "dsls"),
        ("operators", "operators"),
        ("products", "products"),
    ):
        values = sorted(getattr(page, attribute, ()) or ())
        if values and values != ["uncategorized"]:
            metadata[output] = values
    if page.source:
        metadata["source"] = page.source
    if page.area == "reference-kernels":
        metadata["status"] = page.status or "unclassified"
        metadata["kind"] = page.kind
        if Path(page.rel_path).suffix.lower() == ".md":
            metadata["searchable"] = True
    return metadata


def _clean_link_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    elif " " in target:
        target = target.split(" ", 1)[0]
    return unquote(target.split("#", 1)[0])


def _rewrite_missing_links(text: str, source_relative: str, allowed: set[str]) -> str:
    source_parent = Path(source_relative).parent

    def replace(match: re.Match[str]) -> str:
        raw_target = match.group(2).strip()
        if raw_target.startswith("#"):
            return match.group(0)
        if raw_target.startswith(("http://", "https://", "mailto:")):
            return match.group(1)
        target = _clean_link_target(raw_target)
        if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            return match.group(0)
        resolved = (source_parent / target).as_posix()
        normalized = os.path.normpath(resolved).replace(os.sep, "/")
        if normalized in allowed or any(path.startswith(normalized.rstrip("/") + "/") for path in allowed):
            return match.group(0)
        return match.group(1)

    return _MARKDOWN_LINK.sub(replace, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _make_files_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _redacted_exclusion_path(path: str, reason: str) -> str:
    sensitive_reasons = {
        "explicit-path",
        "teacher-source",
        "operator-metadata",
        "operator-identity",
    }
    if reason not in sensitive_reasons:
        return path
    return "redacted:" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def build_knowledge_view(
    source_root: Path | str,
    output_root: Path | str,
    architecture: str,
    framework: str,
    provenance: TeacherProvenance,
) -> KnowledgeView:
    """Create a physical, scoped wiki projection for a hidden-Teacher campaign."""
    source = Path(source_root).resolve()
    destination_base = Path(output_root).resolve()
    if not source.is_dir():
        raise ValueError("gpu-wiki root is missing: %s" % source_root)
    if not (source / "manifest.json").is_file():
        raise ValueError("gpu-wiki manifest.json is missing")
    check_script = source / "scripts" / "check-self-contained.py"
    if not check_script.is_file():
        raise ValueError("gpu-wiki self-containment script is missing")
    if not isinstance(provenance, TeacherProvenance):
        raise ValueError("provenance must be validated before building a knowledge view")

    query = _load_query_module(source)
    try:
        arches = query._resolve_many(
            [architecture], query.ARCH_INPUT_ALIASES, query.ARCH_QUERY_SCOPES, "architecture"
        )
        dsls = query._resolve_many([framework], {}, set(query.DSL_ALIASES), "dsl")
    except ValueError as exc:
        message = str(exc).replace("unknown-arch", "unknown-architecture")
        raise ValueError(message) from exc
    vendors = {query.ARCH_VENDORS[arch] for arch in arches}

    pages = [
        *query.load_pages(source / "docs"),
        *query.load_reference_pages(source / "reference-kernels"),
    ]
    included: list[Any] = []
    excluded: list[dict[str, str]] = []
    for page in pages:
        display_path = "%s/%s" % (page.area, page.rel_path)
        denied = _deny_reason(page, provenance)
        if denied:
            excluded.append(
                {"path": _redacted_exclusion_path(display_path, denied), "reason": denied}
            )
            continue
        if not query.matches_dimension(page, query.ARCH_ALIASES, arches):
            excluded.append({"path": display_path, "reason": "architecture-scope"})
            continue
        if not query.matches_dimension(page, query.VENDOR_ALIASES, vendors):
            excluded.append({"path": display_path, "reason": "vendor-scope"})
            continue
        if not query.matches_dimension(page, query.DSL_ALIASES, dsls):
            excluded.append({"path": display_path, "reason": "dsl-scope"})
            continue
        included.append(page)

    destination_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teacher-knowledge-view-", dir=destination_base) as temp_dir:
        staging = Path(temp_dir)
        selected_paths = {"%s/%s" % (page.area, page.rel_path) for page in included}
        selected_paths.update(
            {
                "README.md",
                "docs/README.md",
                "reference-kernels/README.md",
                "scripts/query.py",
                "scripts/check-self-contained.py",
                "manifest.json",
                "knowledge-view.json",
            }
        )
        for page in included:
            relative = "%s/%s" % (page.area, page.rel_path)
            source_path = source / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source_path.suffix.lower() == ".md":
                content = _rewrite_missing_links(
                    source_path.read_text(encoding="utf-8"), relative, selected_paths
                )
                _write_text(target, content)
            else:
                shutil.copy2(source_path, target)

        (staging / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "scripts" / "query.py", staging / "scripts" / "query.py")
        shutil.copy2(check_script, staging / "scripts" / "check-self-contained.py")
        _write_text(
            staging / "README.md",
            "# Sanitized GPU Wiki\n\n"
            "Read-only architecture-scoped knowledge view for one hidden-audited campaign.\n"
            "Operator-specific Teacher material and external search sources are intentionally absent.\n",
        )
        _write_text(staging / "docs" / "README.md", "# Sanitized documentation\n")
        _write_text(
            staging / "reference-kernels" / "README.md",
            "# Sanitized cross-operator reference kernels\n",
        )

        manifest: dict[str, Any] = {
            "version": 1,
            "docs": {"defaults": [], "entries": {}},
            "reference-kernels": {"defaults": [], "entries": {}},
        }
        for page in included:
            manifest[page.area]["entries"][page.rel_path] = _page_metadata(page)
        _write_text(staging / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        public_exclusions = sorted(excluded, key=lambda row: (row["reason"], row["path"]))
        report = {
            "schema_version": 1,
            "integrity": "hidden-audited",
            "architecture": architecture,
            "framework": framework,
            "included_count": len(included),
            "excluded_count": len(excluded),
            "included": sorted(selected_paths),
            "excluded": public_exclusions,
        }
        _write_text(
            staging / "knowledge-view.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )

        view_hash = _tree_hash(staging)
        final_root = destination_base / view_hash
        if final_root.exists():
            if _tree_hash(final_root) != view_hash:
                raise ValueError("existing knowledge view content does not match its hash")
        else:
            Path(temp_dir).rename(final_root)
        _make_files_read_only(final_root)

    return KnowledgeView(
        root=final_root,
        view_hash=view_hash,
        included_count=len(included),
        excluded_count=len(excluded),
    )
