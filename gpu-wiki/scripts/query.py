#!/usr/bin/env python3

"""Architecture-scoped search for the gpu-wiki knowledge tree.

Documents are physically organized by vendor and architecture first, with the
knowledge role (``kernel-opt``, ``ref-docs``, ``pitfalls``, ...) below that
scope.  This tool applies product inheritance and filters the tree before text
ranking so research for one GPU cannot silently consume sibling-product advice.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


ARCH_ALIASES = {
    "ampere": {"ampere", "sm80", "a100"},
    "hopper": {"hopper", "sm90", "h100", "h20", "h200"},
    "blackwell": {"blackwell", "sm100", "sm_100"},
    "b200": {"b200", "gb200"},
    "blackwell-ultra": {"blackwell-ultra", "sm103", "sm_103", "b300", "gb300"},
    "blackwell-geforce": {
        "blackwell-geforce",
        "pro-5000",
        "pro5000",
        "rtx-pro-5000",
        "rtxpro5000",
        "sm-120",
        "sm120",
        "sm_120",
    },
    "cdna3": {"cdna3", "gfx942"},
    "mi300x": {"mi300x", "mi-300x"},
    "mi308x": {"mi308x", "mi-308x"},
    "cdna4": {"cdna4", "gfx950", "mi355x"},
    "rdna4": {"rdna4", "gfx1250"},
}
ARCH_INPUT_ALIASES = {
    alias: canonical
    for canonical, aliases in ARCH_ALIASES.items()
    for alias in aliases
}
# "blackwell" is a family query, while sm100 is the exact B200-era ISA scope.
ARCH_INPUT_ALIASES["blackwell"] = "blackwell-family"
ARCH_QUERY_SCOPES = set(ARCH_ALIASES) | {"blackwell-family"}
ARCH_VENDORS = {
    "ampere": "nvidia",
    "hopper": "nvidia",
    "blackwell": "nvidia",
    "b200": "nvidia",
    "blackwell-ultra": "nvidia",
    "blackwell-geforce": "nvidia",
    "cdna3": "amd",
    "mi300x": "amd",
    "mi308x": "amd",
    "cdna4": "amd",
    "rdna4": "amd",
    "blackwell-family": "nvidia",
}
# Product scopes inherit architecture-general pages, but product-specific pages
# do not enter a sibling product scope. gfx942 is a deliberate family query and
# therefore includes both MI300X and MI308X product pages.
ARCH_PARENTS = {
    "b200": "blackwell",
    "blackwell-ultra": "blackwell",
    "mi300x": "cdna3",
    "mi308x": "cdna3",
}
ARCH_PARENT_CHILDREN = {
    "blackwell": {"b200"},
    "cdna3": {"mi300x", "mi308x"},
}
ARCH_PRODUCT_SCOPES = set(ARCH_PARENTS)
# Truly cross-architecture articles live in a vendor ``common`` directory.  A
# short explicit registry keeps those pages available to the architectures they
# compare without widening them to every architecture from the same vendor.
ARCH_PATH_SCOPES = {
    "nvidia/common/kernel-opt/thread-block-cluster.md": {
        "hopper", "blackwell", "blackwell-geforce",
    },
    "nvidia/common/ref-docs/h100-to-b200-gpgpu-scaling-analysis.md": {
        "hopper", "b200",
    },
    "nvidia/common/ref-docs/ptx-instruction-evolution-a100-h100-b200.md": {
        "ampere", "hopper", "b200",
    },
    "nvidia/common/ref-docs/sglang-hopper-blackwell-backend-selection.md": {
        "hopper", "blackwell",
    },
    "nvidia/common/ref-docs/cutedsl/cutlass-fmha-mla.md": {
        "hopper", "blackwell",
    },
    "nvidia/common/ref-docs/cutedsl/cutlass-quantization-block-scaled.md": {
        "hopper", "blackwell", "blackwell-geforce",
    },
    "nvidia/common/ref-docs/cutedsl/cutlass-tile-scheduling.md": {
        "hopper", "blackwell", "blackwell-geforce",
    },
    "nvidia/common/ref-docs/cutedsl/quack-architecture-overview.md": {
        "hopper", "blackwell", "blackwell-geforce",
    },
    "nvidia/common/ref-docs/cutedsl/quack-gemm-epilogue.md": {
        "hopper", "blackwell", "blackwell-geforce",
    },
    "nvidia/common/ref-docs/gluon/gluon-07-persistent-kernel-pipeline.md": {
        "hopper", "blackwell",
    },
    "nvidia/common/ref-docs/triton/triton-tile-ir-beyond-simt.md": {
        "blackwell", "blackwell-geforce",
    },
}
VENDOR_ALIASES = {"nvidia": {"nvidia"}, "amd": {"amd"}}
DSL_ALIASES = {
    "aiter": {"aiter"},
    "ck-tile": {"ck-tile", "cktile"},
    "cuda": {"cuda"},
    "cutile": {"cutile", "cu-tile"},
    "cutedsl": {"cutedsl", "cute-dsl"},
    "flydsl": {"flydsl"},
    "gluon": {"gluon"},
    "hip": {"hip"},
    "tilelang": {"tilelang"},
    "triton": {"triton"},
}
SECTIONS = {"converter", "hardware-specs", "kernel-opt", "pitfalls", "ref-docs"}
AREAS = {"docs", "reference-kernels"}
REFERENCE_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".sh",
}
WIKI_MANIFEST_NAME = "manifest.json"
REFERENCE_STATUSES = {
    "unclassified",
    "runnable",
    "requires-external-checkout",
    "diagnostic-archive",
    "historical-snapshot",
}
REFERENCE_KINDS = {
    "guide", "kernel", "wrapper", "benchmark", "support", "test", "build", "package",
}
DEFAULT_EXCLUDED_REFERENCE_KINDS = {"test", "build", "package"}
REFERENCE_PRODUCTS = {
    "a100", "h20", "h100", "h200", "b200", "gb200", "b300", "gb300",
    "pro5000", "rtx-pro-5000", "mi300x", "mi308x", "mi355x",
}
# The reference tree predates product overlays in a few places. Explicit
# architecture markers in a path narrow the physical directory scope; otherwise
# the architecture directory is authoritative.
REFERENCE_ARCH_MARKERS = {
    "mi300x": "mi300x",
    "mi308x": "mi308x",
    "mi355x": "cdna4",
    "gfx942": "cdna3",
    "gfx950": "cdna4",
    "gfx1250": "rdna4",
    "sm120": "blackwell-geforce",
    "pro5000": "blackwell-geforce",
    "sm103": "blackwell-ultra",
    "b300": "blackwell-ultra",
    "gb300": "blackwell-ultra",
    "sm100": "blackwell",
    "b200": "b200",
    "gb200": "b200",
    "sm90": "hopper",
    "sm80": "ampere",
}
REFERENCE_DIRECTORY_SCOPES = (
    ("nvidia/blackwell-geforce/", {"blackwell-geforce"}),
    ("nvidia/blackwell-ultra/", {"blackwell-ultra"}),
    ("nvidia/blackwell/", {"blackwell"}),
    ("nvidia/hopper/", {"hopper"}),
    ("nvidia/ampere/", {"ampere"}),
    ("amd/cdna4/", {"cdna4"}),
    ("amd/cdna3/", {"cdna3"}),
    ("amd/cdna/", {"cdna3", "cdna4"}),
    ("amd/rdna4/", {"rdna4"}),
)
SYMPTOMS = {
    "compute-bound": {"compute bound", "compute-bound", "tensor core throughput"},
    "low-sm-utilization": {
        "low sm utilization", "low-sm-utilization", "persistent kernel", "occupancy tuning",
    },
    "memory-bound": {
        "memory bound", "memory-bound", "memory bandwidth", "coalesced access",
    },
    "moe-load-imbalance": {"moe load imbalance", "moe-load-imbalance", "expert load imbalance"},
    "pipeline-stalls": {
        "pipeline stalls", "pipeline-stalls", "software pipeline", "software pipelining",
        "pipeline depth",
    },
    "register-pressure": {
        "register pressure", "register-pressure", "register spill", "vgpr spill",
    },
    "tail-effect": {"tail effect", "tail-effect", "wave quantization"},
}
KERNEL_TYPES = {
    "attention": {"attention", "flash attention", "flash-attention", "flashmla", "mla"},
    "gemm": {"gemm", "matmul", "matrix multiplication"},
    "gemv": {"gemv", "matrix vector"},
    "moe": {"moe", "mixture of experts"},
    "norm": {"norm", "rmsnorm", "layernorm"},
    "reduction": {"reduction", "softmax"},
}
KERNEL_TYPE_INPUT_ALIASES = {"matmul": "gemm", "rmsnorm": "norm", "softmax": "reduction"}
OPERATORS = {
    "activation": {"activation", "silu", "gelu"},
    "allreduce": {"allreduce", "all reduce"},
    "conv": {"conv", "convolution"},
    "cross-entropy": {"cross entropy", "cross-entropy"},
    "elementwise": {"elementwise", "vector add", "vectoradd"},
    "flash-attention": {
        "flash attention", "flash-attention", "flashattention", "flash attn", "fmha",
    },
    "gdn": {"gdn", "gated delta net", "gated-delta-net"},
    "gemm": {"gemm", "matmul"},
    "gemv": {"gemv"},
    "grouped-gemm": {"grouped gemm", "grouped-gemm"},
    "mamba": {"mamba", "state space model", "ssm"},
    "mla": {"mla", "flashmla", "multi head latent attention"},
    "moe": {"moe", "mixture of experts"},
    "norm": {"rmsnorm", "layernorm", "rms norm", "layer norm"},
    "paged-attention": {"paged attention", "paged-attention"},
    "quantization": {"quantization", "quantize", "quant"},
    "rope": {"rope", "rotary"},
    "softmax": {"softmax"},
    "sort": {"sort", "sorting"},
    "topk": {"topk", "top k"},
    "uncategorized": set(),
}
REFERENCE_METADATA_SET_FIELDS = {
    "architectures": set(ARCH_ALIASES),
    "vendors": set(VENDOR_ALIASES),
    "dsls": set(DSL_ALIASES),
    "operators": set(OPERATORS),
    "products": REFERENCE_PRODUCTS,
}
DEFAULT_FUZZY_THRESHOLD = 0.78


@dataclass(frozen=True)
class Page:
    rel_path: str
    title: str
    summary: str
    body: str
    segments: tuple[str, ...]
    stable_text: str
    area: str = "docs"
    architectures: frozenset[str] = frozenset()
    vendors: frozenset[str] = frozenset()
    dsls: frozenset[str] = frozenset()
    operators: frozenset[str] = frozenset()
    products: frozenset[str] = frozenset()
    status: str | None = None
    source: str | None = None
    kind: str = "document"


def _path_words(value: str) -> str:
    return re.sub(r"[/_.-]+", " ", value.lower())


def _title_and_summary(text: str, fallback: str) -> tuple[str, str]:
    lines = text.splitlines()
    title = fallback.replace("-", " ")
    title_index = -1
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# "):
            title = line[2:].strip()
            title_index = index
            break
    summary_lines: list[str] = []
    in_fence = False
    for line in lines[title_index + 1:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            if summary_lines:
                break
            continue
        if stripped.startswith(("#", "|", "- ", "* ", ">")):
            if summary_lines:
                break
            continue
        summary_lines.append(stripped)
        if len(" ".join(summary_lines)) >= 240:
            break
    return title, " ".join(summary_lines)


def load_pages(docs_dir: Path) -> list[Page]:
    pages: list[Page] = []
    defaults, entries = load_docs_manifest(docs_dir)
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name == "README.md":
            continue
        rel_path = path.relative_to(docs_dir).as_posix()
        # Root-level files such as RELATIONS.md are navigation documents, not
        # architecture-scoped search results.
        if "/" not in rel_path or not any(part in SECTIONS for part in Path(rel_path).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        title, summary = _title_and_summary(text, path.stem)
        segments = tuple(part.lower() for part in rel_path.split("/"))
        metadata = _resolved_manifest_metadata(rel_path, defaults, entries)
        architectures = metadata.get("architectures")
        if architectures is None:
            architectures = frozenset(_document_architectures_from_path(rel_path))
        vendors = metadata.get("vendors")
        if vendors is None:
            vendors = frozenset(_reference_path_values(rel_path, VENDOR_ALIASES))
        dsls = metadata.get("dsls")
        if dsls is None:
            dsls = frozenset(_reference_path_values(rel_path, DSL_ALIASES))
        pages.append(Page(
            rel_path=rel_path,
            title=title,
            summary=summary,
            body=text.lower(),
            segments=segments,
            stable_text=f"{title.lower()} {_path_words(rel_path)}",
            area="docs",
            architectures=architectures,
            vendors=vendors,
            dsls=dsls,
            operators=metadata.get("operators", frozenset()),
            source=metadata.get("source"),
            kind="document",
        ))
    return pages


def _manifest_path(value: object, kind: str, directory: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ValueError(f"invalid-reference-manifest invalid-{kind} {value!r}")
    if directory != value.endswith("/"):
        expected = "directory-prefix" if directory else "file-path"
        raise ValueError(f"invalid-reference-manifest {kind}-must-be-{expected} {value}")
    parts = value.rstrip("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid-reference-manifest invalid-{kind} {value}")
    return value


def _manifest_metadata(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid-reference-manifest metadata-not-object {context}")
    allowed = set(REFERENCE_METADATA_SET_FIELDS) | {
        "status", "source", "kind", "searchable",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"invalid-reference-manifest unknown-field {context} {','.join(sorted(unknown))}"
        )
    metadata: dict[str, object] = {}
    for field, valid in REFERENCE_METADATA_SET_FIELDS.items():
        if field not in value:
            continue
        items = value[field]
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"invalid-reference-manifest {field}-must-be-string-list {context}")
        unknown_values = set(items) - valid
        if unknown_values:
            raise ValueError(
                f"invalid-reference-manifest unknown-{field} {context} "
                f"{','.join(sorted(unknown_values))}"
            )
        metadata[field] = frozenset(items)
    if "status" in value:
        status = value["status"]
        if not isinstance(status, str) or status not in REFERENCE_STATUSES:
            raise ValueError(f"invalid-reference-manifest unknown-status {context} {status!r}")
        metadata["status"] = status
    if "source" in value:
        source = value["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"invalid-reference-manifest invalid-source {context}")
        metadata["source"] = source.strip()
    if "kind" in value:
        kind = value["kind"]
        if not isinstance(kind, str) or kind not in REFERENCE_KINDS:
            raise ValueError(f"invalid-reference-manifest unknown-kind {context} {kind!r}")
        metadata["kind"] = kind
    if "searchable" in value:
        searchable = value["searchable"]
        if not isinstance(searchable, bool):
            raise ValueError(
                f"invalid-reference-manifest searchable-must-be-boolean {context}"
            )
        metadata["searchable"] = searchable
    return metadata


def _validate_manifest_scope(path: str, metadata: dict[str, object]) -> None:
    vendors = metadata.get("vendors")
    path_vendor = path.split("/", 1)[0]
    if vendors is not None and path_vendor in VENDOR_ALIASES and vendors != frozenset({path_vendor}):
        raise ValueError(
            f"invalid-reference-manifest vendor-path-conflict {path} "
            f"{','.join(sorted(vendors))}"
        )
    architectures = metadata.get("architectures")
    if architectures and path_vendor in VENDOR_ALIASES:
        architecture_vendors = {ARCH_VENDORS[architecture] for architecture in architectures}
        if architecture_vendors != {path_vendor}:
            raise ValueError(f"invalid-reference-manifest architecture-path-conflict {path}")
    if architectures and vendors:
        architecture_vendors = {ARCH_VENDORS[architecture] for architecture in architectures}
        if not architecture_vendors <= set(vendors):
            raise ValueError(f"invalid-reference-manifest architecture-vendor-conflict {path}")
    dsls = metadata.get("dsls")
    path_dsls = _reference_path_values(path, DSL_ALIASES)
    if dsls is not None and path_dsls and not (set(dsls) & path_dsls):
        raise ValueError(f"invalid-reference-manifest dsl-path-conflict {path}")


def load_manifest_section(
    base_dir: Path,
    section_name: str,
    allowed_suffixes: set[str],
) -> tuple[list[tuple[str, dict[str, object]]], dict[str, dict[str, object]]]:
    """Load one validated area from the top-level wiki manifest."""
    manifest_path = base_dir.parent / WIKI_MANIFEST_NAME
    if not manifest_path.is_file():
        return [], {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid-reference-manifest unreadable {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("invalid-reference-manifest root-not-object")
    unknown = set(manifest) - {"version", "docs", "reference-kernels"}
    if unknown:
        raise ValueError(f"invalid-reference-manifest unknown-root-field {','.join(sorted(unknown))}")
    if manifest.get("version") != 1:
        raise ValueError(f"invalid-reference-manifest unsupported-version {manifest.get('version')!r}")
    section = manifest.get(section_name)
    if section is None:
        return [], {}
    if not isinstance(section, dict):
        raise ValueError(f"invalid-reference-manifest {section_name}-not-object")
    unknown = set(section) - {"defaults", "entries"}
    if unknown:
        raise ValueError(
            f"invalid-reference-manifest unknown-{section_name}-field "
            f"{','.join(sorted(unknown))}"
        )
    raw_defaults = section.get("defaults", [])
    raw_entries = section.get("entries", {})
    if not isinstance(raw_defaults, list):
        raise ValueError("invalid-reference-manifest defaults-not-list")
    if not isinstance(raw_entries, dict):
        raise ValueError("invalid-reference-manifest entries-not-object")

    defaults: list[tuple[str, dict[str, object]]] = []
    seen_prefixes: set[str] = set()
    for index, item in enumerate(raw_defaults):
        if not isinstance(item, dict) or "prefix" not in item:
            raise ValueError(f"invalid-reference-manifest missing-prefix defaults[{index}]")
        prefix = _manifest_path(item["prefix"], f"prefix[{index}]", directory=True)
        if prefix in seen_prefixes:
            raise ValueError(f"invalid-reference-manifest duplicate-prefix {prefix}")
        if not (base_dir / prefix.rstrip("/")).is_dir():
            raise ValueError(f"invalid-reference-manifest unknown-prefix {prefix}")
        metadata = _manifest_metadata(
            {key: value for key, value in item.items() if key != "prefix"}, prefix
        )
        _validate_manifest_scope(prefix, metadata)
        defaults.append((prefix, metadata))
        seen_prefixes.add(prefix)

    entries: dict[str, dict[str, object]] = {}
    for raw_path, raw_metadata in raw_entries.items():
        rel_path = _manifest_path(raw_path, "entry")
        source_path = base_dir / rel_path
        if not source_path.is_file() or source_path.suffix.lower() not in allowed_suffixes:
            raise ValueError(f"invalid-reference-manifest unknown-entry {rel_path}")
        metadata = _manifest_metadata(raw_metadata, rel_path)
        _validate_manifest_scope(rel_path, metadata)
        entries[rel_path] = metadata
    defaults.sort(key=lambda item: len(item[0]))
    return defaults, entries


def load_reference_manifest(
    reference_dir: Path,
) -> tuple[list[tuple[str, dict[str, object]]], dict[str, dict[str, object]]]:
    return load_manifest_section(
        reference_dir, "reference-kernels", REFERENCE_SOURCE_SUFFIXES | {".md"}
    )


def load_docs_manifest(
    docs_dir: Path,
) -> tuple[list[tuple[str, dict[str, object]]], dict[str, dict[str, object]]]:
    return load_manifest_section(docs_dir, "docs", {".md"})


def _resolved_manifest_metadata(
    rel_path: str,
    defaults: list[tuple[str, dict[str, object]]],
    entries: dict[str, dict[str, object]],
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for prefix, prefix_metadata in defaults:
        if rel_path.startswith(prefix):
            metadata.update(prefix_metadata)
    metadata.update(entries.get(rel_path, {}))
    return metadata


def _reference_path_values(rel_path: str, aliases: dict[str, set[str]]) -> set[str]:
    segments = tuple(part.lower() for part in rel_path.split("/"))
    filename = Path(rel_path).name.lower()
    return {
        canonical
        for canonical, tokens in aliases.items()
        if any(token in segments or token in filename for token in tokens)
    }


def _document_architectures_from_path(rel_path: str) -> set[str]:
    values = _reference_path_values(rel_path, ARCH_ALIASES)
    for prefix, architectures in ARCH_PATH_SCOPES.items():
        if rel_path.startswith(prefix):
            values.update(architectures)
    return values


def _reference_architectures_from_path(rel_path: str) -> set[str]:
    compact_path = re.sub(r"[^a-z0-9]+", "", rel_path.lower())
    explicit = {
        architecture
        for marker, architecture in REFERENCE_ARCH_MARKERS.items()
        if marker in compact_path
    }
    if explicit:
        return explicit
    for prefix, architectures in REFERENCE_DIRECTORY_SCOPES:
        if rel_path.startswith(prefix):
            return set(architectures)
    return set()


def _reference_kind(rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    stem = Path(rel_path).stem.lower()
    if name == "__init__.py":
        return "package"
    if stem.startswith("test_") or stem.endswith("_test"):
        return "test"
    if name.endswith(".sh") or stem.startswith("build_"):
        return "build"
    if stem.startswith(("bench", "benchmark")):
        return "benchmark"
    if any(token in stem for token in ("utils", "helper", "common", "config")):
        return "support"
    if any(token in stem for token in ("runner", "dispatch", "wrapper", "entry", "shim")):
        return "wrapper"
    return "kernel"


def _operators_from_title(title: str) -> frozenset[str]:
    normalized = _path_words(title)
    operators = frozenset(
        name
        for name, markers in OPERATORS.items()
        if any(marker in normalized for marker in markers)
    )
    return operators or frozenset({"uncategorized"})


def load_reference_pages(reference_dir: Path) -> list[Page]:
    pages: list[Page] = []
    if not reference_dir.is_dir():
        return pages
    defaults, entries = load_reference_manifest(reference_dir)
    for path in sorted(reference_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(reference_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix not in REFERENCE_SOURCE_SUFFIXES and suffix != ".md":
            continue
        metadata = _resolved_manifest_metadata(rel_path, defaults, entries)
        if suffix == ".md" and not metadata.get("searchable", False):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".md":
            title, summary = _title_and_summary(text, path.stem)
        else:
            title = path.stem.replace("_", " ").replace("-", " ")
            summary = ""
        architectures = metadata.get("architectures")
        if architectures is None:
            architectures = frozenset(_reference_architectures_from_path(rel_path))
        vendors = metadata.get("vendors")
        if vendors is None:
            vendors = frozenset(_reference_path_values(rel_path, VENDOR_ALIASES))
        dsls = metadata.get("dsls")
        if dsls is None:
            dsls = frozenset(_reference_path_values(rel_path, DSL_ALIASES))
        operators = metadata.get("operators")
        if operators is None:
            operators = _operators_from_title(title)
        products = metadata.get("products", frozenset())
        source = metadata.get("source")
        kind = metadata.get("kind", "guide" if suffix == ".md" else _reference_kind(rel_path))
        metadata_terms = set(operators) | set(products)
        metadata_text = " ".join(sorted(metadata_terms))
        pages.append(Page(
            rel_path=rel_path,
            title=title,
            summary=summary or f"{kind.title()} reference ({suffix}).",
            body=text.lower(),
            segments=tuple(part.lower() for part in rel_path.split("/")),
            stable_text=f"{title.lower()} {metadata_text.lower()}",
            area="reference-kernels",
            architectures=architectures,
            vendors=vendors,
            dsls=dsls,
            operators=operators,
            products=products,
            status=metadata.get("status", "unclassified"),
            source=source,
            kind=kind,
        ))
    return pages


def display_path(page: Page) -> str:
    return f"{page.area}/{page.rel_path}"


def reference_architectures(page: Page) -> set[str]:
    return set(page.architectures) or _reference_architectures_from_path(page.rel_path)


def dimension_values(page: Page, aliases: dict[str, set[str]]) -> set[str]:
    if aliases is ARCH_ALIASES:
        return set(page.architectures)
    if aliases is VENDOR_ALIASES:
        return set(page.vendors)
    if aliases is DSL_ALIASES:
        return set(page.dsls)
    return set()


def section_value(page: Page) -> str | None:
    """Return the role component from an architecture-first path."""
    return next((segment for segment in page.segments if segment in SECTIONS), None)


def _expanded_architectures(requested: set[str]) -> set[str]:
    expanded = set(requested)
    if "blackwell-family" in expanded:
        expanded.remove("blackwell-family")
        expanded.update({"blackwell", "b200", "blackwell-ultra"})
    return expanded


def matches_dimension(page: Page, aliases: dict[str, set[str]], requested: set[str]) -> bool:
    if not requested:
        return True
    if aliases is ARCH_ALIASES:
        requested = _expanded_architectures(requested)
    present = dimension_values(page, aliases)
    if aliases is ARCH_ALIASES and present:
        page_products = present & ARCH_PRODUCT_SCOPES
        requested_products = requested & ARCH_PRODUCT_SCOPES
        if page_products:
            if requested_products:
                return bool(page_products & requested_products)
            return any(
                parent in requested and bool(page_products & children)
                for parent, children in ARCH_PARENT_CHILDREN.items()
            )
        if requested_products:
            inherited_parents = {ARCH_PARENTS[product] for product in requested_products}
            return bool(present & inherited_parents)
    return not present or bool(present & requested)


def architecture_relevance(page: Page, requested: set[str]) -> int:
    """Prefer exact target sources, then inherited parents, then generic pages."""
    if not requested:
        return 0
    expanded = _expanded_architectures(requested)
    present = dimension_values(page, ARCH_ALIASES)
    if present & expanded:
        return 3
    if present and matches_dimension(page, ARCH_ALIASES, requested):
        return 2
    return 1 if not present else 0


def kind_relevance(page: Page) -> int:
    return {
        "document": 6,
        "guide": 5,
        "kernel": 5,
        "wrapper": 4,
        "benchmark": 3,
        "support": 2,
        "test": 1,
        "build": 0,
        "package": 0,
    }.get(page.kind, 0)


def classify_stable(page: Page, vocabulary: dict[str, set[str]]) -> set[str]:
    values = {
        name
        for name, markers in vocabulary.items()
        if any(marker in page.stable_text for marker in markers)
    }
    if vocabulary is OPERATORS:
        values.update(page.operators)
    return values


def _fuzzy_candidates(value: str) -> set[str]:
    """Return normalized words and compact adjacent phrases for fuzzy lookup."""
    words = re.findall(r"[a-z0-9]+", value.casefold())
    candidates = set(words)
    for width in (2, 3):
        candidates.update(
            "".join(words[index:index + width])
            for index in range(len(words) - width + 1)
        )
    return candidates


def _fuzzy_similarity(left: str, right: str) -> float:
    """Combine edit-sequence and trigram similarity for operator-name typos."""
    if left == right:
        return 1.0
    sequence = SequenceMatcher(None, left, right).ratio()
    if min(len(left), len(right)) < 3:
        return sequence
    left_trigrams = {left[index:index + 3] for index in range(len(left) - 2)}
    right_trigrams = {right[index:index + 3] for index in range(len(right) - 2)}
    trigram = 2 * len(left_trigrams & right_trigrams) / (
        len(left_trigrams) + len(right_trigrams)
    )
    return max(sequence, trigram)


def _best_fuzzy_similarity(term: str, candidates: set[str]) -> float:
    normalized = "".join(re.findall(r"[a-z0-9]+", term.casefold()))
    if not normalized or not candidates:
        return 0.0
    return max(_fuzzy_similarity(normalized, candidate) for candidate in candidates)


def normalize_query_terms(items: list[str], fuzzy: bool = False) -> list[str]:
    """Normalize exact filename terms while preserving whole fuzzy tokens."""
    return [
        term
        for item in items
        for term in (
            item.casefold().split()
            if fuzzy
            else re.sub(r"[/_.\\-]+", " ", item.casefold()).split()
        )
    ]


def score(
    page: Page,
    terms: list[str],
    match_any: bool,
    fuzzy: bool = False,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[int, float]:
    matched = 0
    total = 0
    best_fuzzy = 0.0
    title = page.title.lower()
    summary = page.summary.lower()
    stable_candidates = _fuzzy_candidates(page.stable_text) if fuzzy else set()
    summary_candidates = _fuzzy_candidates(summary) if fuzzy else set()
    for term in terms:
        if term in title:
            total += 3
            matched += 1
        elif term in summary or term in page.stable_text:
            total += 2
            matched += 1
        elif term in page.body:
            total += 1
            matched += 1
        elif fuzzy:
            stable_similarity = _best_fuzzy_similarity(term, stable_candidates)
            summary_similarity = _best_fuzzy_similarity(term, summary_candidates)
            similarity = max(stable_similarity, summary_similarity)
            if similarity >= fuzzy_threshold:
                total += 2 if stable_similarity >= fuzzy_threshold else 1
                matched += 1
                best_fuzzy = max(best_fuzzy, similarity)
    if matched == 0 or (not match_any and matched != len(terms)):
        return 0, 0.0
    return total, best_fuzzy


def _resolve_many(values: list[str], aliases: dict[str, str], valid: set[str], kind: str) -> set[str]:
    resolved: set[str] = set()
    for raw in values:
        value = re.sub(r"[\s_]+", "-", raw.lower().strip())
        canonical = aliases.get(value, value if value in valid else None)
        if canonical is None:
            raise ValueError(f"unknown-{kind} {raw}")
        resolved.add(canonical)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture-scoped gpu-wiki search")
    parser.add_argument("query", nargs="*", help="Keywords (AND by default).")
    parser.add_argument("--root", default="gpu-wiki")
    parser.add_argument(
        "--area",
        action="append",
        default=[],
        help="Search area: docs or reference-kernels (default: both).",
    )
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--vendor", action="append", default=[])
    parser.add_argument("--dsl", action="append", default=[])
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--exclude-section", action="append", default=[])
    parser.add_argument("--symptom", action="append", default=[])
    parser.add_argument("--kernel-type", action="append", default=[])
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Reference usability status, including unclassified.",
    )
    parser.add_argument(
        "--source", action="append", default=[], help="Reference upstream source/project."
    )
    parser.add_argument(
        "--kind", action="append", default=[], help="Reference role such as kernel or support."
    )
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Include test, build, and package-support files in normal results.",
    )
    parser.add_argument("--any", action="store_true", dest="match_any")
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Allow typo-tolerant matching in titles, paths, and summaries after scope filtering.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help=f"Minimum fuzzy similarity from 0 to 1 (default: {DEFAULT_FUZZY_THRESHOLD}).",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--list-arch", action="store_true")
    parser.add_argument("--list-operators", action="store_true")
    args = parser.parse_args(argv)

    if not 0.0 <= args.fuzzy_threshold <= 1.0:
        parser.error("--fuzzy-threshold must be between 0 and 1")

    if args.list_arch:
        for name, aliases in ARCH_ALIASES.items():
            if name == "blackwell":
                print("blackwell-sm100: sm100, sm_100")
            else:
                print(f"{name}: {', '.join(sorted(aliases))}")
        print("blackwell-family: blackwell")
        return 0
    if args.list_operators:
        print("\n".join(sorted(OPERATORS)))
        return 0

    root = Path(args.root)
    try:
        requested_areas = _resolve_many(args.area, {}, AREAS, "area")
        arches = _resolve_many(args.arch, ARCH_INPUT_ALIASES, ARCH_QUERY_SCOPES, "arch")
        vendors = _resolve_many(args.vendor, {}, set(VENDOR_ALIASES), "vendor")
        dsls = _resolve_many(args.dsl, {}, set(DSL_ALIASES), "dsl")
        sections = _resolve_many(args.section, {}, SECTIONS, "section")
        excluded = _resolve_many(args.exclude_section, {}, SECTIONS, "section")
        symptoms = _resolve_many(args.symptom, {}, set(SYMPTOMS), "symptom")
        kernel_types = _resolve_many(
            args.kernel_type, KERNEL_TYPE_INPUT_ALIASES, set(KERNEL_TYPES), "kernel-type"
        )
        operators = _resolve_many(args.operator, {}, set(OPERATORS), "operator")
        statuses = _resolve_many(args.status, {}, REFERENCE_STATUSES, "status")
        kinds = _resolve_many(args.kind, {}, REFERENCE_KINDS, "kind")
        sources = {
            re.sub(r"[\s_]+", "-", source.casefold().strip())
            for source in args.source
            if source.strip()
        }
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1

    areas = requested_areas or set(AREAS)
    docs_dir = root / "docs"
    if "docs" in areas and not docs_dir.is_dir():
        print(f"ERROR docs-not-found {docs_dir}", file=sys.stderr)
        return 1
    reference_dir = root / "reference-kernels"
    if "reference-kernels" in requested_areas and not reference_dir.is_dir():
        print(f"ERROR reference-kernels-not-found {reference_dir}", file=sys.stderr)
        return 1

    # Architecture-neutral means neutral within a vendor, not across vendors.
    # Infer the architecture vendor when callers omit --vendor so A100/H20 do
    # not retain AMD common pages and MI300X/MI355X do not retain NVIDIA pages.
    if arches and not vendors:
        vendors = {ARCH_VENDORS[architecture] for architecture in arches}

    pages: list[Page] = []
    if "docs" in areas:
        pages.extend(load_pages(docs_dir))
    if "reference-kernels" in areas:
        try:
            pages.extend(load_reference_pages(reference_dir))
        except ValueError as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 1
    known_sources = {
        re.sub(r"[\s_]+", "-", page.source.casefold())
        for page in pages
        if page.source
    }
    unknown_sources = sources - known_sources
    if unknown_sources:
        print(f"ERROR unknown-source {','.join(sorted(unknown_sources))}", file=sys.stderr)
        return 1
    scoped = []
    for page in pages:
        section = section_value(page)
        if not matches_dimension(page, ARCH_ALIASES, arches):
            continue
        if not matches_dimension(page, VENDOR_ALIASES, vendors):
            continue
        if not matches_dimension(page, DSL_ALIASES, dsls):
            continue
        if sections and section not in sections:
            continue
        if excluded and section in excluded:
            continue
        if symptoms and not (classify_stable(page, SYMPTOMS) & symptoms):
            continue
        if kernel_types and not (classify_stable(page, KERNEL_TYPES) & kernel_types):
            continue
        if operators and not (classify_stable(page, OPERATORS) & operators):
            continue
        if statuses and page.status not in statuses:
            continue
        if sources and (
            page.source is None
            or re.sub(r"[\s_]+", "-", page.source.casefold()) not in sources
        ):
            continue
        if kinds and page.kind not in kinds:
            continue
        if (
            not kinds
            and not args.include_auxiliary
            and page.area == "reference-kernels"
            and page.kind in DEFAULT_EXCLUDED_REFERENCE_KINDS
        ):
            continue
        scoped.append(page)

    filters = []
    for name, values in (
        ("area", requested_areas),
        ("arch", arches), ("vendor", vendors), ("dsl", dsls), ("section", sections),
        ("symptom", symptoms), ("kernel-type", kernel_types), ("operator", operators),
        ("status", statuses),
        ("source", sources), ("kind", kinds),
    ):
        if values:
            filters.append(f"{name}={','.join(sorted(values))}")
    print(f"scope: {'; '.join(filters) if filters else 'no filter'} — {len(scoped)}/{len(pages)} pages")

    terms = normalize_query_terms(args.query, fuzzy=args.fuzzy)
    if not terms:
        for page in scoped[:args.limit]:
            labels = []
            if page.source:
                labels.append(f"source={page.source}")
            if page.status and page.status != "unclassified":
                labels.append(f"status={page.status}")
            metadata_label = f" [{'; '.join(labels)}]" if labels else ""
            print(f"  {display_path(page)}{metadata_label} — {page.summary}")
        return 0

    ranked = sorted(
        (
            (
                *score(page, terms, args.match_any, args.fuzzy, args.fuzzy_threshold),
                architecture_relevance(page, arches),
                kind_relevance(page),
                page,
            )
            for page in scoped
        ),
        key=lambda item: (-item[0], -item[2], -item[3], -item[1], item[4].rel_path),
    )
    hits = [
        (rank, similarity, page)
        for rank, similarity, _, _, page in ranked
        if rank
    ]
    print(f"{len(hits)} match(es) for \"{' '.join(terms)}\"")
    for rank, similarity, page in hits[:args.limit]:
        fuzzy_label = f"; fuzzy={similarity:.2f}" if similarity else ""
        status_label = (
            f"; status={page.status}"
            if page.status and page.status != "unclassified"
            else ""
        )
        source_label = f"; source={page.source}" if page.source else ""
        print(
            f"  [{rank}{fuzzy_label}{status_label}{source_label}] "
            f"{display_path(page)} — {page.summary}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
