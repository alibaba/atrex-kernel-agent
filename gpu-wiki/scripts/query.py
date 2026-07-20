#!/usr/bin/env python3

"""Architecture-scoped search for the role-first gpu-wiki knowledge tree.

The main branch organizes documents by knowledge role first (``kernel-opt``,
``ref-docs``, ``pitfalls``, ...), then vendor/DSL/architecture.  This tool
filters that live tree before ranking text matches so research for one GPU does
not silently consume advice for another architecture.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ARCH_ALIASES = {
    "ampere": {"ampere", "sm80", "a100"},
    "hopper": {"hopper", "sm90", "h100", "h20", "h200"},
    "blackwell": {"blackwell", "sm100", "sm103", "b200", "b300"},
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
    "cdna3": {"cdna3", "gfx942", "mi300x", "mi308x"},
    "cdna4": {"cdna4", "gfx950", "mi355x"},
    "rdna4": {"rdna4", "gfx1250"},
}
ARCH_INPUT_ALIASES = {
    alias: canonical
    for canonical, aliases in ARCH_ALIASES.items()
    for alias in aliases
}
# Legacy role-first directories whose README assigns an architecture even though
# the architecture is not encoded in every child filename.
ARCH_PATH_PREFIXES = {
    "kernel-opt/nvidia/common/hands-on/": "blackwell",
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
SYMPTOMS = {
    "compute-bound",
    "low-sm-utilization",
    "memory-bound",
    "moe-load-imbalance",
    "pipeline-stalls",
    "register-pressure",
    "tail-effect",
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
    "allreduce": {"allreduce", "all reduce"},
    "conv": {"conv", "convolution"},
    "flash-attention": {"flash attention", "flash-attention", "flashattention"},
    "gdn": {"gdn", "gated delta net", "gated-delta-net"},
    "gemm": {"gemm", "matmul"},
    "gemv": {"gemv"},
    "grouped-gemm": {"grouped gemm", "grouped-gemm"},
    "mamba": {"mamba", "state space model", "ssm"},
    "mla": {"mla", "flashmla", "multi head latent attention"},
    "moe": {"moe", "mixture of experts"},
    "norm": {"rmsnorm", "layernorm", "rms norm", "layer norm"},
    "paged-attention": {"paged attention", "paged-attention"},
    "softmax": {"softmax"},
}


@dataclass(frozen=True)
class Page:
    rel_path: str
    title: str
    summary: str
    body: str
    segments: tuple[str, ...]
    stable_text: str


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
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name == "README.md":
            continue
        rel_path = path.relative_to(docs_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        title, summary = _title_and_summary(text, path.stem)
        segments = tuple(part.lower() for part in rel_path.split("/"))
        pages.append(Page(
            rel_path=rel_path,
            title=title,
            summary=summary,
            body=text.lower(),
            segments=segments,
            stable_text=f"{title.lower()} {_path_words(rel_path)}",
        ))
    return pages


def dimension_values(page: Page, aliases: dict[str, set[str]]) -> set[str]:
    values: set[str] = set()
    filename = Path(page.rel_path).name.lower()
    if aliases is ARCH_ALIASES:
        values.update(
            architecture
            for prefix, architecture in ARCH_PATH_PREFIXES.items()
            if page.rel_path.startswith(prefix)
        )
    for canonical, tokens in aliases.items():
        # Scope comes from the taxonomy, never prose. In particular, an SM120
        # page may say "Blackwell" in its title but must not enter the B200 scope.
        if any(token in page.segments or token in filename for token in tokens):
            values.add(canonical)
    return values


def matches_dimension(page: Page, aliases: dict[str, set[str]], requested: set[str]) -> bool:
    if not requested:
        return True
    present = dimension_values(page, aliases)
    return not present or bool(present & requested)


def classify_stable(page: Page, vocabulary: dict[str, set[str]]) -> set[str]:
    return {name for name, markers in vocabulary.items() if any(marker in page.stable_text for marker in markers)}


def score(page: Page, terms: list[str], match_any: bool) -> int:
    matched = 0
    total = 0
    title = page.title.lower()
    summary = page.summary.lower()
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
    if matched == 0 or (not match_any and matched != len(terms)):
        return 0
    return total


def _resolve_many(values: list[str], aliases: dict[str, str], valid: set[str], kind: str) -> set[str]:
    resolved: set[str] = set()
    for raw in values:
        value = re.sub(r"[\s_]+", "-", raw.lower().strip())
        canonical = value if value in valid else aliases.get(value)
        if canonical is None:
            raise ValueError(f"unknown-{kind} {raw}")
        resolved.add(canonical)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture-scoped gpu-wiki search")
    parser.add_argument("query", nargs="*", help="Keywords (AND by default).")
    parser.add_argument("--root", default="gpu-wiki")
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--vendor", action="append", default=[])
    parser.add_argument("--dsl", action="append", default=[])
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--exclude-section", action="append", default=[])
    parser.add_argument("--symptom", action="append", default=[])
    parser.add_argument("--kernel-type", action="append", default=[])
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--any", action="store_true", dest="match_any")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--list-arch", action="store_true")
    parser.add_argument("--list-operators", action="store_true")
    args = parser.parse_args(argv)

    if args.list_arch:
        for name, aliases in ARCH_ALIASES.items():
            print(f"{name}: {', '.join(sorted(aliases))}")
        return 0
    if args.list_operators:
        print("\n".join(sorted(OPERATORS)))
        return 0

    docs_dir = Path(args.root) / "docs"
    if not docs_dir.is_dir():
        print(f"ERROR docs-not-found {docs_dir}", file=sys.stderr)
        return 1
    try:
        arches = _resolve_many(args.arch, ARCH_INPUT_ALIASES, set(ARCH_ALIASES), "arch")
        vendors = _resolve_many(args.vendor, {}, set(VENDOR_ALIASES), "vendor")
        dsls = _resolve_many(args.dsl, {}, set(DSL_ALIASES), "dsl")
        sections = _resolve_many(args.section, {}, SECTIONS, "section")
        excluded = _resolve_many(args.exclude_section, {}, SECTIONS, "section")
        symptoms = _resolve_many(args.symptom, {}, SYMPTOMS, "symptom")
        kernel_types = _resolve_many(
            args.kernel_type, KERNEL_TYPE_INPUT_ALIASES, set(KERNEL_TYPES), "kernel-type"
        )
        operators = _resolve_many(args.operator, {}, set(OPERATORS), "operator")
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1

    pages = load_pages(docs_dir)
    scoped = []
    for page in pages:
        section = page.segments[0]
        stem = Path(page.rel_path).stem.lower()
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
        if symptoms and stem not in symptoms:
            continue
        if kernel_types and not (classify_stable(page, KERNEL_TYPES) & kernel_types):
            continue
        if operators and not (classify_stable(page, OPERATORS) & operators):
            continue
        scoped.append(page)

    filters = []
    for name, values in (
        ("arch", arches), ("vendor", vendors), ("dsl", dsls), ("section", sections),
        ("symptom", symptoms), ("kernel-type", kernel_types), ("operator", operators),
    ):
        if values:
            filters.append(f"{name}={','.join(sorted(values))}")
    print(f"scope: {'; '.join(filters) if filters else 'no filter'} — {len(scoped)}/{len(pages)} pages")

    terms = [word.lower() for item in args.query for word in item.split()]
    if not terms:
        for page in scoped[:args.limit]:
            print(f"  docs/{page.rel_path} — {page.summary}")
        return 0

    ranked = sorted(
        ((score(page, terms, args.match_any), page) for page in scoped),
        key=lambda item: (-item[0], item[1].rel_path),
    )
    hits = [(rank, page) for rank, page in ranked if rank]
    print(f"{len(hits)} match(es) for \"{' '.join(terms)}\"")
    for rank, page in hits[:args.limit]:
        print(f"  [{rank}] docs/{page.rel_path} — {page.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
