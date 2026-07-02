#!/usr/bin/env python3

"""Architecture-scoped keyword search over the gpu-wiki docs.

The wiki is organized vendor-first (`docs/{generic,nvidia,amd}/`) with NVIDIA
split by architecture (`hopper`/`blackwell`/`blackwell-geforce`) and AMD by
`gfx942`/`gfx950`. That taxonomy lives in each file's path, so a scoped search is
a path filter plus keyword ranking. The point is isolation: a ``--arch blackwell``
query returns Blackwell (and architecture-neutral) pages and never leaks Hopper,
CDNA, or Blackwell-GeForce (sm120) results.

Examples:
    python3 gpu-wiki/scripts/query.py "bank conflict" --arch blackwell
    python3 gpu-wiki/scripts/query.py "flash attention" --arch cdna3 --dsl flydsl
    python3 gpu-wiki/scripts/query.py --list-arch

Matching is by path *segment* (a directory name) or a filename substring, so
`blackwell` matches `nvidia/blackwell/...` but not `nvidia/blackwell-geforce/...`.
A page is kept for a requested filter value when it carries that value's token OR
carries no token from that dimension at all (neutral / cross-arch pages such as
`nvidia/common/...` or `RELATIONS.md`). It is dropped only when it belongs to a
*different* value of the same dimension. `generic` pages are vendor-neutral and
survive any `--vendor` filter.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import build_index  # same directory; reuse title/summary extraction + fence masking


# canonical value -> tokens identifying it as a path segment or filename substring
ARCH_ALIASES = {
    "hopper": {"hopper", "sm90", "h100", "h20", "h200"},
    "blackwell": {"blackwell", "sm100", "sm103", "b200", "b300"},
    "blackwell-geforce": {"blackwell-geforce", "sm120"},
    "cdna3": {"cdna3", "gfx942", "mi300x", "mi308x"},
    "cdna4": {"cdna4", "gfx950", "mi355x"},
    "rdna4": {"rdna4", "gfx1250"},
    "ampere": {"ampere", "sm80", "a100"},
}
# `generic` is intentionally omitted: generic pages are vendor-neutral and should
# survive any --vendor filter rather than being treated as a competing vendor.
VENDOR_ALIASES = {
    "nvidia": {"nvidia"},
    "amd": {"amd"},
}
DSL_ALIASES = {
    "cutedsl": {"cutedsl"},
    "flydsl": {"flydsl"},
    "gluon": {"gluon"},
    "triton": {"triton"},
    "cuda": {"cuda"},
}

# non-canonical spellings a user might pass on the command line
ARCH_INPUT_ALIASES = {
    "sm90": "hopper",
    "h100": "hopper",
    "h20": "hopper",
    "sm100": "blackwell",
    "sm103": "blackwell",
    "b200": "blackwell",
    "b300": "blackwell",
    "geforce": "blackwell-geforce",
    "sm120": "blackwell-geforce",
    "gfx942": "cdna3",
    "mi300x": "cdna3",
    "mi308x": "cdna3",
    "gfx950": "cdna4",
    "mi355x": "cdna4",
    "gfx1250": "rdna4",
}

TITLE_WEIGHT = 3
SUMMARY_WEIGHT = 2
BODY_WEIGHT = 1


@dataclass
class Page:
    rel_path: str  # relative to docs/
    title: str
    summary: str
    segments: tuple[str, ...]  # path components (dirs + filename), lowercased
    filename: str  # lowercased filename
    keyword_blob: str  # path words for keyword scoring
    body: str


def path_segments(rel_path: str) -> tuple[str, ...]:
    return tuple(part for part in rel_path.lower().split("/") if part)


def keyword_blob(rel_path: str) -> str:
    text = rel_path.lower()
    for ch in "/-_.":
        text = text.replace(ch, " ")
    return text


def dimension_values(page: "Page", aliases: dict[str, set[str]]) -> set[str]:
    """Which canonical values of a dimension does this page belong to?"""
    found = set()
    for value, value_tokens in aliases.items():
        if any(token in page.segments or token in page.filename for token in value_tokens):
            found.add(value)
    return found


def matches_dimension(page: "Page", aliases: dict[str, set[str]], requested: set[str]) -> bool:
    if not requested:
        return True
    present = dimension_values(page, aliases)
    if not present:
        return True  # neutral page (no token in this dimension) — never excluded
    return bool(present & requested)


def load_pages(docs_dir: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name in {"README.md", "index.md"}:
            continue
        rel = path.relative_to(docs_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        masked = build_index.mask_fences(text.splitlines())
        title, title_index = build_index.extract_title(masked, path.stem)
        summary = build_index.extract_summary(masked, title_index)
        pages.append(
            Page(
                rel_path=rel,
                title=title,
                summary=summary,
                segments=path_segments(rel),
                filename=path.name.lower(),
                keyword_blob=keyword_blob(rel),
                body=text.lower(),
            )
        )
    return pages


def resolve_arch(value: str) -> Optional[str]:
    value = value.lower()
    if value in ARCH_ALIASES:
        return value
    return ARCH_INPUT_ALIASES.get(value)


def score_page(page: Page, terms: list[str], match_any: bool) -> int:
    title_l = page.title.lower()
    summary_l = page.summary.lower()
    matched_terms = 0
    score = 0
    for term in terms:
        if term in title_l:
            score += TITLE_WEIGHT
            matched_terms += 1
        elif term in summary_l or term in page.keyword_blob:
            score += SUMMARY_WEIGHT
            matched_terms += 1
        elif term in page.body:
            score += BODY_WEIGHT
            matched_terms += 1
    if matched_terms == 0:
        return 0
    if not match_any and matched_terms < len(terms):
        return 0  # AND semantics: every term must appear somewhere
    return score


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture-scoped search over gpu-wiki docs.")
    parser.add_argument("query", nargs="*", help="Keywords to search for.")
    parser.add_argument("--root", default="gpu-wiki", help="Path to the gpu-wiki root.")
    parser.add_argument("--arch", action="append", default=[], help="Restrict to an architecture (repeatable).")
    parser.add_argument("--vendor", action="append", default=[], help="Restrict to nvidia / amd (repeatable).")
    parser.add_argument("--dsl", action="append", default=[], help="Restrict to cutedsl / flydsl / gluon / triton / cuda.")
    parser.add_argument("--any", dest="match_any", action="store_true", help="Match any keyword (default: all).")
    parser.add_argument("--limit", type=int, default=20, help="Maximum results to print.")
    parser.add_argument("--list-arch", action="store_true", help="List known architecture values and exit.")
    args = parser.parse_args(argv)

    if args.list_arch:
        for value, tokens in ARCH_ALIASES.items():
            print(f"{value}: {', '.join(sorted(tokens))}")
        return 0

    docs_dir = Path(args.root) / "docs"
    if not docs_dir.is_dir():
        print(f"ERROR docs-not-found {docs_dir}:1 path does not exist", file=sys.stderr)
        return 1

    requested_arch: set[str] = set()
    for value in args.arch:
        resolved = resolve_arch(value)
        if resolved is None:
            print(f"ERROR unknown-arch {value}: try --list-arch", file=sys.stderr)
            return 1
        requested_arch.add(resolved)
    requested_vendor = {v.lower() for v in args.vendor}
    requested_dsl = {v.lower() for v in args.dsl}

    pages = load_pages(docs_dir)
    scoped = [
        page
        for page in pages
        if matches_dimension(page, ARCH_ALIASES, requested_arch)
        and matches_dimension(page, VENDOR_ALIASES, requested_vendor)
        and matches_dimension(page, DSL_ALIASES, requested_dsl)
    ]

    filters = []
    if requested_arch:
        filters.append(f"arch={','.join(sorted(requested_arch))}")
    if requested_vendor:
        filters.append(f"vendor={','.join(sorted(requested_vendor))}")
    if requested_dsl:
        filters.append(f"dsl={','.join(sorted(requested_dsl))}")
    scope_desc = "; ".join(filters) if filters else "no filter"
    print(f"scope: {scope_desc} — {len(scoped)}/{len(pages)} pages in scope")

    # Split every positional arg on whitespace so a quoted multi-word query
    # ("moe gemm") behaves the same as separate tokens (moe gemm) — each word is
    # an independent keyword (AND by default, OR with --any), not a literal phrase.
    terms = [word.lower() for arg in args.query for word in arg.split()]
    if not terms:
        print("(no keywords given; listing in-scope pages)")
        for page in scoped[: args.limit]:
            print(f"  docs/{page.rel_path} — {page.summary}")
        return 0

    ranked = sorted(
        ((score_page(page, terms, args.match_any), page) for page in scoped),
        key=lambda item: (-item[0], item[1].rel_path),
    )
    hits = [(score, page) for score, page in ranked if score > 0]
    print(f'{len(hits)} match(es) for "{" ".join(terms)}"')
    for score, page in hits[: args.limit]:
        print(f"  [{score}] docs/{page.rel_path} — {page.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
