#!/usr/bin/env python3

"""Structural / semantic lint for the gpu-wiki knowledge base.

Complements ``check-self-contained.py`` (which guarantees link integrity and
self-containment). This script flags conventions that the page schema in
``CLAUDE.md`` asks for but that cannot be checked structurally by the
self-containment gate:

  - missing-h1            page has no leading ``# H1`` title
  - missing-summary       no one-line summary under the title
  - missing-related       no ``## Related`` section (legacy names accepted)
  - orphan-page           no inbound relative link except the generated index
  - relations-staleness   pages are newer than anything noted in RELATIONS.md

Findings are warnings (exit 0) by default. Pass ``--strict`` to make any finding
fail the run; that is the intended end state once the normalization sweep lands.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import build_index  # same directory; reuse the index extractors so lint and index agree


EXCLUDED_NAMES = {"README.md", "index.md"}
RELATIONS_NAME = "RELATIONS.md"
INDEX_REL = "docs/index.md"

# Objective, universal conventions every page must satisfy — these gate under
# --strict. The rest (missing-related, orphan-page, relations-staleness) are
# advisory: useful signals, but not worth blocking a PR over, so they never gate.
GATING_CODES = {"missing-h1", "missing-summary"}

H1_RE = re.compile(r"^#\s+\S")
RELATED_RE = re.compile(r"^##\s+Related(\s+Docs|\s+Documents)?\s*$", re.IGNORECASE)
LAST_UPDATED_RE = re.compile(r"\*\*Last updated\*\*:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
URL_PREFIXES = ("http://", "https://", "mailto:")


@dataclass(frozen=True)
class Finding:
    code: str
    path: Path
    line: int
    message: str


def has_h1(lines: list[str]) -> bool:
    return any(H1_RE.match(line) for line in build_index.mask_fences(lines))


def has_summary(lines: list[str]) -> bool:
    # A page "has a summary" iff build_index can extract one. Reusing the index
    # extractor (with the same fence masking) keeps the lint and the generated
    # index.md in agreement.
    masked = build_index.mask_fences(lines)
    _title, title_index = build_index.extract_title(masked, "")
    return build_index.extract_summary(masked, title_index) != "(no summary)"


def has_related(lines: list[str]) -> bool:
    return any(RELATED_RE.match(line) for line in lines)


def iter_content_pages(docs_dir: Path):
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name in EXCLUDED_NAMES:
            continue
        yield path


def relative_link_targets(path: Path, text: str) -> list[Path]:
    targets: list[Path] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        raw = match.group(2).strip()
        if raw.startswith(URL_PREFIXES) or raw.startswith("#"):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
            continue
        target = raw.split("#", 1)[0].strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if not target:
            continue
        targets.append((path.parent / target).resolve())
    return targets


def build_inbound_counts(root: Path) -> dict[Path, int]:
    """Count inbound relative links to each file, ignoring docs/index.md."""
    index_path = (root / INDEX_REL).resolve()
    counts: dict[Path, int] = {}
    for path in sorted(root.rglob("*.md")):
        if path.resolve() == index_path:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in relative_link_targets(path, text):
            counts[target] = counts.get(target, 0) + 1
    return counts


def latest_date(text: str) -> Optional[str]:
    dates = DATE_RE.findall(text)
    return max(dates) if dates else None


def scan(root: Path) -> list[Finding]:
    docs_dir = root / "docs"
    findings: list[Finding] = []
    inbound = build_inbound_counts(root)

    newest_page_date: Optional[str] = None
    for path in iter_content_pages(docs_dir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        if not has_h1(lines):
            findings.append(Finding("missing-h1", path, 1, "page has no leading '# H1' title"))
        if not has_summary(lines):
            findings.append(Finding("missing-summary", path, 1, "no one-line summary under the title"))
        if path.name != RELATIONS_NAME and not has_related(lines):
            findings.append(Finding("missing-related", path, 1, "no '## Related' section"))
        if inbound.get(path.resolve(), 0) == 0:
            findings.append(
                Finding("orphan-page", path, 1, "no inbound relative link except the generated index")
            )

        match = LAST_UPDATED_RE.search(text)
        if match:
            date = match.group(1)
            if newest_page_date is None or date > newest_page_date:
                newest_page_date = date

    relations_path = docs_dir / RELATIONS_NAME
    if newest_page_date and relations_path.is_file():
        relations_date = latest_date(relations_path.read_text(encoding="utf-8", errors="ignore"))
        if relations_date is not None and newest_page_date > relations_date:
            findings.append(
                Finding(
                    "relations-staleness",
                    relations_path,
                    1,
                    f"pages updated through {newest_page_date} but RELATIONS.md latest date is {relations_date}; "
                    "review whether new cross-arch relations need recording",
                )
            )
    return findings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Structural lint for gpu-wiki docs.")
    parser.add_argument("--root", default="gpu-wiki", help="Path to the gpu-wiki root.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat findings as errors (exit 1). Default reports warnings and exits 0.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not (root / "docs").is_dir():
        print(f"ERROR docs-not-found {root / 'docs'}:1 path does not exist", file=sys.stderr)
        return 1

    findings = scan(root.resolve())
    resolved_root = root.resolve()
    gating: list[Finding] = []
    for finding in findings:
        try:
            display = finding.path.relative_to(resolved_root)
        except ValueError:
            display = finding.path
        is_gating = args.strict and finding.code in GATING_CODES
        if is_gating:
            gating.append(finding)
        severity = "ERROR" if is_gating else "WARN"
        print(f"{severity} {finding.code} {display}:{finding.line} {finding.message}")

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.code] = counts.get(finding.code, 0) + 1
    summary = ", ".join(f"{code}={count}" for code, count in sorted(counts.items())) or "none"
    print(f"structure findings: {summary}")

    if gating:
        print(f"Found {len(gating)} gating structural issue(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
