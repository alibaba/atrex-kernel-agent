#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote


MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
MARKDOWN_QUOTED_TITLE_RE = re.compile(r'^(\S+?)\s+(?:"[^"]*"|\'[^\']*\')$')
FENCED_CODE_LINE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})")
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(/(?:Users|home|root|workspace|private|tmp|opt)/[^\s`'\"\\)>,;:]+)"
)
URL_PREFIXES = ("http://", "https://", "mailto:")
ALLOWED_ABSOLUTE_DIRS = (
    "/opt/rocm",
    "/opt/conda",
    "/tmp/flydsl_bench",
    "/tmp/reference-projects",
    "/tmp/gpu-wiki-upstream",
    "/tmp/gpu-wiki-update-report.md",
    "/tmp/fwd.ttgir",
    "/tmp/bwd.ttgir",
    "/tmp/triton_dump",
    "/tmp/${USER}/cutlass_python_cache",
    "/tmp/{user}/cutlass_python_cache",
    "/tmp/$USER/quack_cache",
)
ALLOWED_ABSOLUTE_PREFIXES = (
    "/opt/rocm-",
    "/tmp/{dsl_name}_python_cache_",
)
WIKI_PATH_MARKERS = (
    "/tmp/gpu-wiki",
    "/root/gpu-wiki",
    "/home/liangyan/gpu-wiki",
    "/Users/liangyan/Program/gpu-wiki",
)
SELF_CHECK_EXCLUDED_RELATIVE_PATHS = {
    Path("scripts/check-self-contained.py"),
    Path("scripts/test_check_self_contained.py"),
}
PERSONAL_PREFIXES = (
    "/Users/",
    "/home/",
    "/root/",
    "/workspace/",
    "/private/",
)
PROVENANCE_MARKERS = (
    "historical provenance",
    "provenance:",
    "provenance path",
    "reference source:",
    "reference-source:",
    "source material:",
    "source-material:",
)
RUNTIME_PROTOCOL_PATTERNS = (
    re.compile(r"\bcase/original\.py\b"),
    re.compile(r"\boutput/optimized\.py\b"),
    re.compile(r"\bgenerated_kernel\.py\b"),
    re.compile(r"\btest_kernel\.py\b"),
    re.compile(r"\bkernel\.py\b.*\b(final|submit|submission|artifact|optimized)\b", re.IGNORECASE),
    re.compile(r"\b(final|submit|submission|artifact|optimized)\b.*\bkernel\.py\b", re.IGNORECASE),
    re.compile(r"\batrex-agent-cli\s+kernel\b"),
    re.compile(r"\bRayJob\b"),
    re.compile(r"\bNormandy\s+(?:task|runtime)\s+path\b", re.IGNORECASE),
    re.compile(r"\bNormandy\s+(?:job|task|runtime|submission|protocol)\b", re.IGNORECASE),
    re.compile(r"\bOSS\s+task\s+path\b", re.IGNORECASE),
    re.compile(r"\bAtrex\s+Server\s+runtime\s+path\b", re.IGNORECASE),
    re.compile(r"~/.atrex-agent-infa\b"),
)
DOWNLOAD_COMMAND_RE = re.compile(r"(?<![A-Za-z0-9_-])(git\s+clone|curl\s+|wget\s+)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IP_CONTEXT_RE = re.compile(r"\b(ip|host|server|endpoint|address|ssh|connect|listen|curl|wget|http://|https://)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net"}
MAINTENANCE_DIR_NAMES = {".skill", "tools", "maintenance"}
SKILL_FRONTMATTER_NAME_RE = re.compile(r"^name:\s*[-A-Za-z0-9_]+\s*$")
SKILL_FRONTMATTER_DESCRIPTION_RE = re.compile(r"^description:\s*.+$")


@dataclass(frozen=True)
class Finding:
    code: str
    path: Path
    line: int
    message: str
    blocking: bool = True

    def format(self, root: Path) -> str:
        try:
            display_path = self.path.relative_to(root)
        except ValueError:
            display_path = self.path
        severity = "ERROR" if self.blocking else "INFO"
        return f"{severity} {self.code} {display_path}:{self.line} {self.message}"


def iter_paths(root: Path):
    for path in sorted(root.rglob("*")):
        if path == root:
            continue
        yield path


def iter_files(root: Path):
    for path in iter_paths(root):
        if path.is_file():
            yield path


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def is_url_or_anchor(target: str) -> bool:
    return target.startswith(URL_PREFIXES) or target.startswith("#")


def has_label_text(label: str) -> bool:
    return any(char.isalnum() for char in label)


def mask_fenced_code_blocks(text: str) -> str:
    masked_lines: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        match = FENCED_CODE_LINE_RE.match(stripped)
        if match:
            current_fence = match.group("fence")
            if not in_fence:
                in_fence = True
                fence_char = current_fence[0]
                fence_len = len(current_fence)
            elif current_fence[0] == fence_char and len(current_fence) >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
            masked_lines.append("\n" if line.endswith("\n") else "")
            continue

        if in_fence:
            newline = "\n" if line.endswith("\n") else ""
            masked_lines.append(newline)
            continue

        masked_lines.append(line)

    return "".join(masked_lines)


def clean_markdown_target(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    else:
        target = MARKDOWN_QUOTED_TITLE_RE.sub(r"\1", target)
    target = target.split("#", 1)[0]
    return unquote(target)


def is_allowed_absolute_path(value: str) -> bool:
    return any(value == path or value.startswith(f"{path}/") for path in ALLOWED_ABSOLUTE_DIRS) or value.startswith(
        ALLOWED_ABSOLUTE_PREFIXES
    )


def classify_absolute_path(value: str) -> Optional[str]:
    if is_allowed_absolute_path(value):
        return None
    if value.startswith(WIKI_PATH_MARKERS) or "/gpu-wiki/" in value:
        return "absolute-wiki-path"
    if value.startswith(PERSONAL_PREFIXES):
        return "personal-absolute-path"
    return "absolute-path"


def is_labeled_provenance(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in PROVENANCE_MARKERS)


def is_self_check_path(root: Path, path: Path) -> bool:
    try:
        relative_path = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return relative_path in SELF_CHECK_EXCLUDED_RELATIVE_PATHS


def is_under_maintenance(root: Path, path: Path) -> bool:
    try:
        relative_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return False
    return bool(relative_parts) and relative_parts[0] in MAINTENANCE_DIR_NAMES


def has_skill_frontmatter(text: str) -> bool:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    has_name = False
    has_description = False
    for line in lines[1:26]:
        if line.strip() == "---":
            return has_name and has_description
        if SKILL_FRONTMATTER_NAME_RE.match(line):
            has_name = True
        if SKILL_FRONTMATTER_DESCRIPTION_RE.match(line):
            has_description = True
    return False


def scan_markdown_links(root: Path, path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in MARKDOWN_LINK_RE.finditer(mask_fenced_code_blocks(text)):
        if not has_label_text(match.group(1)):
            continue
        raw_target = match.group(2)
        if is_url_or_anchor(raw_target.strip()):
            continue
        target = clean_markdown_target(raw_target)
        if not target:
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            continue

        resolved = (path.parent / target).resolve()
        line = line_number(text, match.start())
        try:
            resolved.relative_to(root)
        except ValueError:
            findings.append(
                Finding(
                    "markdown-link-escapes",
                    path,
                    line,
                    f"link target escapes gpu-wiki: {raw_target}",
                )
            )
            continue

        if resolved.is_file():
            continue
        if resolved.is_dir():
            continue
        findings.append(
            Finding(
                "markdown-link-missing",
                path,
                line,
                f"link target does not exist: {raw_target}",
            )
        )
    return findings


def scan_absolute_paths(root: Path, path: Path, text: str) -> list[Finding]:
    if is_self_check_path(root, path):
        return []
    findings: list[Finding] = []
    lines = text.splitlines()
    for match in ABSOLUTE_PATH_RE.finditer(text):
        value = match.group(1).rstrip(".,;:")
        code = classify_absolute_path(value)
        if code is None:
            continue
        line = line_number(text, match.start())
        blocking = True
        if is_labeled_provenance(lines[line - 1]):
            blocking = False
        findings.append(
            Finding(
                code,
                path,
                line,
                f"non-portable absolute path: {value}",
                blocking=blocking,
            )
        )
    return findings


def scan_runtime_protocol(root: Path, path: Path, text: str) -> list[Finding]:
    if is_self_check_path(root, path):
        return []
    findings: list[Finding] = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        for pattern in RUNTIME_PROTOCOL_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        "runtime-protocol",
                        path,
                        line_index,
                        "source wiki must not define Atrex task runtime protocol",
                    )
                )
                break
    return findings


def scan_download_steps(root: Path, path: Path, text: str) -> list[Finding]:
    if is_self_check_path(root, path) or is_under_maintenance(root, path):
        return []
    findings: list[Finding] = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        if DOWNLOAD_COMMAND_RE.search(line) and not is_labeled_provenance(line):
            findings.append(
                Finding(
                    "download-step",
                    path,
                    line_index,
                    "external downloads belong in provenance/reference-source notes, not task steps",
                )
            )
    return findings


def scan_skill_package_docs(root: Path, path: Path, text: str) -> list[Finding]:
    if is_self_check_path(root, path) or is_under_maintenance(root, path):
        return []
    findings: list[Finding] = []
    if path.name == "SKILL.md" or path.name.lower().endswith("-skill.md"):
        findings.append(Finding("skill-package-doc", path, 1, "wiki docs must not be skill package entrypoints"))
        return findings
    if path.suffix.lower() == ".md" and has_skill_frontmatter(text):
        findings.append(Finding("skill-package-doc", path, 1, "wiki docs must not contain skill package frontmatter"))
    if ".skill/" in text:
        line = line_number(text, text.index(".skill/"))
        findings.append(Finding("skill-package-doc", path, line, "wiki docs must not reference skill package paths"))
    return findings


def is_public_ipv4(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(part) for part in parts]
    except ValueError:
        return False
    if any(octet < 0 or octet > 255 for octet in octets):
        return False
    first, second, third, _ = octets
    if first == 10:
        return False
    if first == 172 and 16 <= second <= 31:
        return False
    if first == 192 and second == 168:
        return False
    if first == 127 or first == 0:
        return False
    if first == 169 and second == 254:
        return False
    if first == 100 and 64 <= second <= 127:
        return False
    if first == 192 and second == 0 and third == 2:
        return False
    if first == 198 and second == 51 and third == 100:
        return False
    if first == 203 and second == 0 and third == 113:
        return False
    return True


def is_likely_version_or_section_reference(line: str, start: int, end: int) -> bool:
    previous_char = line[start - 1] if start > 0 else ""
    next_char = line[end] if end < len(line) else ""
    return previous_char in {"[", "/", "_"} or next_char in {"_", "/"}


def scan_personal_info(root: Path, path: Path, text: str) -> list[Finding]:
    if is_self_check_path(root, path):
        return []
    findings: list[Finding] = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        line_without_urls = URL_RE.sub("", line)
        for match in EMAIL_RE.finditer(line_without_urls):
            domain = match.group(1).lower()
            if domain not in ALLOWED_EMAIL_DOMAINS:
                findings.append(Finding("personal-info", path, line_index, "remove personal or internal email address"))
                break

        if PHONE_RE.search(line_without_urls):
            findings.append(Finding("personal-info", path, line_index, "remove personal phone number"))

        if IP_CONTEXT_RE.search(line):
            for match in IPV4_RE.finditer(line):
                value = match.group(0)
                if is_likely_version_or_section_reference(line, match.start(), match.end()):
                    continue
                if is_public_ipv4(value):
                    findings.append(Finding("personal-info", path, line_index, f"remove public IPv4 address: {value}"))
                    break
    return findings


def scan_filenames(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_paths(root):
        if path.name == ".DS_Store":
            findings.append(Finding("os-metadata", path, 1, "remove OS metadata file"))
        if re.search(r"\s", path.name):
            findings.append(Finding("unsafe-filename", path, 1, "filename contains unsafe whitespace"))
    return findings


def scan(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    findings.extend(scan_filenames(root))
    for path in iter_files(root):
        if path.suffix.lower() not in {".md", ".py", ".sh", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".md":
            findings.extend(scan_markdown_links(root, path, text))
        findings.extend(scan_absolute_paths(root, path, text))
        findings.extend(scan_runtime_protocol(root, path, text))
        findings.extend(scan_download_steps(root, path, text))
        findings.extend(scan_skill_package_docs(root, path, text))
        findings.extend(scan_personal_info(root, path, text))
    return findings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check gpu-wiki self-containment.")
    parser.add_argument("--root", default="gpu-wiki", help="Path to the gpu-wiki root.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR root-not-found {root}:1 path does not exist", file=sys.stderr)
        return 1

    findings = scan(root)
    for finding in findings:
        print(finding.format(root.resolve()))

    blocking = [finding for finding in findings if finding.blocking]
    if blocking:
        print(f"Found {len(blocking)} blocking self-containment issue(s).", file=sys.stderr)
        return 1

    print("gpu-wiki self-containment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
