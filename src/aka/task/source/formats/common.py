"""Paths and deterministic archives for native task candidates."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gzip
import io
import tarfile

SELECTION = '.atrex_file_system.json'
POLICY = ".atrex_source.json"
CACHES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
# Agent runtime state roots are created as *siblings* of the optimizer workspace
# (home.py: workspace.parent/.atrex-agent-homes; session_capture.py:
# cwd.parent/.atrex-session-traces). When a snapshot root sits above them they are
# walked as nested paths whose first segment is not in WORKSPACE_ONLY, so their
# provider files (e.g. .claude/projects/*/memory/MEMORY.md) used to leak into the
# candidate source tree and fail editable_roots validation, aborting the whole run.
# They are pruned at any depth so runtime state can never be mistaken for kernel
# source; this is a collection filter only and leaves validate_files untouched.
AGENT_RUNTIME_ROOTS = {".atrex-agent-homes", ".atrex-session-traces"}
# Controller/public inputs and runtime state are not Candidate source files.
WORKSPACE_ONLY = {"tools", "skills", "scratch", "memory", "input", "profiles", "eval", "plans",
    "reference", "reference-projects", "gpu-wiki", "atrex-bench", "verification_artifacts",
    "README.md", "CLAUDE.md", "WORKSPACE.md", "agent_problem.json", "reference.py", "input.py",
    "shapes.json", "metadata.json", "roofline.json", "valid.py", "definition.json", "workload.jsonl",
    "solution.json", "config.json", "test_kernel.py", "profile_driver.py", "evaluations.jsonl",
    "numerical_probe.py", "numerical_inputs.py", "probe-request.json", "policy_review_request.json",
    ".atrex_context.json", ".gitignore", ".claude", ".codex", ".qoder", ".agents", ".pi", ".orchestrator_mode.json",
    ".aka-agent-observations", ".aka-agent-sessions", ".aka-agent-session-errors", ".humanize",
    ".gpu_wiki_profile", ".atrex_long_horizon", ".atrex_versions", ".atrex_plugins",
    ".atrex_stall_state.json", ".orchestrator_state.json", "framework_baseline.json",
    "baseline_report.md", "trace-retention-manifest.json", ".atrex_environment", ".qodersec", ".venv", "node_modules", POLICY, SELECTION}


def relative(value: str, *, dot=False) -> str:
    if dot and value == ".":
        return value
    if (not isinstance(value, str) or not value or value.startswith(("/", "-"))
            or any(c in value for c in ("\\", "\x00", "\n", "\r"))
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"Unsafe source path: {value!r}")
    return value


def ignored(name):
    parts = Path(name).parts
    return (any(p in CACHES or p in AGENT_RUNTIME_ROOTS for p in parts)
            or Path(name).suffix in {".pyc", ".pyo"})


def source_name(name):
    first = name.split("/")[0]
    return not (first in WORKSPACE_ONLY or ignored(name))


@dataclass(frozen=True)
class ExecutionSource:
    candidate: str | dict
    archives: dict[str, dict[str, str]]
    requirements: tuple[str, ...] = ()


def archive_bytes(files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, text in sorted(files.items()):
                relative(name)
                data = text.encode()
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                tar.addfile(member, io.BytesIO(data))
    return output.getvalue()
