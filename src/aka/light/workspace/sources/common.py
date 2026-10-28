"""Format-neutral paths, file I/O and execution value objects."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gzip,io,tarfile,subprocess,os
from aka.light.workspace.files import publish
from aka.light.agent.session_tail import read_regular_bytes
SELECTION = '.atrex_file_system.json'
POLICY = ".atrex_source.json"
MAGIC = b"ATREX-SOURCE-TREE-v1\n"
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_SNAPSHOT = 6 * MAX_TOTAL
CACHES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
PROTECTED = {"atrex_bench", "torch", "cutlass", "cuda", "triton", "site", "sitecustomize",
             "usercustomize", "importlib", "packaging", "json", "subprocess", "os", "sys"}
# Controller/public inputs and runtime state are not Candidate source files.
WORKSPACE_ONLY = {"tools", "skills", "scratch", "memory", "input", "profiles", "eval", "plans",
    "reference", "reference-projects", "gpu-wiki", "atrex-bench", "verification_artifacts",
    "README.md", "CLAUDE.md", "WORKSPACE.md", "agent_problem.json", "reference.py", "input.py",
    "shapes.json", "metadata.json", "roofline.json", "valid.py", "definition.json", "workload.jsonl",
    "solution.json", "config.json", "test_kernel.py", "profile_driver.py", "evaluations.jsonl",
    "numerical_probe.py", "numerical_inputs.py", "probe-request.json", "policy_review_request.json",
    ".gitignore", ".claude", ".codex", ".qoder", ".agents", ".pi", ".orchestrator_mode.json",
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
    return any(p in CACHES for p in Path(name).parts) or Path(name).suffix in {".pyc", ".pyo"}

def source_name(name):
    first = name.split("/")[0]
    return not (first in WORKSPACE_ONLY or ignored(name))

@dataclass(frozen=True)
class Layout:
    python_entrypoint: str | None
    framework: str | None = None
    typed_only: bool = False
    candidate_metadata: tuple[str, ...] = ()
    replaceable_baseline: bool = True
    archived: bool = False
    auxiliary_metadata: tuple[str, ...] = ()

    def owns(self, name: str) -> bool:
        return name == SELECTION or name in self.candidate_metadata

    def diagnostic_route(self, kind: str) -> str:
        if self.typed_only:
            if kind != "profile":
                raise ValueError("Compiled manifests support Evaluate and Profile; use Evaluate correctness_only for a build/correctness check")
            return "typed"
        return "dev" if self.archived else "single"

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


def git_blob(workspace, revision, name):
    from aka.legacy.task.git import _git
    return _git(workspace, 'show', f'{revision}:{name}', binary=True).stdout


def git_files(workspace, revision, names=None):
    command = ['git', 'ls-tree', '-r', '-z', revision]
    if names is not None:command += ['--', *names]
    listing = subprocess.run(command, cwd=workspace, capture_output=True, check=True).stdout
    result = {}
    for entry in listing.split(b'\0'):
        if not entry:continue
        meta, name = entry.split(b'\t', 1);name = name.decode()
        if names is None and not source_name(name):continue
        if meta.split()[0] not in {b'100644', b'100755'}:
            raise ValueError('Committed source contains a link or special object')
        result[name] = git_blob(workspace, revision, name)
    return result


def remove_files(workspace, names):
    from aka.light.agent.home import open_private_directory
    for name in names:
        with open_private_directory((workspace / name).parent) as parent:
            try:os.unlink(Path(name).name, dir_fd=parent)
            except FileNotFoundError:pass


from typing import Protocol

class FileSystem(Protocol):
    """One selected implementation owns its layout, files and execution material."""
    name: str
    default_policy: dict | None
    layout: Layout
    def prepare_source(self, request, source_provider, *, options, state_root, native_eval): ...
    def seed_metadata(self, workspace, framework, name, *, native_eval): ...
    def typed_limitation(self, workspace, policy) -> str | None: ...
    def language_hints(self, workspace) -> list[str]: ...
    def inspect_policy(self, get) -> tuple[bool, dict | None]: ...
    def validate_policy(self, policy): ...
    def validate(self, files, policy): ...
    def collect(self, workspace, policy, *, validate=True): ...
    def from_git(self, workspace, revision, policy): ...
    def from_inputs(self, inputs, policy, prefix): ...
    def editable(self, policy, name) -> bool: ...
    def instructions(self, policy) -> str: ...
    def metadata(self, policy) -> dict[str, bytes]: ...
    def validate_restore(self, previous, target) -> None: ...
    def obsolete(self, workspace, policy) -> set[str]: ...
    def install(self, workspace, files, policy) -> None: ...
    def execution_source(self, files, policy, archive) -> ExecutionSource: ...
    def declared_paths(self, workspace, policy) -> set[str]: ...
    def evidence_paths(self, workspace, policy) -> set[str]: ...
