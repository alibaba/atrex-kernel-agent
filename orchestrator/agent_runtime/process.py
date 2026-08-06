from __future__ import annotations

import ast
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


DEPENDENCY_GUARD_POLL_SECONDS = 0.05
DEFAULT_PROTECTED_GATEWAY_SCREEN = "atrex-local-gateway"
DEFAULT_PROTECTED_GATEWAY_STATE_NAME = "atrex-local-gateway"
ACCESS_POLICY_ENV = "ATREX_ACCESS_POLICY_ID"
ACCESS_VIOLATION_FILE_ENV = "ATREX_ACCESS_VIOLATION_FILE"


@dataclass(frozen=True)
class ProcessAccessPolicy:
    forbidden_roots: tuple[Path, ...] = ()
    network_disabled: bool = False
    audit_log: Path | None = None
    label: str = "process-access-policy"

    def __post_init__(self) -> None:
        roots = tuple(Path(path).expanduser().resolve() for path in self.forbidden_roots)
        object.__setattr__(self, "forbidden_roots", roots)
        if self.audit_log is not None:
            object.__setattr__(self, "audit_log", Path(self.audit_log).expanduser().resolve())
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("process access policy label must be non-empty")


_ACCESS_POLICIES: dict[str, ProcessAccessPolicy] = {}
_ACCESS_POLICY_LOCK = threading.Lock()


def register_access_policy(policy: ProcessAccessPolicy) -> str:
    if not isinstance(policy, ProcessAccessPolicy):
        raise TypeError("policy must be a ProcessAccessPolicy")
    policy_id = uuid.uuid4().hex
    with _ACCESS_POLICY_LOCK:
        _ACCESS_POLICIES[policy_id] = policy
    return policy_id


def unregister_access_policy(policy_id: str) -> None:
    with _ACCESS_POLICY_LOCK:
        _ACCESS_POLICIES.pop(policy_id, None)


def resolve_access_policy(policy_id: str) -> ProcessAccessPolicy | None:
    with _ACCESS_POLICY_LOCK:
        return _ACCESS_POLICIES.get(policy_id)


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        cwd: Path,
        timeout: int,
        env: dict | None = None,
    ) -> tuple[str, str, int, bool]:
        ...


def protected_gateway_identity(
    environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve shared gateway protection targets without embedding host paths."""
    values = os.environ if environment is None else environment
    screen = values.get(
        "ATREX_PROTECTED_GATEWAY_SCREEN", DEFAULT_PROTECTED_GATEWAY_SCREEN
    )
    state_dir = values.get("ATREX_PROTECTED_GATEWAY_STATE_DIR")
    if not state_dir:
        cache_home = values.get("XDG_CACHE_HOME")
        cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        state_dir = str(cache_root / DEFAULT_PROTECTED_GATEWAY_STATE_NAME)
    return screen, state_dir


def python_import_roots(code: str, *, _depth: int = 0) -> set[str]:
    """Return real imported top-level modules without matching strings/comments."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and node.args:
            target: str | None = None
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                target = "import"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                target = "import"
            if target and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                if isinstance(module, str) and module:
                    roots.add(module.split(".", 1)[0])
            if (
                _depth < 2
                and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval"}
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                roots.update(python_import_roots(node.args[0].value, _depth=_depth + 1))
    return roots


def dependency_process_violation(
    argv: list[str],
    access_policy: ProcessAccessPolicy | None = None,
    cwd: Path | None = None,
) -> str | None:
    """Describe a forbidden dependency, host GPU, or scoped-access action."""
    if not argv:
        return None

    def unwrap(segment: list[str]) -> list[str]:
        result = list(segment)
        while result and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", result[0]):
            result.pop(0)
        if result and Path(result[0]).name.lower() in {"env", "command"}:
            result.pop(0)
            while result and (result[0].startswith("-") or "=" in result[0]):
                result.pop(0)
        if result and Path(result[0]).name.lower() == "timeout":
            result.pop(0)
            while result and result[0].startswith("-"):
                result.pop(0)
            if result:
                result.pop(0)
        return result

    def command_segments(process_argv: list[str]) -> list[list[str]]:
        tokens = process_argv
        executable = Path(process_argv[0]).name.lower()
        if executable in {"bash", "sh", "dash", "zsh", "ksh"}:
            command_index = next(
                (
                    index + 1
                    for index, value in enumerate(process_argv[:-1])
                    if value.startswith("-") and "c" in value[1:]
                ),
                -1,
            )
            if command_index >= 0:
                try:
                    lexer = shlex.shlex(
                        process_argv[command_index], posix=True, punctuation_chars=";&|"
                    )
                    lexer.whitespace_split = True
                    tokens = list(lexer)
                except ValueError:
                    tokens = process_argv
        segments: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if token and all(character in ";&|" for character in token):
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        expanded = list(segments)
        for segment in segments:
            unwrapped = unwrap(segment)
            if (
                unwrapped
                and Path(unwrapped[0]).name.lower() == "eval"
                and len(unwrapped) > 1
            ):
                expanded.extend(
                    command_segments(["sh", "-c", " ".join(unwrapped[1:])])
                )
        return expanded

    def is_installer(segment: list[str]) -> bool:
        tokens = unwrap(segment)
        if not tokens:
            return False
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        if re.fullmatch(r"pip[0-9.]*", executable):
            return len(lowered) > 1 and lowered[1] in {"install", "wheel"}
        if executable == "uv":
            return lowered[1:3] in (
                ["pip", "install"],
                ["pip", "sync"],
                ["pip", "compile"],
            )
        if executable in {"conda", "mamba", "micromamba"}:
            return len(lowered) > 1 and lowered[1] in {"install", "create"}
        if re.fullmatch(r"python[0-9.]*", executable):
            if len(lowered) > 3 and lowered[1:3] == ["-m", "pip"]:
                return lowered[3] in {"install", "wheel"}
            if len(lowered) > 2 and lowered[1:3] == ["-m", "build"]:
                return True
            for index, token in enumerate(lowered[:-1]):
                if Path(token).name == "setup.py" and lowered[index + 1] in {
                    "install",
                    "build",
                    "build_ext",
                    "bdist_wheel",
                }:
                    return True
            if "--" in lowered:
                boundary = lowered.index("--")
                return is_installer(tokens[boundary + 1 :])
        if Path(executable).name == "setup.py":
            return len(lowered) > 1 and lowered[1] in {
                "install",
                "build",
                "build_ext",
                "bdist_wheel",
            }
        return False

    segments = command_segments(argv)

    if access_policy is not None:
        rendered = " ".join(argv)
        command_tokens = [*argv, *[token for segment in segments for token in segment]]
        for root in access_policy.forbidden_roots:
            root_text = str(root)
            matched = bool(root_text and root_text in rendered)
            if not matched:
                for token in command_tokens:
                    candidate_text = token.strip("'\"").rstrip(";,)")
                    if "=" in candidate_text and not candidate_text.startswith(("http://", "https://")):
                        candidate_text = candidate_text.split("=", 1)[1]
                    if not candidate_text or candidate_text.startswith(("-", "http://", "https://")):
                        continue
                    candidate_path = Path(candidate_text).expanduser()
                    if not candidate_path.is_absolute():
                        if cwd is None or not any(
                            marker in candidate_text for marker in ("/", "..", ".")
                        ):
                            continue
                        candidate_path = Path(cwd) / candidate_path
                    candidate = candidate_path.resolve(strict=False)
                    if (
                        candidate == root
                        or root in candidate.parents
                        or candidate in root.parents
                    ):
                        matched = True
                        break
            if matched:
                return "teacher knowledge access policy violation: forbidden path"
        if access_policy.network_disabled:
            network_tools = {
                "curl", "wget", "ssh", "scp", "sftp", "ftp", "nc", "netcat", "gh", "aria2c"
            }
            git_network_actions = {"clone", "fetch", "pull", "push", "ls-remote", "submodule"}
            for segment in segments:
                tokens = unwrap(segment)
                if not tokens:
                    continue
                executable = Path(tokens[0]).name.lower()
                lowered = [token.lower() for token in tokens]
                if executable in network_tools:
                    return "teacher knowledge access policy violation: network access disabled"
                if executable == "git" and any(action in lowered[1:] for action in git_network_actions):
                    return "teacher knowledge access policy violation: network access disabled"
                if re.fullmatch(r"python[0-9.]*", executable) and "-c" in tokens:
                    code_index = tokens.index("-c") + 1
                    code = tokens[code_index] if code_index < len(tokens) else ""
                    if python_import_roots(code) & {
                        "requests", "urllib", "httpx", "aiohttp", "socket", "ftplib", "smtplib"
                    }:
                        return "teacher knowledge access policy violation: network access disabled"

    def shared_gateway_mutation(segment: list[str]) -> bool:
        tokens = unwrap(segment)
        if not tokens:
            return False
        executable = Path(tokens[0]).name.lower()
        lowered = [token.lower() for token in tokens]
        protected_screen, protected_state = protected_gateway_identity()
        protected_screen = protected_screen.lower()
        protected_state = protected_state.lower()

        if executable == "screen" and any(
            token == protected_screen or token.endswith("." + protected_screen)
            for token in lowered[1:]
        ):
            return True
        if executable in {"rm", "rmdir", "unlink", "shred", "truncate", "mv"} and any(
            token == protected_state
            or token.startswith(protected_state + "/")
            or token == protected_state + ".log"
            for token in lowered[1:]
        ):
            return True
        if re.fullmatch(r"python[0-9.]*", executable):
            if any(Path(token).name == "local_gateway.py" for token in tokens[1:3]):
                return "serve" in lowered[1:]
            if "-c" in tokens:
                code_index = tokens.index("-c") + 1
                code = tokens[code_index].lower() if code_index < len(tokens) else ""
                if protected_state in code and re.search(
                    r"(?:rmtree|unlink|remove|rename|replace|sqlite3)", code
                ):
                    return True
        if executable in {"pkill", "killall"} and any(
            "local_gateway" in token or token == protected_screen
            for token in lowered[1:]
        ):
            return True
        if executable in {"curl", "wget"} and any(
            "/v1/jobs/" in token and "/cancel" in token for token in lowered[1:]
        ):
            return True
        return False

    if any(shared_gateway_mutation(segment) for segment in segments):
        return "shared localhost gateway lifecycle/state mutation"

    if any(is_installer(segment) for segment in segments):
        return "third-party package installation/build command"

    def direct_host_gpu_action(segment: list[str]) -> str | None:
        tokens = unwrap(segment)
        if not tokens:
            return None
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        info_only = (
            any(token in {"--help", "-h", "--version"} for token in lowered[1:])
            or (executable == "nvcc" and "-V" in tokens[1:])
        )
        if executable in {"nvcc", "cicc", "ptxas", "fatbinary", "ninja"} and not info_only:
            return "CUDA/JIT build tool executed directly on the host"
        if executable in {"ncu", "rocprof", "rocprofv3", "compute-sanitizer"}:
            return "GPU profiler executed directly on the host"
        if re.fullmatch(r"python[0-9.]*", executable):
            if len(tokens) > 1 and Path(tokens[1]).name == "sandbox.py":
                return None
            if len(tokens) > 1 and Path(tokens[1]).name in {
                "kernel.py",
                "test_kernel.py",
                "profile_driver.py",
            }:
                return "kernel/evaluator executed directly on the host"
            if "-c" in tokens:
                code_index = tokens.index("-c") + 1
                code = tokens[code_index] if code_index < len(tokens) else ""
                imports = python_import_roots(code)
                if "kernel" in imports:
                    return "kernel imported directly on the host"
                if imports & {"flashinfer", "flash_attn", "xformers", "vllm"}:
                    return "JIT-capable third-party GPU package imported directly on the host"
        if executable in {"bash", "sh", "dash", "zsh", "ksh"} and any(
            Path(token).name in {"profile_nvidia.sh", "profile_kernel.sh"}
            for token in tokens[1:]
        ):
            return "GPU profiler wrapper executed directly on the host"
        return None

    for segment in segments:
        reason = direct_host_gpu_action(segment)
        if reason is not None:
            return reason

    command = " ".join(argv).lower()
    package_build_tree = re.search(
        r"(?:^|[\s=])[^\s]*(?:pip-install-|pip-build-|pip-modern-metadata-)[^\s]*",
        command,
    )
    build_tools = {
        "cicc",
        "nvcc",
        "ninja",
        "cmake",
        "make",
        "gcc",
        "g++",
        "clang",
        "clang++",
    }
    if package_build_tree and any(
        unwrap(segment) and Path(unwrap(segment)[0]).name.lower() in build_tools
        for segment in segments
    ):
        return "compiler/build tool running in a package-manager temporary tree"
    return None


def _ps_descendant_process_commands(root_pid: int) -> list[tuple[int, list[str]]]:
    """Portable fallback for hosts without Linux procfs (notably macOS)."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    children: dict[int, list[tuple[int, list[str]]]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        try:
            argv = shlex.split(parts[2])
        except ValueError:
            argv = [parts[2]]
        children.setdefault(parent, []).append((pid, argv))
    pending = [root_pid]
    seen = {root_pid}
    descendants: list[tuple[int, list[str]]] = []
    while pending:
        parent = pending.pop()
        for pid, argv in children.get(parent, []):
            if pid in seen:
                continue
            seen.add(pid)
            pending.append(pid)
            descendants.append((pid, argv))
    return descendants


def descendant_process_commands(root_pid: int) -> list[tuple[int, list[str]]]:
    """Return live descendants and argv, preferring Linux procfs."""
    if not Path(f"/proc/{root_pid}/task").is_dir():
        return _ps_descendant_process_commands(root_pid)
    pending = [root_pid]
    seen = {root_pid}
    descendants: list[tuple[int, list[str]]] = []
    while pending:
        parent = pending.pop()
        task_dir = Path(f"/proc/{parent}/task")
        try:
            thread_dirs = list(task_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children: set[int] = set()
        for thread_dir in thread_dirs:
            try:
                children.update(
                    int(value)
                    for value in (thread_dir / "children").read_text().split()
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        for pid in children:
            if pid in seen:
                continue
            seen.add(pid)
            pending.append(pid)
            try:
                raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            descendants.append((pid, argv))
    return descendants


def descendant_process_groups(root_pid: int) -> set[int]:
    """Capture every process group in a coding session's live process tree."""
    process_groups: set[int] = set()
    for pid in [root_pid, *[pid for pid, _argv in descendant_process_commands(root_pid)]]:
        try:
            process_groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    return process_groups


def signal_process_groups(process_groups: set[int], sig: signal.Signals) -> None:
    for process_group in process_groups:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass


def _record_access_violation(
    policy: ProcessAccessPolicy | None,
    reason: str,
    argv: list[str],
) -> None:
    if policy is None or policy.audit_log is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": policy.label,
        "reason": reason,
        "argv": argv[:100],
    }
    try:
        policy.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with policy.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass


def _process_cwd(pid: int, fallback: Path) -> Path:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except (OSError, RuntimeError):
        return fallback.resolve()


def dependency_guard(
    proc: subprocess.Popen[str],
    stop: threading.Event,
    violations: list[str],
    access_policy: ProcessAccessPolicy | None = None,
    cwd: Path = Path.cwd(),
    violation_marker: Path | None = None,
) -> None:
    """Kill a coding session as soon as it starts a forbidden dependency job."""
    while not stop.wait(DEPENDENCY_GUARD_POLL_SECONDS):
        if proc.poll() is not None:
            return
        if violation_marker is not None:
            try:
                marker_reasons = [
                    line.strip()
                    for line in violation_marker.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                marker_reasons = []
            if marker_reasons:
                violation_marker.write_text("", encoding="utf-8")
                for reason in marker_reasons:
                    violations.append("shell guard: " + reason)
                    _record_access_violation(access_policy, reason, ["shell-guard"])
                process_groups = descendant_process_groups(proc.pid)
                signal_process_groups(process_groups, signal.SIGTERM)
                deadline = time.monotonic() + 1.0
                while proc.poll() is None and time.monotonic() < deadline:
                    if stop.wait(0.05):
                        return
                signal_process_groups(process_groups, signal.SIGKILL)
                return
        for pid, argv in descendant_process_commands(proc.pid):
            reason = dependency_process_violation(
                argv,
                access_policy=access_policy,
                cwd=_process_cwd(pid, cwd),
            )
            if reason is None:
                continue
            rendered = " ".join(argv)
            violations.append(f"pid={pid}: {reason}: {rendered[:1000]}")
            _record_access_violation(access_policy, reason, argv)
            process_groups = descendant_process_groups(proc.pid)
            signal_process_groups(process_groups, signal.SIGTERM)
            deadline = time.monotonic() + 1.0
            while proc.poll() is None and time.monotonic() < deadline:
                if stop.wait(0.05):
                    return
            signal_process_groups(process_groups, signal.SIGKILL)
            return


def run_bounded(
    command: list[str],
    cwd: Path,
    timeout: int,
    env: dict | None = None,
) -> tuple[str, str, int, bool]:
    """Run a command in its own process group with timeout and policy enforcement."""
    policy_id = (env or {}).get(ACCESS_POLICY_ENV, "")
    access_policy = resolve_access_policy(policy_id) if policy_id else None
    if policy_id and access_policy is None:
        return "", "[orchestrator] unknown or expired process access policy\n", 126, False
    child_env = dict(env or {})
    violation_marker: Path | None = None
    if access_policy is not None:
        descriptor, marker_name = tempfile.mkstemp(prefix="atrex-access-violation-")
        os.close(descriptor)
        violation_marker = Path(marker_name)
        child_env[ACCESS_VIOLATION_FILE_ENV] = marker_name
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=child_env or None,
    )
    guard_stop = threading.Event()
    dependency_violations: list[str] = []
    guard = threading.Thread(
        target=dependency_guard,
        args=(
            proc,
            guard_stop,
            dependency_violations,
            access_policy,
            Path(cwd),
            violation_marker,
        ),
        name=f"dependency-guard-{proc.pid}",
        daemon=True,
    )
    guard.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process_groups = descendant_process_groups(proc.pid)
        signal_process_groups(process_groups, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    except BaseException:
        process_groups = descendant_process_groups(proc.pid)
        signal_process_groups(process_groups, signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            signal_process_groups(process_groups, signal.SIGKILL)
            proc.communicate()
        raise
    finally:
        guard_stop.set()
        guard.join(timeout=1)
    returncode = proc.returncode
    if violation_marker is not None:
        try:
            marker_reasons = [
                line.strip()
                for line in violation_marker.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            marker_reasons = []
        finally:
            violation_marker.unlink(missing_ok=True)
        for reason in marker_reasons:
            dependency_violations.append("shell guard: " + reason)
            _record_access_violation(access_policy, reason, ["shell-guard"])
    if dependency_violations:
        policy_message = (
            "[orchestrator] dependency policy violation; terminated coding session:\n"
            + "\n".join(dependency_violations)
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + policy_message + "\n"
        returncode = 126
    return stdout or "", stderr or "", returncode, timed_out
