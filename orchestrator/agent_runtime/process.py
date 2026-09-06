from __future__ import annotations

import ast
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Protocol

from ..recovery_processes import spawn_owned_session

DEPENDENCY_GUARD_POLL_SECONDS = 0.25
ENVIRONMENT_TEMPFAIL = 75
DEFAULT_PROTECTED_GATEWAY_SCREEN = "atrex-local-gateway"
DEFAULT_PROTECTED_GATEWAY_STATE_NAME = "atrex-local-gateway"


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        cwd: Path,
        timeout: int | None,
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


def dependency_process_violation(argv: list[str]) -> str | None:
    """Describe a forbidden dependency build or host GPU action, if any."""
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


def descendant_process_commands(root_pid: int) -> list[tuple[int, list[str]]]:
    """Return live descendants and argv using Linux procfs, tolerating races."""
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


def dependency_guard(
    proc: subprocess.Popen[str],
    stop: threading.Event,
    violations: list[str],
    environment_state_file: str = "",
    environment_failures: list[str] | None = None,
) -> None:
    """Kill a coding session on a policy violation or environment failure."""
    while not stop.wait(DEPENDENCY_GUARD_POLL_SECONDS):
        if proc.poll() is not None:
            return
        if environment_state_file and Path(environment_state_file).is_file():
            if environment_failures is not None:
                environment_failures.append(environment_state_file)
            process_groups = descendant_process_groups(proc.pid)
            signal_process_groups(process_groups, signal.SIGTERM)
            deadline = time.monotonic() + 1.0
            while proc.poll() is None and time.monotonic() < deadline:
                if stop.wait(0.05):
                    return
            signal_process_groups(process_groups, signal.SIGKILL)
            return
        for pid, argv in descendant_process_commands(proc.pid):
            reason = dependency_process_violation(argv)
            if reason is None:
                continue
            rendered = " ".join(argv)
            violations.append(f"pid={pid}: {reason}: {rendered[:1000]}")
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
    timeout: int | None,
    env: dict | None = None,
) -> tuple[str, str, int, bool]:
    """Run a guarded command, optionally without a wall-clock deadline."""
    proc = spawn_owned_session(
        command,
        role="coding-agent",
        environment=env,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    guard_stop = threading.Event()
    dependency_violations: list[str] = []
    environment_failures: list[str] = []
    environment_values = os.environ if env is None else env
    environment_state_file = str(
        environment_values.get("ATREX_ENVIRONMENT_STATE_FILE", "")
    )
    guard = threading.Thread(
        target=dependency_guard,
        args=(
            proc,
            guard_stop,
            dependency_violations,
            environment_state_file,
            environment_failures,
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
    if dependency_violations:
        policy_message = (
            "[orchestrator] dependency policy violation; terminated coding session:\n"
            + "\n".join(dependency_violations)
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + policy_message + "\n"
        if returncode == 0:
            returncode = 126
    if environment_failures:
        environment_message = (
            "[orchestrator] remote GPU environment became unavailable; "
            "terminated coding session for durable recovery"
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + environment_message + "\n"
        returncode = ENVIRONMENT_TEMPFAIL
    return stdout or "", stderr or "", returncode, timed_out
