from __future__ import annotations

import ast
import os
import re
import shlex
import signal
import subprocess
import threading
from pathlib import Path
from aka.legacy.agent import process as legacy_process
from aka.legacy.agent.process import (
    ProcessRunner, protected_gateway_identity,
    DEFAULT_PROTECTED_GATEWAY_SCREEN, DEFAULT_PROTECTED_GATEWAY_STATE_NAME,
)

from aka.light.agent.recovery_processes import spawn_owned_session

ENVIRONMENT_TEMPFAIL = 75
TRUSTED_SANDBOX_ENTRYPOINTS = frozenset({
    (Path(__file__).resolve().parents[2] / "runtime" / "client" / "sandbox.py").resolve(),
})


def register_sandbox_entrypoints(*paths: Path) -> None:
    """Register supervisor-owned transport scripts before launching agents."""
    global TRUSTED_SANDBOX_ENTRYPOINTS
    if any(not path.is_absolute() for path in paths):
        raise ValueError("sandbox entrypoints must be absolute paths")
    TRUSTED_SANDBOX_ENTRYPOINTS |= frozenset(path.resolve() for path in paths)


def _python_entrypoint(tokens: list[str]) -> tuple[str, str]:
    """Parse interpreter options without treating script arguments as Python code."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return "script", tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "--check-hash-based-pycs":
            index += 2
        elif token.startswith("--"):
            index += 1
        elif token.startswith("-") and token != "-":
            # Python accepts combined flags such as -uc and attached -cCODE.
            for offset, flag in enumerate(token[1:], start=1):
                if flag in {"c", "m"}:
                    value = token[offset + 1:] or (tokens[index + 1] if index + 1 < len(tokens) else "")
                    return ("code" if flag == "c" else "module"), value
                if flag in {"W", "X"}:
                    if offset == len(token) - 1:
                        index += 1
                    break
            index += 1
        else:
            return "script", token
    return "", ""






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
    argv: list[str], *, cwd: Path | None = None,
    trusted_sandbox_entrypoints: frozenset[Path] | None = None,
) -> str | None:
    """Report confirmed policy violations for the shared dependency guard.

    ``cwd`` must be the inspected process's directory, not the Supervisor's.
    Missing process/file evidence is inconclusive (None), not authorization or
    evidence of a violation. In particular, procfs can disappear during exit.
    """
    if not argv:
        return None
    trusted = (TRUSTED_SANDBOX_ENTRYPOINTS if trusted_sandbox_entrypoints is None
               else trusted_sandbox_entrypoints)
    transport_names = {"sandbox.py", *(path.name for path in trusted)}

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
        executable = Path(process_argv[0]).name.lower()
        # A real argv already preserves argument boundaries. A quoted ';' in
        # Dev's remote command is data, not a new local shell command.
        if executable not in {"bash", "sh", "dash", "zsh", "ksh"}:
            return [process_argv]
        command_index = next(
            (index + 1 for index, value in enumerate(process_argv[:-1])
             if value.startswith("-") and "c" in value[1:]),
            -1,
        )
        if command_index < 0:
            return [process_argv]

        # Split before unquoting: shlex's punctuation tokens otherwise lose the
        # distinction between a shell operator and a literal quoted ';'/'|'.
        command = process_argv[command_index]
        boundaries = [-1]
        quote = ""
        escaped = False
        for index, character in enumerate(command):
            if escaped:
                escaped = False
            elif character == "\\" and quote != "'":
                escaped = True
            elif quote:
                if character == quote:
                    quote = ""
            elif character in "'\"`":
                quote = character
            elif character in ";&|":
                boundaries.append(index)
        boundaries.append(len(command))
        segments: list[list[str]] = []
        try:
            for left, right in zip(boundaries, boundaries[1:]):
                segment = shlex.split(command[left + 1:right], comments=True)
                if segment:
                    segments.append(segment)
        except ValueError:
            return [process_argv]
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

    def transport_status(segment: list[str]) -> str | None:
        tokens = unwrap(segment)
        if not tokens or not re.fullmatch(r"python[0-9.]*", Path(tokens[0]).name.lower()):
            return None
        mode, entry = _python_entrypoint(tokens)
        if mode != "script" or Path(entry).name not in transport_names:
            return None
        script = Path(entry)
        if not script.is_absolute():
            if cwd is None or not cwd.is_absolute():
                return "unknown"
            script = cwd / script
        from aka.light.agent.session_tail import read_regular_bytes
        try:
            script = script.resolve(strict=True)
            if script in trusted:
                return "trusted"
            # Workspace tools are regular copies of the HTTP-only client.
            # Bound both reads; a failed read is not a content mismatch.
            canonical = Path(__file__).resolve().parents[2] / "runtime/client/sandbox.py"
            limit = 128 * 1024
            expected = read_regular_bytes(canonical, limit=limit + 1)
            if len(expected) > limit:
                return "unknown"
            actual = read_regular_bytes(script, limit=limit + 1)
            return "trusted" if actual == expected else "untrusted"
        except (OSError, ValueError, RuntimeError):
            # Includes inaccessible/deleted cwd, disappearing processes/files
            # and symlink loops. Never fall back to Path.cwd() here.
            return "unknown"

    segments = command_segments(argv)
    local_segments = []
    for segment in segments:
        status = transport_status(segment)
        if status == "untrusted":
            return "unregistered sandbox transport executed on the host"
        if status is None:
            local_segments.append(segment)
        # A verified transport's arguments belong to the remote executor. An
        # unreadable transport cannot be classified; neither permits inferring
        # a local pip/build/import from its opaque request payload.
    segments = local_segments

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
            mode, entry = _python_entrypoint(tokens)
            if mode == "script" and Path(entry).name == "local_gateway.py":
                return "serve" in lowered[1:]
            if mode == "code":
                code = entry.lower()
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
            mode, entry = _python_entrypoint(tokens)
            if mode == "script" and Path(entry).name in {
                "kernel.py",
                "test_kernel.py",
                "profile_driver.py",
            }:
                return "kernel/evaluator executed directly on the host"
            if mode == "module" and entry.split(".", 1)[0] in {"kernel", "test_kernel", "profile_driver"}:
                return "kernel/evaluator executed directly on the host"
            if mode == "code":
                imports = python_import_roots(entry)
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
    """Use Legacy's always-on guard with Light's process-aware HTTP classifier.

    The mode and sandbox selection do not control this guard. Procfs/cwd races
    are inconclusive; never resolve a relative client in the Supervisor's cwd.
    """
    def classify(pid: int, argv: list[str]) -> str | None:
        try:
            cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
        except (OSError, RuntimeError):
            cwd = None
        return dependency_process_violation(argv, cwd=cwd)

    legacy_process.dependency_guard(
        proc, stop, violations, environment_state_file, environment_failures,
        classify_process=classify,
    )


def run_bounded(
    command: list[str],
    cwd: Path,
    timeout: int | None,
    env: dict | None = None,
    *, auxiliary_input_files: dict[str, Path] | None = None,
) -> tuple[str, str, int, bool]:
    from aka.light.agent.execution_clock import SessionControl
    environment = dict(os.environ if env is None else env)
    control = SessionControl()
    if not environment.get("ATREX_AKA_RUNTIME_OWNER"):
        return _run_bounded(command, cwd, timeout, environment,
                            auxiliary_input_files=auxiliary_input_files, session_control=control)
    from aka.light.runtime.server import session_environment

    with session_environment(cwd, environment, session_control=control) as environment:
        if environment.get("ATREX_EPISODE_WORKSPACE") and command[:2] == ["codex", "exec"]:
            command = [*command[:2], "--skip-git-repo-check", *command[2:]]
        return _run_bounded(command, cwd, timeout, environment,
                            auxiliary_input_files=auxiliary_input_files, session_control=control)


def _run_bounded(
    command: list[str], cwd: Path, timeout: int | None, env: dict | None = None,
    *, auxiliary_input_files: dict[str, Path] | None = None,
    session_control=None,
) -> tuple[str, str, int, bool]:
    """Run with an active-time deadline excluding only configured suspended waits."""
    from aka.light.agent.execution_clock import (
        ActiveCommunication, DEFAULT_TIMEOUT_SECONDS, SessionControl, current_budget, validate_timeout,
    )
    from aka.light.agent.home import prepare_agent_environment
    from aka.light.agent.sandbox import wrap_agent_command
    from aka.light.agent.session_capture import clear_capture, finish_session_capture, start_session_capture

    clear_capture()
    control = session_control or SessionControl()
    budget = current_budget()
    default_timeout = float((env or {}).get("ATREX_AGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    limit = validate_timeout(timeout if timeout is not None else default_timeout, "Agent timeout_seconds")
    if budget is not None:
        limit = min(limit, budget.remaining)
    if limit <= 0:
        control.close()
        return "", "Agent active-time budget exhausted", -signal.SIGKILL, True
    environment_values = prepare_agent_environment(
        cwd, dict(os.environ if env is None else env),
        (env or {}).get("ATREX_TELEMETRY_ATTEMPT_ID") or "\0".join(command),
    )
    if environment_values.get("ATREX_AKA_RUNTIME_URL"):
        from aka.light.agent.claude_tools import foreground_command
        # Like Runtime, keep these calls foreground; the authenticated server
        # suspension remains authoritative if a shell backgrounds/disconnects.
        command = foreground_command(command, cwd, limit)
    launch, view = wrap_agent_command(
        command, cwd, environment_values, auxiliary_input_files=auxiliary_input_files,
    )
    capture = None
    proc = None
    try:
        capture = start_session_capture(command, cwd, environment_values)
        proc = spawn_owned_session(
            launch.command,
            role="coding-agent",
            environment=launch.environment,
            **({"inherited_fds": launch.pass_fds} if launch.pass_fds else {}),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        control.attach(proc.pid)
    except BaseException:
        control.close()
        if proc is not None:
            signal_process_groups(descendant_process_groups(proc.pid) | {proc.pid}, signal.SIGKILL)
            proc.communicate()
        finish_session_capture(capture, interrupted=True)
        if view:
            view.close()
        raise
    finally:
        launch.close()
    guard_stop = threading.Event()
    dependency_violations: list[str] = []
    environment_failures: list[str] = []
    environment_state_file = str(
        environment_values.get("ATREX_ENVIRONMENT_STATE_FILE", "")
    )
    # Match Legacy: every launched Agent is guarded, independent of production
    # mode, capture, Bubblewrap or whether an outage marker is configured.
    guard = threading.Thread(
        target=dependency_guard,
        args=(proc, guard_stop, dependency_violations, environment_state_file, environment_failures),
        name=f"dependency-guard-{proc.pid}",
        daemon=True,
    )
    guard.start()
    completed = False
    timed_out = False
    interrupted = False
    raw_communicate = (
        (lambda timeout=None: capture.communicate(proc, timeout))
        if capture is not None else proc.communicate
    )
    communicate = ActiveCommunication(raw_communicate, proc, control)
    try:
        stdout, stderr = communicate(timeout=limit)
        completed = True
    except subprocess.TimeoutExpired:
        timed_out = True
        # spawn_owned_session creates this PGID. Keep it even if the group
        # leader has exited/reaped while descendants still hold stdout/stderr.
        control.kill()
        stdout, stderr = communicate()
    except BaseException:
        interrupted = True
        # SIGTERM alone cannot terminate a SIGSTOP'ed Provider. Cancellation
        # kills the owned tree and never briefly resumes it into more work.
        control.kill()
        communicate()
        raise
    finally:
        try:
            control.close()
            if budget is not None:
                budget.used += control.active_seconds()
            guard_stop.set()
            guard.join(timeout=1)
            finish_session_capture(
                capture, exit_status=proc.returncode, timed_out=timed_out, interrupted=interrupted,
            )
            # A killed/failed auxiliary session may leave a syntactically valid
            # but incomplete report. Only normal, successful completion can
            # replace the caller's report; timeout draining is not completion.
            if (
                view is not None
                and completed
                and not timed_out
                and not interrupted
                and proc.returncode == 0
                and not dependency_violations
                and not environment_failures
            ):
                view.publish()
            elif (view is not None and view.role == "numerical-review" and timed_out
                  and not interrupted and not dependency_violations and not environment_failures):
                # Unlike an auxiliary verdict, a probe plan is untrusted data:
                # the controller still validates its schema/evidence and executes
                # it. A truncated file can never become an admission verdict.
                try:
                    view.recover_numerical_plan()
                except (OSError, ValueError):
                    pass  # The planner timeout path will record unusable evidence.
        finally:
            if view is not None:
                view.close()
    returncode = proc.returncode
    if dependency_violations:
        policy_message = (
            "[orchestrator] dependency policy violation; terminated coding session:\n"
            + "\n".join(dependency_violations)
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + policy_message + "\n"
        if returncode == 0:
            returncode = 126
    if timed_out:
        stderr = (stderr or "") + (
            f"\n[orchestrator] Agent active-time limit exceeded ({limit:g}s); "
            f"{control.wait_seconds:.3f}s of suspended Runtime waiting excluded.\n"
        )
    if environment_failures:
        environment_message = (
            "[orchestrator] remote GPU environment became unavailable; "
            "terminated coding session for durable recovery"
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + environment_message + "\n"
        returncode = ENVIRONMENT_TEMPFAIL
    return stdout or "", stderr or "", returncode, timed_out
