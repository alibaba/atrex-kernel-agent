from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from aka.legacy.agent import process as legacy_process
from aka.legacy.agent.process import (
    ProcessRunner, protected_gateway_identity,
    DEFAULT_PROTECTED_GATEWAY_SCREEN, DEFAULT_PROTECTED_GATEWAY_STATE_NAME,
)

from aka.light.agent.recovery_processes import spawn_owned_session
from aka.light.agent.dependency_policy import (
    canonical_sandbox_entrypoint,
    dependency_command_violation,
    dependency_guard_enabled,
    python_import_roots,
)

ENVIRONMENT_TEMPFAIL = 75
OUTPUT_LIMIT_EXIT_STATUS = 91
OUTPUT_LIMIT_INTERRUPT_MARKER = (
    "[orchestrator] Claude max_output_tokens observed; "
    "terminated CLI wrapper for same-session recovery"
)
TRUSTED_SANDBOX_ENTRYPOINTS = frozenset({
    (Path(__file__).resolve().parents[2] / "runtime" / "client" / "sandbox.py").resolve(),
})


def register_sandbox_entrypoints(*paths: Path) -> None:
    """Register supervisor-owned transport scripts before launching agents."""
    global TRUSTED_SANDBOX_ENTRYPOINTS
    if any(not path.is_absolute() for path in paths):
        raise ValueError("sandbox entrypoints must be absolute paths")
    TRUSTED_SANDBOX_ENTRYPOINTS |= frozenset(path.resolve() for path in paths)


def dependency_process_violation(
    argv: list[str], *, cwd: Path | None = None,
    trusted_sandbox_entrypoints: frozenset[Path] | None = None,
    enforce_dependencies: bool = True,
) -> str | None:
    """Report a confirmed violation using the shared hook/guard policy."""
    protected = protected_gateway_identity()
    return dependency_command_violation(
        argv,
        cwd=cwd,
        trusted_sandbox_entrypoints=(
            TRUSTED_SANDBOX_ENTRYPOINTS
            if trusted_sandbox_entrypoints is None
            else trusted_sandbox_entrypoints
        ),
        canonical_sandbox_entrypoint=canonical_sandbox_entrypoint(),
        protected_gateway=protected,
        enforce_dependencies=enforce_dependencies,
    )


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
    enforce_policy: bool = True,
) -> None:
    """Use Legacy's monitor with Light's mode-aware dependency classifier.

    The monitor always retains environment-outage recovery. Leaderboard can
    disable dependency-policy classification without disabling that recovery.
    Procfs/cwd races are inconclusive; never resolve a relative client in the
    Supervisor's cwd.
    """
    def classify(pid: int, argv: list[str]) -> str | None:
        try:
            cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
        except (OSError, RuntimeError):
            cwd = None
        return dependency_process_violation(
            argv, cwd=cwd, enforce_dependencies=enforce_policy
        )

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
        if command and Path(command[0]).name == "dsh":
            from aka.legacy.agent.dsh import runtime_connection_environment
            # Managed credentials exist only inside session_environment. The
            # adapter's earlier environment hook cannot bridge them yet.
            with runtime_connection_environment(environment) as dsh_environment:
                return _run_bounded(command, cwd, timeout, dsh_environment,
                                    auxiliary_input_files=auxiliary_input_files, session_control=control)
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
    from aka.light.agent.home import open_private_directory
    from aka.light.agent.sandbox import wrap_agent_command
    from aka.light.agent.session_capture import clear_capture, finish_session_capture, start_session_capture
    from aka.light.workspace.auxiliary import ensure_workspace_scratch

    clear_capture()
    # Agent prompts and Runtime examples may redirect directly into scratch/.
    # Create it at the final launch boundary as well as workspace initialization,
    # covering restored and older workspaces without clearing resumed contents.
    ensure_workspace_scratch(cwd)
    control = session_control or SessionControl()
    budget = current_budget()
    # ``None`` is an explicit unbounded request.  Callers that want the Agent
    # profile's default deadline pass that finite value in AgentRunRequest;
    # silently replacing None with the profile default would make an allegedly
    # unbounded auxiliary review stop after eight hours.
    limit = (
        None
        if timeout is None
        else validate_timeout(timeout, "Agent timeout_seconds")
    )
    if limit is not None and budget is not None and budget.limited:
        limit = min(limit, budget.remaining)
    if limit is not None and limit <= 0:
        control.close()
        return "", "Agent active-time budget exhausted", -signal.SIGKILL, True
    from aka.light.runtime.paths import supervisor_agent_control_root
    control_root = supervisor_agent_control_root(cwd)
    with open_private_directory(control_root):
        pass
    budget_state_path = control_root / "agent-active-budget.json"
    if budget is not None and budget.limited:
        control.configure_budget(
            budget_state_path,
            budget_id=budget.id,
            stage_name=(env or {}).get("ATREX_AGENT_STAGE_NAME", "Agent stage"),
            total_seconds=budget.seconds,
            remaining_seconds=budget.remaining,
        )
    environment_values = prepare_agent_environment(
        cwd, dict(os.environ if env is None else env),
        (env or {}).get("ATREX_TELEMETRY_ATTEMPT_ID") or "\0".join(command),
    )
    from aka.light.agent.claude_tools import CONTROL_ROOT_ENV
    environment_values[CONTROL_ROOT_ENV] = str(control_root)
    output_limit_signal = None
    if Path(command[0]).name == "claude":
        from aka.light.agent.claude_tools import (
            OUTPUT_LIMIT_RECOVERY_ENV,
            OUTPUT_LIMIT_SIGNAL_ENV,
            OUTPUT_LIMIT_WORKSPACE_ENV,
        )
        try:
            recovery_enabled = int(
                environment_values.get(OUTPUT_LIMIT_RECOVERY_ENV, "0")
            ) > 0
        except (TypeError, ValueError):
            recovery_enabled = False
        if recovery_enabled:
            output_limit_signal = (
                control_root
                / "output-limit-signals"
                / f"{uuid.uuid4().hex}.json"
            )
            output_limit_signal.unlink(missing_ok=True)
            environment_values[OUTPUT_LIMIT_SIGNAL_ENV] = str(output_limit_signal)
            environment_values[OUTPUT_LIMIT_WORKSPACE_ENV] = str(cwd.resolve())
    if (
        environment_values.get("ATREX_AKA_RUNTIME_URL")
        or budget is not None
        or output_limit_signal is not None
    ):
        from aka.light.agent.claude_tools import FOREGROUND_HOOKS_ENV, foreground_command
        environment_values[FOREGROUND_HOOKS_ENV] = "1"
        # Runtime calls remain foreground, and the same invocation-local hook
        # carries active-time reminders even in stages without Runtime tools.
        hook_timeout = limit
        if hook_timeout is None:
            hook_timeout = validate_timeout(
                float(
                    (env or {}).get(
                        "ATREX_AGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
                    )
                ),
                "Agent timeout_seconds",
            )
        command = foreground_command(
            command,
            cwd,
            hook_timeout,
            budget_state_path=(
                budget_state_path
                if budget is not None and budget.limited else None
            ),
        )
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
    # Every launched Agent retains environment-outage monitoring. Dependency
    # policy classification is mode-selectable and defaults off in Leaderboard.
    guard = threading.Thread(
        target=dependency_guard,
        args=(proc, guard_stop, dependency_violations, environment_state_file,
              environment_failures, dependency_guard_enabled(environment_values)),
        name=f"dependency-guard-{proc.pid}",
        daemon=True,
    )
    guard.start()
    completed = False
    timed_out = False
    interrupted = False
    output_limit_detected = False
    raw_communicate = (
        (lambda timeout=None: capture.communicate(proc, timeout))
        if capture is not None else proc.communicate
    )
    def output_limit_ready():
        return bool(
            (output_limit_signal is not None and output_limit_signal.is_file())
            or (
                capture is not None
                and capture.output_limit_detected.is_set()
            )
        )

    communicate = ActiveCommunication(
        raw_communicate,
        proc,
        control,
        interrupt_when=(output_limit_ready if output_limit_signal is not None else None),
    )
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
            output_limit_detected = bool(
                communicate.interrupted
                or (output_limit_signal is not None and output_limit_signal.is_file())
                or (
                    capture is not None
                    and capture.output_limit_detected.is_set()
                )
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
                and not output_limit_detected
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
            if output_limit_signal is not None:
                output_limit_signal.unlink(missing_ok=True)
    returncode = proc.returncode
    if dependency_violations:
        policy_message = (
            "[orchestrator] dependency policy violation; terminated coding session:\n"
            + "\n".join(dependency_violations)
        )
        stderr = (stderr or "") + ("\n" if stderr else "") + policy_message + "\n"
        # The guard normally terminates the Agent process group with SIGTERM,
        # which would otherwise surface as -SIGTERM and be indistinguishable
        # from an external shutdown. Pair the dedicated status with the
        # explicit marker above so ordinary shell exit 126 remains ordinary.
        returncode = 126
    elif output_limit_detected:
        stderr = (stderr or "") + ("\n" if stderr else "") + (
            OUTPUT_LIMIT_INTERRUPT_MARKER + "\n"
        )
        returncode = OUTPUT_LIMIT_EXIT_STATUS
    if timed_out and limit is not None:
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
