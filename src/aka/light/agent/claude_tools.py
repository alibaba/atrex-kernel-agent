"""Invocation-local Claude hooks for policy and Supervisor HTTP clients."""
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

try:
    from .dependency_policy import (
        canonical_sandbox_entrypoint,
        dependency_command_violation,
        dependency_guard_enabled,
        protected_gateway_identity,
    )
except ImportError:  # Executed directly by Claude's system-Python hook.
    from dependency_policy import (
        canonical_sandbox_entrypoint,
        dependency_command_violation,
        dependency_guard_enabled,
        protected_gateway_identity,
    )


OUTPUT_LIMIT_RECOVERY_ENV = "ATREX_OUTPUT_LIMIT_RECOVERY_RETRIES"
OUTPUT_LIMIT_SCOPE_ENV = "ATREX_OUTPUT_LIMIT_RECOVERY_SCOPE"
OUTPUT_LIMIT_SIGNAL_ENV = "ATREX_OUTPUT_LIMIT_SIGNAL"
OUTPUT_LIMIT_WORKSPACE_ENV = "ATREX_OUTPUT_LIMIT_WORKSPACE"
FOREGROUND_HOOKS_ENV = "ATREX_CLAUDE_FOREGROUND_HOOKS"
CONTROL_ROOT_ENV = "ATREX_AGENT_CONTROL_ROOT"
_STOP_HOOK_STATE = "output-limit-stop-hook.json"


def output_limit_continuation_guidance(retry, maximum, stage_name="Episode"):
    """Return concise same-loop guidance shared by Stop-hook recovery."""
    return f"""Continue the same {stage_name} after a truncated response.

Your previous response exceeded the 32K output-token limit and was truncated. Continue in this
same provider session with the existing conversation, workspace, candidate, tools and remaining
budget. Do not restart the task or repeat the whole previous explanation.

Do not try to think through, explain or implement everything in one response. Choose one small,
concrete modification or targeted probe and complete it. If you have several useful ideas, first
write a short checklist into a scratch file so they survive, then implement and verify them one at
a time. Keep each response concise and save progress to files before moving to the next item.

This is output-limit continuation {retry}/{maximum}."""


def _control_root(workspace):
    configured = os.environ.get(CONTROL_ROOT_ENV, "")
    if configured:
        return Path(configured).expanduser()
    # Normal Supervisor code imports this module as part of the package.  The
    # direct stdlib-only Claude hook receives CONTROL_ROOT_ENV and never needs
    # to import the package from inside the sandbox.
    try:
        from aka.light.runtime.paths import supervisor_agent_control_root
    except ImportError:
        return Path(workspace).expanduser().parent / ".atrex-agent-control"
    return supervisor_agent_control_root(Path(workspace))


def _output_limit_state_path(workspace):
    return _control_root(workspace) / _STOP_HOOK_STATE


def _claim_output_limit_event(path, scope, event_key, maximum):
    """Atomically claim one transcript event; fail closed if state is unavailable."""
    try:
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            state = {"schema_version": 1, "scopes": {}}
        if not isinstance(state, dict) or not isinstance(state.get("scopes"), dict):
            return None
        scopes = state["scopes"]
        if scope not in scopes and len(scopes) >= 128:
            # A workspace can host many sequential Episodes. Keep the hook
            # receipt bounded without touching the active scope.
            scopes.pop(next(iter(scopes)))
        record = scopes.get(scope, {"recoveries": 0, "event_keys": []})
        if not isinstance(record, dict):
            return None
        recoveries = record.get("recoveries", 0)
        event_keys = record.get("event_keys", [])
        if type(recoveries) is not int or not isinstance(event_keys, list):
            return None
        if event_key in event_keys or recoveries >= maximum:
            return None
        recoveries += 1
        scopes[scope] = {
            "recoveries": recoveries,
            "event_keys": [*event_keys[-15:], event_key],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, sort_keys=True))
        os.replace(temporary, path)
        return recoveries
    except (OSError, TypeError, ValueError):
        return None


def output_limit_recovery_count(workspace, scope):
    """Return the number of claimed hook/outer recoveries in one runner scope."""
    try:
        state = json.loads(_output_limit_state_path(workspace).read_text())
        record = state["scopes"][scope]
        recoveries = record["recoveries"]
        return recoveries if type(recoveries) is int and recoveries >= 0 else 0
    except (OSError, KeyError, TypeError, ValueError):
        return 0


def claim_outer_output_limit_recovery(workspace, scope, event_key, maximum):
    """Share the Stop-hook allowance with terminal StopFailure recovery."""
    return _claim_output_limit_event(
        _output_limit_state_path(workspace),
        scope,
        f"outer:{event_key}",
        maximum,
    )


def _signal_output_limit_failure(payload):
    """Wake the Supervisor after Claude's non-controllable StopFailure event."""
    if payload.get("error") != "max_output_tokens":
        return
    session_id = payload.get("session_id")
    workspace_root = os.environ.get(OUTPUT_LIMIT_WORKSPACE_ENV, "")
    raw_path = os.environ.get(OUTPUT_LIMIT_SIGNAL_ENV, "")
    if not all(
        isinstance(value, str) and value
        for value in (session_id, workspace_root, raw_path)
    ):
        return
    try:
        workspace = Path(workspace_root).resolve(strict=True)
        control_root = _control_root(workspace).resolve(strict=True)
        path = Path(raw_path)
        if not path.is_absolute():
            path = control_root / path
        path = path.resolve(strict=False)
        if not path.is_relative_to(control_root):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({
            "schema_version": 1,
            "session_id": session_id,
            "error": "max_output_tokens",
        }, sort_keys=True))
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        # StopFailure ignores hook output. Failure to signal leaves the native
        # transcript watcher and terminal controller as recovery fallbacks.
        return


def preflight_dependency_violation(command):
    """Reject only violations proven by the same policy as the live guard."""
    canonical = canonical_sandbox_entrypoint()
    try:
        cwd = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        cwd = None
    return dependency_command_violation(
        ["bash", "-lc", command],
        cwd=cwd,
        trusted_sandbox_entrypoints=(canonical,),
        canonical_sandbox_entrypoint=canonical,
        protected_gateway=protected_gateway_identity(),
        enforce_dependencies=dependency_guard_enabled(),
    )


def _format_duration(seconds):
    minutes = max(1, int(math.ceil(seconds / 60)))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, remainder = divmod(minutes, 60)
    if not remainder:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours}h {remainder}m"


def _budget_reminder(state_path):
    """Return one advisory reminder per crossed active-time threshold."""
    if state_path is None:
        return None
    path = Path(state_path)
    try:
        state = json.loads(path.read_text())
        budget_id = state["budget_id"]
        total = float(state["total_seconds"])
        remaining = float(state["remaining_seconds"])
        updated = float(state["updated_monotonic"])
        paused = state["paused"]
        stage_name = state.get("stage_name", "Agent stage")
        if (
            not isinstance(budget_id, str)
            or not budget_id
            or not isinstance(stage_name, str)
            or not stage_name
            or not isinstance(paused, bool)
            or not all(math.isfinite(value) for value in (total, remaining, updated))
            or total <= 0
        ):
            return None
        if not paused:
            remaining -= max(0.0, time.monotonic() - updated)
        remaining = max(0.0, remaining)
        thresholds = sorted({
            value for value in (
                total * 0.5,
                total * 0.25,
                60 * 60,
                30 * 60,
                15 * 60,
                5 * 60,
            )
            if 0 < value < total
        }, reverse=True)

        notice_path = path.with_name("agent-active-budget-notifications.json")
        try:
            notices = json.loads(notice_path.read_text())
        except (OSError, ValueError, TypeError):
            notices = {}
        notified = (
            set(notices.get("notified_thresholds", []))
            if notices.get("budget_id") == budget_id else set()
        )
        crossed = [value for value in thresholds if remaining <= value and value not in notified]
        if not crossed:
            return None
        notified.update(crossed)
        document = {
            "schema_version": 1,
            "budget_id": budget_id,
            "notified_thresholds": sorted(notified, reverse=True),
        }
        notice_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = notice_path.with_name(f".{notice_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, sort_keys=True))
        os.replace(temporary, notice_path)
    except (OSError, KeyError, TypeError, ValueError):
        # Reminders are advisory and must always fail open.
        return None
    return (
        f"Active-time reminder: {stage_name} has approximately "
        f"{_format_duration(remaining)} remaining. Waiting for Runtime or Agate does "
        "not consume this budget. Use the remaining active time on the smallest concrete "
        "evidence-backed step. Publish a handoff only if the stage's documented handoff "
        "condition is actually satisfied; budget pressure does not make a merely promotable "
        "checkpoint terminal."
    )


def _with_reminder(output, reminder):
    if reminder:
        output.setdefault("hookSpecificOutput", {})["additionalContext"] = reminder
    return output


def hook_output(
    payload,
    timeout_ms,
    budget_state_path=None,
    output_limit_recovery_retries=None,
):
    reminder = _budget_reminder(budget_state_path)
    if not isinstance(payload, dict):
        return None
    if payload.get("hook_event_name") == "StopFailure":
        _signal_output_limit_failure(payload)
        return None
    if payload.get("tool_name") != "Bash":
        return _with_reminder({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
        }}, reminder) if reminder else None
    value = payload.get("tool_input")
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return _with_reminder({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
        }}, reminder) if reminder else None
    violation = preflight_dependency_violation(value["command"])
    if violation:
        return _with_reminder({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "The local execution boundary rejected this command before execution. "
                f"Reason: {violation}. Keep development inside the documented workspace, "
                "transport, and shared-service boundaries; use the permitted Runtime operation "
                "when remote execution is needed, then continue the same task."
            ),
        }}, reminder)
    if not any(name in value["command"] for name in ("tools/sandbox.py", "tools/plugin.py")):
        return _with_reminder({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
        }}, reminder) if reminder else None
    return _with_reminder({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow",
        "updatedInput": {**value, "run_in_background": False, "timeout": timeout_ms},
    }}, reminder)


def foreground_command(command, workspace, timeout_seconds, budget_state_path=None):
    if Path(command[0]).name != "claude":
        return command
    result = list(command)
    index = result.index("--settings") if "--settings" in result else None
    raw = (result[index + 1] if index is not None else "") or "{}"
    if not raw.lstrip().startswith("{"):
        path = Path(raw).expanduser()
        raw = (path if path.is_absolute() else workspace / path).read_text()
    settings = json.loads(raw)
    if not isinstance(settings, dict):
        raise ValueError("Claude settings must be a JSON object")
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Claude hooks must be an object")
    for event in ("PreToolUse", "StopFailure"):
        if not isinstance(hooks.setdefault(event, []), list):
            raise ValueError(f"Claude {event} hooks must be an array")
    # A Supervisor virtualenv path is deliberately absent from the Agent view.
    # This hook is stdlib-only: use the mounted system interpreter on Linux.
    interpreter = Path("/usr/bin/python3") if sys.platform == "linux" else Path(sys.executable).resolve()
    if not interpreter.is_file():
        raise ValueError("Runtime foreground hooks require a system Python interpreter")
    hook_command = (
        f"{shlex.quote(str(interpreter))} "
        f"{shlex.quote(str(Path(__file__).resolve()))} "
        f"{int(timeout_seconds * 1000)}"
    )
    if budget_state_path is not None:
        hook_command += f" {shlex.quote(str(budget_state_path))}"
    hooks["PreToolUse"].append({"matcher": "*", "hooks": [{
        "type": "command",
        "command": hook_command,
    }]})
    hooks["StopFailure"].append({"matcher": "max_output_tokens", "hooks": [{
        "type": "command",
        "command": hook_command,
    }]})
    raw = json.dumps(settings)
    if index is None:
        result[1:1] = ["--settings", raw]
    else:
        result[index + 1] = raw
    return result


if __name__ == "__main__":
    result = hook_output(
        json.load(sys.stdin),
        int(sys.argv[1]),
        Path(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
    if result is not None:
        json.dump(result, sys.stdout)
