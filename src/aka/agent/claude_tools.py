"""Persistent counters for Claude output-limit continuation."""
import json
import os
from pathlib import Path


OUTPUT_LIMIT_RECOVERY_ENV = "ATREX_OUTPUT_LIMIT_RECOVERY_RETRIES"
OUTPUT_LIMIT_SCOPE_ENV = "ATREX_OUTPUT_LIMIT_RECOVERY_SCOPE"
OUTPUT_LIMIT_SIGNAL_ENV = "ATREX_OUTPUT_LIMIT_SIGNAL"
OUTPUT_LIMIT_WORKSPACE_ENV = "ATREX_OUTPUT_LIMIT_WORKSPACE"
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
    try:
        from aka.workflow.light.support.paths import supervisor_agent_control_root
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
    """Return the number of claimed continuation recoveries in one runner scope."""
    try:
        state = json.loads(_output_limit_state_path(workspace).read_text())
        record = state["scopes"][scope]
        recoveries = record["recoveries"]
        return recoveries if type(recoveries) is int and recoveries >= 0 else 0
    except (OSError, KeyError, TypeError, ValueError):
        return 0


def claim_outer_output_limit_recovery(workspace, scope, event_key, maximum):
    """Claim one terminal output-limit recovery for this runner scope."""
    return _claim_output_limit_event(
        _output_limit_state_path(workspace),
        scope,
        f"outer:{event_key}",
        maximum,
    )
