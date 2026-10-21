"""Invocation-local Claude foreground hook for the Supervisor HTTP clients."""
import json
from pathlib import Path
import shlex
import sys


def hook_output(payload, timeout_ms):
    if not isinstance(payload, dict) or payload.get("tool_name") != "Bash":
        return None
    value = payload.get("tool_input")
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        return None
    if not any(name in value["command"] for name in ("tools/sandbox.py", "tools/plugin.py")):
        return None
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow",
        "updatedInput": {**value, "run_in_background": False, "timeout": timeout_ms},
    }}


def foreground_command(command, workspace, timeout_seconds):
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
    if not isinstance(hooks, dict) or not isinstance(hooks.setdefault("PreToolUse", []), list):
        raise ValueError("Claude PreToolUse hooks must be an array")
    # A Supervisor virtualenv path is deliberately absent from the Agent view.
    # This hook is stdlib-only: use the mounted system interpreter on Linux.
    interpreter = Path("/usr/bin/python3") if sys.platform == "linux" else Path(sys.executable).resolve()
    if not interpreter.is_file():
        raise ValueError("Runtime foreground hooks require a system Python interpreter")
    hooks["PreToolUse"].append({"matcher": "Bash", "hooks": [{
        "type": "command",
        "command": f"{shlex.quote(str(interpreter))} {shlex.quote(str(Path(__file__).resolve()))} {int(timeout_seconds * 1000)}",
    }]})
    raw = json.dumps(settings)
    if index is None:
        result[1:1] = ["--settings", raw]
    else:
        result[index + 1] = raw
    return result


if __name__ == "__main__":
    result = hook_output(json.load(sys.stdin), int(sys.argv[1]))
    if result is not None:
        json.dump(result, sys.stdout)
