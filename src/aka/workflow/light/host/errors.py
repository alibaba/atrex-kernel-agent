"""Agent-facing errors with Runtime-style issues and actionable recovery steps."""

from __future__ import annotations

from typing import Any

REPAIR_REQUEST = (
    "Correct the indicated input before retrying. Use python3 tools/sandbox.py "
    "--kind OPERATION --help for CLI options and the session instructions for JSON fields."
)
ESCALATE_RUNTIME = (
    "Report this infrastructure blocker to the operator. Do not install packages, edit "
    "private Runtime files, change credentials, or bypass tools/sandbox.py."
)
UNKNOWN_OUTCOME = (
    "The operation may already have taken effect. Inspect available records before "
    "resubmitting a mutation or GPU request; do not retry blindly. If its outcome cannot "
    "be confirmed, report the blocker to the operator."
)


def error_response(
    message: str,
    *,
    code: str = "invalid_request",
    next_action: str = REPAIR_REQUEST,
    repairable: bool = True,
    **details: Any,
) -> dict[str, Any]:
    issues = details.pop("issues", None)
    recovery = details.pop("recovery", None)
    return {
        "ok": False,
        # Repairable means the Agent can fix its input/state, not automatic replay safety.
        "repairable": repairable,
        "detail": message[:2000],
        "issues": issues if issues is not None else [
            {"path": "$", "code": code, "message": message[:2000]}
        ],
        "recovery": recovery if recovery is not None else [{"instruction": next_action}],
        "error": {
            "code": code,
            "message": message[:2000],
            "next_action": next_action,
            **details,
        },
    }


class AgentRequestError(ValueError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.response = error_response(message, **details)


def field_error(path: str, message: str, *, code="invalid_value", **details) -> AgentRequestError:
    return AgentRequestError(message, code=code,
                             issues=[{"path": path, "code": code, "message": message}], **details)


class RuntimeStateError(RuntimeError):
    """A Supervisor-owned failure, not a malformed Agent request."""

    def __init__(self, message: str, *, code: str = "runtime_state_unavailable",
                 next_action: str = ESCALATE_RUNTIME) -> None:
        super().__init__(message)
        self.response = error_response(
            message, code=code, repairable=False, next_action=next_action
        )


class GatewayConfigurationError(RuntimeStateError):
    """Invalid operator Gateway configuration, not an Agent input error."""

    def __init__(self, message: str | None = None, *, next_action: str | None = None) -> None:
        super().__init__(
            message or "Supervisor ATREX_AGATE_EXECUTABLE does not resolve to an executable.",
            code="gateway_configuration_invalid",
            next_action=next_action or (
                "Ask the operator to correct ATREX_AGATE_EXECUTABLE to an executable path "
                "or command on the Supervisor PATH, or unset it to use default discovery, "
                "then restart the Supervisor. Do not change Agent arguments, install a "
                "replacement client, or bypass the Runtime."
            ),
        )


def require_fields(
    value: dict[str, Any], required: set[str], *, optional: set[str] | None = None, label: str
) -> None:
    allowed = required | (optional or set())
    missing, unexpected = sorted(required - value.keys()), sorted(value.keys() - allowed)
    if missing or unexpected:
        raise AgentRequestError(
            f"{label} fields do not match the request format.",
            code="invalid_fields",
            missing_fields=missing,
            unexpected_fields=unexpected,
            allowed_fields=sorted(allowed),
            issues=[{"path": field, "code": "missing", "message": "Field required"}
                    for field in missing]
                   + [{"path": field, "code": "extra_forbidden", "message": "Unexpected field"}
                      for field in unexpected],
            next_action=(
                "Add missing_fields and remove unexpected_fields in the request JSON, then "
                "resubmit. Read the session instructions for field types and meanings."
            ),
        )
