"""Execution failures are facts, not guessed numerical verdicts."""
from aka.contracts.task import Assessment


def execution_failure(receipt):
    payload = receipt.payload
    error = payload.get("error") or {}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    category = error.get("code", "")
    origin = error.get("failure_origin", payload.get("failure_origin", ""))
    output = payload.get("result", payload)
    diagnostics = {key: output[key][-8192:] for key in ("stdout", "stderr") if isinstance(output.get(key), str)}
    reason = error.get("message") or payload.get("reason") or f"Execution {receipt.status}; inspect the retained receipt"
    if receipt.status in {"unknown", "cancelled"}:
        kind = "unknown"
    elif category.startswith(("infra/", "timeout/")) or origin in {"infra", "infrastructure"}:
        kind = "infrastructure"
    elif category in {"code/native_build_failed", "code/compile_error", "compilation_failed"}:
        kind = "build"
    elif category or origin or type(payload.get("result", payload).get("exit_code")) is int:
        kind = "execution"
    else:
        kind = "unknown"
    return Assessment("unknown" if kind in {"unknown", "infrastructure"} else "fail", False,
                      reason=reason, failure_kind=kind,
                      details={"execution_status": receipt.status, "error": error, **diagnostics})
