"""Public plugin output contract and classified private-executor failures."""
from __future__ import annotations

from typing import NoReturn

from .errors import AgentRequestError, RuntimeStateError

MAX_OUTPUT_BYTES = 256 * 1024


def raise_output_error(name: str, code: str) -> NoReturn:
    if code == "result_too_large":
        raise AgentRequestError(
            f"Plugin {name!r} completed execution, but its result exceeds the "
            f"{MAX_OUTPUT_BYTES}-byte public output limit.",
            code="plugin_result_too_large", tool=name, max_output_bytes=MAX_OUTPUT_BYTES,
            execution_completed=True,
            next_action=(
                "Narrow the question or reduce max_records/max_bytes if the tool's input schema "
                "supports them (python3 tools/plugin.py list), then submit the adjusted request. "
                "The original call already ran: verify any side effects before reissuing a "
                "mutating call. If output cannot be narrowed, ask the plugin operator to fix it."
            ),
        )
    if code == "invalid_output":
        raise RuntimeStateError(
            f"Plugin {name!r} completed execution but returned invalid JSON or a result that "
            "violates its declared output schema.",
            code="plugin_output_invalid",
            next_action=(
                f"Report the output-contract failure for {name!r} to the plugin operator. "
                "Changing request arguments or repeating the same call is not a repair; "
                "the operator must inspect the private diagnostics and fix the plugin. "
                "The call already ran, so verify any side effects before retrying after repair."
            ),
        )
    raise RuntimeError("Unrecognized plugin output classification")
