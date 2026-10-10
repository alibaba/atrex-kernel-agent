"""Public plugin output contract and classified private-executor failures."""
from __future__ import annotations

import json
import logging
from typing import NoReturn

from plugin_runtime.schema import decode_json, validate_schema

from .errors import AgentRequestError, RuntimeStateError

MAX_OUTPUT_BYTES = 256 * 1024
# Only the Supervisor wrapper emits this code/envelope. A plugin's own nonzero
# exit is wrapped as tool_failed and must not be interpreted as a known outcome.
OUTPUT_ERROR_EXIT_CODE = 65
OUTPUT_ERROR_FIELD = "plugin_output_error"
OUTPUT_ERROR_CODES = frozenset({"invalid_output", "result_too_large"})


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
    raise RuntimeError("Unrecognized private plugin output classification")


def project_plugin_output(name: str, schema: dict, stdout: str) -> str:
    """Validate successful output; never include its content in public errors."""
    try:
        result = decode_json(stdout)
        validate_schema(schema, result, "output")
        rendered = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
    except (ValueError, TypeError, RecursionError):
        logging.getLogger(__name__).exception("Plugin %s violated its output contract", name)
        raise_output_error(name, "invalid_output")
    if len(rendered.encode()) > MAX_OUTPUT_BYTES:
        raise_output_error(name, "result_too_large")
    return rendered


def check_executor_output_failure(name: str, returncode: int, stdout: str) -> None:
    """Recognize only the wrapper's small, fixed error protocol after dispatch."""
    if returncode == OUTPUT_ERROR_EXIT_CODE and len(stdout) <= 1024:
        try:
            value = decode_json(stdout)
        except ValueError:
            return
        if (isinstance(value, dict) and set(value) == {OUTPUT_ERROR_FIELD}
                and isinstance(value[OUTPUT_ERROR_FIELD], str)
                and value[OUTPUT_ERROR_FIELD] in OUTPUT_ERROR_CODES):
            raise_output_error(name, value[OUTPUT_ERROR_FIELD])
