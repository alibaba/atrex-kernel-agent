#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 --input <draft.md> [--context <file>]... [--timeout <seconds>]" >&2
    exit 2
}

input_file=""
context_files=()
qoder_timeout="${ATREX_ASK_QODER_TIMEOUT:-600}"
# Keep the independent review at maximum reasoning depth regardless of the active episode or
# settings file. The explicit CLI flag below has precedence over configured defaults.
reasoning_effort="max"
max_output_tokens="${ATREX_ASK_QODER_MAX_OUTPUT_TOKENS:-4096}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            input_file="$2"
            shift 2
            ;;
        --context)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            context_files+=("$2")
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            qoder_timeout="$2"
            shift 2
            ;;
        --reasoning-effort)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            echo "ask-qoder: ignoring --reasoning-effort=$2; reviewer effort is fixed at max" >&2
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ask-qoder: unknown option: $1" >&2
            usage
            ;;
    esac
done

[[ -n "$input_file" ]] || { echo "ask-qoder: --input is required" >&2; usage; }
[[ "$qoder_timeout" =~ ^[1-9][0-9]*$ ]] || {
    echo "ask-qoder: timeout must be a positive integer" >&2
    exit 2
}
[[ "$max_output_tokens" =~ ^[1-9][0-9]*$ ]] || {
    echo "ask-qoder: ATREX_ASK_QODER_MAX_OUTPUT_TOKENS must be a positive integer" >&2
    exit 2
}
if [[ ! -f "$input_file" || ! -s "$input_file" ]]; then
    echo "ask-qoder: input draft is missing or empty: $input_file" >&2
    exit 1
fi
for ((index = 0; index < ${#context_files[@]}; index++)); do
    context_file="${context_files[$index]}"
    if [[ ! -f "$context_file" ]]; then
        echo "ask-qoder: context file not found: $context_file" >&2
        exit 1
    fi
done

# A Qoder-owned episode already has Qoder's analysis available in the current session. Starting a
# second Qoder process would add recursion without providing an independent backend perspective.
if [[ "${ATREX_AGENT_CLI:-}" == "qodercli" ]]; then
    echo "ASK_QODER_SKIPPED: current agent backend is qodercli; use the current session's review"
    exit 0
fi

qoder_bin="${ATREX_QODER_BIN:-qodercli}"
if [[ "$qoder_bin" == */* ]]; then
    if [[ ! -x "$qoder_bin" ]]; then
        echo "ask-qoder: Qoder executable not found: $qoder_bin" >&2
        exit 127
    fi
elif ! command -v "$qoder_bin" >/dev/null 2>&1; then
    echo "ask-qoder: qodercli is not installed or not on PATH" >&2
    exit 127
fi

query="$(printf '%s\n' \
    'Act as an independent reviewer for a GPU-kernel implementation plan.' \
    'The first attachment is the original planning draft. Remaining attachments are bounded repository context.' \
    'Do not edit files, implement code, invoke tools, or broaden the draft into multiple optimization categories.' \
    'Challenge unsupported inferences, identify missing correctness/performance requirements, and recommend one coherent direction.' \
    'Return concise plain text using exactly these section markers:' \
    'QODER_SUMMARY:' \
    'RISKS:' \
    'MISSING_REQUIREMENTS:' \
    'DIRECTION_RECOMMENDATIONS:' \
    'VALIDATION_RECOMMENDATIONS:' \
    'QUESTIONS_OR_ASSUMPTIONS:')"

command=(
    "$qoder_bin"
    --print
    --no-session-persistence
    --permission-mode dont_ask
    --reasoning-effort "$reasoning_effort"
    --max-output-tokens "$max_output_tokens"
)
if [[ -n "${ATREX_QODER_SESSION_SETTINGS:-}" ]]; then
    command+=(--settings "$ATREX_QODER_SESSION_SETTINGS")
fi
command+=(--attachment "$input_file")
for ((index = 0; index < ${#context_files[@]}; index++)); do
    context_file="${context_files[$index]}"
    command+=(--attachment "$context_file")
done
# Empty tools keeps the consultation read-only. Attachments provide all required context.
command+=(--tools "" -- "$query")

echo "ask-qoder: running read-only consultation (timeout=${qoder_timeout}s, effort=$reasoning_effort)" >&2
if qoder_response="$(python3 - "$qoder_timeout" "${command[@]}" <<'PY'
import os
import subprocess
import sys

timeout = int(sys.argv[1])
command = sys.argv[2:]
environment = os.environ.copy()
environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
try:
    completed = subprocess.run(
        command, check=False, timeout=timeout, env=environment
    )
except subprocess.TimeoutExpired:
    print(f"ask-qoder: timed out after {timeout}s", file=sys.stderr)
    raise SystemExit(124)
except OSError as exc:
    print(f"ask-qoder: failed to start qodercli: {exc}", file=sys.stderr)
    raise SystemExit(127)

return_code = completed.returncode
if return_code < 0:
    return_code = 128 + abs(return_code)
raise SystemExit(return_code)
PY
)"; then
    missing_markers=""
    for marker in \
        "QODER_SUMMARY:" \
        "RISKS:" \
        "MISSING_REQUIREMENTS:" \
        "DIRECTION_RECOMMENDATIONS:" \
        "VALIDATION_RECOMMENDATIONS:" \
        "QUESTIONS_OR_ASSUMPTIONS:"
    do
        if [[ "$qoder_response" != *"$marker"* ]]; then
            missing_markers="${missing_markers}${missing_markers:+, }${marker%:}"
        fi
    done
    if [[ -n "$missing_markers" ]]; then
        echo "ask-qoder: malformed response; missing section(s): $missing_markers" >&2
        exit 3
    fi
    printf '%s\n' "$qoder_response"
    exit 0
else
    qoder_status=$?
    echo "ask-qoder: consultation failed with exit code $qoder_status" >&2
    exit "$qoder_status"
fi
