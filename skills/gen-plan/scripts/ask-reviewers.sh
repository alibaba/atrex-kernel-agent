#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 --input <draft.md> [--context <file>]..." >&2
    exit 2
}

input_file=""
context_files=()

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
        -h|--help)
            usage
            ;;
        *)
            echo "ask-reviewers: unknown option: $1" >&2
            usage
            ;;
    esac
done

[[ -n "$input_file" ]] || { echo "ask-reviewers: --input is required" >&2; usage; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
reviewer_args=(--input "$input_file")
for ((index = 0; index < ${#context_files[@]}; index++)); do
    reviewer_args+=(--context "${context_files[$index]}")
done

# Keep the two consultations independent: both receive the same immutable evidence packet, start
# concurrently, and cannot observe the other review before producing their own response.
python3 - "$script_dir" "${reviewer_args[@]}" <<'PY'
import concurrent.futures
import os
import subprocess
import sys

script_dir = sys.argv[1]
reviewer_args = sys.argv[2:]
reviewers = (
    ("CODEX", os.path.join(script_dir, "ask-codex.sh"), "ASK_CODEX_SKIPPED:"),
    ("QODER", os.path.join(script_dir, "ask-qoder.sh"), "ASK_QODER_SKIPPED:"),
)


def run_reviewer(spec):
    name, helper, skip_marker = spec
    try:
        completed = subprocess.run(
            ["bash", helper, *reviewer_args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        reason = f"failed to start helper: {exc}"
        return name, "unavailable", "", reason, reason

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0 and stdout.startswith(skip_marker):
        status = f"current_{name.lower()}_session"
    elif completed.returncode == 0:
        status = "completed"
    else:
        status = f"unavailable (exit {completed.returncode})"
    failure_reason = ""
    if completed.returncode != 0:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        failure_reason = next(
            (
                line
                for line in reversed(lines)
                if "consultation failed with exit code" not in line
                and "running read-only consultation" not in line
                and "running isolated read-only consultation" not in line
            ),
            f"helper exited with code {completed.returncode}",
        )
    return name, status, stdout, stderr, failure_reason


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(run_reviewer, spec) for spec in reviewers]
    results = [future.result() for future in futures]

for name, status, stdout, stderr, failure_reason in results:
    if stderr:
        for line in stderr.splitlines():
            print(f"ask-reviewers[{name.lower()}]: {line}", file=sys.stderr)
    print(f"=== {name} REVIEW ===")
    print(f"{name}_CONSULTATION_STATUS: {status}")
    if failure_reason:
        print(f"{name}_FAILURE_REASON: {failure_reason}")
    if stdout and not stdout.startswith(f"ASK_{name}_SKIPPED:"):
        print(stdout)
    print(f"=== END {name} REVIEW ===")
PY
