#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 --input <draft.md> --proposal <candidate.md> [--context <file>]..." >&2
    exit 2
}

input_file=""
proposal_file=""
context_files=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            input_file="$2"
            shift 2
            ;;
        --proposal)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            proposal_file="$2"
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
[[ -n "$proposal_file" ]] || { echo "ask-reviewers: --proposal is required" >&2; usage; }
if [[ ! -f "$input_file" || ! -s "$input_file" ]]; then
    echo "ask-reviewers: input draft is missing or empty: $input_file" >&2
    exit 1
fi
if [[ ! -f "$proposal_file" || ! -s "$proposal_file" ]]; then
    echo "ask-reviewers: candidate proposal is missing or empty: $proposal_file" >&2
    exit 1
fi
for ((index = 0; index < ${#context_files[@]}; index++)); do
    if [[ ! -f "${context_files[$index]}" ]]; then
        echo "ask-reviewers: context file not found: ${context_files[$index]}" >&2
        exit 1
    fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
reviewer_args=(--input "$input_file" --proposal "$proposal_file")
for ((index = 0; index < ${#context_files[@]}; index++)); do
    reviewer_args+=(--context "${context_files[$index]}")
done

# Keep the two consultations independent: both review the same immutable candidate proposal against
# the same evidence packet, start concurrently, and cannot observe the other review before producing
# their own response.
python3 - "$script_dir" "${reviewer_args[@]}" <<'PY'
import concurrent.futures
import os
import pathlib
import subprocess
import sys
import tempfile

script_dir = sys.argv[1]
raw_reviewer_args = sys.argv[2:]
reviewers = (
    (
        "CODEX",
        os.path.join(script_dir, "ask-codex.sh"),
        "ASK_CODEX_SKIPPED:",
        "ASK_CODEX_DISABLED:",
    ),
    (
        "QODER",
        os.path.join(script_dir, "ask-qoder.sh"),
        "ASK_QODER_SKIPPED:",
        "ASK_QODER_DISABLED:",
    ),
)
non_retryable_markers = (
    "quota",
    "rate limit",
    "usage limit",
    "insufficient credit",
    "authentication",
    "unauthorized",
    "forbidden",
)


def shared_reviewer_args(raw_args, scratch):
    base_args = []
    context_files = []
    index = 0
    while index < len(raw_args):
        option = raw_args[index]
        if index + 1 >= len(raw_args):
            raise ValueError(f"missing value for {option}")
        value = raw_args[index + 1]
        if option == "--context":
            context_files.append(pathlib.Path(value))
        else:
            base_args.extend((option, value))
        index += 2
    # Qoder supports at most five attachments. Proposal and draft consume two, so
    # collapse larger context packets into one labeled text attachment and give that
    # exact same bundle to Codex to preserve independent-review parity.
    if len(context_files) <= 3:
        return [
            *base_args,
            *(
                item
                for context_file in context_files
                for item in ("--context", str(context_file))
            ),
        ]
    bundle = pathlib.Path(scratch) / "shared-context-bundle.md"
    parts = ["# Shared Plan Review Context Bundle\n"]
    for context_file in context_files:
        parts.extend(
            (
                f"\n--- CONTEXT FILE: {context_file} ---\n",
                context_file.read_text(encoding="utf-8", errors="replace"),
                "\n--- END CONTEXT FILE ---\n",
            )
        )
    bundle.write_text("".join(parts), encoding="utf-8")
    print(
        "ask-reviewers: bundled "
        f"{len(context_files)} context files into one shared reviewer attachment",
        file=sys.stderr,
    )
    return [*base_args, "--context", str(bundle)]


def run_once(helper, reviewer_args):
    try:
        return subprocess.run(
            ["bash", helper, *reviewer_args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            ["bash", helper, *reviewer_args],
            127,
            "",
            f"failed to start helper: {exc}",
        )


def should_retry(completed):
    if completed.returncode not in {1, 3}:
        return False
    diagnostic = f"{completed.stdout}\n{completed.stderr}".casefold()
    return not any(marker in diagnostic for marker in non_retryable_markers)


def run_reviewer(spec, reviewer_args):
    name, helper, skip_marker, disabled_marker = spec
    completed = run_once(helper, reviewer_args)
    stderr_parts = [completed.stderr.strip()] if completed.stderr.strip() else []
    if should_retry(completed):
        retry_notice = (
            f"retrying only failed {name.lower()} consultation once "
            f"(exit {completed.returncode})"
        )
        stderr_parts.append(retry_notice)
        completed = run_once(helper, reviewer_args)
        if completed.stderr.strip():
            stderr_parts.append(completed.stderr.strip())

    stdout = completed.stdout.strip()
    stderr = "\n".join(stderr_parts)
    if completed.returncode == 0 and stdout.startswith(skip_marker):
        status = f"current_{name.lower()}_session"
    elif completed.returncode == 0 and stdout.startswith(disabled_marker):
        status = "disabled_after_startup_probe"
    elif completed.returncode == 0:
        status = "completed"
    else:
        status = f"unavailable (exit {completed.returncode})"
    failure_reason = ""
    if completed.returncode == 0 and stdout.startswith(disabled_marker):
        failure_reason = stdout.removeprefix(disabled_marker).strip()
    elif completed.returncode != 0:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        failure_reason = next(
            (
                line
                for line in reversed(lines)
                if "consultation failed with exit code" not in line
                and "running read-only consultation" not in line
                and "running isolated read-only consultation" not in line
                and "running campaign-persistent read-only consultation" not in line
            ),
            f"helper exited with code {completed.returncode}",
        )
    if stdout.startswith((skip_marker, disabled_marker)):
        stdout = ""
    return name, status, stdout, stderr, failure_reason


with tempfile.TemporaryDirectory(prefix="atrex-plan-review-packet-") as scratch:
    reviewer_args = shared_reviewer_args(raw_reviewer_args, scratch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_reviewer, spec, reviewer_args) for spec in reviewers
        ]
        results = [future.result() for future in futures]

    for name, status, stdout, stderr, failure_reason in results:
        if stderr:
            for line in stderr.splitlines():
                print(f"ask-reviewers[{name.lower()}]: {line}", file=sys.stderr)
        print(f"=== {name} REVIEW ===")
        print(f"{name}_CONSULTATION_STATUS: {status}")
        if failure_reason:
            print(f"{name}_FAILURE_REASON: {failure_reason}")
        if stdout:
            print(stdout)
        print(f"=== END {name} REVIEW ===")
PY
