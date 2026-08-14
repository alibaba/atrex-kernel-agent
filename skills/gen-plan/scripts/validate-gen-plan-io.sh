#!/usr/bin/env bash
# Adapted from PolyArch Humanize's validate-gen-plan-io.sh (MIT).

set -euo pipefail

usage() {
    echo "Usage: $0 --input <draft.md> --output <plan.md> [--direct|--discussion]"
    exit 6
}

input_file=""
output_file=""
mode="discussion"
mode_was_set="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            input_file="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 && "$2" != --* ]] || usage
            output_file="$2"
            shift 2
            ;;
        --direct|--discussion)
            requested_mode="${1#--}"
            if [[ "$mode_was_set" == "true" && "$mode" != "$requested_mode" ]]; then
                echo "ERROR: --direct and --discussion are mutually exclusive" >&2
                exit 6
            fi
            mode="$requested_mode"
            mode_was_set="true"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage
            ;;
    esac
done

[[ -n "$input_file" ]] || { echo "ERROR: --input is required" >&2; usage; }
[[ -n "$output_file" ]] || { echo "ERROR: --output is required" >&2; usage; }

case "$input_file" in
    /*) ;;
    *) input_file="$PWD/$input_file" ;;
esac
case "$output_file" in
    /*) ;;
    *) output_file="$PWD/$output_file" ;;
esac

output_dir="$(dirname "$output_file")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template_file="$script_dir/../templates/gen-plan-template.md"

if [[ ! -f "$input_file" ]]; then
    echo "VALIDATION_ERROR: INPUT_NOT_FOUND: $input_file" >&2
    exit 1
fi
if [[ ! -s "$input_file" ]]; then
    echo "VALIDATION_ERROR: INPUT_EMPTY: $input_file" >&2
    exit 2
fi
if [[ ! -d "$output_dir" ]]; then
    echo "VALIDATION_ERROR: OUTPUT_DIR_NOT_FOUND: $output_dir" >&2
    exit 3
fi
if [[ -e "$output_file" || -L "$output_file" ]]; then
    echo "VALIDATION_ERROR: OUTPUT_EXISTS: $output_file" >&2
    exit 4
fi
if [[ ! -w "$output_dir" ]]; then
    echo "VALIDATION_ERROR: NO_WRITE_PERMISSION: $output_dir" >&2
    exit 5
fi
if [[ ! -f "$template_file" ]]; then
    echo "VALIDATION_ERROR: TEMPLATE_NOT_FOUND: $template_file" >&2
    exit 7
fi

echo "VALIDATION_SUCCESS"
echo "INPUT_FILE: $input_file"
echo "OUTPUT_FILE: $output_file"
echo "TEMPLATE_FILE: $template_file"
echo "GEN_PLAN_MODE: $mode"
