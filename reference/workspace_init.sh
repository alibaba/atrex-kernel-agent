#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${AKA_TASK_PYTHON:-python3}" "$SCRIPT_DIR/kernel_source.py" --initialize "$SCRIPT_DIR" "$0" "$@"
