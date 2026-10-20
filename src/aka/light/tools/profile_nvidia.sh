#!/bin/bash
# Copyright 2026 Alibaba Group. Licensed under the Apache License, Version 2.0.
# Run through sandbox.py --kind profile, on the GPU worker. Collection and
# parsing are remote; the coordinator needs neither NCU nor ncu_report.
# --source exports source-correlated CUDA/SASS text using the NCU CLI.
# Raw binary reports stay on the worker unless explicitly requested.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/ncu_profile.py" "$@"
