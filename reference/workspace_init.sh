#!/bin/bash
# Copyright 2026 Alibaba Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Workspace Initialization Script for GPU Kernel Optimizer
#
# This is the FIRST file to land when starting an optimization session.
# It creates the workspace structure, copies the kernel demo, and initializes git.
#
# Usage:
#     bash workspace_init.sh <name> <kernel_demo_path>
#
# Example:
#     bash workspace_init.sh mla_decode /path/to/mla_decode_kernel.py

set -euo pipefail

NAME="${1:-}"
KERNEL_DEMO="${2:-}"

if [[ -z "$NAME" ]]; then
    echo "Error: workspace name is required"
    echo "Usage: $0 <name> <kernel_demo_path>"
    exit 1
fi

if [[ -z "$KERNEL_DEMO" ]]; then
    echo "Error: kernel_demo path is required"
    echo "Usage: $0 <name> <kernel_demo_path>"
    exit 1
fi

if [[ ! -f "$KERNEL_DEMO" ]]; then
    echo "Error: kernel_demo file not found: $KERNEL_DEMO"
    exit 1
fi

WORKSPACE="/tmp/kernel_opt_${NAME}"

echo "=========================================="
echo "  GPU Kernel Optimizer - Workspace Init"
echo "=========================================="
echo "  Name:       $NAME"
echo "  Workspace:  $WORKSPACE"
echo "  Kernel:     $KERNEL_DEMO"
echo "=========================================="

# Step 1: Create workspace directory structure
mkdir -p "$WORKSPACE"/{memory,plans,profiles}

# Step 2: Initialize git
cd "$WORKSPACE"
if [[ ! -d .git ]]; then
    git init
    git config user.email "gpu-kernel-optimizer@local"
    git config user.name "GPU Kernel Optimizer"
fi

# Step 3: Copy kernel demo as kernel.py
cp "$KERNEL_DEMO" "$WORKSPACE/kernel.py"

# Step 4: Create .gitignore
cat > "$WORKSPACE/.gitignore" << 'EOF'
__pycache__/
*.pyc
*.ncu-rep
profiles/*/att/*.att
profiles/*/att/*.out
profiles/*/att/*.pftrace
profiles/*/att/*.otf2
EOF

echo ""
echo "Workspace initialized at: $WORKSPACE"
echo ""
echo "Directory structure:"
echo "  $WORKSPACE/"
echo "  ├── kernel.py          (copied from kernel_demo)"
echo "  ├── .gitignore"
echo "  ├── memory/            (iteration JSON files)"
echo "  ├── plans/             (optimization plans)"
echo "  └── profiles/          (profiling artifacts)"
echo ""
echo "Next steps:"
echo "  1. Parse user input (platform, framework, dtype, shapes)"
echo "  2. Run Step 0: Hardware spec lookup + Roofline analysis"
echo "  3. Write README.md with Stop Conditions"
echo "  4. Enter Stage 1: Baseline Implementation"
