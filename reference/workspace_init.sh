#!/bin/bash
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

WORKSPACE="$(pwd)/kernel_opt_${NAME}"

echo "=========================================="
echo "  GPU Kernel Optimizer - Workspace Init"
echo "=========================================="
echo "  Name:       $NAME"
echo "  Workspace:  $WORKSPACE"
echo "  Kernel:     $KERNEL_DEMO"
echo "=========================================="

# Step 1: Create workspace directory structure
mkdir -p "$WORKSPACE"/{memory,scratch}

# Step 2: Initialize git
cd "$WORKSPACE"
if [[ ! -d .git ]]; then
    git init
    git config user.email "gpu-kernel-optimizer@local"
    git config user.name "GPU Kernel Optimizer"
fi

# Step 3: Copy kernel demo as kernel.py
cp "$KERNEL_DEMO" "$WORKSPACE/kernel.py"

# Step 4: Install private Git excludes (no Agent-facing .gitignore).
PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}" python3 -m orchestrator.git_metadata "$WORKSPACE"

# Step 5: Deploy the orchestrator's Agent behavior constraints.
if [[ ! -f "$SCRIPT_DIR/CLAUDE.md" ]]; then
    echo "Error: orchestrator constraints not found: $SCRIPT_DIR/CLAUDE.md"
    exit 1
fi
cp "$SCRIPT_DIR/CLAUDE.md" "$WORKSPACE/CLAUDE.md"

echo ""
echo "Workspace initialized at: $WORKSPACE"
echo ""
echo "Directory structure:"
echo "  $WORKSPACE/"
echo "  ├── kernel.py          (copied from kernel_demo)"
echo "  ├── CLAUDE.md          (agent behavior constraints)"
echo "  ├── memory/            (iteration JSON files)"
echo "  └── scratch/           (temporary requests and optional diagnostics)"
echo ""
echo "Supervisor will write README.md, measure V0, and record the baseline."
echo "Framework Baseline and optimization sessions follow according to campaign settings."
