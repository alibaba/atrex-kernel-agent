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
mkdir -p "$WORKSPACE/scratch"

# Step 2: Initialize git
cd "$WORKSPACE"
if [[ ! -d .git ]]; then
    git init
    git config user.email "gpu-kernel-optimizer@local"
    git config user.name "GPU Kernel Optimizer"
fi

# Step 3: Initialize the unified editable source tree
cp "$KERNEL_DEMO" "$WORKSPACE/kernel.py"

# Step 4: Create .gitignore
cat > "$WORKSPACE/.gitignore" << 'EOF'
# Runtime-managed Agent repository exclusions
__pycache__/
*.pyc
*.ncu-rep
profiles/*/att/*.att
profiles/*/att/*.out
profiles/*/att/*.pftrace
profiles/*/att/*.otf2
/scratch
/tools
/skills
/research-library
/reference
/reference-projects
/references
/gpu-wiki
/.claude
/.qoder
/.agents
/.orchestrator_mode.json
/.orchestrator_state.json
/.atrex_long_horizon/
/verification_artifacts/.atrex_long_horizon_verify/
/atrex-bench
EOF

# Step 5: Deploy the orchestrator's Agent behavior constraints.
if [[ ! -f "$SCRIPT_DIR/CLAUDE.md" ]]; then
    echo "Error: orchestrator constraints not found: $SCRIPT_DIR/CLAUDE.md"
    exit 1
fi
cp "$SCRIPT_DIR/CLAUDE.md" "$WORKSPACE/CLAUDE.md"

echo ""
echo "Workspace initialized at: $WORKSPACE"
echo ""
echo "Supervisor staging ready: kernel.py + impl/, CLAUDE.md, scratch/."
echo "kernel.py is the evaluator entry point; optional host/CUDA helpers belong under impl/."
