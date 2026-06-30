#!/bin/bash
#
# Workspace Initialization Script for GPU Kernel Optimizer
#
# This is the FIRST file to land when starting an optimization session.
# It creates the workspace structure, copies the kernel demo, and initializes git.
#
# Usage:
#     bash workspace_init.sh <name> <kernel_demo_path>
#     bash workspace_init.sh --sol-execbench <problem_dir> [<name>]
#
# Example:
#     bash workspace_init.sh mla_decode /path/to/mla_decode_kernel.py
#     bash workspace_init.sh --sol-execbench /path/SOL-ExecBench/data/benchmark/L1/067_...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" 2>/dev/null && pwd || echo "$SCRIPT_DIR/../tools")"
PYTHON="${PYTHON:-python3}"

# --- SOL-ExecBench mode: parse a problem dir into an AKA workspace ----------
# Delegates to tools/sol_adapter.py, which generates reference.py, a DPS kernel.py
# stub, a SOL-faithful test_kernel.py (static anti-cheat gate + authoritative
# sol-execbench eval + honest T_b), and a frozen baseline.json.
if [[ "${1:-}" == "--sol-execbench" ]]; then
    PROBLEM_DIR="${2:-}"
    SOL_NAME="${3:-}"
    if [[ -z "$PROBLEM_DIR" || ! -d "$PROBLEM_DIR" ]]; then
        echo "Error: --sol-execbench requires an existing <problem_dir>"
        echo "Usage: $0 --sol-execbench <problem_dir> [<name>]"
        exit 1
    fi
    exec "$PYTHON" "$TOOLS_DIR/sol_adapter.py" materialize "$PROBLEM_DIR" ${SOL_NAME:+"$SOL_NAME"} --dest "$(pwd)"
fi

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
