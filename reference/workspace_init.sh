#!/bin/bash
#
# Workspace Initialization Script for GPU Kernel Optimizer
#
# Initializes the optimization workspace IN THE CURRENT DIRECTORY (the run root) — it creates the
# workspace structure, copies the kernel demo as kernel.py, initializes git, writes .gitignore +
# CLAUDE.md, and drops a workspace sentinel so the hooks gate this directory by marker (not by
# directory name). All agent operations then happen in this directory.
#
# Usage:
#     cd <run-root> && bash workspace_init.sh <name> <kernel_demo_path>
#
# <name> is a label (recorded in the sentinel, used to warn on reuse of a shared run root); it does
# NOT create a subdirectory. Example:
#     mkdir -p /work/mla_decode && cd /work/mla_decode
#     bash /path/to/reference/workspace_init.sh mla_decode /path/to/mla_decode_kernel.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NAME="${1:-}"
KERNEL_DEMO="${2:-}"

if [[ -z "$NAME" ]]; then
    echo "Error: workspace name (label) is required"
    echo "Usage: $0 <name> <kernel_demo_path>   (run from the target run-root directory)"
    exit 1
fi

if [[ -z "$KERNEL_DEMO" ]]; then
    echo "Error: kernel_demo path is required"
    echo "Usage: $0 <name> <kernel_demo_path>   (run from the target run-root directory)"
    exit 1
fi

if [[ ! -f "$KERNEL_DEMO" ]]; then
    echo "Error: kernel_demo file not found: $KERNEL_DEMO"
    exit 1
fi

WORKSPACE="$(pwd)"

# Guard: never initialize a workspace directly in the source repo root — that would git-init over the
# repo and dump kernel.py / memory into it. Run from a dedicated run-root directory instead.
if [[ -f "$WORKSPACE/install.sh" && -f "$WORKSPACE/SKILL.md" ]]; then
    echo "Error: refusing to initialize a workspace in the source repo root ($WORKSPACE)."
    echo "       cd into a dedicated run-root directory first (e.g. mkdir -p /work/<name> && cd /work/<name>)."
    exit 1
fi

echo "=========================================="
echo "  GPU Kernel Optimizer - Workspace Init"
echo "=========================================="
echo "  Name:       $NAME"
echo "  Workspace:  $WORKSPACE   (current directory)"
echo "  Kernel:     $KERNEL_DEMO"
echo "=========================================="

# Step 1: Create workspace directory structure
mkdir -p "$WORKSPACE"/{memory,plans,profiles}

# Step 2: Initialize git (local identity; guarded on a missing .git)
if [[ ! -d "$WORKSPACE/.git" ]]; then
    git init
    git config user.email "gpu-kernel-optimizer@local"
    git config user.name "GPU Kernel Optimizer"
fi

# Step 3: Copy kernel demo as kernel.py
cp "$KERNEL_DEMO" "$WORKSPACE/kernel.py"

# Step 4: Create .gitignore (also excludes run-root install artifacts co-located with the workspace)
cat > "$WORKSPACE/.gitignore" << 'EOF'
__pycache__/
*.pyc
*.ncu-rep
profiles/*/att/*.att
profiles/*/att/*.out
profiles/*/att/*.pftrace
profiles/*/att/*.otf2
# run-root install artifacts (not part of the kernel workspace)
/.claude/
/orchestrator/
/gpu-wiki
/reference-projects/
/tools
/reference
/skills
# gpu-kernel-optimizer runtime markers
.gpu_kernel_optimizer_*
EOF

# Step 5: Deploy CLAUDE.md (agent behavior constraints)
cp "$SCRIPT_DIR/CLAUDE.md" "$WORKSPACE/CLAUDE.md"

# Step 6: Drop the workspace sentinel (marks this dir as a workspace for the hooks)
printf '%s\n' "$NAME" > "$WORKSPACE/.gpu_kernel_optimizer_workspace"

echo ""
echo "Workspace initialized at: $WORKSPACE"
echo ""
echo "Directory structure:"
echo "  $WORKSPACE/   (your current working directory — all agent operations happen here)"
echo "  ├── kernel.py          (copied from kernel_demo)"
echo "  ├── CLAUDE.md          (agent behavior constraints)"
echo "  ├── .gitignore"
echo "  ├── .gpu_kernel_optimizer_workspace   (workspace sentinel)"
echo "  ├── memory/            (iteration JSON files)"
echo "  ├── plans/             (optimization plans)"
echo "  └── profiles/          (profiling artifacts)"
echo ""
echo "Next steps:"
echo "  1. Parse user input (platform, framework, dtype, shapes)"
echo "  2. Run Step 0: Hardware spec lookup + Roofline analysis"
echo "  3. Write README.md with Stop Conditions"
echo "  4. Enter Stage 1: Baseline Implementation"
