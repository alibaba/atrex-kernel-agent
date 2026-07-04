#!/usr/bin/env bash
# One-shot installer for the gpu-kernel-optimizer skill + hooks.
#
# Default install base: /tmp/aka-opt (customizable via AKA_KERNEL_OPT_HOME)
# Files are installed under <base>/.codex and/or <base>/.claude:
#   - Codex:  /tmp/aka-opt/.codex/skills/gpu-kernel-optimizer/
#             /tmp/aka-opt/.codex/hooks/
#   - Claude: /tmp/aka-opt/.claude/skills/gpu-kernel-optimizer/
#             /tmp/aka-opt/.claude/hooks/
#   - Shared: /tmp/aka-opt/gpu-wiki/
#             /tmp/aka-opt/reference-projects/
#             /tmp/aka-opt/orchestrator/   (optimize.py + prompts/ + anchor_bench.py)
#
# The orchestrator makes <base> a self-contained per-kernel run root — one <base> per kernel,
# fully isolated from every other kernel:
#     cd <base> && python orchestrator/optimize.py --op-dir <op> --platform <hw> --framework <dsl>
# At runtime optimize.py reads its helpers from <base>/.claude (populated above) and pins each
# nested session's CLAUDE_CONFIG_DIR to <base>/.claude, so config/hooks/history never cross.
#
# Skill directory whitelist (only these are copied):
#   reference/  skills/  tools/  SKILL.md
#
# agents/ is copied separately to $TARGET_DIR/agents/ (not inside the skill dir).
#
# Usage:
#   ./install.sh                       # install/update all detected targets
#   ./install.sh --prefix /my/path     # specify custom install directory
#   ./install.sh --hooks-only          # only install/update hooks for detected targets
#   ./install.sh --without-github      # install/update, but skip GitHub reference repos from gpu-wiki
#   ./install.sh --uninstall           # remove hooks installed by this script from detected targets
#
# Idempotent. Safe to re-run.

set -euo pipefail

SKILL_NAME="gpu-kernel-optimizer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_WIKI_SOURCE_DIR="$SCRIPT_DIR/gpu-wiki"

ROCPROF_TRACE_DECODER_REPO="https://github.com/ROCm/rocprof-trace-decoder.git"

MODE="install"
WITHOUT_GITHUB="0"
CLI_PREFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "ERROR: --prefix requires a directory path."
        exit 2
      fi
      CLI_PREFIX="$2"
      shift 2
      ;;
    --prefix=*)
      CLI_PREFIX="${1#*=}"
      if [[ -z "$CLI_PREFIX" ]]; then
        echo "ERROR: --prefix requires a directory path."
        exit 2
      fi
      shift
      ;;
    --hooks-only)     MODE="hooks-only"; shift ;;
    --without-github) WITHOUT_GITHUB="1"; shift ;;
    --uninstall)      MODE="uninstall"; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# Resolve install base: --prefix > AKA_KERNEL_OPT_HOME > /tmp/aka-opt
if [[ -n "$CLI_PREFIX" ]]; then
  INSTALL_BASE="$CLI_PREFIX"
else
  INSTALL_BASE="${AKA_KERNEL_OPT_HOME:-/tmp/aka-opt}"
fi

# Per-target paths
CODEX_TARGET_DIR="$INSTALL_BASE/.codex"
CODEX_SKILL_DIR="$CODEX_TARGET_DIR/skills/$SKILL_NAME"
CODEX_HOOKS_DIR="$CODEX_TARGET_DIR/hooks"
CODEX_HOOKS_FILE="$CODEX_TARGET_DIR/hooks.json"
CODEX_CONFIG_FILE="$CODEX_TARGET_DIR/config.toml"
CODEX_HOOK_SCRIPT="$CODEX_HOOKS_DIR/gpu_kernel_optimizer_hook.py"
CODEX_HOOK_TAG="gpu-kernel-optimizer-codex-hook-v1"

CLAUDE_TARGET_DIR="$INSTALL_BASE/.claude"
CLAUDE_SKILL_DIR="$CLAUDE_TARGET_DIR/skills/$SKILL_NAME"
CLAUDE_HOOKS_DIR="$CLAUDE_TARGET_DIR/hooks"
CLAUDE_HOOKS_FILE="$CLAUDE_TARGET_DIR/settings.json"
CLAUDE_HOOK_SCRIPT="$CLAUDE_HOOKS_DIR/gpu_kernel_optimizer_hook.py"
CLAUDE_HOOK_TAG="gpu-kernel-optimizer-claude-hook-v1"

GPU_WIKI_DIR="$INSTALL_BASE/gpu-wiki"
REFERENCE_PROJECTS_DIR="$INSTALL_BASE/reference-projects"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required. Install it with your system package manager."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required. Install git and retry."
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Prepare default knowledge repositories under $INSTALL_BASE
# ---------------------------------------------------------------------------
link_gpu_wiki() {
  if [ "$GPU_WIKI_SOURCE_DIR" = "$GPU_WIKI_DIR" ]; then
    echo "[repo] gpu-wiki source is already $GPU_WIKI_DIR (skip)"
    return
  fi

  local real_source
  real_source="$(cd "$GPU_WIKI_SOURCE_DIR" && pwd)"

  if [ -L "$GPU_WIKI_DIR" ]; then
    local current_target
    current_target="$(readlink "$GPU_WIKI_DIR")"
    if [ "$current_target" = "$real_source" ]; then
      echo "[repo] gpu-wiki symlink up-to-date"
      return
    fi
    rm -f "$GPU_WIKI_DIR"
  elif [ -e "$GPU_WIKI_DIR" ]; then
    # 旧的拷贝目录存在，替换为软链接
    rm -rf "$GPU_WIKI_DIR"
  fi

  ln -s "$real_source" "$GPU_WIKI_DIR"
  echo "[repo] Symlinked gpu-wiki: $GPU_WIKI_DIR -> $real_source"
}

prepare_knowledge_repos() {
  link_gpu_wiki
  mkdir -p "$REFERENCE_PROJECTS_DIR"

  # 初始化 submodules（如果在源码目录中）
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local submodule_dir="$script_dir/reference-projects"

  if [ -f "$script_dir/.gitmodules" ]; then
    # No --depth 1: it overrides each submodule's .gitmodules `shallow` setting, and a forced-shallow
    # fetch can fail to check out the recorded commit (leaving only a .git file, as happened with
    # DeepGEMM which is shallow=false). Let git honor per-submodule shallow flags; --force completes a
    # checkout a prior run left half-done.
    echo "[repo] Initializing reference-projects submodules..."
    (cd "$script_dir" && git submodule update --init --force -- reference-projects/) || {
      echo "[repo] Warning: Failed to initialize reference-projects submodules (network unavailable?), will symlink existing dirs"
    }
    echo "[repo] Initializing gpu-wiki/3rdparty submodules..."
    (cd "$script_dir" && git submodule update --init --force -- gpu-wiki/3rdparty/) || {
      echo "[repo] Warning: Failed to initialize gpu-wiki/3rdparty submodules (network unavailable?), skipping"
    }
  fi

  # 将 submodule 目录软链接到工作目录
  if [ -d "$submodule_dir" ]; then
    for repo_dir in "$submodule_dir"/*/; do
      [ -d "$repo_dir" ] || continue
      local repo_name
      repo_name="$(basename "$repo_dir")"

      # 跳过 README.md 等非目录项
      [ -d "$repo_dir/.git" ] || [ -f "$repo_dir/.git" ] || continue

      # 如果设置了 --without-github，检查该 submodule 的 URL 是否是 GitHub
      if [ "$WITHOUT_GITHUB" = "1" ]; then
        local repo_url
        repo_url="$(git config --file "$script_dir/.gitmodules" "submodule.reference-projects/$repo_name.url" 2>/dev/null || true)"
        if [[ "$repo_url" == *"github.com"* ]]; then
          echo "[repo] Skipping GitHub repo (--without-github): $repo_name"
          continue
        fi
      fi

      # 创建软链接到目标工作目录
      local link_target="$REFERENCE_PROJECTS_DIR/$repo_name"
      local real_repo_dir
      real_repo_dir="$(cd "$repo_dir" && pwd)"

      if [ -L "$link_target" ]; then
        # 已是软链接，检查是否指向正确目标
        local current_target
        current_target="$(readlink "$link_target")"
        if [ "$current_target" = "$real_repo_dir" ]; then
          echo "[repo] Symlink up-to-date: $repo_name"
        else
          rm -f "$link_target"
          ln -s "$real_repo_dir" "$link_target"
          echo "[repo] Updated symlink: $repo_name -> $real_repo_dir"
        fi
      elif [ -e "$link_target" ]; then
        # 旧的拷贝目录存在，替换为软链接
        rm -rf "$link_target"
        ln -s "$real_repo_dir" "$link_target"
        echo "[repo] Replaced copy with symlink: $repo_name -> $real_repo_dir"
      else
        ln -s "$real_repo_dir" "$link_target"
        echo "[repo] Symlinked: $repo_name -> $real_repo_dir"
      fi
    done
  else
    echo "[repo] No reference-projects submodule directory found at $submodule_dir"
    echo "[repo] Please run 'git submodule update --init --depth 1' first"
  fi

  echo "[repo] gpu-wiki: $GPU_WIKI_DIR"
  echo "[repo] reference_project: $REFERENCE_PROJECTS_DIR"
}

# ---------------------------------------------------------------------------
# Helper: clone or update a git repo
# ---------------------------------------------------------------------------
update_or_clone_repo() {
  local repo_url="$1"
  local target_dir="$2"

  if [ -d "$target_dir/.git" ] || [ -f "$target_dir/.git" ]; then
    echo "[repo] Updating $target_dir ..."
    (cd "$target_dir" && git pull --ff-only 2>/dev/null) || true
  else
    echo "[repo] Cloning $repo_url -> $target_dir ..."
    git clone --depth 1 "$repo_url" "$target_dir" || {
      echo "[repo] Warning: Failed to clone $repo_url (network unavailable?), skipping."
      return 0
    }
  fi
}

# ---------------------------------------------------------------------------
# Clone rocprof-trace-decoder into the source tools/ directory (once)
# ---------------------------------------------------------------------------
clone_decoder_to_tools() {
  local tools_dir="$SCRIPT_DIR/tools"
  mkdir -p "$tools_dir"
  update_or_clone_repo "$ROCPROF_TRACE_DECODER_REPO" "$tools_dir/rocprof-trace-decoder"
}

# ---------------------------------------------------------------------------
# 2. Detect installed targets and configure the active target
# ---------------------------------------------------------------------------
detect_targets() {
  DETECTED_TARGETS=()

  # Detect Codex: check system ~/.codex OR existing install at $INSTALL_BASE/.codex
  if [ -d "$HOME/.codex" ] || [ -d "$CODEX_TARGET_DIR" ]; then
    DETECTED_TARGETS+=("codex")
  fi

  # Detect Claude: check system ~/.claude OR existing install at $INSTALL_BASE/.claude
  if [ -d "$HOME/.claude" ] || [ -d "$CLAUDE_TARGET_DIR" ]; then
    DETECTED_TARGETS+=("claude")
  fi

  if [ "${#DETECTED_TARGETS[@]}" -eq 0 ]; then
    echo "ERROR: Codex or Claude Code does not appear to be installed."
    echo "  Checked Codex path:      $HOME/.codex"
    echo "  Checked Claude Code path: $HOME/.claude"
    echo "Install Codex/Claude Code first, or manually create $INSTALL_BASE/.codex or $INSTALL_BASE/.claude."
    exit 1
  fi

  echo "[detect] Targets: ${DETECTED_TARGETS[*]}"
}

configure_codex_target() {
  TARGET_NAME="codex"
  TARGET_SKILL_DIR="$CODEX_SKILL_DIR"
  HOOKS_DIR="$CODEX_HOOKS_DIR"
  HOOKS_FILE="$CODEX_HOOKS_FILE"
  CONFIG_FILE="$CODEX_CONFIG_FILE"
  HOOK_SCRIPT="$CODEX_HOOK_SCRIPT"
  HOOK_TAG="$CODEX_HOOK_TAG"
}

configure_claude_target() {
  TARGET_NAME="claude"
  TARGET_SKILL_DIR="$CLAUDE_SKILL_DIR"
  HOOKS_DIR="$CLAUDE_HOOKS_DIR"
  HOOKS_FILE="$CLAUDE_HOOKS_FILE"
  CONFIG_FILE=""
  HOOK_SCRIPT="$CLAUDE_HOOK_SCRIPT"
  HOOK_TAG="$CLAUDE_HOOK_TAG"
}

# ---------------------------------------------------------------------------
# 3. Copy skill files into the active target skill directory
# ---------------------------------------------------------------------------
# Whitelist of paths to copy into the skill directory
SKILL_WHITELIST=(reference skills tools SKILL.md)

copy_skill() {
  if [ "$SCRIPT_DIR" = "$TARGET_SKILL_DIR" ]; then
    echo "[$TARGET_NAME][skill] Already at $TARGET_SKILL_DIR (skip copy)"
    return
  fi

  echo "[$TARGET_NAME][skill] Copying whitelisted paths $SCRIPT_DIR -> $TARGET_SKILL_DIR"
  mkdir -p "$TARGET_SKILL_DIR"
  if command -v rsync >/dev/null 2>&1; then
    for item in "${SKILL_WHITELIST[@]}"; do
      if [ -e "$SCRIPT_DIR/$item" ]; then
        rsync -a --delete "$SCRIPT_DIR/$item" "$TARGET_SKILL_DIR/"
      fi
    done
  else
    for item in "${SKILL_WHITELIST[@]}"; do
      if [ -d "$SCRIPT_DIR/$item" ]; then
        mkdir -p "$TARGET_SKILL_DIR/$item"
        cp -R "$SCRIPT_DIR/$item"/. "$TARGET_SKILL_DIR/$item"/
      elif [ -f "$SCRIPT_DIR/$item" ]; then
        cp "$SCRIPT_DIR/$item" "$TARGET_SKILL_DIR/$item"
      fi
    done
  fi
}

copy_agents() {
  local agents_src="$SCRIPT_DIR/agents"
  local agents_dst

  if [ "$TARGET_NAME" = "codex" ]; then
    agents_dst="$CODEX_TARGET_DIR/agents"
  elif [ "$TARGET_NAME" = "claude" ]; then
    agents_dst="$CLAUDE_TARGET_DIR/agents"
  else
    echo "[$TARGET_NAME][agents] Unknown target, skipping agents copy"
    return
  fi

  if [ ! -d "$agents_src" ]; then
    echo "[$TARGET_NAME][agents] No agents/ directory in source, skipping"
    return
  fi

  echo "[$TARGET_NAME][agents] Copying $agents_src -> $agents_dst"
  mkdir -p "$agents_dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$agents_src/" "$agents_dst/"
  else
    cp -R "$agents_src"/. "$agents_dst/"
  fi
}

# ---------------------------------------------------------------------------
# 3b. Install the orchestrator (optimize.py + prompts/ + anchor_bench.py) into
#     $INSTALL_BASE, making <base> a self-contained per-kernel run root:
#         cd <base> && python orchestrator/optimize.py --op-dir ...
#     Target-independent: the orchestrator drives `claude` sessions and reads its
#     runtime helpers (reference/ tools/ skills/, agents/) from <base>/.claude,
#     which the per-target install already populated — so we copy no duplicates.
# ---------------------------------------------------------------------------
install_orchestrator() {
  local src="$SCRIPT_DIR/orchestrator"
  local dst="$INSTALL_BASE/orchestrator"

  if [ ! -d "$src" ]; then
    echo "[orchestrator] No orchestrator/ in source, skipping"
    return
  fi
  if [ "$src" = "$dst" ]; then
    echo "[orchestrator] Source already at $dst (skip copy)"
    return
  fi

  echo "[orchestrator] Installing $src -> $dst"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude '__pycache__' "$src/" "$dst/"
  else
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -R "$src"/. "$dst"/
    rm -rf "$dst/__pycache__"
  fi
  echo "[orchestrator] Installed $dst (run: cd $INSTALL_BASE && python orchestrator/optimize.py --op-dir ...)"
}

uninstall_orchestrator() {
  local dst="$INSTALL_BASE/orchestrator"
  if [ "$SCRIPT_DIR/orchestrator" = "$dst" ]; then
    echo "[orchestrator] Refusing to remove source orchestrator/ (skip)"
    return
  fi
  if [ -d "$dst" ]; then
    rm -rf "$dst"
    echo "[orchestrator] Removed $dst"
  fi
}


# ---------------------------------------------------------------------------
# 4. Ensure Codex hooks are enabled in config.toml
# ---------------------------------------------------------------------------
enable_hooks_feature() {
  mkdir -p "$(dirname "$CONFIG_FILE")"

  if [ ! -f "$CONFIG_FILE" ]; then
    printf '[features]\nhooks = true\n' > "$CONFIG_FILE"
    echo "[codex][config] Created $CONFIG_FILE with hooks enabled"
    return
  fi

  if grep -Eq '^[[:space:]]*codex_hooks[[:space:]]*=' "$CONFIG_FILE"; then
    perl -0pi -e 's/^[[:space:]]*codex_hooks[[:space:]]*=.*$/hooks = true/m' "$CONFIG_FILE"
    echo "[codex][config] Migrated deprecated codex_hooks to hooks in $CONFIG_FILE"
    return
  fi

  if grep -Eq '^[[:space:]]*hooks[[:space:]]*=' "$CONFIG_FILE"; then
    perl -0pi -e 's/^[[:space:]]*hooks[[:space:]]*=.*$/hooks = true/m' "$CONFIG_FILE"
    echo "[codex][config] Ensured hooks = true in $CONFIG_FILE"
    return
  fi

  if grep -Eq '^\[features\]' "$CONFIG_FILE"; then
    local tmp
    tmp=$(mktemp)
    awk 'BEGIN{inserted=0} /^\[features\]$/ {print; print "hooks = true"; inserted=1; next} {print} END{if(!inserted){print ""; print "[features]"; print "hooks = true"}}' "$CONFIG_FILE" > "$tmp"
    mv "$tmp" "$CONFIG_FILE"
    echo "[codex][config] Added hooks = true to [features] in $CONFIG_FILE"
    return
  fi

  printf '\n[features]\nhooks = true\n' >> "$CONFIG_FILE"
  echo "[$TARGET_NAME][config] Appended [features].hooks = true to $CONFIG_FILE"
}

ensure_codex_agents_config() {
  mkdir -p "$(dirname "$CONFIG_FILE")"
  [ -f "$CONFIG_FILE" ] || : > "$CONFIG_FILE"

  local tmp
  tmp=$(mktemp)

  python3 - "$CONFIG_FILE" "$tmp" <<'PY_CONFIG'
from __future__ import annotations

import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
config_lines = config_path.read_text().splitlines()
agent_settings = {
    "max_threads": "max_threads = 6",
    "max_depth": "max_depth = 2",
    "interrupt_message": "interrupt_message = true",
}

output_lines: list[str] = []
in_agents_section = False
agents_section_found = False
seen_agent_settings = {key: False for key in agent_settings}


def is_section_header(line: str) -> bool:
    stripped_line = line.strip()
    return stripped_line.startswith("[") and stripped_line.endswith("]")


def append_missing_agent_settings() -> None:
    for setting_key, setting_line in agent_settings.items():
        if seen_agent_settings[setting_key]:
            continue
        output_lines.append(setting_line)
        seen_agent_settings[setting_key] = True


for config_line in config_lines:
    stripped_line = config_line.strip()
    if is_section_header(config_line):
        if in_agents_section:
            append_missing_agent_settings()

        in_agents_section = stripped_line == "[agents]"
        if in_agents_section:
            agents_section_found = True
            seen_agent_settings = {key: False for key in agent_settings}

        output_lines.append(config_line)
        continue

    if in_agents_section:
        setting_replaced = False
        for setting_key, setting_line in agent_settings.items():
            if re.match(rf"^\s*{re.escape(setting_key)}\s*=", config_line):
                if not seen_agent_settings[setting_key]:
                    output_lines.append(setting_line)
                    seen_agent_settings[setting_key] = True
                setting_replaced = True
                break
        if setting_replaced:
            continue

    output_lines.append(config_line)

if in_agents_section:
    append_missing_agent_settings()
elif not agents_section_found:
    if output_lines and output_lines[-1].strip():
        output_lines.append("")
    output_lines.append("[agents]")
    output_lines.extend(agent_settings.values())

output_path.write_text("\n".join(output_lines) + "\n")
PY_CONFIG

  mv "$tmp" "$CONFIG_FILE"
  echo "[$TARGET_NAME][config] Ensured [agents] permissions in $CONFIG_FILE"
}

# ---------------------------------------------------------------------------
# 5. Install hook script into the active target
# ---------------------------------------------------------------------------
install_hook_script() {
  mkdir -p "$HOOKS_DIR"
  # The hook script is the single shared source of truth in the repo's hooks/ dir
  # (extracted from this installer so optimize.py's self-bootstrap installs the SAME
  # script without duplicating it). Both codex and claude targets use it verbatim;
  # the target is selected at runtime via the hook's `--target` argv, not the file.
  local hook_src="$SCRIPT_DIR/hooks/gpu_kernel_optimizer_hook.py"
  if [ ! -f "$hook_src" ]; then
    echo "ERROR: missing $hook_src (the shared gpu-kernel-optimizer hook script)."
    exit 1
  fi
  cp "$hook_src" "$HOOK_SCRIPT"
  chmod +x "$HOOK_SCRIPT"
  echo "[$TARGET_NAME][hooks] Installed $HOOK_SCRIPT (from $hook_src)"
}

# ---------------------------------------------------------------------------
# 6. Merge hooks into the active target hook config
# ---------------------------------------------------------------------------
merge_hooks() {
  mkdir -p "$(dirname "$HOOKS_FILE")"
  if [ ! -f "$HOOKS_FILE" ]; then
    echo '{"hooks":{}}' > "$HOOKS_FILE"
  fi

  local backup="$HOOKS_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$HOOKS_FILE" "$backup"
  echo "[$TARGET_NAME][hooks] Backup: $backup"

  local pre_cmd post_cmd tmp
  pre_cmd="python3 \"$HOOK_SCRIPT\" pre --target $TARGET_NAME --tag $HOOK_TAG"
  post_cmd="python3 \"$HOOK_SCRIPT\" post --target $TARGET_NAME --tag $HOOK_TAG"
  tmp=$(mktemp)

  jq \
    --arg tag "$HOOK_TAG" \
    --arg pre "$pre_cmd" \
    --arg post "$post_cmd" '
    def strip_tagged:
      (. // [])
      | map(.hooks |= (map(select((.command // "") | contains($tag) | not))))
      | map(select((.hooks | length) > 0));

    .hooks = (.hooks // {})
    | .hooks.PreToolUse = ((.hooks.PreToolUse | strip_tagged) + [{
        matcher:"Write|Edit|MultiEdit|apply_patch|file_replace|shell|Bash",
        hooks:[{type:"command", command:$pre, statusMessage:"gpu-kernel-optimizer pre-edit guard", timeout:10}]
      }])
    | .hooks.PostToolUse = ((.hooks.PostToolUse | strip_tagged) + [{
        matcher:"Read|read_file|Write|Edit|MultiEdit|apply_patch|file_replace|shell|Bash",
        hooks:[{type:"command", command:$post, statusMessage:"gpu-kernel-optimizer memory/kernel edit gate", timeout:10}]
      }])
    | .hooks.Stop = ((.hooks.Stop // []) | strip_tagged)
    | if (.hooks.Stop | length) == 0 then del(.hooks.Stop) else . end
  ' "$HOOKS_FILE" > "$tmp"

  jq empty "$tmp" >/dev/null
  mv "$tmp" "$HOOKS_FILE"
  echo "[$TARGET_NAME][hooks] Merged into $HOOKS_FILE"
}

# ---------------------------------------------------------------------------
# 7. Uninstall hooks installed by this script
# ---------------------------------------------------------------------------
strip_hooks() {
  [ -f "$HOOKS_FILE" ] || { echo "[$TARGET_NAME][hooks] No hooks file to clean"; return; }

  local backup="$HOOKS_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$HOOKS_FILE" "$backup"
  echo "[$TARGET_NAME][hooks] Backup: $backup"

  local tmp
  tmp=$(mktemp)

  jq --arg tag "$HOOK_TAG" '
    def strip_tagged:
      (. // [])
      | map(.hooks |= (map(select((.command // "") | contains($tag) | not))))
      | map(select((.hooks | length) > 0));

    .hooks = (.hooks // {})
    | .hooks.PreToolUse = ((.hooks.PreToolUse // []) | strip_tagged)
    | .hooks.PostToolUse = ((.hooks.PostToolUse // []) | strip_tagged)
    | .hooks.Stop = ((.hooks.Stop // []) | strip_tagged)
    | if (.hooks.PreToolUse | length) == 0 then del(.hooks.PreToolUse) else . end
    | if (.hooks.PostToolUse | length) == 0 then del(.hooks.PostToolUse) else . end
    | if (.hooks.Stop | length) == 0 then del(.hooks.Stop) else . end
  ' "$HOOKS_FILE" > "$tmp"

  jq empty "$tmp" >/dev/null
  mv "$tmp" "$HOOKS_FILE"
  rm -f "$HOOK_SCRIPT"
  echo "[$TARGET_NAME][hooks] Removed tagged hooks from $HOOKS_FILE"
  echo "[$TARGET_NAME][hooks] Removed $HOOK_SCRIPT"
}

# ---------------------------------------------------------------------------
# Per-target orchestration
# ---------------------------------------------------------------------------
install_codex() {
  configure_codex_target
  if [ "$MODE" != "hooks-only" ]; then
    copy_skill
    copy_agents
  fi
  enable_hooks_feature
  ensure_codex_agents_config
  install_hook_script
  merge_hooks
}

install_claude() {
  configure_claude_target
  if [ "$MODE" != "hooks-only" ]; then
    copy_skill
    copy_agents
  fi
  install_hook_script
  merge_hooks
}

uninstall_codex() {
  configure_codex_target
  strip_hooks
}

uninstall_claude() {
  configure_claude_target
  strip_hooks
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "[install] Base directory: $INSTALL_BASE"
mkdir -p "$INSTALL_BASE"

detect_targets

case "$MODE" in
  install|hooks-only)
    prepare_knowledge_repos
    clone_decoder_to_tools
    for target in "${DETECTED_TARGETS[@]}"; do
      case "$target" in
        codex) install_codex ;;
        claude) install_claude ;;
      esac
    done
    # Install the orchestrator only for a full install (hooks-only must not touch it).
    [ "$MODE" = "install" ] && install_orchestrator
    ;;
  uninstall)
    for target in "${DETECTED_TARGETS[@]}"; do
      case "$target" in
        codex) uninstall_codex ;;
        claude) uninstall_claude ;;
      esac
    done
    uninstall_orchestrator
    ;;
esac

echo ""
echo "Done."
[ "$MODE" = "install" ] && echo "  gpu-wiki:       $GPU_WIKI_DIR"
[ "$MODE" = "install" ] && echo "  ref-projects:   $REFERENCE_PROJECTS_DIR"
[ "$MODE" = "install" ] && [ -d "$INSTALL_BASE/orchestrator" ] && echo "  orchestrator:   $INSTALL_BASE/orchestrator"
for target in "${DETECTED_TARGETS[@]}"; do
  case "$target" in
    codex)
      [ "$MODE" = "install" ] && echo "  Codex skill:    $CODEX_SKILL_DIR"
      [ "$MODE" != "uninstall" ] && echo "  Codex hooks:    $CODEX_HOOKS_FILE"
      [ "$MODE" != "uninstall" ] && echo "  Codex hook bin: $CODEX_HOOK_SCRIPT"
      ;;
    claude)
      [ "$MODE" = "install" ] && echo "  Claude skill:   $CLAUDE_SKILL_DIR"
      [ "$MODE" != "uninstall" ] && echo "  Claude hooks:   $CLAUDE_HOOKS_FILE"
      [ "$MODE" != "uninstall" ] && echo "  Claude hook bin:$CLAUDE_HOOK_SCRIPT"
      ;;
  esac
done

if [ "$MODE" = "install" ] && [ -d "$INSTALL_BASE/orchestrator" ]; then
  echo ""
  echo "Self-contained per-kernel run root: $INSTALL_BASE"
  echo "  cd $INSTALL_BASE && python orchestrator/optimize.py \\"
  echo "      --op-dir <atrex-bench op dir> --platform <hw> --framework <dsl> [--max-iters N ...]"
  echo "  Nested sessions auto-isolate via CLAUDE_CONFIG_DIR=$INSTALL_BASE/.claude"
  echo "  (use one install base per kernel to keep campaigns fully independent)."
fi

echo ""
echo "Restart Codex/Claude Code or open a new session if hooks are not picked up immediately."
