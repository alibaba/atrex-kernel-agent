#!/usr/bin/env bash
# One-shot installer for the gpu-kernel-optimizer skill + hooks.
#
# Default install base: ~/aka_kernel_opt (customizable via AKA_KERNEL_OPT_HOME)
# Files are installed under <base>/.codex and/or <base>/.claude:
#   - Codex:  ~/aka_kernel_opt/.codex/skills/gpu-kernel-optimizer/
#             ~/aka_kernel_opt/.codex/hooks/
#   - Claude: ~/aka_kernel_opt/.claude/skills/gpu-kernel-optimizer/
#             ~/aka_kernel_opt/.claude/hooks/
#   - Shared: ~/aka_kernel_opt/gpu-wiki/
#             ~/aka_kernel_opt/reference-projects/
#
# Skill directory whitelist (only these are copied):
#   reference/  skills/  tools/  agents/  SKILL.md
#
# Usage:
#   ./install.sh                       # install/update all detected targets
#   ./install.sh --prefix /my/path     # specify custom install directory
#   ./install.sh --hooks-only          # only install/update hooks for detected targets
#   ./install.sh --without-github      # install/update, but skip GitHub reference repos from gpu-wiki
#   ./install.sh --max-iterations N    # allow Stop hooks after memory/vN.json exceeds N iterations
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
MAX_ITERATIONS="${GPU_KERNEL_MAX_ITERATIONS:-0}"
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
    --max-iterations)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --max-iterations requires a non-negative integer."
        exit 2
      fi
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --max-iterations=*)
      MAX_ITERATIONS="${1#*=}"
      if [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --max-iterations requires a non-negative integer."
        exit 2
      fi
      shift
      ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# Resolve install base: --prefix > AKA_KERNEL_OPT_HOME > ~/aka_kernel_opt
if [[ -n "$CLI_PREFIX" ]]; then
  INSTALL_BASE="$CLI_PREFIX"
else
  INSTALL_BASE="${AKA_KERNEL_OPT_HOME:-$HOME/aka_kernel_opt}"
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

if [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_KERNEL_MAX_ITERATIONS must be a non-negative integer."
  exit 2
fi

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
    echo "[repo] Initializing reference-projects submodules..."
    (cd "$script_dir" && git submodule update --init --depth 1 -- reference-projects/) || {
      echo "[repo] Warning: Failed to initialize reference-projects submodules (network unavailable?), will symlink existing dirs"
    }
    echo "[repo] Initializing gpu-wiki/3rdparty submodules..."
    (cd "$script_dir" && git submodule update --init --depth 1 -- gpu-wiki/3rdparty/) || {
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
SKILL_WHITELIST=(reference skills tools agents SKILL.md)

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
  cat > "$HOOK_SCRIPT" <<'PY_HOOK'
#!/usr/bin/env python3
"""Hook guard for gpu-kernel-optimizer.

The hook keeps long-running kernel optimization sessions from stopping while
state is inconsistent. It is intentionally conservative: it only enforces rules
inside directories named kernel_opt_*.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

RECENT_SECONDS = 6 * 60 * 60
MEMORY_READ_MARKER = ".gpu_kernel_optimizer_memory_read_marker"  # tracks memory/ dir reads
PLAN_READ_MARKER = ".gpu_kernel_optimizer_plan_read_marker"
GOAL_CHECK_MARKER = ".gpu_kernel_optimizer_goal_check_marker"
GOAL_CHECK_LEVEL_MARKER = ".gpu_kernel_optimizer_goal_check_level"  # tracks progressive prompt level
SCHEMA_READ_MARKER = ".gpu_kernel_optimizer_schema_read_marker"  # tracks v_iteration.schema.json reads
OUTPUT_CONTRACT_SKILL = "skills/gpu-kernel-output-contract/SKILL.md"
OUTPUT_CONTRACT_MARKER = ".gpu_kernel_optimizer_output_contract_marker"
GOAL_CHECK_MAX_LEVEL = 3


def load_payload() -> dict:
    raw_payload = sys.stdin.read().strip()
    if not raw_payload:
        return {}
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        return {"raw_stdin": raw_payload}


def iter_values(value: object) -> Iterable[object]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_values(child)


def extract_file_path(payload: dict) -> Path | None:
    preferred_keys = ("file_path", "path", "filename", "relative_workspace_path")
    for container_key in ("tool_input", "input", "arguments", "params"):
        container = payload.get(container_key)
        if isinstance(container, dict):
            for key in preferred_keys:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return Path(value).expanduser()

    for value in iter_values(payload):
        if not isinstance(value, dict):
            continue
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return Path(candidate).expanduser()
    return None


def extract_tool_name(payload: dict) -> str:
    preferred_keys = ("tool_name", "tool", "name", "matcher")
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.lower()

    for container_key in ("tool", "tool_input", "input", "arguments", "params"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in preferred_keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.lower()

    return ""


def is_read_tool(payload: dict) -> bool:
    tool_name = extract_tool_name(payload)
    return "read" in tool_name


def extract_command_text(payload: dict) -> str:
    command_keys = ("command", "cmd", "script")
    for key in command_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for container_key in ("tool_input", "input", "arguments", "params"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in command_keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return ""


def is_shell_tool(payload: dict) -> bool:
    tool_name = extract_tool_name(payload)
    return "shell" in tool_name or "bash" in tool_name or bool(extract_command_text(payload))


def command_may_modify_kernel(command_text: str) -> bool:
    if "kernel.py" not in command_text:
        return False

    mutating_tokens = (
        " >", ">>", "cat >", "tee ", "sed -i", "perl -pi", "python - <<", "python3 - <<",
        "mv ", "cp ", "rsync ", "truncate ", "touch ", "apply_patch", "file_replace",
    )
    readonly_tokens = (
        "git add", "git commit", "git status", "grep ", "rg ", "cat ", "python kernel.py", "python3 kernel.py",
    )
    stripped_command = command_text.strip()
    if any(token in stripped_command for token in readonly_tokens):
        return False
    return any(token in stripped_command for token in mutating_tokens)


def command_may_modify_memory(command_text: str) -> bool:
    """Detect shell commands that write memory/v*.json via memory_manager.py."""
    stripped = command_text.strip()
    if "memory_manager" not in stripped:
        return False
    return any(sub in stripped for sub in ("create", "update", "mask", "unmask"))


def command_is_memory_manager_update(command_text: str) -> bool:
    """Detect 'memory_manager update' specifically (not init/create)."""
    stripped = command_text.strip()
    if "memory_manager" not in stripped:
        return False
    if "update" not in stripped:
        return False
    # Exclude init and create invocations
    if " init" in stripped or " create" in stripped:
        return False
    return True


def resolve_path(path: Path, payload: dict) -> Path:
    if path.is_absolute():
        return path
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return Path(cwd).expanduser() / path
    return Path.cwd() / path


def find_workspace_from_path(path: Path) -> Path | None:
    resolved = path.expanduser()
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part.startswith("kernel_opt_"):
            return Path(*parts[: index + 1])
    return None


def deny(reason: str, target: str) -> int:
    if target in {"codex", "claude"}:
        output = {
            "decision": "block",
            "reason": reason,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }
        print(json.dumps(output, ensure_ascii=False))
        print(reason, file=sys.stderr)
        return 2

    output = {
        "decision": "deny",
        "reason": reason,
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


def latest_iteration_plan(workspace: Path) -> Path | None:
    plans_dir = workspace / "plans"
    try:
        plan_paths = [path for path in plans_dir.glob("v*_plan.md") if path.is_file()]
    except OSError:
        return None
    if not plan_paths:
        return None
    return max(plan_paths, key=lambda path: path.stat().st_mtime)


def is_iteration_plan(path: Path, workspace: Path) -> bool:
    try:
        relative_path = path.resolve().relative_to(workspace.resolve())
    except (OSError, ValueError):
        return False
    return len(relative_path.parts) == 2 and relative_path.parts[0] == "plans" and path.match("v*_plan.md")


def handle_pre_tool_use(payload: dict, target: str) -> int:
    file_path = extract_file_path(payload)
    command_text = extract_command_text(payload)

    # Schema gate for shell commands that write memory via memory_manager.py
    if file_path is None and is_shell_tool(payload) and command_may_modify_memory(command_text):
        for ws in iter_workspaces(payload):
            schema_marker = ws / SCHEMA_READ_MARKER
            if not schema_marker.exists():
                return deny(
                    "Before writing memory/v<N>.json via memory_manager.py, read reference/v_iteration.schema.json to ensure the JSON conforms to the required schema.",
                    target,
                )
        # schema checks passed; fall through to normal checks

    if file_path is None:
        if not (is_shell_tool(payload) and command_may_modify_kernel(command_text)):
            return 0
        file_path = Path("kernel.py")

    resolved_path = resolve_path(file_path, payload)
    workspace = find_workspace_from_path(resolved_path)
    if workspace is None and resolved_path.name == "kernel.py":
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            workspace = find_workspace_from_path(Path(cwd).expanduser())
            if workspace is not None:
                resolved_path = workspace / "kernel.py"
    if workspace is None:
        return 0

    if resolved_path.name == "README.md":
        return 0
    memory_dir = workspace / "memory"
    is_memory_write = (
        (resolved_path.is_relative_to(memory_dir) if hasattr(resolved_path, 'is_relative_to') else str(resolved_path).startswith(str(memory_dir)))
        and resolved_path.suffix == ".json"
    )
    if is_memory_write:
        schema_marker = workspace / SCHEMA_READ_MARKER
        if not schema_marker.exists():
            return deny(
                f"Before writing {resolved_path.name}, read reference/v_iteration.schema.json to ensure the JSON conforms to the required schema.",
                target,
            )
        return 0
    if resolved_path.is_relative_to(memory_dir) if hasattr(resolved_path, 'is_relative_to') else str(resolved_path).startswith(str(memory_dir)):
        return 0

    if not memory_dir.exists() or not any(memory_dir.glob("v*.json")):
        return deny(
            f"memory/ directory is missing or empty in {workspace}. Initialize memory with 'python tools/memory_manager.py init' before editing kernel files.",
            target,
        )

    marker = workspace / MEMORY_READ_MARKER
    newest_memory_mtime = max(
        (f.stat().st_mtime for f in memory_dir.glob("v*.json")),
        default=0.0,
    )
    if not marker.exists() or newest_memory_mtime > marker.stat().st_mtime:
        return deny(f"Read memory/v*.json files in {workspace} before editing files in this kernel optimization workspace.", target)

    if resolved_path.name == "kernel.py":
        plan_path = latest_iteration_plan(workspace)
        if plan_path is None:
            return deny(
                f"Before editing {resolved_path}, create plans/v<N>_plan.md in {workspace} and read it.",
                target,
            )

        plan_marker = workspace / PLAN_READ_MARKER
        if not plan_marker.exists() or plan_path.stat().st_mtime > plan_marker.stat().st_mtime:
            return deny(
                f"Before editing {resolved_path}, read the current iteration plan first: {plan_path}.",
                target,
            )

    return 0


def handle_post_tool_use(payload: dict, target: str) -> int:
    file_path = extract_file_path(payload)
    command_text = extract_command_text(payload)

    # Schema read tracking: v_iteration.schema.json lives outside the workspace
    # (in the skill's reference/ dir), so handle it before the workspace gate.
    if file_path is not None and is_read_tool(payload):
        rp = resolve_path(file_path, payload)
        if rp.name == "v_iteration.schema.json":
            for ws in iter_workspaces(payload):
                try:
                    ws.joinpath(SCHEMA_READ_MARKER).touch()
                except OSError:
                    pass
            return 0

    if file_path is None:
        # Detect memory_manager update via shell command
        if is_shell_tool(payload) and command_is_memory_manager_update(command_text):
            for ws in iter_workspaces(payload):
                if should_trigger_memory_write_prompt(ws):
                    version = f"v{latest_memory_version(ws)}"
                    prompt = memory_write_continuation_prompt(ws, version)
                    print(json.dumps({"decision": "block", "reason": prompt}, ensure_ascii=False))
                    print(prompt, file=sys.stderr)
                    return 2
            return 0
        if not (is_shell_tool(payload) and command_may_modify_kernel(command_text)):
            return 0
        file_path = Path("kernel.py")

    resolved_path = resolve_path(file_path, payload)
    workspace = find_workspace_from_path(resolved_path)
    if workspace is None and resolved_path.name == "kernel.py":
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            workspace = find_workspace_from_path(Path(cwd).expanduser())
            if workspace is not None:
                resolved_path = workspace / "kernel.py"
    if workspace is None:
        return 0

    if is_read_tool(payload):
        memory_dir = workspace / "memory"
        is_memory_file = (
            (resolved_path.is_relative_to(memory_dir) if hasattr(resolved_path, 'is_relative_to') else str(resolved_path).startswith(str(memory_dir)))
            and resolved_path.suffix == ".json"
        )
        if is_memory_file:
            try:
                workspace.joinpath(MEMORY_READ_MARKER).touch()
            except OSError:
                return 0
            return 0

        if is_iteration_plan(resolved_path, workspace):
            try:
                workspace.joinpath(PLAN_READ_MARKER).touch()
            except OSError:
                return 0
            return 0

        return 0

    # Memory write detection — remind to continue Stage 1 for next iteration
    memory_dir = workspace / "memory"
    is_memory_write = (
        (resolved_path.is_relative_to(memory_dir) if hasattr(resolved_path, 'is_relative_to') else str(resolved_path).startswith(str(memory_dir)))
        and resolved_path.suffix == ".json"
        and resolved_path.stem.startswith("v")
        and resolved_path.stem[1:].isdigit()
    )
    if is_memory_write and should_trigger_memory_write_prompt(workspace):
        version = resolved_path.stem
        prompt = memory_write_continuation_prompt(workspace, version)
        print(json.dumps({"decision": "block", "reason": prompt}, ensure_ascii=False))
        print(prompt, file=sys.stderr)
        return 2

    if resolved_path.name != "kernel.py":
        return 0

    continuation_prompt = (
        "gpu-kernel-optimizer immediate kernel edit gate: "
        f"{resolved_path} was just modified. Do not continue with unrelated work. "
        "Immediately run the relevant correctness test for this kernel change. "
        "If the test failed, update the current memory/v<N>.json with the failure, error message, and next fix; "
        "do not enter performance validation or commit the failed kernel change, and continue fixing it. "
        "If the test passed, immediately enter gpu-kernel-profile-optimizer Stage 4 validation: "
        "measure kernel performance (latency, TFLOPS, bandwidth) using triton do_bench or the workspace timing harness, "
        "then calculate peak utilization with tools/compute_utilization.py. "
        "Compare metrics against the previous version, "
        "verify correctness and ISA target progress, and update memory/v<N>.json with the results. "
        "Note: full hardware profiling (profile_kernel.sh for AMD, profile_nvidia.sh for NVIDIA) is performed "
        "in Stage 1 of the NEXT iteration to identify new bottlenecks — it is not required for Stage 4 validation. "
        "Only after Stage 4 has updated memory/v<N>.json may you proceed to the next stage, "
        "including finalizing the memory JSON and performing the git operation required by the profile optimizer workflow."
    )
    print(json.dumps({"decision": "block", "reason": continuation_prompt}, ensure_ascii=False))
    print(continuation_prompt, file=sys.stderr)
    return 2


def candidate_roots(payload: dict) -> list[Path]:
    roots: list[Path] = []
    for key in ("cwd", "workspace", "workspace_root", "project_root"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            roots.append(Path(value).expanduser())

    roots.extend([Path.cwd(), Path("/tmp"), Path("/private/tmp")])

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique_roots.append(resolved)
    return unique_roots


def iter_workspaces(payload: dict) -> Iterable[Path]:
    seen: set[str] = set()

    for root in candidate_roots(payload):
        current = root
        for candidate in [current, *current.parents]:
            if candidate.name.startswith("kernel_opt_"):
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    yield candidate
                break

        for pattern_root in (root, root.parent if root.parent != root else root):
            try:
                children = list(pattern_root.glob("kernel_opt_*"))
            except OSError:
                children = []
            for child in children:
                if not child.is_dir():
                    continue
                key = str(child)
                if key not in seen:
                    seen.add(key)
                    yield child


def recently_active(workspace: Path) -> bool:
    memory_dir = workspace / "memory"
    if not memory_dir.exists() or not any(memory_dir.glob("v*.json")):
        return False
    newest_mtime = max(
        (f.stat().st_mtime for f in memory_dir.glob("v*.json")),
        default=0.0,
    )
    for child in workspace.glob("*.py"):
        try:
            newest_mtime = max(newest_mtime, child.stat().st_mtime)
        except OSError:
            continue
    return time.time() - newest_mtime <= RECENT_SECONDS

def kernel_files_newer_than_memory(workspace: Path) -> list[Path]:
    memory_dir = workspace / "memory"
    if not memory_dir.exists():
        return []
    memory_mtime = max(
        (f.stat().st_mtime for f in memory_dir.glob("v*.json")),
        default=0.0,
    )
    if memory_mtime <= 0:
        return []
    changed_files: list[Path] = []
    for pattern in ("*.py", "plans/v*_plan.md"):
        for child in workspace.glob(pattern):
            try:
                if child.stat().st_mtime > memory_mtime:
                    changed_files.append(child)
            except OSError:
                continue
    return changed_files

def newest_iteration_artifact_mtime(workspace: Path) -> float:
    newest_mtime = 0.0
    memory_dir = workspace / "memory"
    if memory_dir.exists():
        for child in memory_dir.glob("v*.json"):
            try:
                newest_mtime = max(newest_mtime, child.stat().st_mtime)
            except OSError:
                continue
    for pattern in ("*.py", "profiles/*/iteration_report.md", "plans/v*_plan.md"):
        for child in workspace.glob(pattern):
            try:
                newest_mtime = max(newest_mtime, child.stat().st_mtime)
            except OSError:
                continue
    return newest_mtime


def goal_check_already_processed_current_artifacts(workspace: Path, newest_mtime: float) -> bool:
    marker = workspace / GOAL_CHECK_MARKER
    try:
        return marker.exists() and marker.stat().st_mtime >= newest_mtime
    except OSError:
        return False


def get_goal_check_level(workspace: Path) -> int:
    """Read the current progressive prompt level (0-based) from the level marker."""
    level_file = workspace / GOAL_CHECK_LEVEL_MARKER
    try:
        content = level_file.read_text().strip()
        return int(content) if content.isdigit() else 0
    except (OSError, ValueError):
        return 0


def set_goal_check_level(workspace: Path, level: int) -> None:
    """Persist the current progressive prompt level."""
    level_file = workspace / GOAL_CHECK_LEVEL_MARKER
    try:
        level_file.write_text(str(level))
    except OSError:
        pass


def reset_goal_check_level(workspace: Path) -> None:
    """Reset level back to 0 when a new artifact cycle begins."""
    level_file = workspace / GOAL_CHECK_LEVEL_MARKER
    try:
        if level_file.exists():
            level_file.unlink()
    except OSError:
        pass


def latest_memory_version(workspace: Path) -> int:
    memory_dir = workspace / "memory"
    latest_version = 0
    if not memory_dir.exists():
        return latest_version

    for memory_path in memory_dir.glob("v*.json"):
        stem = memory_path.stem
        if not stem.startswith("v"):
            continue
        version_text = stem[1:]
        if not version_text.isdigit():
            continue
        latest_version = max(latest_version, int(version_text))
    return latest_version


def should_trigger_memory_write_prompt(workspace: Path) -> bool:
    """Check exclusion conditions for memory write continuation prompt."""
    # If kernel.py doesn't exist, workspace is still initializing
    kernel_py = workspace / "kernel.py"
    if not kernel_py.exists():
        return False
    # If only v0.json exists (baseline stage), don't trigger
    memory_dir = workspace / "memory"
    if not memory_dir.exists():
        return False
    version_files = [f for f in memory_dir.glob("v*.json") if f.stem[1:].isdigit()]
    if len(version_files) <= 1:
        return False
    return True


def memory_write_continuation_prompt(workspace: Path, version: str) -> str:
    """Generate continuation prompt after memory/v<N>.json is updated."""
    return (
        "gpu-kernel-optimizer Stage 5 memory update detected: "
        f"memory/{version}.json in workspace {workspace} has been updated. "
        "Check Stop Conditions in README.md: if the optimization target is met, "
        "proceed to finalize (git commit if not already done, then stop). "
        "If the target is NOT met, continue to the next iteration \u2014 "
        "enter Stage 1: run full hardware profiling using profile_kernel.sh (AMD) or profile_nvidia.sh (NVIDIA) "
        "on the current kernel.py to identify the next bottleneck. "
        "Do NOT skip profiling or rely on previous profile data \u2014 "
        "the kernel has changed and fresh profile evidence is required for the next optimization plan."
    )


def iteration_limit_reached(workspace: Path, max_iterations: int) -> bool:
    if max_iterations <= 0:
        return False
    return latest_memory_version(workspace) > max_iterations


def output_path_for_workspace(workspace: Path) -> Path:
    return workspace.parent / "generated_kernel.py"


def first_existing_output_path(payload: dict, workspace: Path) -> Path | None:
    path = output_path_for_workspace(workspace)
    if path.exists() and path.is_file():
        return path
    return None


def generated_kernel_violations(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ["generated_kernel.py is not UTF-8 text"]
    except OSError as error:
        return [f"generated_kernel.py cannot be read: {error}"]

    violations: list[str] = []
    forbidden_patterns = {
        "Markdown fence": "```",
        "main entry block": "if __name__",
        "torch.compile": "torch.compile",
        "torch.jit": "torch.jit",
        "shapes.json read": "shapes.json",
        "_make_inputs redefinition": "def _make_inputs",
        "debug print": "print(",
        "benchmark code": "benchmark",
        "test code": "pytest",
        "test code unittest": "unittest",
        "custom C++ extension": "cpp_extension",
        "external dynamic import": "importlib.import_module",
    }
    for label, pattern in forbidden_patterns.items():
        if pattern in source:
            violations.append(label)

    if "class Model(nn.Module)" not in source and "class Model(torch.nn.Module)" not in source:
        violations.append("missing class Model(nn.Module)")
    if "def forward(" not in source:
        violations.append("missing Model.forward")
    if "reference.Model" in source or "from reference import Model" in source or "import reference" in source:
        violations.append("reference Model fallback")
    _src_lower = source.lower()
    _has_kernel_framework = any(kw in _src_lower for kw in (
        "flydsl", "triton", "gluon", "cutedsl",
        "cpp_inline", "cuda_source", "__global__",
    ))
    if not _has_kernel_framework:
        violations.append("missing visible kernel implementation path (expected one of: triton, gluon, flydsl, cutedsl, or C++ inline)")

    return violations


def output_contract_prompt(workspace: Path, output_path: Path | None, violations: list[str]) -> str:
    if output_path is None:
        output_status = f"No generated_kernel.py was found in the project root ({workspace.parent})."
    elif violations:
        output_status = f"generated_kernel.py exists at {output_path}, but violates the final output contract: {', '.join(violations)}."
    else:
        output_status = f"generated_kernel.py exists at {output_path}, but the output contract gate requires one final confirmation pass."

    return (
        "gpu-kernel-optimizer final output contract gate: "
        f"workspace {workspace} is about to end the current optimization session. "
        f"{output_status} "
        f"Do not modify the root SKILL.md. Read the child skill `{OUTPUT_CONTRACT_SKILL}` from the installed gpu-kernel-optimizer skill. "
        f"Then convert the validated optimized implementation into `generated_kernel.py` in the project root directory ({workspace.parent}), NOT inside the {workspace.name}/ subdirectory. "
        "The final file is the only candidate evaluated by the hidden evaluator and must be a self-contained Python module with valid Python source only. "
        "It must define `class Model(nn.Module)`, preserve the reference Model init and forward signatures, return the same externally observable output structure, shapes, device, dtype, and numerical behavior, "
        "and launch the main compute path through GPU kernels from a supported framework (Triton, Gluon, FlyDSL, CuteDSL, or C++ inline CUDA) from `Model.forward`. "
        "Use PyTorch only for setup or glue logic such as allocation, reshape/view, indexing, metadata preparation, and launch orchestration. "
        "Do not include Markdown, explanatory prose, tests, benchmarks, debug prints, command-line entry points, `__main__`, `_make_inputs`, `shapes.json` reads, `torch.compile`, `torch.jit`, custom C++ extensions, external files, or a reference Model fallback. "
        "After writing `generated_kernel.py`, run syntax/import checks outside the file and stop only when the final candidate is clean."
    )


def goal_check_prompt_level_1(workspace: Path) -> str:
    """Layer 1: Core goal verification — concise, lightweight."""
    return (
        "gpu-kernel-optimizer goal check: before stopping, read README.md in "
        f"workspace {workspace} to confirm the exact optimization target (e.g. TFLOPS, bandwidth, utilization threshold). "
        "Then explicitly verify whether this target has been achieved with real benchmark/profile evidence. "
        "If the target is not met, do not stop. Read memory/v*.json files, current plans, "
        "and profile reports, then continue the optimization. Only stop after the target is met, "
        "or the user explicitly stops."
    )


def goal_check_prompt_level_2(workspace: Path) -> str:
    """Layer 2: Broaden search space — inline_asm/ptx + web search guidance."""
    return (
        "gpu-kernel-optimizer goal check (escalated): the optimization target in "
        f"workspace {workspace} is still not met. Treat any conclusion that the goal cannot be reached with skepticism: "
        "assume the search space is still incomplete. You must broaden the optimization search space "
        "and keep trying different optimization directions so the session accumulates evidence, experience, "
        "and reusable lessons toward eventually reaching the target. "
        "When the framework's high-level API does not expose the instruction or scheduling you need, "
        "use CuteDSL's inline_ptx() or FlyDSL's inline_asm() to gain direct ISA-level control within the framework. "
        "These framework-native inline_asm/inline_ptx mechanisms are a first-class optimization tool "
        "when profile evidence shows the generated code is suboptimal at the instruction scheduling, "
        "memory ordering, or synchronization level. Search gpu-wiki for inline_asm/inline_ptx API usage and examples. "
        "Search public web sources such as Google, papers, blogs, "
        "vendor docs, GitHub issues, and open-source kernels for relevant GPU optimization methods "
        "including inline_asm/inline_ptx patterns and ISA-level scheduling techniques. "
        "Read memory/v*.json files, current plans, "
        "and profile reports, then produce a concise follow-up optimization path that maps external ideas to the current kernel "
        "bottlenecks and constraints. Public web findings may be used only as optimization ideas and must not be used as hardware "
        "spec values; hardware specs still require gpu-wiki evidence."
    )


def goal_check_prompt_level_3(workspace: Path) -> str:
    """Layer 3: Full escalation — partial restart + Stage 2 continuation."""
    return (
        "gpu-kernel-optimizer goal check (final escalation): the optimization target in "
        f"workspace {workspace} remains unmet after multiple attempts. "
        "If no new optimization direction is available, read and follow the gpu-kernel-partial-restart agent: "
        "randomly mask half of the optimization experience in memory/v*.json files, then restart optimization work. "
        "Then continue directly into the gpu-kernel-optimizer "
        "Stage 2 flow from the installed skill entry (`SKILL.md`): update the Stage 2 optimization plan, "
        "implement the selected path, validate correctness/performance with real benchmark/profile evidence, write/update the "
        "Stage 2 iteration report, update memory/v<N>.json, and keep iterating in Stage 2. Only stop after the target is met, "
        "or the user explicitly stops."
    )


def goal_check_prompt_for_level(workspace: Path, level: int) -> str:
    """Return the appropriate prompt for the given progressive level."""
    if level <= 0:
        return goal_check_prompt_level_1(workspace)
    elif level == 1:
        return goal_check_prompt_level_2(workspace)
    else:
        return goal_check_prompt_level_3(workspace)


def handle_goal_check(payload: dict, target: str, max_iterations: int) -> int:
    stop_hook_active = payload.get("stop_hook_active", False)

    for workspace in iter_workspaces(payload):
        if not recently_active(workspace):
            continue
        if iteration_limit_reached(workspace, max_iterations):
            continue

        newest_mtime = newest_iteration_artifact_mtime(workspace)
        if newest_mtime <= 0:
            continue

        marker = workspace / GOAL_CHECK_MARKER
        try:
            # When stop_hook_active is true, Claude is already continuing
            # because a prior stop hook blocked — escalate to the next level.
            if not stop_hook_active and marker.exists() and marker.stat().st_mtime >= newest_mtime:
                continue

            if stop_hook_active:
                # Escalate: advance to the next prompt level
                current_level = get_goal_check_level(workspace)
                next_level = min(current_level + 1, GOAL_CHECK_MAX_LEVEL - 1)
                set_goal_check_level(workspace, next_level)
            else:
                # New artifact cycle: reset to level 0
                reset_goal_check_level(workspace)
                next_level = 0
                set_goal_check_level(workspace, 0)

            marker.touch()
        except OSError:
            continue

        goal_check_prompt = goal_check_prompt_for_level(workspace, next_level)
        print(json.dumps({"decision": "block", "reason": goal_check_prompt}, ensure_ascii=False))
        print(goal_check_prompt, file=sys.stderr)
        return 2

    return 0

def handle_stop(payload: dict, target: str, max_iterations: int) -> int:
    for workspace in iter_workspaces(payload):
        if not recently_active(workspace):
            continue

        newest_mtime = newest_iteration_artifact_mtime(workspace)
        if newest_mtime <= 0:
            continue

        iteration_limit_hit = iteration_limit_reached(workspace, max_iterations)

        if iteration_limit_hit:
            continue

        output_path = first_existing_output_path(payload, workspace)
        violations = generated_kernel_violations(output_path) if output_path is not None else []
        if output_path is None or violations:
            try:
                (workspace / OUTPUT_CONTRACT_MARKER).touch()
            except OSError:
                pass
            continuation_prompt = output_contract_prompt(workspace, output_path, violations)
            print(json.dumps({"decision": "block", "reason": continuation_prompt}, ensure_ascii=False))
            print(continuation_prompt, file=sys.stderr)
            return 2

        changed_files = kernel_files_newer_than_memory(workspace)
        if changed_files:
            relative_changes = ", ".join(str(path.relative_to(workspace)) for path in changed_files[:5])
            continuation_prompt = (
                "gpu-kernel-optimizer workflow gate: "
                f"workspace {workspace} has iteration artifacts newer than the latest memory/v*.json ({relative_changes}). "
                "Do not stop yet. Resume the gpu-kernel-optimizer workflow from "
                "the installed skill entry (`SKILL.md`). "
                "Required next steps: read memory/v*.json files, validate correctness if needed, "
                "update the current iteration's memory/v<N>.json with real evidence, commit the accepted iteration, "
                "then check Stop Conditions. Continue Stage 2 unless target performance is met, "
                "no applicable optimization remains, or the user explicitly stops."
            )
            print(json.dumps({"decision": "block", "reason": continuation_prompt}, ensure_ascii=False))
            print(continuation_prompt, file=sys.stderr)
            return 2

        if not goal_check_already_processed_current_artifacts(workspace, newest_mtime):
            continue
    return 0


def parse_args() -> tuple[str, str, int]:
    mode = "stop"
    target = "generic"
    max_iterations = 0
    args = sys.argv[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"pre", "post", "stop", "goal"}:
            mode = arg
        elif arg == "--target" and index + 1 < len(args):
            target = args[index + 1]
            index += 1
        elif arg.startswith("--target="):
            target = arg.split("=", 1)[1]
        elif arg == "--max-iterations" and index + 1 < len(args):
            value = args[index + 1]
            if value.isdigit():
                max_iterations = int(value)
            index += 1
        elif arg.startswith("--max-iterations="):
            value = arg.split("=", 1)[1]
            if value.isdigit():
                max_iterations = int(value)
        index += 1
    return mode, target, max_iterations


def main() -> int:
    mode, target, max_iterations = parse_args()
    payload = load_payload()
    if mode == "pre":
        return handle_pre_tool_use(payload, target)
    if mode == "post":
        return handle_post_tool_use(payload, target)
    if mode == "stop":
        return handle_stop(payload, target, max_iterations)
    if mode == "goal":
        return handle_goal_check(payload, target, max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY_HOOK

  chmod +x "$HOOK_SCRIPT"
  echo "[$TARGET_NAME][hooks] Installed $HOOK_SCRIPT"
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

  local pre_cmd post_cmd stop_cmd goal_cmd tmp
  pre_cmd="python3 \"$HOOK_SCRIPT\" pre --target $TARGET_NAME --tag $HOOK_TAG"
  post_cmd="python3 \"$HOOK_SCRIPT\" post --target $TARGET_NAME --tag $HOOK_TAG"
  stop_cmd="python3 \"$HOOK_SCRIPT\" stop --target $TARGET_NAME --max-iterations $MAX_ITERATIONS --tag $HOOK_TAG"
  goal_cmd="python3 \"$HOOK_SCRIPT\" goal --target $TARGET_NAME --max-iterations $MAX_ITERATIONS --tag $HOOK_TAG"
  tmp=$(mktemp)

  jq \
    --arg tag "$HOOK_TAG" \
    --arg pre "$pre_cmd" \
    --arg post "$post_cmd" \
    --arg stop "$stop_cmd" \
    --arg goal "$goal_cmd" '
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
    | .hooks.Stop = ((.hooks.Stop | strip_tagged) + [{
        hooks:[{
          type:"command",
          command:$stop,
          statusMessage:"gpu-kernel-optimizer workflow gate",
          timeout:30
        }]
      }, {
        hooks:[{
          type:"command",
          command:$goal,
          statusMessage:"gpu-kernel-optimizer goal check",
          timeout:30
        }]
      }])
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
  [ "$MODE" = "hooks-only" ] || copy_skill
  enable_hooks_feature
  ensure_codex_agents_config
  install_hook_script
  merge_hooks
}

install_claude() {
  configure_claude_target
  [ "$MODE" = "hooks-only" ] || copy_skill
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
    ;;
  uninstall)
    for target in "${DETECTED_TARGETS[@]}"; do
      case "$target" in
        codex) uninstall_codex ;;
        claude) uninstall_claude ;;
      esac
    done
    ;;
esac

echo ""
echo "Done."
[ "$MODE" = "install" ] && echo "  gpu-wiki:       $GPU_WIKI_DIR"
[ "$MODE" = "install" ] && echo "  ref-projects:   $REFERENCE_PROJECTS_DIR"
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

echo ""
echo "Restart Codex/Claude Code or open a new session if hooks are not picked up immediately."
