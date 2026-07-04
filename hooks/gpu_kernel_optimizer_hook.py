#!/usr/bin/env python3
"""Hook guard for gpu-kernel-optimizer.

The hook keeps long-running kernel optimization sessions from stopping while
state is inconsistent. It is intentionally conservative: it only enforces rules
inside a workspace — a directory holding the WORKSPACE_SENTINEL file.
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
# A workspace is any directory containing this sentinel (written by optimize.py init_workspace /
# workspace_init.sh), so gates fire in a run root of ANY name. MUST match optimize.py's
# WORKSPACE_SENTINEL.
WORKSPACE_SENTINEL = ".gpu_kernel_optimizer_workspace"


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
    """Nearest ancestor of `path` (including itself) that is a workspace — a directory holding the
    WORKSPACE_SENTINEL file. Returns None if none."""
    resolved = path.expanduser()
    for candidate in [resolved, *resolved.parents]:
        try:
            if (candidate / WORKSPACE_SENTINEL).is_file():
                return candidate
        except OSError:
            pass
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

    def _emit(ws: Path) -> Iterable[Path]:
        key = str(ws)
        if key not in seen:
            seen.add(key)
            yield ws

    for root in candidate_roots(payload):
        # The workspace is the run root (the nearest sentinel'd ancestor of a candidate root) — nested
        # sessions run with cwd == the workspace, so the ancestor walk finds it. We deliberately do NOT
        # glob candidate roots (incl. /tmp, /) for sibling workspaces: that would scan /tmp on every
        # tool call and could match an UNRELATED recently-active workspace and gate the wrong session.
        ws = find_workspace_from_path(root)
        if ws is not None:
            yield from _emit(ws)


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


def output_path_for_workspace(workspace: Path) -> Path:
    # The workspace is the run root (the project root), so the final artifact lives in it (there is
    # no separate per-op subdirectory).
    return workspace / "generated_kernel.py"


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
        output_status = f"No generated_kernel.py was found in the workspace root ({workspace})."
    elif violations:
        output_status = f"generated_kernel.py exists at {output_path}, but violates the final output contract: {', '.join(violations)}."
    else:
        output_status = f"generated_kernel.py exists at {output_path}, but the output contract gate requires one final confirmation pass."

    return (
        "gpu-kernel-optimizer final output contract gate: "
        f"workspace {workspace} is about to end the current optimization session. "
        f"{output_status} "
        f"Do not modify the root SKILL.md. Read the child skill `{OUTPUT_CONTRACT_SKILL}` from the installed gpu-kernel-optimizer skill. "
        f"Then convert the validated optimized implementation into `generated_kernel.py` in the workspace root directory ({workspace}) — your current working directory. "
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
        "gpu-kernel-optimizer goal check (escalated): before stopping, read README.md in "
        f"workspace {workspace} to confirm the exact optimization target (e.g. TFLOPS, bandwidth, utilization threshold). "
        "If the target is not met, do not stop. Treat any conclusion that the goal cannot be reached with skepticism: "
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
        "gpu-kernel-optimizer goal check (final escalation): before stopping, read README.md in "
        f"workspace {workspace} to confirm the exact optimization target (e.g. TFLOPS, bandwidth, utilization threshold). "
        "If the target is not met, do not stop. If no new optimization direction is available, launch the gpu-kernel-partial-restart agent: "
        "randomly mask half of the optimization experience in memory/v*.json files, then restart optimization work. "
        "Then continue directly into the gpu-kernel-optimizer "
        "Stage 2 flow from the installed skill entry (`gpu-kernel-profile-optimizer/SKILL.md`): update the Stage 2 optimization plan, "
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


def handle_goal_check(payload: dict, target: str) -> int:
    stop_hook_active = payload.get("stop_hook_active", False)

    for workspace in iter_workspaces(payload):
        if not recently_active(workspace):
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

def handle_stop(payload: dict, target: str) -> int:
    for workspace in iter_workspaces(payload):
        if not recently_active(workspace):
            continue

        newest_mtime = newest_iteration_artifact_mtime(workspace)
        if newest_mtime <= 0:
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


def parse_args() -> tuple[str, str]:
    mode = "stop"
    target = "generic"
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
        index += 1
    return mode, target


def main() -> int:
    mode, target = parse_args()
    payload = load_payload()
    if mode == "pre":
        return handle_pre_tool_use(payload, target)
    if mode == "post":
        return handle_post_tool_use(payload, target)
    if mode == "stop":
        return handle_stop(payload, target)
    if mode == "goal":
        return handle_goal_check(payload, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
