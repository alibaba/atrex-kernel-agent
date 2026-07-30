"""Optimization-mode policy and mechanical production-kernel enforcement."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")
MODE_STATE_FILE = ".orchestrator_mode.json"
POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"
POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"
AGGREGATE_DISPATCH_FILE = "aggregate_dispatch.json"
AGGREGATE_KERNELS_DIR = "aggregate_kernels"


def _framework_key(framework: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", framework.strip().lower())
    aliases = {
        "triton": "triton",
        "gluon": "gluon",
        "tritongluon": "gluon",
        "cutedsl": "cutedsl",
        "cute": "cutedsl",
        "cuda": "cuda",
        "cudac": "cuda",
        "flydsl": "flydsl",
        "fly": "flydsl",
    }
    return aliases.get(token, token)


def optimization_mode_directive(mode: str, framework: str) -> str:
    """Self-contained policy block injected into every coding-agent prompt."""
    if mode == "leaderboard":
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` exactly. Its existing framework guidance remains "
            "unchanged: the requested framework is a recommended direction, compatible mixed/alternate "
            "implementations are allowed when evidence supports them, and third-party helper/kernel "
            "libraries may be used.\n"
        )
    if mode != "production":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    return (
        "## Optimization mode: production (hard gate)\n\n"
        "This generated section overrides any conflicting permissive framework or third-party-library "
        "guidance elsewhere in `CLAUDE.md`.\n\n"
        f"- The implementation framework is exactly **{framework}**. It is a hard constraint, not a "
        "recommendation. Do not switch to another DSL, mix another kernel framework into the candidate, "
        "or replace the implementation with a prebuilt operator.\n"
        "- The V0 PyTorch reference wrapper is the only baseline exception. Every optimized candidate "
        f"committed after V0 must implement the GPU computation directly in **{framework}**.\n"
        "- Do not import, call, dispatch to, or depend on third-party kernel/operator libraries "
        "(for example FlashInfer, FlashAttention, xFormers, vLLM custom ops, cuBLAS/cuDNN wrappers, "
        "CUTLASS kernels outside CuteDSL itself, or another DSL). Documentation and source may be read "
        "for research, but the submitted candidate must be self-contained apart from Python/PyTorch "
        "plumbing and the selected framework/toolchain.\n"
        "- Update `solution.json` so its languages and dependencies contain only PyTorch/evaluator "
        "plumbing plus the selected framework. Before committing, inspect `kernel.py` and "
        "`solution.json` against these rules. The orchestrator will mechanically reject and revert a "
        "kernel-changing commit that violates them, even if it is faster and correct.\n"
    )


def workspace_policy_block(mode: str, framework: str) -> str:
    directive = optimization_mode_directive(mode, framework).rstrip()
    return f"{POLICY_BEGIN}\n\n{directive}\n\n{POLICY_END}\n"


def install_workspace_policy(workspace: Path, mode: str, framework: str) -> None:
    """Persist mode identity and append/replace the generated CLAUDE.md overlay.

    The ignored state file prevents accidentally resuming one workspace under a
    different policy. The production overlay explicitly overrides the permissive
    framework section in the base CLAUDE.md.
    """
    if mode not in OPTIMIZATION_MODE_CHOICES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / MODE_STATE_FILE
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid optimization-mode state: {state_path}") from exc
        existing_mode = state.get("mode")
        existing_framework = state.get("framework")
        if existing_mode != mode or existing_framework != framework:
            raise RuntimeError(
                "workspace policy mismatch: "
                f"recorded mode/framework={existing_mode}/{existing_framework}, "
                f"requested={mode}/{framework}"
            )
    else:
        state_path.write_text(
            json.dumps({"mode": mode, "framework": framework}, indent=2) + "\n",
            encoding="utf-8",
        )

    claude_path = workspace / "CLAUDE.md"
    current = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
    generated = workspace_policy_block(mode, framework)
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        current = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    claude_path.write_text(current, encoding="utf-8")

    gitignore = workspace / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    entry = f"/{MODE_STATE_FILE}"
    if entry not in ignored.splitlines():
        with gitignore.open("a", encoding="utf-8") as handle:
            if ignored and not ignored.endswith("\n"):
                handle.write("\n")
            handle.write("\n# orchestrator optimization-mode identity (local policy state)\n")
            handle.write(entry + "\n")


_STDLIB_IMPORTS = frozenset(sys.stdlib_module_names) | {"__future__"}
_ALLOWED_IMPORTS = {
    "triton": {"torch", "triton", "sol_execbench"},
    "gluon": {"torch", "triton", "sol_execbench"},
    "cutedsl": {"torch", "cutlass", "cuda", "sol_execbench"},
    "cuda": {"torch", "cuda", "sol_execbench"},
    "flydsl": {"torch", "flydsl", "sol_execbench"},
}
_ALLOWED_DEPENDENCY_TOKENS = {
    "triton": {"torch", "triton"},
    "gluon": {"torch", "triton"},
    "cutedsl": {"torch", "cutlass", "nvidiacutlassdsl", "cuda", "cudapython"},
    "cuda": {"torch", "cuda", "cudapython"},
    "flydsl": {"torch", "flydsl"},
}


def _import_roots(tree: ast.AST) -> tuple[set[str], bool]:
    roots: set[str] = set()
    relative = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = True
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots, relative


def _normalized_dependency(value: object) -> str:
    text = re.split(r"[<>=!~\[; ]", str(value).strip(), maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


_BANNED_TORCH_COMPUTE = {
    "addmm", "amax", "amin", "bmm", "conv1d", "conv2d", "conv3d", "cumprod", "cumsum",
    "einsum", "exp", "gelu", "layer_norm", "log", "log_softmax", "matmul", "max", "mean",
    "min", "mm", "rms_norm", "scaled_dot_product_attention", "sigmoid", "silu", "softmax",
    "sort", "sum", "topk",
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _torch_compute_violations(tree: ast.AST) -> list[str]:
    torch_aliases = {"torch"}
    functional_aliases: set[str] = set()
    direct_functional_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_aliases.add(alias.asname or "torch")
                elif alias.name == "torch.nn.functional":
                    functional_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch.nn.functional":
            direct_functional_calls.update(alias.asname or alias.name for alias in node.names)

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append("Python/PyTorch matrix multiplication is not the selected kernel framework")
            continue
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if not dotted:
            continue
        parts = dotted.split(".")
        if parts[0] in direct_functional_calls:
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
            continue
        if any(dotted == alias or dotted.startswith(alias + ".") for alias in functional_aliases):
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
            continue
        if parts[0] not in torch_aliases or len(parts) < 2:
            continue
        suffix = parts[1:]
        if suffix[0] == "ops":
            violations.append(f"torch.ops dispatch is forbidden in production candidate: {dotted}")
        elif suffix[:2] == ["nn", "functional"]:
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
        elif suffix[-1] in _BANNED_TORCH_COMPUTE:
            violations.append(f"PyTorch compute call is forbidden in production candidate: {dotted}")
    return list(dict.fromkeys(violations))


def production_kernel_violations(workspace: Path, framework: str) -> list[str]:
    """Return static production-policy violations for the current candidate.

    This is deliberately a fail-closed commit gate. Runtime correctness and
    performance still use the normal sandbox; this gate proves implementation
    provenance and dependency discipline, which benchmark results cannot prove.
    """
    key = _framework_key(framework)
    errors: list[str] = []
    if key not in _ALLOWED_IMPORTS:
        return [f"unsupported production framework: {framework}"]
    kernel_path = workspace / "kernel.py"
    if not kernel_path.is_file():
        return ["kernel.py is missing"]
    source = kernel_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(kernel_path))
    except SyntaxError as exc:
        return [f"kernel.py is not valid Python: {exc.msg} (line {exc.lineno})"]

    policy_sources = [source]
    policy_trees = [tree]
    aggregate_dispatch = workspace / AGGREGATE_DISPATCH_FILE
    aggregate_mode = aggregate_dispatch.is_file()
    if aggregate_mode:
        try:
            dispatch = json.loads(aggregate_dispatch.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{AGGREGATE_DISPATCH_FILE} is invalid: {exc}")
            dispatch = {}
        if dispatch.get("schema_version") != 1 or dispatch.get("mode") != "deterministic_dispatch":
            errors.append("aggregate dispatcher manifest is not deterministic schema v1")
        modules = dispatch.get("modules")
        if not isinstance(modules, dict) or not modules:
            errors.append("aggregate dispatcher manifest has no bucket modules")
            modules = {}
        for bucket, record in sorted(modules.items()):
            relative = record.get("path") if isinstance(record, dict) else None
            if not isinstance(relative, str):
                errors.append(f"aggregate bucket {bucket} has no source path")
                continue
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or len(relative_path.parts) != 2
                or relative_path.parts[0] != AGGREGATE_KERNELS_DIR
                or relative_path.suffix != ".py"
            ):
                errors.append(f"invalid aggregate bucket source path: {relative}")
                continue
            module_path = workspace / relative_path
            if not module_path.is_file():
                errors.append(f"aggregate bucket source is missing: {relative}")
                continue
            module_source = module_path.read_text(encoding="utf-8", errors="replace")
            try:
                module_tree = ast.parse(module_source, filename=str(module_path))
            except SyntaxError as exc:
                errors.append(
                    f"aggregate bucket {bucket} is not valid Python: "
                    f"{exc.msg} (line {exc.lineno})"
                )
                continue
            policy_sources.append(module_source)
            policy_trees.append(module_tree)

    roots: set[str] = set()
    has_relative_import = False
    for policy_tree in policy_trees:
        tree_roots, tree_relative = _import_roots(policy_tree)
        roots.update(tree_roots)
        has_relative_import = has_relative_import or tree_relative
    if has_relative_import:
        errors.append("relative/local-module imports are not self-contained")
    allowed = _STDLIB_IMPORTS | _ALLOWED_IMPORTS[key]
    if aggregate_mode:
        allowed = allowed | {AGGREGATE_KERNELS_DIR}
    for root in sorted(roots - allowed):
        errors.append(f"third-party import is not allowed in production mode: {root}")
    for root in sorted(roots & {"ctypes", "importlib", "pkgutil", "runpy", "subprocess"}):
        errors.append(f"dynamic external-code loading is forbidden in production candidate: {root}")
    for policy_tree in policy_trees:
        errors.extend(_torch_compute_violations(policy_tree))

    policy_source = "\n".join(policy_sources)

    marker_checks = {
        "triton": (
            bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+triton\b", policy_source)),
            "missing Triton implementation/import",
        ),
        "gluon": ("triton.experimental" in policy_source and "gluon" in policy_source, "missing Gluon implementation"),
        "cutedsl": ("cutlass.cute" in policy_source or "@cute.kernel" in policy_source, "missing CuteDSL implementation"),
        "cuda": (
            "__global__" in policy_source
            and bool(re.search(r"load_inline|cpp_extension|CUDAExtension|nvrtc|cuda\.bindings", policy_source)),
            "missing self-authored CUDA kernel/loader",
        ),
        "flydsl": (bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+flydsl\b", policy_source)), "missing FlyDSL implementation"),
    }
    marker_ok, marker_error = marker_checks[key]
    if not marker_ok:
        errors.append(marker_error)

    foreign_markers = {
        "triton": bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+triton\b", policy_source)),
        "gluon": "triton.experimental" in policy_source and "gluon" in policy_source,
        "cutedsl": "cutlass.cute" in policy_source or "@cute.kernel" in policy_source,
        "cuda": "__global__" in policy_source,
        "flydsl": bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+flydsl\b", policy_source)),
    }
    compatible_markers = {key}
    if key == "gluon":
        compatible_markers.add("triton")  # Gluon is distributed under triton.experimental.
    for other, present in foreign_markers.items():
        if present and other not in compatible_markers:
            errors.append(f"mixed/alternate framework marker is forbidden: {other}")

    banned_source_patterns = {
        r"\btorch\.ops\b": "torch.ops dispatch is a prebuilt/custom operator call",
        r"\btorch\.nn\.functional\b": "torch.nn.functional is not the selected kernel framework",
        r"\btorch\.(?:linalg|_scaled_mm)\b": "PyTorch compute fallback is not the selected kernel framework",
        r"\b(?:flashinfer|flash_attn|xformers|vllm|sglang|bitsandbytes)\b": "third-party kernel/operator library reference",
        r"\b(?:cublas|cudnn)[A-Za-z0-9_]*\b": "prebuilt CUDA math/operator library call",
        r"#\s*include\s*[<\"]cutlass/": "prebuilt CUTLASS C++ kernels are forbidden outside CuteDSL",
    }
    for pattern, message in banned_source_patterns.items():
        if re.search(pattern, policy_source, flags=re.IGNORECASE) and not (
            key == "cutedsl" and "CUTLASS" in message
        ):
            errors.append(message)

    solution_path = workspace / "solution.json"
    if solution_path.is_file():
        try:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"solution.json is invalid: {exc}")
        else:
            dependencies = ((solution.get("spec") or {}).get("dependencies") or [])
            allowed_dependencies = _ALLOWED_DEPENDENCY_TOKENS[key]
            for dependency in dependencies:
                token = _normalized_dependency(dependency)
                if token and token not in allowed_dependencies:
                    errors.append(f"third-party solution dependency is not allowed: {dependency}")
    return list(dict.fromkeys(errors))


def reject_production_commit(
    workspace: Path,
    version: int,
    pre_head: str,
    violations: list[str],
) -> Path:
    """Revert a violating kernel commit and preserve an actionable local record."""
    memory_path = workspace / "memory" / f"v{version}.json"
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        memory = {}
    if pre_head:
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory["version"] = f"v{version}"
    memory["masked"] = False
    memory["git_commit_hash"] = None
    memory["quality_gate"] = {
        "result": "FAIL",
        "failure_reason": "production policy violation: " + "; ".join(violations),
    }
    memory["optimization"] = {
        "action_category": "production_policy_rejection",
        "action_description": "reverted candidate that used a forbidden dependency or wrong framework",
    }
    pitfalls = memory.setdefault("pitfalls_and_fixes", [])
    pitfalls.append({
        "error_type": "production_policy",
        "error_message": "; ".join(violations),
        "lesson": "implement the candidate directly and exclusively in the selected framework",
    })
    memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return memory_path
