"""Legacy source-contract and measurement route policy."""
import ast
from pathlib import Path
from aka.execution.command_syntax import is_test_kernel_command as _is_test_kernel_command, test_kernel_script_index as _test_kernel_script_index, option_value as _option_value
from .gateway_request import _json_object

def _typed_workspace_limitation(
    workspace: Path, command: list[str], kind: str, *, reference_dir=None, reference_content=None, private=False
) -> str | None:
    """Explain why the typed run/profile source contract cannot represent a workspace."""
    required = ("kernel.py", "reference.py", "input.py")
    missing = [name for name in required if not (workspace / name).is_file() and name not in (reference_content or {})]
    if missing:
        return "missing " + ", ".join(missing)
    try:
        if "shapes.json" not in (reference_content or {}) and not ((reference_dir or workspace) / "shapes.json").is_file():
            raise ValueError("missing shapes.json")
    except ValueError as exc:
        return str(exc)
    if (
        kind == "run"
        and _is_test_kernel_command(command)
        and _test_kernel_script_index(command, typed_launcher=True) is None
    ):
        return "evaluator launcher semantics require the dev route"
    if kind == "profile" and private:
        return "generalized tasks inject one private real shape through the dev profile route"
    if (workspace / "workload.jsonl").is_file():
        return (
            "SOL-ExecBench workload.jsonl is not supported by the Atrex-Bench typed API"
        )

    solution = _json_object(workspace / "solution.json")
    if solution is not None:
        sources = solution.get("sources")
        if isinstance(sources, list):
            source_paths = {
                str(item.get("path"))
                for item in sources
                if isinstance(item, dict) and item.get("path")
            }
            if source_paths - {"kernel.py"}:
                return "solution.json declares auxiliary candidate sources"

    try:
        tree = ast.parse((workspace / "kernel.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        tree = None
    if tree is not None:
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        local_imports = sorted(
            root
            for root in imported_roots
            if root not in {"input", "reference", "kernel"}
            and (
                (workspace / f"{root}.py").is_file()
                or (workspace / root / "__init__.py").is_file()
            )
        )
        if local_imports:
            return "candidate imports local auxiliary modules: " + ", ".join(
                local_imports
            )

    # These test-harness controls do not exist in the typed request contract.
    # Preserve their exact semantics through dev instead of silently dropping them.
    unsupported_options = (
        "--seed",
        "--workspace",
        "--candidate-timeout-s",
        "--perf-timeout-s",
    )
    for option in unsupported_options:
        if _option_value(command, option) is not None:
            return f"{option} is not supported by the typed API"
    warmup = _option_value(command, "--warmup")
    if warmup is not None and str(warmup) != "5":
        return "non-default --warmup is not supported by the typed API"
    for option, default in (("--atol", 1e-2), ("--rtol", 0.05)):
        value = _option_value(command, option)
        if value is not None:
            try:
                matches_default = float(value) == default
            except (TypeError, ValueError):
                matches_default = False
            if not matches_default:
                return f"non-default {option} is not exposed by agate run"
    return None
