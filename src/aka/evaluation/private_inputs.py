"""Freeze selected Task inputs for the unchanged portable measurement method."""
from dataclasses import replace
import json
from pathlib import Path
import stat

from aka.contracts.execution import ExecutionFile, PreparedExecutionInputs, UnsupportedExecution
from aka.execution.command_syntax import option_value

_PRIVATE_FILES = ("shapes.json", "metadata.json", "roofline.json")
PROFILE_CASE_PATH = ".atrex_private_profile_case.json"


def selected_content(evaluation_input):
    """Read only the explicitly injected Task bundle, without environment inference."""
    if evaluation_input is None:
        return {}
    result = {}
    for item in evaluation_input.files:
        if item.path in result:
            raise ValueError("duplicate Task evaluation input")
        if item.source is not None:
            data = item.source.read_bytes()
            mode = stat.S_IMODE(item.source.stat().st_mode)
        else:
            data, mode = item.text.encode("utf-8"), 0o644
        result[item.path] = ExecutionFile(item.path, data, mode)
    return result


def portable_inputs(command, evaluation_input, content, *, evaluator, kind, capabilities, inject_private=True, prepared_files=()):
    """Construct a closed remote-only view without writing the Agent workspace."""
    if evaluation_input is None:
        return command
    from aka.execution.input_snapshot import snapshot_command_inputs
    files = {}
    private = evaluation_input.private_reference_dir is not None
    # Private Task members replace only matching public allowlist paths.
    if evaluator and private and inject_private:
        for name in _PRIVATE_FILES:
            item = content.get(name)
            if item is None and name in {"shapes.json", "metadata.json"}:
                raise ValueError("required private evaluator input is missing: " + name)
            if item is not None:
                files[name] = item
    private_paths = set(files)
    if kind == "profile" and private:
        shapes = json.loads(content["shapes.json"].content)
        if not isinstance(shapes, dict) or not shapes:
            raise ValueError("private shapes.json must contain a non-empty object")
        explicit = option_value(list(command.command), "--shape-id")
        shape_id = explicit or command.environment.get("PROFILE_SHAPE_ID") or sorted(
            (str(value) for value in shapes), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))[0]
        entry = shapes.get(shape_id)
        if not isinstance(entry, dict):
            raise ValueError(f"PROFILE_SHAPE_ID={shape_id!r} is not a real evaluator shape id")
        payload = {"schema_version": 1, "shape_id": shape_id,
                   "init_kwargs": entry.get("init_kwargs") or {}, "input_kwargs": entry.get("input_kwargs") or {}}
        files[PROFILE_CASE_PATH] = ExecutionFile(PROFILE_CASE_PATH,
            (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"), 0o400)
        private_paths.add(PROFILE_CASE_PATH)
    runtime_files = ()
    if evaluator and evaluation_input.runtime_root is not None:
        runtime = evaluation_input.runtime_root
        package = runtime / "src" / "atrex_bench"
        if not (runtime / "scripts/run_eval.py").is_file() or not (package / "utils.py").is_file():
            raise RuntimeError("invalid selected Task Atrex-Bench runtime")
        # Preserve the evaluator-only runtime closure; Profile imports its public
        # driver and does not upload the evaluator runtime.
        selections = ["scripts/run_eval.py", "src/atrex_bench/__init__.py", "src/atrex_bench/utils.py"]
        selections += ["src/atrex_bench/" + name for name in ("sdk.py", "shape_contracts.py", "eval", "cli")
                       if (package / name).exists()]
        raw, skipped = snapshot_command_inputs(runtime, tuple(selections), filesystem_access=None, max_file_bytes=None)
        runtime_files = tuple(replace(item, path="atrex-bench/" + item.path) for item in raw)
    if not files and not runtime_files:
        return command
    if capabilities is None or not capabilities.prepared_inputs:
        raise UnsupportedExecution("selected executor cannot transport the closed Task evaluation inputs")
    from .portable_selection import public_inputs
    public = {item.path: item for item in public_inputs(command, evaluator=evaluator, prepared_files=prepared_files)}
    public.update(files)
    files = public
    if kind == "profile" and private:
        from .command_inputs import PROFILE_ENVIRONMENT_KEYS
        controls = tuple(f"{key}={value}" for key, value in command.environment.items() if key in PROFILE_ENVIRONMENT_KEYS)
        if controls:
            command = replace(command, command=("env", *controls, *command.command),
                environment={key: value for key, value in command.environment.items() if key not in PROFILE_ENVIRONMENT_KEYS})
    return replace(command, prepared_inputs=PreparedExecutionInputs(tuple(files[name] for name in sorted(files)), tuple(sorted(runtime_files, key=lambda item: item.path)), tuple(sorted(private_paths))))
