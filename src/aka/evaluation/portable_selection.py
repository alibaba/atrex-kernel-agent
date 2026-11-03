"""Select the original portable evaluator and Profile workspace workload."""
import json
from pathlib import Path, PurePosixPath

from aka.execution.command_syntax import command_parts
from aka.execution.input_snapshot import read_command_input, snapshot_command_inputs

_EVALUATION_FILES = ("agent_problem.json", "definition.json", "input.py", "kernel.py", "metadata.json",
                     "reference.py", "roofline.json", "shapes.json", "solution.json", "test_kernel.py", "workload.jsonl")
_CANDIDATE_FILES = ("agent_problem.json", "definition.json", "input.py", "kernel.py", "reference.py",
                    "shapes.json", "solution.json", "workload.jsonl")
_OUTPUT_FLAGS = frozenset(("-o", "--output", "--output-dir"))


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("portable input must be workspace-relative")
    return path.as_posix()


def public_inputs(command, *, evaluator, prepared_files=()):
    """Freeze the method allowlist, including yet-unpublished prepared byte files."""
    workspace = command.workspace
    generated = {item.path: item for item in prepared_files}
    selected = set(_EVALUATION_FILES if evaluator else _CANDIDATE_FILES)
    selected.update(item.path for item in command.input_files)
    referenced = set()
    parts = command_parts(list(command.command))
    skip_next = False
    for index, token in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if token in _OUTPUT_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(flag + "=") for flag in _OUTPUT_FLAGS):
            continue
        if index > 0 and parts[index - 1] in {"-c", "--command"}:
            continue
        try:
            name = _relative(token)
        except ValueError:
            continue
        path = workspace / name
        if name in generated or path.is_file() or path.is_dir():
            referenced.add(name)
    selected.update(referenced)
    solution = workspace / "solution.json"
    if solution.is_file():
        try:
            frozen = read_command_input(command.filesystem_access, solution.absolute())
            value = json.loads(frozen[0])
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid workspace solution.json: {error}") from error
        sources = value.get("sources", []) if isinstance(value, dict) else []
        if not isinstance(sources, list):
            raise RuntimeError("workspace solution.json sources must be a list of paths")
        for item in sources:
            name = item if isinstance(item, str) else item.get("path") if isinstance(item, dict) else None
            if not isinstance(name, str):
                raise RuntimeError("workspace solution.json source entries must be paths or path objects")
            selected.add(_relative(name))
    if evaluator:
        # The ABBA/numerical launcher resolves its frozen manifests relative to
        # each selected request. Preserve that original sibling snapshots view.
        for name in referenced:
            if "verification_artifacts" not in PurePosixPath(name).parts:
                continue
            directory = PurePosixPath(name).parent / "snapshots"
            selected.update(path for path in generated if PurePosixPath(path).is_relative_to(directory))
            if (workspace / directory).is_dir():
                selected.add(directory.as_posix())
    else:
        basenames = {Path(token).name for token in parts}
        if "profile_nvidia.sh" in basenames:
            selected.update(("profiling/nvidia/profile_nvidia.sh", "profiling/nvidia/classify_ncu.py",
                             "profiling/nvidia/ncu_classification.py"))
            helpers = "profiling/nvidia/ncu_helpers"
            if (workspace / helpers).is_dir():
                selected.add(helpers)
        if "profile_kernel.sh" in basenames:
            selected.add("profiling/amd/profile_kernel.sh")
        for name in tuple(selected):
            path = PurePosixPath(name)
            if "harness" in path.parts:
                index = path.parts.index("harness")
                directory = PurePosixPath(*path.parts[:index + 1])
                if (workspace / directory).is_dir():
                    selected.add(directory.as_posix())
    explicit = {item.path for item in command.input_files}
    # Explicit leaves remain in CommandSpec.input_files; the physical payload
    # appends that group, so duplicating them here would break its path identity.
    generated_selected = {name: generated[name] for name in selected if name in generated and name not in explicit}
    existing = tuple(sorted(name for name in selected if name not in generated and name not in explicit
                            and ((workspace / name).is_file() or (workspace / name).is_dir())))
    files, skipped = snapshot_command_inputs(workspace, existing, filesystem_access=command.filesystem_access,
                                            max_file_bytes=None)
    from aka.contracts.execution import ExecutionFile
    result = {item.path: item for item in files if item.path not in explicit}
    result.update({name: ExecutionFile(item.path, item.content, item.mode) for name, item in generated_selected.items()})
    return tuple(result[name] for name in sorted(result))
