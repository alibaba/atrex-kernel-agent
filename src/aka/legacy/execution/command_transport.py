"""Adapt explicit command inputs to the retained trusted sandbox launcher."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import base64
import io
import tarfile

from aka.contracts.execution import CommandSpec, UnsupportedExecution

RAW_PROFILE_SUFFIXES = (".ncu-rep", ".nsys-rep", ".att", ".pftrace", ".otf2")


def raw_profile_path(name):
    path = Path(name)
    return path.suffix.lower() in RAW_PROFILE_SUFFIXES or any(part.lower() == "att" for part in path.parts)


def freeze_cli_inputs(workspace, input_paths, max_file_bytes):
    """Snapshot the retained CLI's explicit input selection and skip policy.

    Trusted CLI compatibility uses the launcher's existing resource-link,
    directory, size and exclusion rules. Scoped callers must first authorize
    and pin their inputs; this function does not supply filesystem isolation.
    Returned files contain only relative names, frozen bytes and Unix modes.
    """
    from aka.contracts.execution import ExecutionFile
    from aka.legacy.execution.gateway.sandbox import _expand_workspace_input, _make_input_bundle
    selected = set()
    workspace = Path(workspace).resolve(strict=True)
    for value in input_paths:
        selected.update(_expand_workspace_input(workspace, value))
    if not selected:
        return (), ()
    bundle, _, skipped = _make_input_bundle(workspace, max_file_bytes, selected)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz") as archive:
        files = tuple(ExecutionFile(item.name, archive.extractfile(item).read(), item.mode & 0o777)
                      for item in archive.getmembers() if item.isfile())
    return files, tuple(skipped)


def sandbox_options(command):
    """Map public inputs to launcher options, separately from workload argv."""
    arguments = []
    for item in command.input_files:
        arguments.extend(("--input", item.path))
    profile = command.profile_options
    if profile is not None:
        for flag, value in (("--profile-level", profile.level), ("--profiler", profile.profiler),
                            ("--kernel-regex", profile.kernel_regex), ("--top-kernels", profile.top_kernels)):
            if value is not None:
                arguments.extend((flag, str(value)))
        for counter in profile.counters:
            arguments.extend(("--profile-counter", counter))
    output = command.output_options
    if output is not None:
        for flag, value in (("--inline-output", output.inline), ("--include-raw-profile", output.include_raw_profile)):
            if value is not None:
                arguments.append(flag if value else "--no-" + flag[2:])
        if output.max_file_bytes is not None:
            arguments.extend(("--max-output-file-bytes", str(output.max_file_bytes)))
    return tuple(arguments)


_PROTECTED_INPUTS = frozenset(("metadata.json", "reference.py", "input.py", "test_kernel.py", "solution.json",
                              "performance_baseline.json", "__aka_eval_driver.py", "__atrex_eval_driver.py"))
_PROTECTED_DIRS = frozenset((".git", "memory", ".aka-runtime", ".aka-verify", ".atrex-harness", "private-review-observations"))


def validate_input_files(command):
    """An explicit input cannot replace existing or protected Task content."""
    workspace = command.workspace.resolve(strict=True)
    from aka.legacy.execution.gateway.sandbox import INPUT_SKIP_PATHS, INPUT_SKIP_SUFFIXES, build_parser
    from aka.legacy.execution.legacy.runtime import current_transport_arguments
    input_limit = build_parser().parse_args(list(current_transport_arguments())).max_input_file_mb * 1024 * 1024
    if input_limit <= 0:
        raise UnsupportedExecution("input file size limit must be positive")
    for item in command.input_files:
        relative = Path(item.path)
        if relative.as_posix() in INPUT_SKIP_PATHS or relative.suffix in INPUT_SKIP_SUFFIXES:
            raise UnsupportedExecution("explicit input file is excluded by the selected launcher")
        if len(item.content) > input_limit:
            raise UnsupportedExecution("explicit input file exceeds the selected launcher size limit")
        if any(part in _PROTECTED_DIRS for part in relative.parts):
            raise UnsupportedExecution("input file targets protected workspace content")
        source = workspace / relative
        if source.exists():
            # The trusted retained launcher already materializes selected
            # Resources links. Matching frozen bytes/modes preserve that route;
            # only the temporary view is written, never the link destination.
            stat = source.stat()
            if not source.is_file() or stat.st_nlink != 1:
                raise UnsupportedExecution("input file must be an unlinked regular file")
            if source.read_bytes() != item.content or (stat.st_mode & 0o777) != item.mode:
                raise UnsupportedExecution("input file differs from existing workspace content")
        elif relative.name in _PROTECTED_INPUTS:
            raise UnsupportedExecution("input file cannot introduce trusted Task content")
        else:
            current = source.parent
            while current != workspace:
                if current.is_symlink():
                    raise UnsupportedExecution("new input file traverses a symbolic link")
                current = current.parent


@contextmanager
def prepared_input_workspace(command):
    """Freeze explicit leaves without writing the caller's workspace.

    This adapter is for trusted launchers with no filesystem scope. Existing
    selected workspace/resource files remain readable by that trusted launcher;
    untrusted argv never runs on the host here. Scoped executors must implement
    actual isolation and cannot use this compatibility view as that guarantee.
    """
    if not command.input_files:
        yield command
        return
    if command.filesystem_access is not None:
        raise UnsupportedExecution("trusted launcher input adaptation cannot enforce a filesystem scope")
    validate_input_files(command)
    original = command.workspace.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="aka-command-inputs-") as raw:
        workspace = Path(raw)
        # Shallow references preserve explicit immutable Resources mounts and
        # legacy private-input association without copying a whole workspace.
        for source in original.iterdir():
            if source.name == ".git":
                continue
            (workspace / source.name).symlink_to(source, target_is_directory=source.is_dir())
        # Returned paths are validated in this physical view before the caller
        # publishes them. Output destinations must therefore be local empty
        # directories, rather than references back into the caller's workspace.
        for relative in command.artifacts:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path == Path("."):
                raise UnsupportedExecution("command output paths must be workspace-relative")
            directory = workspace
            source_directory = original
            for component in path.parts[:-1]:
                target = directory / component
                source_directory /= component
                if target.is_symlink():
                    target.unlink()
                    target.mkdir()
                    if source_directory.is_dir():
                        for child in source_directory.iterdir():
                            (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
                else:
                    target.mkdir(exist_ok=True)
                directory = target
            target = directory / path.name
            if target.is_symlink():
                target.unlink()
            if (original / path).is_dir():
                target.mkdir(exist_ok=True)
        for item in command.input_files:
            directory = workspace
            source_directory = original
            for component in Path(item.path).parts[:-1]:
                target = directory / component
                source_directory /= component
                if target.is_symlink():
                    target.unlink()
                    target.mkdir()
                    if source_directory.is_dir():
                        for child in source_directory.iterdir():
                            (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
                else:
                    target.mkdir(exist_ok=True)
                directory = target
            target = directory / Path(item.path).name
            if target.is_symlink():
                target.unlink()
            with target.open("xb") as stream:
                stream.write(item.content)
            os.chmod(target, item.mode)
        yield replace(command, workspace=workspace)


def output_limit(args):
    """Use the exact explicit byte limit, otherwise the existing CLI fallback."""
    value = getattr(args, "max_output_file_bytes", None)
    return value if value is not None else args.max_output_file_mb * 1024 * 1024
