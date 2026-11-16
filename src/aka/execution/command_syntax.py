"""One lexical grammar for command words and retained supported launchers."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any


def _parsed_command_parts(parts: list[str]) -> tuple[list[str], bool]:
    """Return command words and whether a single shell string stayed opaque."""
    command = parts[1:] if parts and parts[0] == "--" else list(parts)
    if len(command) != 1:
        return command, False
    try:
        parsed = shlex.split(command[0])
    except ValueError:
        return command, True
    if shlex.join(parsed) == command[0]:
        return parsed, False
    return command, True

def command_parts(parts: list[str]) -> list[str]:
    return _parsed_command_parts(parts)[0]

def _standard_command_name(value: str, names: set[str]) -> str | None:
    """Return a command name only for PATH lookup or a conventional system path."""
    name = Path(value).name
    if name in names and value in {name, f"/bin/{name}", f"/usr/bin/{name}"}:
        return name
    return None

def _command_executable_index(
    command: list[str], *, typed_launcher: bool = False
) -> int | None:
    """Skip supported shell assignments, env, and execution wrappers."""
    def assignment_end(start: int, *, shell_prefix: bool = False) -> int | None:
        while start < len(command):
            name, separator, _ = command[start].partition("=")
            if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                break
            if shell_prefix and shlex.quote(command[start]) != command[start]:
                break
            if typed_launcher and name not in {
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONUNBUFFERED",
            }:
                return None
            start += 1
        return start

    index = assignment_end(0, shell_prefix=True)
    if index is None:
        return None
    if index < len(command) and command[index] in {"env", "/usr/bin/env"}:
        index += 1
        while index < len(command):
            option = command[index]
            if option == "--":
                index += 1
                break
            if option == "-" or option == "--ignore-environment" or option == "--debug":
                if typed_launcher:
                    return None
                index += 1
                continue
            if re.fullmatch(r"-[iv]+", option):
                if typed_launcher:
                    return None
                index += 1
                continue
            if option in {"-C", "--chdir", "-u", "--unset"}:
                if index + 1 >= len(command):
                    return None
                if typed_launcher:
                    return None
                index += 2
                continue
            if (option.startswith(("-C", "-u")) and len(option) > 2) or (
                option.startswith(("--chdir=", "--unset="))
                and option.partition("=")[2]
            ):
                if typed_launcher:
                    return None
                index += 1
                continue
            if option.startswith("-"):
                return None
            break
        index = assignment_end(index)
        if index is None:
            return None
    while index < len(command):
        launcher = command[index]
        wrapper = _standard_command_name(
            launcher,
            {"command", "exec", "nice", "nohup", "stdbuf", "time", "timeout"},
        )
        if wrapper is None:
            break
        if typed_launcher:
            return None
        index += 1

        if wrapper == "command":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option == "-p":
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "exec":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option == "-a":
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if re.fullmatch(r"-[cl]+", option):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "nice":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-n", "--adjustment"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if (
                    re.fullmatch(r"-(?:n)?\d+", option)
                    or option.startswith("--adjustment=")
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "time":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-f", "--format", "-o", "--output"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if (
                    re.fullmatch(r"-[fo].+", option)
                    or option.startswith("--format=")
                    or option.startswith("--output=")
                    or re.fullmatch(r"-[apqv]+", option)
                    or option in {"--append", "--portability", "--quiet", "--verbose"}
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "timeout":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-k", "--kill-after", "-s", "--signal"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if (
                    re.fullmatch(r"-[ks].+", option)
                    or option.startswith(("--kill-after=", "--signal="))
                    or option in {"--foreground", "--preserve-status", "--verbose"}
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
            if index >= len(command):
                return None
            index += 1
        elif wrapper == "stdbuf":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-e", "--error", "-i", "--input", "-o", "--output"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if re.fullmatch(r"-[eio].+", option) or option.startswith(
                    ("--error=", "--input=", "--output=")
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        else:
            if index < len(command) and command[index] == "--":
                index += 1
            elif index < len(command) and command[index].startswith("-"):
                return None
    if index < len(command) and command[index] in {"env", "/usr/bin/env"}:
        nested = _command_executable_index(
            command[index:], typed_launcher=typed_launcher
        )
        return index + nested if nested is not None else None
    return index if index < len(command) else None

def _python_script_index(
    parts: list[str], script_name: str, *, typed_launcher: bool = False
) -> int | None:
    """Locate a Python script; optionally require a prefix typed run may omit."""
    command, opaque = _parsed_command_parts(parts)
    if opaque:
        return None
    index = _command_executable_index(command, typed_launcher=typed_launcher)

    if (
        index is None
        or re.fullmatch(r"python(?:3(?:\.\d+)*)?", Path(command[index]).name) is None
    ):
        return None
    index += 1
    while index < len(command):
        option = command[index]
        if option == "--":
            index += 1
            break
        if re.fullmatch(r"-[bBdEiIOPqRsSuvx]+", option):
            if typed_launcher and re.fullmatch(r"-[Bu]+", option) is None:
                return None
            index += 1
            continue
        if option in {"-W", "-X"}:
            if index + 1 >= len(command):
                return None
            if typed_launcher:
                return None
            index += 2
            continue
        if len(option) > 2 and option.startswith(("-W", "-X")):
            if typed_launcher:
                return None
            index += 1
            continue
        if option == "--check-hash-based-pycs":
            if index + 1 >= len(command) or command[index + 1] not in {
                "always",
                "default",
                "never",
            }:
                return None
            if typed_launcher:
                return None
            index += 2
            continue
        if option.startswith("-"):
            return None
        break

    if index < len(command) and Path(command[index]).name == script_name:
        return index
    return None

def test_kernel_script_index(
    parts: list[str], *, typed_launcher: bool = False
) -> int | None:
    return _python_script_index(
        parts, "test_kernel.py", typed_launcher=typed_launcher
    )

def is_test_kernel_command(parts: list[str]) -> bool:
    return test_kernel_script_index(parts) is not None

def _shell_command_operand(
    command: list[str], executable_index: int
) -> tuple[str, int] | None:
    """Locate a shell script or the command string consumed by ``-c``."""
    shell = _standard_command_name(command[executable_index], {"bash", "sh"})
    if shell is None:
        return None
    index = executable_index + 1
    while index < len(command):
        option = command[index]
        if option == "--":
            index += 1
            break
        if shell == "bash" and option in {"--init-file", "--rcfile"}:
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        if shell == "bash" and (
            option.startswith("--init-file=") or option.startswith("--rcfile=")
        ):
            index += 1
            continue
        if shell == "bash" and option in {
            "--debug",
            "--debugger",
            "--login",
            "--noediting",
            "--noprofile",
            "--norc",
            "--posix",
            "--protected",
            "--restricted",
            "--verbose",
        }:
            index += 1
            continue
        if shell == "bash" and option in {
            "--dump-po-strings",
            "--dump-strings",
            "--help",
            "--version",
            "--wordexp",
        }:
            return None
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]*c", option):
            return ("command", index + 1) if index + 1 < len(command) else None
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]*[oO]", option):
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]+", option):
            index += 1
            continue
        if option.startswith("-"):
            return None
        break
    return ("script", index) if index < len(command) else None

def is_profile_command(parts: list[str]) -> bool:
    """Return whether argv invokes one of the repository profiler wrappers."""
    command, opaque = _parsed_command_parts(parts)
    if opaque:
        return False
    if _python_script_index(command, "profile_driver.py") is not None:
        return True
    index = _command_executable_index(command)
    if index is None:
        return False
    frontend = _standard_command_name(command[index], {"ncu", "nsys", "rocprofv3"})
    if frontend is not None and (
        frontend != "nsys"
        or (index + 1 < len(command) and command[index + 1] == "profile")
    ):
        for nested_index in range(index + 1, len(command)):
            if (
                _python_script_index(
                    command[nested_index:], "profile_driver.py"
                )
                is not None
            ):
                return True
    wrappers = {"profiling/nvidia/profile_nvidia.sh", "profiling/amd/profile_kernel.sh"}
    executable = PurePosixPath(command[index]).as_posix()
    if Path(executable).name == "profile_driver.py" or executable in wrappers:
        return True
    operand = _shell_command_operand(command, index)
    return bool(
        operand
        and operand[0] == "script"
        and PurePosixPath(command[operand[1]]).as_posix() in wrappers
    )

def _mentions_evaluator_target(value: str) -> bool:
    """Find target names in shell text without pretending to parse shell grammar."""
    unquoted = value.translate(str.maketrans("", "", "\\'\""))
    return re.search(
        r"(?<![A-Za-z0-9_.-])"
        r"(?:test_kernel\.py|profile_driver\.py|profile_nvidia\.sh|profile_kernel\.sh)"
        r"(?![A-Za-z0-9_.-])",
        unquoted,
    ) is not None

def is_unsafe_target_command(parts: list[str]) -> bool:
    """Reject target-bearing commands outside the supported launcher grammar."""
    command, _ = _parsed_command_parts(parts)
    if is_test_kernel_command(command) or is_profile_command(command):
        return False
    return any(_mentions_evaluator_target(token) for token in command)

def option_value(parts: list[str], name: str, default: Any = None) -> Any:
    """Read a simple ``--flag value``/``--flag=value`` option from command argv."""
    command = command_parts(parts)
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1] if index + 1 < len(command) else default
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default

def requested_operation(requested: str, command: list[str]) -> str:
    if requested != "auto":
        return requested
    if is_test_kernel_command(command):
        return "run"
    if is_profile_command(command):
        return "profile"
    return "dev"
