"""Validate explicit command inputs before preparing any backend request."""
from dataclasses import asdict
from pathlib import Path
import stat

from aka.contracts.execution import UnsupportedExecution
from .gateway_request import _parse_env_items


PROFILE_ENVIRONMENT_KEYS = frozenset(("PROFILE_ITERS", "PROFILE_WARMUP",
    "PROFILE_WORKLOAD_IDX", "PROFILE_SHAPE_ID", "PROFILE_DEVICE"))
_CONFIG_KEYS = frozenset(("ATREX_PRIVATE_REFERENCE_DIR", "ATREX_EVAL_SHAPE_BATCH_SIZE"))
_CONFIG_PREFIXES = ("AGATE_", "ATREX_SANDBOX_", "AKA_GATEWAY_", "AKA_EXECUTION_", "AKA_EXPERIMENT_COMMAND_")
_PROTECTED_FILES = frozenset(("kernel.py", "test_kernel.py", "reference.py", "input.py",
    "shapes.json", "metadata.json", "roofline.json", "solution.json", "requirements.txt",
    "agent_problem.json"))


def transport_environment_key(key):
    return key in _CONFIG_KEYS or key.startswith(_CONFIG_PREFIXES)


def legacy_command_environment(environment, env_items=()):
    """Keep legacy host launch configuration outside explicit workload inputs."""
    result = {key: value for key, value in environment.items()
              if transport_environment_key(key) or key in PROFILE_ENVIRONMENT_KEYS}
    result.update(_parse_env_items(env_items))
    return result


def workload_environment(command, env_items=()):
    # CLI arguments are compatibility inputs. The public mapping is authoritative
    # when both channels contain the same explicitly supplied workload variable.
    values = _parse_env_items(env_items)
    if any(transport_environment_key(key) for key in values):
        raise ValueError("transport configuration cannot be supplied as workload environment")
    parsed = _parse_env_items(f"{key}={value}" for key, value in command.environment.items()
        if isinstance(key, str) and isinstance(value, str))
    if len(parsed) != len(command.environment):
        raise TypeError("command environment requires string keys and values")
    values.update({key: value for key, value in parsed.items() if not transport_environment_key(key)})
    return values


def validate_command(command):
    if command.profile_options is not None and command.operation != "profile":
        raise UnsupportedExecution("profile options require a profile operation")
    for name in command.artifacts:
        if not isinstance(name, str):
            raise TypeError("command output paths require strings")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("command output paths must be workspace-relative")
    for item in command.input_files:
        path = Path(item.path)
        if path.as_posix() in _PROTECTED_FILES or path.parts[0] in {"atrex-bench", "profiling"}:
            from aka.contracts.filesystem_access import require_path_access
            original = require_path_access(command.filesystem_access, command.workspace.absolute() / path)
            if (not original.is_file() or original.read_bytes() != item.content
                    or stat.S_IMODE(original.stat().st_mode) != item.mode):
                raise ValueError("explicit command inputs cannot introduce or replace trusted Task files")
    if command.submission is not None and command.input_files:
        raise UnsupportedExecution("prepared typed requests do not support custom input files")
    workload_environment(command)


def apply_profile_options(args, options):
    if options is None:
        return
    for source, target in (("level", "profile_level"), ("profiler", "profiler"),
                           ("kernel_regex", "kernel_regex"), ("top_kernels", "top_kernels")):
        value = getattr(options, source)
        if value is not None:
            setattr(args, target, value)
    if options.counters:
        args.profile_counter = list(options.counters)


def nondefault_profile(options):
    return options is not None and (options.level not in {None, "sol"}
        or options.profiler is not None or bool(options.counters)
        or options.kernel_regex is not None or options.top_kernels is not None)


def validate_prepared_profile(request, options):
    from aka.contracts.execution import ProfileOptions
    captured = ProfileOptions(level=request.get("level"), profiler=request.get("profiler"),
        counters=tuple(request.get("counters", ())), kernel_regex=request.get("kernel_regex"),
        top_kernels=request.get("top_kernels"))
    if captured.level == "deep" and not captured.kernel_regex:
        raise ValueError("deep profile requires a kernel regex")
    if options is None:
        return
    actual = asdict(captured)
    actual["level"] = actual["level"] or "sol"
    for key, value in asdict(options).items():
        if value is not None and value != () and value != actual[key]:
            raise ValueError("profile options differ from the frozen prepared request")
