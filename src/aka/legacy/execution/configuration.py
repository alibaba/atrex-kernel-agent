"""Queue budget shared by the sandbox transport and its supervising process."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

DEFAULT_QUEUE_WAIT_GRACE = 14_400


def distributed_evaluation_world_size(metadata: object) -> int:
    """GPU count shared by Agate submission and production source-policy checks."""
    if not isinstance(metadata, dict):
        return 1
    contract = metadata.get("benchmark_contract")
    # Legacy metadata may describe multi-GPU hardware without opting into a
    # distributed evaluator. Preserve its single-GPU submission and source gate.
    if not isinstance(contract, dict) or "distributed_evaluation" not in contract:
        return 1
    evaluation = contract["distributed_evaluation"]
    prefix = "benchmark_contract.distributed_evaluation"
    if not isinstance(evaluation, dict):
        raise ValueError(f"{prefix} must be an object")
    hardware = metadata.get("hardware", {})
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object")
    if "required_gpus" in hardware:
        required = hardware["required_gpus"]
        if type(required) is not int or required < 1:
            raise ValueError("hardware.required_gpus must be a positive integer")
    if evaluation.get("launcher") != "torchrun":
        raise ValueError(f"{prefix}.launcher must be 'torchrun'")
    if evaluation.get("backend") != "nccl":
        raise ValueError(f"{prefix}.backend must be 'nccl'")
    count = evaluation.get("world_size")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{prefix}.world_size must be an integer")
    if not 2 <= count <= 64:
        raise ValueError(f"{prefix}.world_size must be between 2 and 64")
    if "required_gpus" in hardware and hardware["required_gpus"] != count:
        raise ValueError("hardware.required_gpus differs from distributed_evaluation.world_size")
    return count


def evaluator_num_gpus(reference_dir: Path) -> int:
    """Read evaluator-owned metadata, never the candidate's dependency manifest."""
    path = reference_dir / "metadata.json"
    if not path.is_file():
        return 1
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON in metadata.json") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object")
    return distributed_evaluation_world_size(metadata)


def queue_wait_grace(environment: Mapping[str, str]) -> int:
    name = "ATREX_SANDBOX_QUEUE_WAIT_GRACE"
    try:
        value = int(environment.get(name, str(DEFAULT_QUEUE_WAIT_GRACE)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def legacy_settings(environment=None):
    """Read connection and supervision settings inside Execution."""
    import os
    from types import SimpleNamespace
    e = os.environ if environment is None else environment
    raw_gpu = e.get("ATREX_SANDBOX_SSH_GPU", "")
    values = dict(profile=e.get("ATREX_SANDBOX_PROFILE", ""), url=e.get("ATREX_SANDBOX_URL", ""),
                  ssh=e.get("ATREX_SANDBOX_SSH", ""), ssh_init=e.get("ATREX_SANDBOX_SSH_INIT", ""),
                  ssh_gpu=int(raw_gpu) if raw_gpu else None,
                  health_command=e.get("ATREX_SANDBOX_HEALTH_COMMAND", ""))
    if sum(bool(values[k]) for k in ("profile", "url", "ssh")) > 1:
        raise ValueError("execution endpoints are mutually exclusive")
    return SimpleNamespace(**values)


def configure_legacy_arguments(values):
    """Translate retained CLI knobs once, before selection is frozen."""
    import os
    mapping = {"sandbox_profile": "ATREX_SANDBOX_PROFILE", "sandbox_url": "ATREX_SANDBOX_URL",
               "sandbox_ssh": "ATREX_SANDBOX_SSH", "sandbox_ssh_init": "ATREX_SANDBOX_SSH_INIT",
               "sandbox_ssh_gpu": "ATREX_SANDBOX_SSH_GPU", "sandbox_health_command": "ATREX_SANDBOX_HEALTH_COMMAND",
               "environment_poll_interval": "ATREX_ENVIRONMENT_POLL_INTERVAL"}
    for key, env in mapping.items():
        if key in values:
            value = values.pop(key)
            if value is not None:
                os.environ[env] = str(value)
    if "sandbox_ssh_runtime_bind" in values:
        os.environ["ATREX_SANDBOX_SSH_RUNTIME_BINDS"] = json.dumps(values.pop("sandbox_ssh_runtime_bind") or [])


def selected_ssh():
    """Expose the execution mode needed by immutable policy and recovery coordination."""
    from .selection import selected_execution
    return selected_execution().get("profile", "legacy") == "legacy" and bool(legacy_settings().ssh)


def prepare_runtime(launch, problem):
    import os
    from .recovery import configure_recovery, raise_if_environment_blocked
    from .health import runtime_health_command
    from .sandbox import check_ssh_environment, detect_arch
    from .selection import selected_execution
    from aka.contracts.workflow_launch import encode_launch
    from types import SimpleNamespace
    settings = legacy_settings() if selected_execution().get("profile", "legacy") == "legacy" else SimpleNamespace(
        profile="", url="", ssh="", ssh_init="", health_command="")
    hardware = launch.request.runtime.hardware or launch.request.settings.platform
    if selected_ssh():
        # Retired checkout-script recovery commands cannot identify an installed run.
        from .recovery import environment_state_file
        state = environment_state_file()
        if state is not None and state.with_name("restart.json").is_file():
            record = json.loads(state.with_name("restart.json").read_text())
            command = record.get("command", [])
            if len(command) < 6 or command[1:6] != ["-m", "aka", "run", "--resume", "--"]:
                raise RuntimeError("retired script recovery command; use the installed profile continuation")
        raw_argv = launch.restart_arguments or encode_launch(launch)
        configure_recovery(workspace_base=launch.workspace_base,
            raw_argv=raw_argv, optimize_script=None,
            sandbox_hardware=hardware, ssh_target=settings.ssh, ssh_init=settings.ssh_init,
            ssh_runtime_binds=json.loads(os.environ.get("ATREX_SANDBOX_SSH_RUNTIME_BINDS", "[]")),
            ssh_gpu=settings.ssh_gpu, health_command=settings.health_command,
            poll_interval=int(os.environ.get("ATREX_ENVIRONMENT_POLL_INTERVAL", "60")),
            runtime_health_command=runtime_health_command(sol=problem.initial_source.kind == "sol",
                framework=launch.request.settings.framework))
        raise_if_environment_blocked()
        check_ssh_environment(launch.workspace_base, hardware, settings.ssh, settings.ssh_init, settings.health_command)
    arch = launch.request.settings.arch or detect_arch(hardware, settings.profile, settings.url,
        settings.ssh, settings.ssh_init, settings.health_command)
    return arch, settings.ssh or settings.url or settings.profile or "selected execution"
