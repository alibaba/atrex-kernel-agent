"""Select and validate trusted adapters without inspecting benchmark filenames."""
from __future__ import annotations
import importlib
import importlib.util
import sys
from .content import digest, relative
from .objective import MetricObjective

BUILTINS = {"command": "aka.light.tasks.adapters.command", "atrex-bench": "aka.light.tasks.adapters.atrex"}


def load_adapter(name, contract):
    if name in BUILTINS:
        module = importlib.import_module(BUILTINS[name])
        cls = module.Adapter
    else:
        if not name.startswith("python:"):
            raise ValueError("Adapter must be built-in or python:relative_file.py:Class")
        filename, clsname = name[7:].rsplit(":", 1)
        path = contract / relative(filename)
        key = "aka_task_adapter_" + digest({"path": str(path), "source": path.read_text()})
        # A contract-local adapter may use relative helpers without installing
        # a host plugin or sharing module names with another contract package.
        spec = importlib.util.spec_from_file_location(key, path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        cls = getattr(module, clsname)
    adapter = cls()
    if type(adapter.protocol_version) is not int or adapter.protocol_version != 1:
        raise ValueError("Unsupported task adapter protocol version")
    if "acceptance" not in adapter.capabilities:
        raise ValueError("Task adapter must support acceptance evaluation")
    for method in ("validate", "instructions", "prepare", "interpret", "delivery"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"Task adapter is missing {method}")
    return adapter


def objective(adapter, config):
    if hasattr(adapter, "objective"):
        result = adapter.objective(config)
        for method in ("describe", "eligible", "compare"):
            if not callable(getattr(result, method, None)):
                raise ValueError(f"Objective is missing {method}")
        return result
    return MetricObjective(config)


def validate_budget(budget):
    if not isinstance(budget, dict):
        raise ValueError("Budget must be an object")
    for key, value in budget.items():
        if key not in {"evaluations", "executions", "episodes", "agent_seconds", "tokens"} or type(value) is not int or value <= 0:
            raise ValueError("Budgets require positive integer evaluations/executions/episodes/agent_seconds/tokens")


def validate_spec(spec):
    required = {"schema_version", "adapter", "contract", "seed", "target", "objective"}
    allowed = required | {"baseline", "config", "budget", "agent", "parent_task", "name", "notes"}
    if (not isinstance(spec, dict) or required - set(spec) or set(spec) - allowed
            or type(spec["schema_version"]) is not int or spec["schema_version"] != 1):
        raise ValueError("Invalid task specification fields/version")
    if "notes" in spec and not isinstance(spec["notes"], str):
        raise ValueError("Task notes must be text")
    target = spec["target"]
    for key in ("target", "objective", "config", "budget", "agent"):
        if key in spec and not isinstance(spec[key], dict):
            raise ValueError(f"Task {key} must be an object")
    if not isinstance(target.get("backend"), str) or not target["backend"] or set(target) - {"backend", "gpu", "profile", "url", "num_gpus", "timeout_s", "config"}:
        raise ValueError("Invalid task execution target")
    if target["backend"] == "agate" and not target.get("gpu"):
        raise ValueError("Agate target requires an explicit GPU")
    if target["backend"] == "remote":
        from aka.light.execution.remote import validate_config
        validate_config(target.get("config", {}))
    if target.get("profile", "prod") not in {"prod", "pre"}:
        raise ValueError("Unsupported Agate environment profile")
    for key in ("timeout_s", "num_gpus"):
        if key in target and (type(target[key]) is not int or target[key] <= 0):
            raise ValueError(f"Execution {key} must be a positive integer")
    for key in ("contract", "seed", "adapter"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"Task {key} must be nonempty text")
    if "baseline" in spec and (not isinstance(spec["baseline"], str) or not spec["baseline"]):
        raise ValueError("Task baseline must be nonempty text")
    validate_budget(spec.get("budget", {}))
    from .implementation import validate
    validate(spec.get("config", {}).get("implementation", {}))
    if set(spec.get("agent", {})) - {"runtime", "reasoning_effort", "sandbox", "profile", "model", "provider"}:
        raise ValueError("Unknown agent configuration")
    return spec
