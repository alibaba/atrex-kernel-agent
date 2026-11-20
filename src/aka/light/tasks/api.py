"""Short parent API over the same task service used by the CLI and worker."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import uuid
from importlib.metadata import entry_points

from .content import atomic_json, materialize, tree
from .service import TaskService


class Job:
    def __init__(self, task_id, root=None):
        self.id = task_id
        self.service = TaskService(root or Path.home() / ".atrex/aka-tasks")

    def status(self): return self.service.status(self.id)
    def events(self, after=0): return self.service.events(self.id, after=after)
    def results(self): return self.service.results(self.id)
    def export(self, candidate_id, destination): return self.service.export(self.id, candidate_id, destination)
    def feedback(self, candidate_id, value): return self.service.feedback(self.id, candidate_id, value)
    def pause(self): return self.service.control(self.id, "pause")
    def resume(self, *, budget=None): return self.service.control(self.id, "resume", budget=budget)
    def cancel(self): return self.service.control(self.id, "cancel")


def submit(*, contract=None, adapter=None, source=None, baseline, seed=None, workload=None, validation=None,
           execution, objective, budget=None, agent=None, root=None, key=None,
           parent_task=None, notes=None, start=True):
    """Bind caller inputs once; the resulting frozen task uses the common workflow.

Installed bindings contribute an `aka.task_bindings` entry point. Its callable
receives source/baseline/seed/workload/validation and a staging directory, and
returns adapter/contract/config/seed/baseline manifest fields. It may copy its
versioned verifier and evaluation adapter into the contract. Nothing from the
binding is required in the caller's editable source tree.
"""
    service = TaskService(root or Path.home() / ".atrex/aka-tasks")
    with tempfile.TemporaryDirectory(prefix="aka-bind-") as temporary:
        staging = Path(temporary)
        inputs = dict(source=source, baseline=baseline, seed=baseline if seed is None else seed,
                      workload=workload, validation=validation)
        if contract is not None:
            if adapter is not None or source is not None:
                raise ValueError("Use contract, or the compatibility adapter/source pair")
            bound = bind_contract(staging, contract=contract, baseline=baseline,
                seed=baseline if seed is None else seed, workload=workload, validation=validation)
        elif adapter in {"module.v1", "atrex-bench.v1"}:
            bound = bind_package(staging, adapter=adapter, **inputs)
        else:
            matches = list(entry_points(group="aka.task_bindings", name=adapter))
            if len(matches) != 1:
                raise ValueError(f"Expected one installed task binding for {adapter!r}")
            bound = matches[0].load()(staging=staging, **inputs)
        spec = {**bound, "schema_version": 1, "target": execution, "objective": objective,
                "budget": budget or {}, "agent": {"runtime": "claude"} if agent is None else agent}
        if notes is not None:
            spec["notes"] = notes
        if parent_task:
            spec["parent_task"] = parent_task.id if isinstance(parent_task, Job) else parent_task
        atomic_json(staging / "task.json", spec)
        result = service.submit(staging / "task.json", idempotency_key=key or str(uuid.uuid4()), start=start)
    return Job(result["id"], service.root)


def bind_contract(staging, *, contract, baseline, seed=None, workload=None, validation=None):
    """Prepare a caller-provided contract package without installing a plugin.

contract.json selects an adapter and its config. All other package files, including
custom Python adapters, are frozen unchanged. A named binding remains optional.
"""
    if isinstance(contract, dict):
        binding = contract["binding"]
        inputs = dict(source=contract["source"], baseline=baseline, seed=seed or baseline,
                      workload=workload, validation=validation)
        if binding in {"module.v1", "atrex-bench.v1"}:
            return bind_package(staging, adapter=binding, **inputs)
        matches = list(entry_points(group="aka.task_bindings", name=binding))
        if len(matches) != 1:
            raise ValueError(f"Expected one installed task binding for {binding!r}")
        return matches[0].load()(staging=staging, **inputs)
    path = Path(contract).resolve()
    if path.is_file():
        path = path.parent
    if not (path / "contract.json").is_file() and (path / "module.json").is_file():
        return bind_package(staging, adapter="module.v1", source=path, baseline=baseline,
                            seed=seed or baseline, workload=workload, validation=validation)
    definition = json.loads((path / "contract.json").read_text())
    materialize(staging / "contract", tree(path))
    for name, value in (("workload", workload), ("validation", validation)):
        if value is not None:
            if isinstance(value, (str, Path)):
                value = json.loads(Path(value).read_text())
            atomic_json(staging / "contract" / (name + ".json"), value)
    for name, value in (("baseline", baseline), ("seed", seed or baseline)):
        materialize(staging / name, tree(Path(value).resolve()))
    return {"adapter": definition.get("adapter", "command"), "contract": "contract",
            "config": definition.get("config", {}), "baseline": "baseline", "seed": "seed"}


def bind_package(staging, *, adapter, source, baseline, seed, workload, validation):
    """Reusable package binding; callers supply values, not a new harness per job.

module.v1 source is a reusable module package with contract/ and module.json
(command, instructions, delivery). Workload/validation become frozen JSON files
for its existing harness. A domain binding may obtain these from a package or
caller object and remove even this generic package convention from the parent.
"""
    source = Path(source).resolve()
    config = json.loads((source / "module.json").read_text()) if adapter == "module.v1" else {}
    contract = source / "contract" if adapter == "module.v1" else source
    materialize(staging / "contract", tree(contract))
    for name, value in (("workload", workload), ("validation", validation)):
        if value is not None:
            if isinstance(value, (str, Path)):
                value = json.loads(Path(value).read_text())
            if adapter == "atrex-bench.v1":
                if name == "workload":
                    atomic_json(staging / "contract/shapes.json", value)
                else:
                    config["runner_overrides"] = value
            else:
                atomic_json(staging / "contract" / (name + ".json"), value)
    for name, value in (("baseline", baseline), ("seed", seed)):
        materialize(staging / name, tree(Path(value).resolve()))
    return {"adapter": "command" if adapter == "module.v1" else "atrex-bench",
            "contract": "contract", "config": config, "baseline": "baseline", "seed": "seed"}
