"""Task selects evaluator content; transports only consume the frozen roles."""
from pathlib import Path
from aka.contracts.content import WorkspaceFile


def validate_resources(evaluation):
    if evaluation is None:
        raise TypeError("Task has no evaluation content")
    if evaluation.adapter in {"legacy-native-v1", "legacy-sol-v1"} and evaluation.harness is None:
        raise ValueError("Task requires an explicit evaluator harness")
    for name, item in (("harness", evaluation.harness), ("worker requirements", evaluation.worker_requirements)):
        if item is None:
            if name == "worker requirements":
                raise ValueError("Task requires an explicit worker requirements role")
            continue
        if name == "harness" and item.path != "test_kernel.py":
            raise ValueError("legacy Task harness must declare test_kernel.py")
        if item.source is not None:
            if not item.source.is_file():
                raise FileNotFoundError(f"missing Task {name}: {item.source}")
            text = item.source.read_text(encoding="utf-8")
        else:
            text = item.text
        if name == "worker requirements":
            # A flat file remains self-contained after snapshot/dispatch. Relative
            # includes and local artifacts cannot be resolved on another worker.
            for line in text.splitlines():
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                if (value.startswith(("-", ".", "/", "file:")) or "\\" in value
                        or (" @ " in value and not value.split(" @ ", 1)[1].startswith("https://"))
                        or ("/" in value and not value.startswith("https://") and " @ https://" not in value)):
                    raise ValueError("Task worker requirements must contain flat package specifications or HTTPS URLs")


def selected_resources(directory, *, sol, native, native_harness=None, worker_requirements=None):
    from aka.legacy.resources.paths import ATREX_BENCH_HARNESS, REPO_ROOT
    harness = (native_harness or ATREX_BENCH_HARNESS) if native else (
        REPO_ROOT / "reference/test_kernel.py" if sol else directory / "test_kernel.py")
    item = WorkspaceFile("test_kernel.py", source=Path(harness)) if (native or sol or harness.is_file()) else None
    requirements = (WorkspaceFile("worker-requirements.txt", source=Path(worker_requirements))
                    if worker_requirements is not None else WorkspaceFile("worker-requirements.txt", text=""))
    from aka.contracts.problem import EvaluationInput
    validate_resources(EvaluationInput("legacy-native-v1" if native else "legacy-sol-v1" if sol else "legacy-kernel-v1",
                                       directory, (), harness=item, worker_requirements=requirements))
    return item, requirements
