"""Trusted execution assets: never copied into the coding Agent's workspace."""

from pathlib import Path

RUNNERS_ROOT = Path(__file__).resolve().parent / "runners"
PROFILE_DRIVER = RUNNERS_ROOT / "profile_driver.py"


def evaluator_path(workspace: Path) -> Path:
    """Select a fixed evaluator from the immutable workload format."""
    name = (
        "sol_test_kernel.py"
        if (workspace / "workload.jsonl").is_file()
        else "atrex_bench_test_kernel.py"
    )
    return RUNNERS_ROOT / name


def evaluation_inputs(workspace: Path) -> dict[str, Path]:
    return {"test_kernel.py": evaluator_path(workspace)}
