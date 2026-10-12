"""Adapt typed problem facts to the original Optimization Application options."""
from pathlib import Path
from functools import partial

from aka.contracts.problem import LoadedProblem
from aka.contracts.problem_provider import ProblemProvider
from aka.core.effects import call_sync
from aka.legacy.task.problem.composition import compose_problem


def resolve_operator_directory(directory: str, optimization_mode: str = "leaderboard",
                               *, provider: ProblemProvider | None = None, **selection) -> dict:
    loaded = load_operator_problem(directory, optimization_mode, provider=provider, **selection)
    return {
        "name": loaded.name,
        "reference": str(loaded.reference),
        "op_dir": str(loaded.directory),
        "atrex_bench_root": str(loaded.bench_root) if loaded.bench_root else "",
        "agent_problem": str(loaded.public_contract) if loaded.public_contract else "",
        "agent_problem_source": loaded.public_contract_source,
    }


def load_operator_problem(directory: str, optimization_mode: str = "leaderboard", *,
                          provider: ProblemProvider | None = None,
                          snapshot_directory: Path | None = None, **selection) -> LoadedProblem:
    options = {"snapshot_directory": snapshot_directory} if snapshot_directory is not None else {}
    if provider is not None:
        if selection:
            raise ValueError("an injected problem provider cannot also select a plugin")
        loaded = call_sync(partial(provider.load, optimization_mode=optimization_mode, **options), Path(directory))
    else:
        with compose_problem(optimization_mode=optimization_mode, **selection) as composition:
            loaded = call_sync(partial(composition.provider.load, **options), Path(directory))
    if not isinstance(loaded, LoadedProblem):
        raise TypeError("ProblemProvider.load must return LoadedProblem")
    return loaded

