"""Adapt typed problem facts to the original Optimization Application options."""
from pathlib import Path
from functools import partial

from aka.contracts.problem import LoadedProblem
from aka.contracts.problem_provider import ProblemProvider
from aka.core.effects import call_sync
from aka.legacy.task.problem.composition import compose_problem


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

