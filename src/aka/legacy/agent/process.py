"""Process call shape and shared gateway identity, without launching a process.

The legacy runner is loaded only when selected. Light supplies its own runner,
so importing common Agent mechanics does not import checkout-only executors.
"""
import os
from pathlib import Path
from typing import Protocol

DEFAULT_PROTECTED_GATEWAY_SCREEN = "atrex-local-gateway"
DEFAULT_PROTECTED_GATEWAY_STATE_NAME = "atrex-local-gateway"


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        cwd: Path,
        timeout: int | None,
        env: dict | None = None,
    ) -> tuple[str, str, int, bool]:
        ...


def protected_gateway_identity(
    environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve shared gateway protection targets without embedding host paths."""
    values = os.environ if environment is None else environment
    screen = values.get(
        "ATREX_PROTECTED_GATEWAY_SCREEN", DEFAULT_PROTECTED_GATEWAY_SCREEN
    )
    state_dir = values.get("ATREX_PROTECTED_GATEWAY_STATE_DIR")
    if not state_dir:
        cache_home = values.get("XDG_CACHE_HOME")
        cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        state_dir = str(cache_root / DEFAULT_PROTECTED_GATEWAY_STATE_NAME)
    return screen, state_dir



def run_bounded(command, cwd, timeout, env=None):
    from orchestrator.agent_runtime.process import run_bounded as legacy_run
    return legacy_run(command, cwd=cwd, timeout=timeout, env=env)
