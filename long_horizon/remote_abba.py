"""Historical entry for the Evaluation-owned standalone driver."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator._bootstrap import task_modules

task_modules()
import importlib

_module = importlib.import_module("aka.legacy.evaluation.drivers.remote_abba")
if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
