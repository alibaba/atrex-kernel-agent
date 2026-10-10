"""Source-checkout launcher using the same Bootstrap profile as aka run."""
import sys
from pathlib import Path

from aka.bootstrap.host import run_profile
from aka.contracts.startup import Invocation

if __name__ == "__main__":
    raise SystemExit(run_profile(Path(__file__).parent / "profiles/dashboard.json",
                                Invocation(tuple(sys.argv[1:]))))
