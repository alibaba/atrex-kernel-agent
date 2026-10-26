"""Compatibility access to the shared Agent process projection."""
import sys
from aka.agent.execution import proc_view

if __name__ == "__main__":
    raise SystemExit(proc_view.main())

sys.modules[__name__] = proc_view
