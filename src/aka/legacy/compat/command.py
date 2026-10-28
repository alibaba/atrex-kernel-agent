"""Forward the historical top-level optimize command to its Workflow owner."""
from aka.legacy.workflow.optimize_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
