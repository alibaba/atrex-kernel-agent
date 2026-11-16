"""Execution-owned SSH health and recovery probe entry."""
from .cli_adapter import run_cli


def main(argv=None):
    return run_cli(argv, prepare_command=None, run_command=None, preview_command=None, health_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
