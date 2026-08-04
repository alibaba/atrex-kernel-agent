if __package__:
    from .cli import main
else:  # main's framework dispatcher executes this file by absolute path.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from long_horizon.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
