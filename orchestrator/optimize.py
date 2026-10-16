#!/usr/bin/env python3
"""Retained script forwarding to the installed ``aka optimize`` command."""
import sys


def main(argv=None):
    from aka.cli import main as launch
    arguments = sys.argv[1:] if argv is None else argv
    return launch(["optimize", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
