"""Dispatch explicitly named commands contributed by installed packages."""
from __future__ import annotations

import sys
from importlib.metadata import entry_points, version
from typing import Sequence

from .core.effects import call_sync


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--version"]:
        print("python -m aka " + version("atrex-aka-framework"))
        return 0
    available = entry_points()
    entries = (available.select(group="aka.commands") if hasattr(available, "select")
               else available.get("aka.commands", ()))
    commands = {}
    for entry in entries:
        if entry.name in commands:
            print(f'Duplicate AKA command contribution: "{entry.name}".', file=sys.stderr)
            return 2
        commands[entry.name] = entry
    if not args or args == ["--help"] or args == ["-h"]:
        print("usage: python -m aka COMMAND [ARGS ...]")
        print("Commands: " + (", ".join(sorted(commands)) or "none installed"))
        return 0
    command = commands.get(args[0])
    if command is None:
        print(f'Unknown AKA command: "{args[0]}". Run python -m aka --help.', file=sys.stderr)
        return 2
    result = call_sync(command.load(), args[1:])
    if type(result) is not int:
        raise TypeError("AKA command must return an integer exit code")
    return result
