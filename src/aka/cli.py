"""Commands for submitting tasks and running task-bound experiments."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: aka {task,execution} [ARGS ...]")
        return 0
    if args[0] == "task":
        from .light.tasks.cli import main as command
    elif args[0] == "execution":
        from .light.execution.command_cli import main as command
    else:
        print(f"Unknown AKA command: {args[0]}", file=sys.stderr)
        return 2
    return command(args[1:])
