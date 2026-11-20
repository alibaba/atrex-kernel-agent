"""JSON command facade for detached task-contract workflows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .content import document
from .service import TaskService


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka task")
    parser.add_argument("--root", default=str(Path.home() / ".atrex" / "aka-tasks"))
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("spec")
    submit.add_argument("--key", required=True)
    submit.add_argument("--no-start", action="store_true")
    for name in ("status", "start", "work", "results"):
        commands.add_parser(name).add_argument("task_id")
    events = commands.add_parser("events")
    events.add_argument("task_id")
    events.add_argument("--after", type=int, default=0)
    control = commands.add_parser("control")
    control.add_argument("task_id")
    control.add_argument("action", choices=("pause", "resume", "cancel"))
    control.add_argument("--budget", metavar="JSON_FILE")
    measure = commands.add_parser("measure")
    measure.add_argument("task_id")
    measure.add_argument("--role", choices=("candidate", "baseline"), default="candidate")
    measure.add_argument("--scope", default="acceptance")
    measure.add_argument("--evaluation-id")
    candidate = commands.add_parser("submit-candidate")
    candidate.add_argument("task_id")
    candidate.add_argument("evaluation_id")
    export = commands.add_parser("export")
    export.add_argument("task_id")
    export.add_argument("candidate_id")
    export.add_argument("destination")
    feedback = commands.add_parser("feedback")
    feedback.add_argument("task_id")
    feedback.add_argument("candidate_id")
    feedback.add_argument("json_file")
    invoke = commands.add_parser("invoke")
    invoke.add_argument("operation", choices=("measure", "submit-candidate", "results", "events"))
    invoke.add_argument("--evaluation-id")
    invoke.add_argument("--scope")
    invoke.add_argument("--after", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "invoke":
            from .runtime import invoke
            result = invoke(args.operation, {key: getattr(args, key) for key in ("evaluation_id", "scope", "after")
                                             if getattr(args, key) is not None})
            print(document(result))
            return 0
        service = TaskService(args.root)
        if args.command == "submit":
            result = service.submit(args.spec, idempotency_key=args.key, start=not args.no_start)
        elif args.command == "work":
            from .worker import run
            run(args.root, args.task_id)
            result = service.status(args.task_id)
        elif args.command == "events":
            result = service.events(args.task_id, after=args.after)
        elif args.command == "control":
            result = service.control(args.task_id, args.action,
                                     budget=json.loads(Path(args.budget).read_text()) if args.budget else None)
        elif args.command == "measure":
            record = service.measure(args.task_id, role=args.role, scope=args.scope, evaluation_id=args.evaluation_id)
            result = {key: record[key] for key in ("id", "candidate_id", "revision", "scope", "state", "job_id", "assessment")}
        elif args.command == "submit-candidate":
            result = service.submit_candidate(args.task_id, args.evaluation_id)
        elif args.command == "export":
            result = service.export(args.task_id, args.candidate_id, args.destination)
        elif args.command == "feedback":
            service.feedback(args.task_id, args.candidate_id, json.loads(Path(args.json_file).read_text()))
            result = {"recorded": True}
        else:
            result = getattr(service, args.command)(args.task_id)
        print(document(result))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(document({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
