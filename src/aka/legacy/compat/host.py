"""Convert the retained optimization CLI once before canonical Workflow launch."""
from pathlib import Path
import os
import sys
from aka.legacy.workflow.launch import run_workflow, PROFILES_DIR
from aka.legacy.workflow.process_environment import invocation_environment


def run_application(request, *, profile="application", **options):
    argv = tuple(sys.argv[1:] if request.argv is None else request.argv)
    if argv[:1] != ("--workflow-request",):
        from aka.bootstrap.continuation import read, restore
        from .cli import adapt_argv
        from aka.legacy.workflow.process_environment import requires_default_recovery, pending_recovery_directory
        payload = None if pending_recovery_directory() is not None or requires_default_recovery() else read(os.environ)
        environment = restore(payload).environment if payload is not None else {}
        with invocation_environment(environment):
            argv = adapt_argv(argv)
    return run_workflow(request.repo_root, argv, profile="workflow" if profile == "application" else profile, **options)
