"""Native CLI fields construct public Workflow DTOs without a legacy argument protocol."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import get_type_hints

from aka.contracts.workflow import CampaignRuntimeSettings, WorkflowSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch


def register_request_options(parser):
    parser.add_argument("--workflow-request", help="serialized public WorkflowLaunch")
    parser.add_argument("--problem", type=Path, help="Task input directory for a new request")
    parser.add_argument("--workspace-base", type=Path, help="base directory for new campaigns")
    parser.add_argument("--workspace-suffix", default=None)
    parser.add_argument("--settings", type=_object, default={}, help="public WorkflowSettings JSON object")
    parser.add_argument("--runtime", type=_object, default={}, help="public CampaignRuntimeSettings JSON object")
    for kind in (WorkflowSettings, CampaignRuntimeSettings):
        hints = get_type_hints(kind)
        for field in fields(kind):
            flag = "--" + field.name.replace("_", "-")
            annotation = hints[field.name]
            options = {"dest": field.name, "default": argparse.SUPPRESS}
            if annotation is bool:
                options["action"] = argparse.BooleanOptionalAction
            elif annotation in (str, int, float):
                options["type"] = annotation
            else:
                options["action"] = "append"
            parser.add_argument(flag, **options)


def _object(value):
    try:
        result = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("requires a JSON object") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("requires a JSON object")
    return result


def request_arguments(options, parser):
    values = vars(options)
    selected = set(values) & {field.name for kind in (WorkflowSettings, CampaignRuntimeSettings)
                             for field in fields(kind)}
    if options.workflow_request is not None:
        if (options.problem is not None or options.workspace_base is not None
                or options.workspace_suffix is not None or options.settings or options.runtime or selected):
            parser.error("a recorded request cannot be combined with new request fields")
        argv = ("--workflow-request", options.workflow_request)
        decode_launch(argv)
        return argv
    if options.problem is None:
        parser.error("provide --workflow-request or --problem")
    groups = []
    for kind, supplied in ((WorkflowSettings, options.settings), (CampaignRuntimeSettings, options.runtime)):
        names = {field.name for field in fields(kind)}
        if set(supplied) - names:
            parser.error("unknown public request fields: " + ", ".join(sorted(set(supplied) - names)))
        explicit = {name: values[name] for name in selected & names}
        if set(explicit) & set(supplied):
            parser.error("request fields supplied twice: " + ", ".join(sorted(set(explicit) & set(supplied))))
        supplied = {**supplied, **explicit}
        for field in fields(kind):
            if field.name.endswith("_reviewers") and field.name in supplied:
                reviewers = supplied[field.name]
                if not isinstance(reviewers, (list, tuple)) or any(not isinstance(name, str) for name in reviewers):
                    parser.error("reviewers require a list of names")
                supplied[field.name] = tuple(reviewers)
        groups.append(supplied)
    settings, runtime = groups
    if not settings.get("platform"):
        parser.error("new requests require --platform or settings.platform")
    provided_settings = tuple(sorted(settings))
    settings.setdefault("framework", "")
    from aka.legacy.workflow.requests import create_launch
    launch = create_launch(options.problem, workspace_base=options.workspace_base or Path.cwd(),
                           workspace_suffix=options.workspace_suffix or "",
                           settings=WorkflowSettings(**settings), runtime=CampaignRuntimeSettings(**runtime),
                           provided_settings=provided_settings,
                           provided_runtime=tuple(sorted(runtime)))
    return encode_launch(launch)
