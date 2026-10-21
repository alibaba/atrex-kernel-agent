"""Execute the existing Wiki query CLI against the explicitly bound store."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROFILE_ENV = "ATREX_WIKI_PROFILE_ROOT"
TASK_ENV = "ATREX_WIKI_TASK_ID"


def invoke(request, context):
    if not isinstance(request, dict) or not isinstance(context, dict):
        raise TypeError("request and tool context must be objects")
    resources = context.get("resource_paths")
    if not isinstance(resources, dict) or 'gpu-wiki-public-store' not in resources:
        raise ValueError("the frozen plan does not contain the selected Wiki store")
    root = Path(resources['gpu-wiki-public-store']).resolve()
    script = root / "tools" / "query_nl.py"
    if not root.is_dir() or not script.is_file():
        raise ValueError(f"selected Wiki store is unavailable: {root}")
    workspace = Path(context["workspace"]).resolve()
    campaign_id = context["campaign_id"]
    profile_root = workspace / PROFILE_DIR
    ambient = dict(os.environ)
    selected = context.get("environment", {})
    if not isinstance(selected, dict):
        selected = {}
    env = {**ambient, **{str(key): str(value) for key, value in selected.items() if value is not None}}
    for key, value in ((PROFILE_ENV, str(profile_root)), (TASK_ENV, str(campaign_id))):
        existing = ambient.get(key)
        selected_value = selected.get(key)
        if ((existing and existing != value) or (selected_value and selected_value != value)):
            raise ValueError(f"{key} conflicts with the bound Wiki profile")
        env[key] = value
    with tempfile.TemporaryDirectory(prefix="aka-wiki-query-") as temporary:
        request_file = Path(temporary) / "request.txt"
        request_file.write_text(request["request"], encoding="utf-8")
        command = [sys.executable, str(script), "--store-root", str(root), "--file", str(request_file)]
        for field in ("max_records", "max_bytes", "exclude"):
            if field in request:
                command.extend(["--" + field.replace("_", "-"), str(request[field])])
        process = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True,
                                 timeout=360, check=False)
    if process.returncode:
        diagnostic = (process.stderr.strip().splitlines() or ["query failed"])[0][:500]
        raise RuntimeError(f"Wiki query failed with exit {process.returncode}: {diagnostic}")
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Wiki query returned invalid JSON") from exc
    if (not isinstance(result, dict) or not isinstance(result.get("records"), dict)
            or not isinstance(result.get("query_id"), str)
            or not re.fullmatch(r"wiki-query-[0-9a-f]{32}", result["query_id"])):
        raise ValueError("Wiki query returned an invalid result object or query ID")
    for record in result["records"].values():
        if not isinstance(record, dict) or not isinstance(record.get("wiki_id"), str) or not record["wiki_id"]:
            raise ValueError("Wiki query record is missing its canonical wiki_id")
        if not record["wiki_id"].startswith(CANONICAL_PREFIX):
            raise ValueError("Wiki query record has the wrong canonical namespace")
        if record.get("store") != STORE_ID:
            raise ValueError("Wiki query record has the wrong store identity")
    return result

STORE_RESOURCE = 'gpu-wiki-public-store'
STORE_ID = 'gpu_wiki'
CANONICAL_PREFIX = 'gpu_wiki::'
PROFILE_DIR = '.gpu_wiki_profile'
