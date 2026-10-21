"""Canonical Wiki source attribution and profile artifact projection."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path

from aka.contracts.attribution import (
    ArtifactDescriptor,
    AttributionValidation,
    SourceReference,
)

PROVIDER = 'gpu-wiki'
TOOL_NAME = 'gpu-wiki.query'
STORE_ID = 'gpu_wiki'
PREFIX = 'gpu_wiki::'
PROFILE_DIR = '.gpu_wiki_profile'


def extract(payload, context):
    data = payload.get("result") if isinstance(payload, dict) else None
    call_id = payload.get("call_id") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("records"), dict) or not isinstance(call_id, str):
        raise TypeError("Wiki result and call ID are required for source extraction")
    refs = []
    for record in data["records"].values():
        if not isinstance(record, dict):
            raise TypeError("Wiki result contains a non-object record")
        # Canonical identity is read only from the record itself; the result-map key is never used.
        wiki_id = record.get("wiki_id")
        store = record.get("store")
        if (not isinstance(wiki_id, str)
                or not re.fullmatch(r"(?:gpu_wiki|internal_gpu_wiki)::[A-Za-z0-9][A-Za-z0-9._:-]*", wiki_id)
                or not wiki_id.startswith(PREFIX) or store != STORE_ID):
            raise ValueError("Wiki record lacks a valid canonical ID and store")
        query_id = data.get("query_id")
        if not isinstance(query_id, str) or not re.fullmatch(r"wiki-query-[0-9a-f]{32}", query_id):
            raise ValueError("Wiki result has no emitted canonical query ID")
        refs.append(SourceReference(PROVIDER, call_id, wiki_id, {
            "query_id": query_id, **{key: record[key] for key in ("store", "source", "type") if isinstance(record.get(key), str)}
        }))
    return [asdict(reference) for reference in refs]


def validate(payload, context):
    declaration = payload.get("declaration", {}) if isinstance(payload, dict) else {}
    call = payload.get("call", {}) if isinstance(payload, dict) else {}
    refs = payload.get("refs", []) if isinstance(payload, dict) else []
    if declaration.get("provider") != PROVIDER or call.get("tool") != TOOL_NAME:
        return AttributionValidation(False, "invalid", ({"code": "provider_mismatch", "message": "declaration or call belongs to another Wiki"},))
    expected_call = call.get("call_id")
    for reference in refs:
        value = reference if isinstance(reference, dict) else vars(reference)
        if value.get("provider") != PROVIDER or value.get("call_id") != expected_call:
            return AttributionValidation(False, "invalid", ({"code": "call_mismatch", "message": "source reference is not from this call"},))
        query_id = value.get("metadata", {}).get("query_id") if isinstance(value.get("metadata"), dict) else None
        if not isinstance(query_id, str) or not re.fullmatch(r"wiki-query-[0-9a-f]{32}", query_id):
            return AttributionValidation(False, "invalid", ({"code": "query_id_invalid", "message": "source reference has no emitted query ID"},))
        if (not isinstance(value.get("reference_id"), str)
                or not re.fullmatch(r"(?:gpu_wiki|internal_gpu_wiki)::[A-Za-z0-9][A-Za-z0-9._:-]*", value["reference_id"])
                or not value["reference_id"].startswith(PREFIX)):
            return AttributionValidation(False, "invalid", ({"code": "namespace_mismatch", "message": "source reference has the wrong canonical namespace"},))
    return AttributionValidation(True, "verified")


def artifacts(payload, context):
    workspace = Path(payload["workspace"]).resolve()
    root = workspace / PROFILE_DIR
    if root.is_symlink() or not root.resolve().is_relative_to(workspace):
        raise ValueError("Wiki profile root must be a workspace-owned directory")
    if not root.is_dir():
        return []
    rows = []
    run = root / "run.json"
    if run.is_file() and not run.is_symlink():
        rows.append(_descriptor(run, workspace, "wiki-run-identity", "atrex-wiki-profile-run-v1", PROVIDER))
    events = root / "raw" / "query_events"
    for path in sorted(events.glob("*/*.json")) if events.is_dir() else ():
        if path.is_file() and not path.is_symlink():
            rows.append(_descriptor(path, workspace, "wiki-query-event", "atrex-wiki-query-event-v2", PROVIDER))
    return rows


def _descriptor(path, workspace, role, schema, provider):
    if path.is_symlink() or not path.resolve().is_relative_to(workspace):
        raise ValueError("Wiki profile artifacts must be regular files inside the workspace")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactDescriptor(path.relative_to(workspace).as_posix(), role, schema, False, "complete", "sha256:" + digest, provider)
