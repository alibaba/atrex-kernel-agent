"""Use public resource evidence without coupling Workflow to a Wiki store."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

DISPOSITIONS = {"applied", "partially_applied", "reference_only", "rejected"}


def recorded_resources(workspace: Path):
    """Reconstruct the recorded resource owner, never discover new plugins."""
    workspace = Path(workspace).resolve()
    manifest = workspace / ".atrex_plugins/agent-resources.json"
    if not manifest.is_file():
        return None
    if manifest.is_symlink():
        raise ValueError("resource evidence manifest must be an ordinary file")
    value = json.loads(manifest.read_text())
    if value.get("workspace") != str(workspace):
        raise ValueError("resource evidence belongs to another workspace")
    from aka.resources import restore_bound

    binding = {
        "schema": "aka.bound-agent-resources.v1",
        "workspace": value["workspace"],
        "campaign_id": value["campaign_id"],
        "plan": value["plan"],
        "digest": value["plan_digest"],
    }
    from aka.bootstrap.continuation import base_payload

    launch = Path(value["plan"]).with_name("launch-selection.json")
    base = base_payload(launch.read_text()) if launch.exists() else None
    return restore_bound(binding, base_selection=base)


def _selected_resources(workspace=None):
    from aka.legacy.resources.core_adapter import selected_core_resources

    selected = selected_core_resources()
    return (
        selected
        if selected is not None
        else (recorded_resources(workspace) if workspace is not None else None)
    )


def collect_artifacts(workspace: Path):
    """Collect declared artifacts with safe paths; auxiliary errors fail open."""
    workspace = Path(workspace).resolve()
    try:
        selected = recorded_resources(workspace)
        if selected is None:
            return None  # Caller retains its explicit legacy collector.
        result = []
        occupied = set()
        for row in selected.artifacts():
            if "path" not in row:
                print(
                    f"[workflow] resource artifact evidence unavailable: {row.get('diagnostics', row)}",
                    file=sys.stderr,
                )
                continue
            name = row["path"]
            relative = Path(name)
            source = workspace / relative
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != name
                or source.is_symlink()
                or any(
                    p.is_symlink()
                    for p in source.parents
                    if p != workspace and workspace in p.parents
                )
                or not source.is_file()
                or not source.resolve().is_relative_to(workspace)
            ):
                raise ValueError("provider artifact is outside its original workspace")
            key = (row["provider"], name)
            if key in occupied:
                continue
            expected = row.get("sha256", "").removeprefix("sha256:")
            if expected and hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "provider artifact content differs from its descriptor"
                )
            occupied.add(key)
            result.append(row)
        return result
    except Exception as error:  # noqa: BLE001 - auxiliary evidence must not change experiment verdicts
        print(
            f"[workflow] resource artifact evidence unavailable: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return []


def historical_attribution(entry):
    """Read legacy claims without rewriting them or manufacturing call IDs."""
    if not any(
        key in entry for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")
    ):
        return {
            "schema": "aka.attribution.v1",
            "declarations": [],
            "verification": "unknown",
        }
    status = entry.get("wiki_usage_status")
    if status is None and entry.get("wiki_usage"):
        status = "declared"
    return {
        "schema": "aka.attribution.v1",
        "declarations": [],
        "verification": "unverified",
        "legacy": {
            key: entry[key]
            for key in (
                "wiki_usage",
                "wiki_query_ids",
                "wiki_usage_status",
                "wiki_usage_errors",
            )
            if key in entry
        },
        "legacy_status": "not_invoked"
        if status == "not_queried"
        else status or "unknown",
    }


def artifact_export_name(row):
    """Keep the historical public Wiki transport name in this legacy adapter."""
    source = Path(row["path"])
    if row["provider"] == "gpu-wiki" and row.get("role") == "wiki-query-event":
        return f"wiki/{source.parent.name}/{source.name}"
    return f"resources/{row['provider']}/{source.as_posix()}"


def normalize_attribution(value, resources, *, experiment_id):
    """Validate generic claims against persisted call/member evidence."""
    rows = value.get("declarations") if isinstance(value, dict) else value
    diagnostics = []
    normalized = []
    declared_sources = {}
    if not isinstance(rows, list) or len(rows) > 1000:
        return {
            "schema": "aka.attribution.v1",
            "experiment_id": experiment_id,
            "declarations": [],
            "diagnostics": [
                {
                    "code": "invalid_attribution",
                    "message": "declarations must be a bounded list",
                }
            ],
        }
    for index, declaration in enumerate(rows):
        try:
            if not isinstance(declaration, dict):
                raise TypeError("declaration must be an object")
            provider = declaration.get("provider")
            status = declaration.get("status")
            calls = declaration.get("call_ids", [])
            uses = declaration.get("uses", [])
            if (
                not isinstance(provider, str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]*", provider)
                or status not in {"declared", "no_material_use", "not_invoked"}
                or not isinstance(calls, list)
                or len(calls) > 1000
                or any(
                    not isinstance(c, str)
                    or not re.fullmatch(r"tool-call-[0-9a-f]{32}", c)
                    for c in calls
                )
                or len(set(calls)) != len(calls)
                or not isinstance(uses, list)
                or len(uses) > 1000
            ):
                raise ValueError("invalid provider, status, call IDs or uses")
            if declaration.get("experiment_id", experiment_id) != experiment_id:
                raise ValueError("declaration belongs to another experiment")
            if (
                status == "declared"
                and not uses
                or status == "no_material_use"
                and (not calls or uses)
                or status == "not_invoked"
                and (calls or uses)
            ):
                raise ValueError("attribution status contradicts its call/use members")
            accepted_uses = []
            seen = {}
            verification = (
                "unverified"
                if resources is None or status == "not_invoked"
                else "verified"
            )
            call_facts = {}
            for call in calls:
                if resources is None:
                    verification = "unverified"
                    continue
                try:
                    facts = resources.call_evidence(call)
                except (FileNotFoundError, OSError):
                    verification = "unverified"
                    continue
                if (
                    facts["receipt"].get("status") != "ok"
                    or facts["receipt"].get("call_id") != call
                ):
                    raise ValueError(
                        "claim refers to an unsuccessful or different call"
                    )
                if (
                    facts.get("coverage", {}).get("receipt") != "persisted"
                    or facts.get("coverage", {}).get("references") != "complete"
                ):
                    verification = "unverified"
                call_facts[call] = facts
                if facts.get("coverage", {}).get("references") == "complete":
                    result = resources.validate_declaration(
                        provider,
                        {
                            "declaration": declaration,
                            "call": facts["receipt"],
                            "refs": facts["references"],
                        },
                    )
                    if not result.get("accepted"):
                        raise ValueError(
                            "attribution provider rejected this call or source"
                        )
            for use in uses:
                if not isinstance(use, dict):
                    raise TypeError("source use must be an object")
                call, reference = use.get("call_id"), use.get("reference_id")
                disposition, reason = use.get("disposition"), use.get("reason", "")
                evidence = use.get("evidence_refs", [])
                if (
                    call not in calls
                    or not isinstance(reference, str)
                    or not reference
                    or len(reference) > 1024
                    or disposition not in DISPOSITIONS
                    or not isinstance(reason, str)
                    or len(reason) > 4000
                    or not isinstance(evidence, list)
                    or len(evidence) > 64
                    or any(not isinstance(e, str) or len(e) > 1024 for e in evidence)
                ):
                    raise ValueError("invalid source use")
                if set(use) - {
                    "call_id",
                    "reference_id",
                    "disposition",
                    "reason",
                    "evidence_refs",
                    "legacy_evidence",
                }:
                    raise ValueError("source use contains undeclared fields")
                facts = call_facts.get(call)
                refs = facts["references"] if facts else []
                matching = [
                    r
                    for r in refs
                    if r.get("provider") == provider
                    and r.get("call_id") == call
                    and r.get("reference_id") == reference
                ]
                if (
                    facts
                    and facts.get("coverage", {}).get("references") == "complete"
                    and not matching
                ):
                    raise ValueError("source ID was not emitted by the referenced call")
                key = (call, reference)
                if key in seen:
                    if seen[key] != use:
                        raise ValueError(
                            "conflicting dispositions for the same experiment source"
                        )
                    continue
                seen[key] = dict(use)
                accepted_uses.append(dict(use))
            candidate = {
                "provider": provider,
                "status": status,
                "call_ids": calls,
                "uses": accepted_uses,
                "verification": verification,
            }
            for use in accepted_uses:
                key = (provider, use["call_id"], use["reference_id"])
                if key in declared_sources and declared_sources[key] != use:
                    raise ValueError(
                        "conflicting declarations for the same experiment source"
                    )
            if candidate not in normalized:
                normalized.append(candidate)
                for use in accepted_uses:
                    declared_sources[
                        (provider, use["call_id"], use["reference_id"])
                    ] = use
        except Exception as error:  # noqa: BLE001 - auxiliary evidence must not change experiment verdicts
            context = {"declaration_index": index}
            if isinstance(declaration, dict):
                # Retain bounded identity evidence for rejected claims without
                # copying arbitrary source payloads into the journal.
                for key in ("provider", "status"):
                    if isinstance(declaration.get(key), str):
                        context[key] = declaration[key][:1024]
                context["call_ids"] = (
                    [
                        call[:1024]
                        for call in declaration.get("call_ids", [])[:64]
                        if isinstance(call, str)
                    ]
                    if isinstance(declaration.get("call_ids"), list)
                    else []
                )
                context["reference_ids"] = (
                    [
                        use["reference_id"][:1024]
                        for use in declaration.get("uses", [])[:64]
                        if isinstance(use, dict)
                        and isinstance(use.get("reference_id"), str)
                    ]
                    if isinstance(declaration.get("uses"), list)
                    else []
                )
            diagnostics.append(
                {"code": "attribution_invalid", "message": str(error)[:1000], **context}
            )
    return {
        "schema": "aka.attribution.v1",
        "experiment_id": experiment_id,
        "declarations": normalized,
        "diagnostics": diagnostics,
    }


_UNSELECTED = object()


def apply_attribution(entry, *, experiment_id, workspace=None, resources=_UNSELECTED):
    """Canonical facts are written once; old Wiki fields are a compatibility view."""
    if "attribution" not in entry:
        return
    if resources is _UNSELECTED:
        try:
            resources = _selected_resources(workspace)
        except Exception:  # noqa: BLE001 - preserve the business verdict when auxiliary evidence fails
            resources = None
    normalized = normalize_attribution(
        entry["attribution"], resources, experiment_id=experiment_id
    )
    entry["attribution"] = normalized
    # Only emitted reference metadata may provide the legacy query ID.
    usage, query_ids = [], []
    statuses = []
    for row in normalized["declarations"]:
        if row["provider"] not in {"gpu-wiki", "internal-gpu-wiki"}:
            continue
        statuses.append(row["status"])
        for use in row["uses"]:
            try:
                facts = resources.call_evidence(use["call_id"])
                reference = next(
                    r
                    for r in facts["references"]
                    if r["reference_id"] == use["reference_id"]
                    and r["provider"] == row["provider"]
                )
                query_id = reference["metadata"]["query_id"]
                if not re.fullmatch(r"wiki-query-[0-9a-f]{32}", query_id):
                    raise ValueError("invalid emitted Wiki query ID")
                query_ids.append(query_id)
                usage.append(
                    {
                        "query_id": query_id,
                        "wiki_id": use["reference_id"],
                        "disposition": use["disposition"],
                        "use": use.get("reason", ""),
                        "evidence": use.get(
                            "legacy_evidence", "\n".join(use.get("evidence_refs", []))
                        ),
                    }
                )
            except Exception:  # noqa: BLE001 - preserve the business verdict when auxiliary evidence fails
                normalized["diagnostics"].append(
                    {
                        "code": "legacy_projection_unavailable",
                        "message": "source lacks emitted query evidence",
                    }
                )
    if not statuses:
        return
    projected = {
        "wiki_usage": usage,
        "wiki_query_ids": list(dict.fromkeys(query_ids)),
        "wiki_usage_status": "declared"
        if usage
        else (
            "not_queried"
            if all(s == "not_invoked" for s in statuses)
            else "no_material_use"
        ),
    }
    conflicts = {
        key: entry[key]
        for key in projected
        if key in entry and entry[key] != projected[key]
    }
    if conflicts:
        normalized["diagnostics"].append(
            {
                "code": "legacy_projection_conflict",
                "message": "legacy input differs from canonical attribution",
            }
        )
        normalized["legacy_input"] = conflicts
    entry.update(projected)
