#!/usr/bin/env python3
"""Serve a local, read-only AKA Light dashboard from observability snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.dashboard.server import DashboardHTTPServer
from aka.evaluation.tool_results import compact_receipt
from observability._internal.common import (
    discover_scopes,
    read_json,
    runtime_audit_paths,
)
from observability.export_light_gateway_record import (
    snapshot as gateway_evidence_snapshot,
)
from observability.light_agate_calls import snapshot as agate_snapshot
from observability.light_kernel_best import snapshot as best_snapshot
from observability.light_kernel_history import snapshot as history_snapshot
from observability.light_lineage_status import snapshot as lineage_snapshot
from observability.light_runtime_calls import snapshot as runtime_snapshot

STATUSES = frozenset({"succeeded", "failed", "pending", "unknown"})
GATEWAY_ROUTE = re.compile(r"/api/gateway-records/(gateway-[a-f0-9]{32})")
RUNTIME_REQUEST_ROUTE = re.compile(r"/api/runtime-requests/([a-f0-9]{32})")
SNAPSHOT_SCHEMA = "atrex.light.dashboard-snapshot.v1"


def _with_agent_receipt(evidence: dict) -> dict:
    """Attach the complete canonical receipt derived at the Agent tool boundary."""
    record = evidence.get("gateway_record")
    response = record.get("response") if isinstance(record, dict) else None
    agent_result = response.get("agent_result") if isinstance(response, dict) else None
    if not isinstance(agent_result, dict):
        return evidence
    value = dict(evidence)
    value["agent_receipt"] = compact_receipt(agent_result)
    return value


def _single(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name, [])
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"{name} may be supplied only once")
    return values[0]


def _last(query: dict[str, list[str]], *, default: int = 50) -> int:
    raw = _single(query, "last")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("last must be a positive integer no greater than 500") from exc
    if value <= 0 or value > 500:
        raise ValueError("last must be a positive integer no greater than 500")
    return value


def _filters(query: dict[str, list[str]]) -> dict:
    statuses = tuple(dict.fromkeys(query.get("status", [])))
    invalid = set(statuses) - STATUSES
    if invalid:
        raise ValueError("status must be one of: " + ", ".join(sorted(STATUSES)))
    return {
        "kinds": tuple(dict.fromkeys(query.get("kind", []))),
        "statuses": statuses,
        "lineages": tuple(dict.fromkeys(query.get("lineage", []))),
        "last": _last(query),
    }


def _filter_captured_calls(payload: dict, filters: dict) -> dict:
    calls = list(payload.get("calls", []))
    if filters["kinds"]:
        calls = [row for row in calls if row.get("kind") in filters["kinds"]]
    if filters["statuses"]:
        calls = [row for row in calls if row.get("status") in filters["statuses"]]
    if filters["lineages"]:
        calls = [row for row in calls if row.get("scope") in filters["lineages"]]
    calls.sort(
        key=lambda row: str(row.get("created_at") or row.get("checkpoint_updated_at") or "")
    )
    result = dict(payload)
    result["calls"] = calls[-filters["last"] :]
    result["detail_selection"] = {
        "kinds": list(filters["kinds"]),
        "statuses": list(filters["statuses"]),
        "lineages": list(filters["lineages"]),
        "last": filters["last"],
        "returned": len(result["calls"]),
        "campaign_total": sum(row.get("calls", 0) for row in payload.get("summary", [])),
        "summary_scope": "campaign",
    }
    return result


def _agate_phase(call: dict) -> str:
    raw = str(
        call.get("execution_phase")
        or call.get("checkpoint_phase")
        or call.get("raw_status")
        or call.get("status")
        or "pending"
    ).lower()
    if raw in {"queued", "waiting", "submitted", "accepted", "created"}:
        return "queued"
    if raw in {"running", "executing", "gpu_running"}:
        return "running_on_gpu"
    if raw in {"collecting", "finalizing", "uploading", "downloading"}:
        return "collecting_result"
    return raw


def _annotate_current_work(lineages: dict, agate_calls: dict) -> None:
    active_by_request: dict[str, dict] = {}
    for call in agate_calls.get("calls", []):
        if call.get("status") != "pending":
            continue
        for request_id in call.get("runtime_request_ids") or ():
            active_by_request[request_id] = call

    for lineage in lineages.get("lineages", []):
        runtime = lineage.get("runtime") or {}
        supervisor = runtime.get("supervisor_activity") or {}
        session = next(iter(runtime.get("active_sessions") or ()), {})
        request_id = supervisor.get("runtime_request_id")
        agate = active_by_request.get(request_id)
        if agate is not None:
            runtime["current_work"] = {
                "layer": "agate",
                "operation": agate.get("kind") or agate.get("operation"),
                "state": _agate_phase(agate),
                "elapsed_seconds": agate.get("pending_age_seconds")
                or agate.get("total_seconds"),
                "agate_job_id": agate.get("agate_job_id"),
                "gateway_record_id": agate.get("gateway_record_id"),
                "runtime_request_id": request_id,
            }
        elif supervisor.get("active") is True and supervisor.get("state") == "running":
            runtime["current_work"] = {
                "layer": "runtime",
                "operation": supervisor.get("operation"),
                "state": supervisor.get("state"),
                "elapsed_seconds": supervisor.get("elapsed_seconds"),
                "runtime_request_id": request_id,
            }
        elif session.get("state") == "running":
            runtime["current_work"] = {
                "layer": "agent",
                "operation": session.get("tool") or session.get("phase"),
                "state": session.get("phase"),
                "elapsed_seconds": session.get("phase_seconds"),
            }
        else:
            runtime["current_work"] = {
                "layer": "lineage",
                "operation": runtime.get("phase") or runtime.get("liveness"),
                "state": runtime.get("liveness"),
                "elapsed_seconds": runtime.get("stagnation_seconds"),
            }


def _runtime_request_detail(workspace: Path, request_id: str) -> dict:
    for scope in discover_scopes(workspace):
        for path in runtime_audit_paths(scope):
            value = read_json(path, default={})
            if not isinstance(value, dict):
                continue
            observed = value.get("request_id") or path.stem.removeprefix("request-")
            if observed != request_id:
                continue
            record_id = value.get("gateway_record_id")
            evidence = None
            if isinstance(record_id, str):
                try:
                    evidence = gateway_evidence_snapshot(workspace, record_id)
                except (OSError, ValueError, KeyError, TypeError, RecursionError):
                    evidence = None
            return {
                "schema_version": 1,
                "runtime_request_id": request_id,
                "scope": scope.label,
                "runtime_audit": value,
                "gateway_record_id": record_id,
                "gateway_evidence": evidence,
            }
    raise ValueError("Runtime request is not available in this Campaign")


class LightDashboardServer(DashboardHTTPServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


def collect_snapshot(
    workspace: Path,
    *,
    refresh_ms: int = 5000,
    source_mode: str = "live_observability",
    include_details: bool = False,
    call_detail_limit: int = 50,
) -> dict:
    """Collect authoritative dashboard data where the Campaign is running."""
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Campaign workspace does not exist: {workspace}")
    lineages = lineage_snapshot(workspace)
    agate_calls = agate_snapshot(workspace, last=call_detail_limit)
    _annotate_current_work(lineages, agate_calls)
    runtime_calls = runtime_snapshot(workspace, last=call_detail_limit)
    gateway_records = {}
    runtime_request_details = {}
    if include_details:
        gateway_ids = {
            call.get("gateway_record_id")
            for call in [
                *runtime_calls.get("calls", []),
                *agate_calls.get("calls", []),
            ]
            if isinstance(call.get("gateway_record_id"), str)
        }
        for record_id in sorted(gateway_ids):
            try:
                gateway_records[record_id] = _with_agent_receipt(
                    gateway_evidence_snapshot(workspace, record_id)
                )
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                continue
        for call in runtime_calls.get("calls", []):
            request_id = call.get("runtime_request_id")
            if not isinstance(request_id, str):
                continue
            try:
                runtime_request_details[request_id] = _runtime_request_detail(
                    workspace, request_id
                )
            except (OSError, ValueError, json.JSONDecodeError):
                runtime_request_details[request_id] = {
                    "schema_version": 1,
                    "runtime_request_id": request_id,
                    "runtime_audit": call,
                    "gateway_record_id": call.get("gateway_record_id"),
                    "gateway_evidence": gateway_records.get(
                        call.get("gateway_record_id")
                    ),
                }
    return {
        "schema": SNAPSHOT_SCHEMA,
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_mode": source_mode,
        "refresh_ms": refresh_ms,
        "workspace": str(workspace),
        "lineages": lineages,
        "kernels": history_snapshot(workspace),
        "winners": best_snapshot(workspace),
        "runtime_calls": runtime_calls,
        "agate_calls": agate_calls,
        "runtime_request_details": runtime_request_details,
        "gateway_records": gateway_records,
    }


def write_snapshot(
    workspace: Path, destination: Path, *, refresh_ms: int = 5000
) -> Path:
    payload = collect_snapshot(
        workspace,
        refresh_ms=refresh_ms,
        source_mode="captured_observability",
        include_details=True,
        call_detail_limit=500,
    )
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _load_captured_snapshot(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Unsupported dashboard snapshot schema in {path}")
    required = {"lineages", "kernels", "winners", "runtime_calls", "agate_calls"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("Dashboard snapshot is missing: " + ", ".join(missing))
    return payload


def create_server(
    source: Path, host: str, port: int, refresh_ms: int
) -> LightDashboardServer:
    source = source.expanduser().resolve()
    captured = _load_captured_snapshot(source) if source.is_file() else None
    if captured is None and not source.is_dir():
        raise ValueError(
            f"Campaign workspace or dashboard snapshot does not exist: {source}"
        )
    workspace = source if captured is None else None
    page = (Path(__file__).parent / "static" / "light-dashboard.html").read_bytes()
    fonts = {
        f"/assets/{name}": files("aka.dashboard")
        .joinpath(f"static/fonts/{name}")
        .read_bytes()
        for name in ("geist-sans.woff2", "geist-mono.woff2")
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            path = request.path.rstrip("/") or "/"
            query = parse_qs(request.query, keep_blank_values=False)
            if path == "/":
                self._respond(200, page, "text/html; charset=utf-8")
                return
            if path in fonts:
                self._respond(200, fonts[path], "font/woff2")
                return
            try:
                if path == "/api/summary":
                    if captured is not None:
                        payload = dict(captured)
                        payload.pop("runtime_request_details", None)
                        payload.pop("gateway_records", None)
                        for name in ("runtime_calls", "agate_calls"):
                            section = dict(payload[name])
                            section["calls"] = section.get("calls", [])[-50:]
                            payload[name] = section
                    else:
                        payload = collect_snapshot(workspace, refresh_ms=refresh_ms)
                elif path == "/api/lineages":
                    payload = (
                        captured["lineages"]
                        if captured is not None
                        else lineage_snapshot(
                            workspace,
                            lineages=tuple(dict.fromkeys(query.get("lineage", []))),
                            research_limit=_last(query),
                        )
                    )
                elif path == "/api/kernels":
                    payload = (
                        captured["kernels"]
                        if captured is not None
                        else history_snapshot(
                            workspace,
                            lineages=tuple(dict.fromkeys(query.get("lineage", []))),
                            commits=tuple(dict.fromkeys(query.get("commit", []))),
                            last=_last(query),
                        )
                    )
                elif path == "/api/winners":
                    payload = (
                        captured["winners"]
                        if captured is not None
                        else best_snapshot(workspace)
                    )
                elif path == "/api/runtime-calls":
                    filters = _filters(query)
                    if captured is not None:
                        payload = _filter_captured_calls(
                            captured["runtime_calls"], filters
                        )
                    else:
                        payload = runtime_snapshot(workspace, **filters)
                elif path == "/api/agate-calls":
                    filters = _filters(query)
                    if captured is not None:
                        payload = _filter_captured_calls(
                            captured["agate_calls"], filters
                        )
                    else:
                        payload = agate_snapshot(workspace, **filters)
                elif match := GATEWAY_ROUTE.fullmatch(path):
                    if captured is not None:
                        payload = captured.get("gateway_records", {}).get(
                            match.group(1)
                        )
                        if payload is None:
                            self._json(
                                404,
                                {
                                    "error": "Gateway Record was not included in the captured snapshot"
                                },
                            )
                            return
                    else:
                        payload = gateway_evidence_snapshot(workspace, match.group(1))
                    payload = _with_agent_receipt(payload)
                elif match := RUNTIME_REQUEST_ROUTE.fullmatch(path):
                    if captured is not None:
                        payload = captured.get("runtime_request_details", {}).get(
                            match.group(1)
                        )
                        if payload is None:
                            self._json(
                                404,
                                {
                                    "error": "Runtime request detail was not included in the captured snapshot"
                                },
                            )
                            return
                    else:
                        payload = _runtime_request_detail(workspace, match.group(1))
                else:
                    self._json(404, {"error": "Route not found", "path": path})
                    return
                self._json(200, payload)
            except ValueError as exc:
                self._json(400, {"error": str(exc), "code": "invalid_request"})
            except (OSError, json.JSONDecodeError) as exc:
                self._json(
                    503, {"error": str(exc), "code": "campaign_temporarily_unavailable"}
                )

        def do_POST(self):
            self._json(405, {"error": "AKA Light dashboard is read-only"})

        do_PUT = do_PATCH = do_DELETE = do_POST

        def _json(self, status: int, payload: dict):
            self._respond(
                status,
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def _respond(self, status: int, payload: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; font-src 'self'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return LightDashboardServer((host, port), Handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Campaign root or captured dashboard JSON"
    )
    parser.add_argument(
        "--capture",
        type=Path,
        metavar="PATH",
        help="collect observability once and write a portable JSON snapshot instead of serving",
    )
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument(
        "--port", type=int, default=8765, help="loopback port; 0 selects a free port"
    )
    parser.add_argument("--refresh-ms", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    if not 0 <= options.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if options.refresh_ms <= 0:
        raise SystemExit("--refresh-ms must be positive")
    if options.capture is not None:
        try:
            destination = write_snapshot(
                options.source, options.capture, refresh_ms=options.refresh_ms
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot capture AKA Light dashboard data: {exc}") from exc
        print(destination, flush=True)
        return 0
    try:
        server = create_server(
            options.source, options.host, options.port, options.refresh_ms
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot start AKA Light dashboard: {exc}") from exc
    url = f"http://{options.host}:{server.server_port}/"
    print(f"AKA Light dashboard: {url} (read-only; Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
