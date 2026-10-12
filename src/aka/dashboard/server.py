"""Explicit read-only routes; no filesystem paths supplied by the browser."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import cast
from urllib.parse import quote, urlsplit

from aka.contracts.observation import (OBSERVATION_VERSION, ArtifactReader, ObservationReader,
                                      ObservationProtocolError, check_observation_version)


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = False

    def get_request(self):
        connection, address = super().get_request()
        # Browser preconnects can leave an idle request; bound cleanup's wait for it.
        connection.settimeout(2)
        return connection, address


def create_server(reader: ObservationReader, host: str, port: int, refresh_ms: int, *, demo=False):
    page = files("aka.dashboard").joinpath("static/index.html").read_bytes()
    fonts = {f"/assets/{name}": files("aka.dashboard").joinpath(f"static/fonts/{name}").read_bytes()
             for name in ("geist-sans.woff2", "geist-mono.woff2")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
                return
            if path in fonts:
                self.respond(200, fonts[path], "font/woff2")
                return
            try:
                download = re.fullmatch(r"/api/campaigns/([A-Za-z0-9_-]+)/artifacts/([A-Za-z0-9_-]+)", path)
                if download:
                    campaign, artifact_id = download.groups()
                    try:
                        payload = reader.campaign(campaign)
                    except KeyError:
                        self.respond_json(404, {"error": "Campaign not found"})
                        return
                    check_observation_version(payload.get("observation_version"))
                    metadata = next((a for a in payload.get("artifacts", []) if a["id"] == artifact_id), None)
                    if metadata is None:
                        raise KeyError(artifact_id)
                    data = cast(ArtifactReader, reader).artifact(campaign, artifact_id)
                    self.respond(200, data, metadata["content_type"], filename=metadata["filename"])
                    return
                if path == "/api/campaigns":
                    payload = {**reader.campaigns(), "refresh_ms": refresh_ms, "demo": demo}
                else:
                    match = re.fullmatch(r"/api/campaigns/([A-Za-z0-9_-]+)(?:/episodes/([1-9][0-9]*))?", path)
                    if match is None:
                        raise KeyError(path)
                    campaign, episode = match.groups()
                    payload = (reader.episode(campaign, int(episode)) if episode
                               else reader.campaign(campaign))
                check_observation_version(payload.get("observation_version"))
                self.respond_json(200, payload)
            except ObservationProtocolError as exc:
                self.respond_json(409, {"error": str(exc), "code": "unsupported_observation_version"})
            except KeyError:
                self.respond_json(404, {"error": "Artifact is no longer available" if download else "Record not found"})
            except OSError:
                self.respond_json(503, {"error": "Workspace temporarily unavailable"})

        def do_POST(self):
            self.respond_json(405, {"error": "Dashboard is read-only"})

        do_PUT = do_PATCH = do_DELETE = do_POST

        def respond_json(self, status, payload):
            self.respond(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                         "application/json; charset=utf-8")

        def respond(self, status, payload, content_type, *, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename is not None:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(filename, safe=""))
                self.send_header("X-AKA-Observation-Version", str(OBSERVATION_VERSION))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return DashboardHTTPServer((host, port), Handler)
