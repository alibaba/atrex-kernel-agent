"""Explicit read-only routes; no filesystem paths supplied by the browser."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from .reader import WorkspaceReader


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = False

    def get_request(self):
        connection, address = super().get_request()
        # Browser preconnects can leave an idle request; bound cleanup's wait for it.
        connection.settimeout(2)
        return connection, address


def create_server(reader: WorkspaceReader, host: str, port: int, refresh_ms: int, *, demo=False):
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
                if path == "/api/campaigns":
                    payload = {**reader.campaigns(), "refresh_ms": refresh_ms, "demo": demo}
                else:
                    match = re.fullmatch(r"/api/campaigns/([0-9a-f]{16})(?:/episodes/([1-9][0-9]*))?", path)
                    if match is None:
                        raise KeyError(path)
                    campaign, episode = match.groups()
                    payload = (reader.episode(campaign, int(episode)) if episode
                               else reader.campaign(campaign))
                self.respond_json(200, payload)
            except KeyError:
                self.respond_json(404, {"error": "Record not found"})
            except OSError:
                self.respond_json(503, {"error": "Workspace temporarily unavailable"})

        def do_POST(self):
            self.respond_json(405, {"error": "Dashboard is read-only"})

        do_PUT = do_PATCH = do_DELETE = do_POST

        def respond_json(self, status, payload):
            self.respond(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                         "application/json; charset=utf-8")

        def respond(self, status, payload, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return DashboardHTTPServer((host, port), Handler)
