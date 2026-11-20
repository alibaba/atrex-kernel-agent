"""HTTP server lifecycle shared by the read-only Light dashboard."""
from __future__ import annotations

from http.server import ThreadingHTTPServer


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = False

    def get_request(self):
        connection, address = super().get_request()
        # Browser preconnects can leave an idle request; bound cleanup's wait for it.
        connection.settimeout(2)
        return connection, address
