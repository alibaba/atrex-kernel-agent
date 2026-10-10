"""Own the observer's HTTP service for one invocation."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from threading import Thread

from aka.contracts.startup import Invocation
from .reader import WorkspaceReader
from .server import create_server
from .demo import create_demo_workspace


class DashboardStartup:
    def __init__(self, host="127.0.0.1", port=8765, refresh_ms=2000):
        self.host, self.port, self.refresh_ms = host, port, refresh_ms
        self._server = None
        self._thread = None

    def start(self, workspace: Path, *, port: int | None = None, demo: bool = False) -> str:
        """Bind only at invocation time; return the actual loopback address."""
        if self._server is not None:
            raise RuntimeError("dashboard is already started")
        workspace = Path(workspace).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace directory does not exist: {workspace}")
        server = create_server(WorkspaceReader(workspace), self.host,
                               self.port if port is None else port, self.refresh_ms, demo=demo)
        self._server = server
        self._thread = Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.1}, name="aka-dashboard", daemon=True)
        self._thread.start()
        return f"http://{self.host}:{server.server_port}/"

    def run(self, invocation: Invocation) -> int:
        parser = argparse.ArgumentParser(prog="aka dashboard", description=__doc__)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--workspace", type=Path,
                            help="One campaign or a parent containing kernel_opt_* directories")
        source.add_argument("--demo", action="store_true", help="Preview sample campaigns without running optimization")
        parser.add_argument("--port", type=int, default=self.port,
                            help="Loopback port; 0 selects an available port (default: %(default)s)")
        args = parser.parse_args(invocation.argv)
        if not 0 <= args.port <= 65535:
            parser.error("--port must be between 0 and 65535")
        temporary = None
        try:
            if args.demo:
                temporary = tempfile.TemporaryDirectory(prefix="aka-dashboard-demo-")
                workspace = create_demo_workspace(Path(temporary.name))
            else:
                workspace = args.workspace
            url = self.start(workspace, port=args.port, demo=args.demo)
            print(f"AKA dashboard: {url} (read-only; Ctrl+C to stop)", flush=True)
            self._thread.join()
        except KeyboardInterrupt:
            return 0
        except (OSError, ValueError) as exc:
            print(f"Cannot start dashboard: {exc}", file=sys.stderr)
            return 2
        finally:
            self.close()
            if temporary is not None:
                temporary.cleanup()
        return 0

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()
        self._server = self._thread = None
