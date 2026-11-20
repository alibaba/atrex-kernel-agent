"""Task-scoped tools: Agent processes never open the parent's ledger or credentials."""
from __future__ import annotations
from contextlib import nullcontext
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from .content import document
from .service import TaskBusy

_REGISTRY_LOCK = threading.RLock()
_REGISTRY = {}


def attach_task_control(environment, control):
    """Host-only bridge to the existing active-time clock; never send control to an Agent."""
    key = (environment.get('AKA_TASK_ENDPOINT'), environment.get('AKA_TASK_TOKEN'))
    with _REGISTRY_LOCK:
        owner = _REGISTRY.get(key)
        if owner is None or not owner.suspend_waits or owner.session_control is not None:
            return lambda: None
        owner.session_control = control
    def release():
        with _REGISTRY_LOCK:
            if owner.session_control is control:
                owner.session_control = None
    return release


class TaskRuntime:
    def __init__(self, service, task_id):
        self.service, self.task_id = service, task_id
        self.token = secrets.token_urlsafe(32)
        self.session_control = None
        try:
            manifest = service.store.task(task_id)['manifest']
            self.suspend_waits = bool(manifest.get('target', {}).get('config', {}).get('scheduling'))
        except AttributeError:
            self.suspend_waits = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.token):
                    self.reply(403, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    # Source uploads share the source bundle's 64 MiB limit,
                    # with room for JSON escaping and the command envelope.
                    if not 0 < length <= 6 * 64 * 1024 * 1024:
                        raise ValueError("Tool request must be a bounded JSON object")
                    args = json.loads(self.rfile.read(length))
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")
                    operation = self.path.removeprefix("/")
                    controller = owner.session_control
                    scope = (controller.request() if controller and operation in {'execute', 'measure'}
                             else nullcontext())
                    with scope:
                        result = owner.call(operation, args)
                    self.reply(200, result)
                except TaskBusy as error:
                    self.reply(409, {"error": "TaskBusy", "message": str(error)})
                except (ValueError, KeyError, TypeError) as error:
                    self.reply(400, {"error": type(error).__name__, "message": str(error)})
                except Exception as error:
                    # Source and credentials are host-private; unexpected exception text may
                    # contain transport internals. Full diagnosis stays with the host log.
                    import logging
                    logging.getLogger(__name__).exception("Task tool failed")
                    self.reply(503, {"error": type(error).__name__, "message": "Host operation interrupted; inspect task events"})

            def reply(self, status, value):
                data = document(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def environment(self):
        result = {"AKA_TASK_ENDPOINT": f"http://127.0.0.1:{self.server.server_port}", "AKA_TASK_TOKEN": self.token}
        if self.suspend_waits:
            result['AKA_TASK_POOLED_WAIT'] = '1'
        return result

    def __enter__(self):
        environment = self.environment
        with _REGISTRY_LOCK:
            _REGISTRY[(environment['AKA_TASK_ENDPOINT'], environment['AKA_TASK_TOKEN'])] = self
        self.thread.start()
        return self

    def __exit__(self, *args):
        environment = self.environment
        with _REGISTRY_LOCK:
            _REGISTRY.pop((environment['AKA_TASK_ENDPOINT'], environment['AKA_TASK_TOKEN']), None)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def call(self, operation, args):
        allowed = {"execute": {"argv", "files", "outputs", "environment", "execution_id", "launch_mode", "source_mode"},
                   "measure": {"scope", "evaluation_id"}, "submit-candidate": {"evaluation_id"},
                   "artifact": {"artifact_id", "offset", "size"},
                   "results": set(), "events": {"after"}}
        if operation not in allowed or set(args) - allowed[operation]:
            raise ValueError("Unsupported task tool or arguments")
        if operation == "execute":
            return self.service.command(self.task_id, **args)
        if operation=='artifact':return self.service.artifact(self.task_id,**args)
        if operation == "measure":
            result = self.service.measure(self.task_id, **args)
            return {k: result[k] for k in ("id", "candidate_id", "revision", "scope", "state", "job_id", "assessment")}
        if operation == "submit-candidate":
            return self.service.submit_candidate(self.task_id, args["evaluation_id"])
        if operation == "results":
            return self.service.results(self.task_id)
        return self.service.events(self.task_id, after=args.get("after", 0))


def invoke(operation, args):
    endpoint, token = os.environ.get("AKA_TASK_ENDPOINT"), os.environ.get("AKA_TASK_TOKEN")
    if not endpoint or not token:
        raise ValueError("Task tools must be invoked inside a bound Agent session")
    request = urllib.request.Request(endpoint + "/" + operation, data=document(args).encode(),
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        # The host owns queue, cancellation and execution deadlines. A pooled
        # wait can exceed the legacy HTTP timeout without spending Agent time.
        timeout = (None if os.environ.get('AKA_TASK_POOLED_WAIT') == '1'
                   and operation in {'execute', 'measure'} else 10860)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            value = json.load(error)
        if error.code == 409 and value.get("error") == "TaskBusy":
            raise TaskBusy(value["message"]) from None
        raise ValueError(value.get("message", value.get("error", "Task tool failed"))) from None
