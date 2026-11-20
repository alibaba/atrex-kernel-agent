"""Durable execution receipts. Domain result parsing belongs to task adapters."""
from __future__ import annotations
import subprocess
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from aka.contracts.task import ExecutionReceipt
from aka.light.tasks.content import materialize
from .package import decode_files


def validate_plan(plan, target):
    allowed = ({"local", "command"} if target["backend"] == "local" else
               {"command"} if target["backend"] == "remote" else {"eval", "dev", "command"})
    if plan.kind not in allowed:
        if target["backend"] == "remote":
            raise ValueError("REMOTE requires a command plan; use a command contract")
        raise ValueError("Unsupported execution operation; GPU plans use eval or dev. "
                         "Run profiling, compilation and disassembly as dev commands.")


def gateway_client(target):
    # Keep client-version-specific credential resolution in the transport boundary.
    # Credentials are never copied into task manifests, plans or Agent workspaces.
    from atrex_gateway_client.cli import _resolve
    from atrex_gateway_client.client import Client
    url, auth = _resolve(SimpleNamespace(url=target.get("url"), profile=target.get("profile", "prod"),
                                        token=None, ak=None, sk=None))
    return Client(url, auth=auth, timeout=45, transport_retries=1)


def execute(plan, target, directory, evaluation_id, previous_job, on_job, control, *, recover_unrecorded=False):
    validate_plan(plan, target)
    if recover_unrecorded and (target['backend'] != 'remote' or previous_job):
        raise ValueError('Unrecorded recovery requires REMOTE and an absent saved handle')
    request = dict(plan.request)
    if not previous_job and control() != "run" and not recover_unrecorded:
        return ExecutionReceipt("cancelled", {"reason": "Stopped before execution admission"})
    if plan.kind == "command":
        from aka.light.execution.contract import CommandSpec, validate_executor
        from aka.light.execution.task import TaskExecutor
        from importlib.metadata import entry_points
        # Keep persisted transport handles/artifacts out of recovered source inputs.
        workspace = Path(directory) / "input" if target["backend"] == "remote" else Path(directory)
        if request.get('files_ref'):
            from .remote_cache import Store
            directory=Path(directory)
            if directory.parent.name!='evaluations' or directory.parents[2].name!='tasks':raise ValueError('Content reference outside a task evaluation')
            store=Store(directory.parents[3]/'content-v2')
            declared_files=store.files(request['files_ref'])
            if not workspace.exists():store.materialize(request['files_ref'],workspace)
            else:
                from aka.light.tasks.content import identity,tree
                if identity(tree(workspace,include_ignored=True,allow_empty=True))!=identity(declared_files):raise ValueError('Frozen execution input changed')
        else:
            declared_files = decode_files(request["files"])
            materialize(workspace, declared_files)
        factory = TaskExecutor
        if target["backend"] == "remote":
            from .remote import Executor
            factory = Executor
        elif target["backend"] not in {"local", "agate"}:
            entries = list(entry_points(group="aka.task_executors", name=target["backend"]))
            if len(entries) != 1:
                raise ValueError("Expected one installed executor for " + target["backend"])
            factory = entries[0].load()
        executor = factory(target=target, directory=directory, execution_id=evaluation_id,
                           previous_job=previous_job, on_job=on_job, control=control)
        if target['backend']=='remote':
            executor.launch_mode=request.get('launch_mode')
            # Explicit research inputs may live under cache/report paths which
            # normal source discovery deliberately ignores. Transfer precisely
            # the declared frozen input map, without adding live runtime files.
            executor.declared_input_files = declared_files
        validate_executor(executor)
        result = executor.reconcile_unrecorded() if recover_unrecorded else None
        if result is None:
            result = executor.run(CommandSpec(workspace=workspace, command=tuple(request["argv"]),
                timeout=target.get("timeout_s", 600), environment=request.get("environment", {}),
                hardware=target.get("gpu", ""), outputs=tuple(request.get("outputs", []))))
        if "receipt" in result.details:
            return ExecutionReceipt(**result.details["receipt"])
        return ExecutionReceipt("completed" if result.transport_returncode == 0 and result.exit_code == 0 else "failed",
            {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code},
            result.resource.identifier or "" if result.resource else "")
    if target["backend"] == "local":
        if plan.kind != "local":
            raise ValueError("Local execution requires a local plan")
        materialize(directory, decode_files(request["files"]))
        process = subprocess.Popen(request["argv"], cwd=directory, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
            env={**os.environ, **request.get("environment", {})})
        deadline = time.monotonic() + target.get("timeout_s", 600)
        while True:
            cancelled = control() == "cancel"
            timed_out = time.monotonic() >= deadline
            if cancelled or timed_out:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                return ExecutionReceipt("cancelled" if cancelled else "failed",
                    {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode,
                     "reason": "Local command cancelled" if cancelled else "Local command timed out"})
            try:
                stdout, stderr = process.communicate(timeout=max(0, min(0.2, deadline - time.monotonic())))
                return ExecutionReceipt("completed" if process.returncode == 0 else "failed",
                    {"stdout": stdout, "stderr": stderr, "exit_code": process.returncode})
            except subprocess.TimeoutExpired:
                continue
    if target["backend"] != "agate":
        raise ValueError("Native vendor plans require the Agate backend")
    client = gateway_client(target)
    if not previous_job:
        request["idempotency_key"] = evaluation_id
        request["spec"] = {**request.get("spec", {}), "target_hardware": [target["gpu"]],
                           "num_gpus": target.get("num_gpus", 1)}
        request["timeout_s"] = target.get("timeout_s", 600)
        from aka.light.evaluation.source_transport import uploaded_request
        def upload(url, data):
            import tempfile
            with tempfile.NamedTemporaryFile() as stream:
                stream.write(data)
                stream.flush()
                client.upload_file(url, stream.name)
        request = uploaded_request(request, evaluation_id, kind=plan.kind, state_directory=directory,
            json_call=lambda method, path, body, timeout: client.prepare_uploads(
                target["gpu"], body["files"], kind=plan.kind, num_gpus=target.get("num_gpus", 1)),
            upload=upload)
        try:
            accepted = client.submit_job(plan.kind, request)
        except Exception as error:
            from atrex_gateway_client.errors import GatewayError
            if not isinstance(error, GatewayError) or error.status not in {400, 422}:
                raise
            # A rejected declaration did not create a remote execution. Keep the
            # diagnostic as a failed receipt; only ambiguous writes need recovery.
            return ExecutionReceipt("failed", {"error": {
                "code": "validation/" + (error.reason or "request_rejected"),
                "message": str(error), "details": error.details}, "response": error.payload})
        previous_job = accepted["job_id"]
        on_job(previous_job)
    cancelled = False
    while True:
        if control() == "cancel" and not cancelled:
            client.cancel_job(previous_job)
            cancelled = True
        # Short status reads survive proxies that time out held long-poll responses.
        raw = client.get_job(previous_job, wait=False)
        state = raw.get("status")
        if state not in {"succeeded", "failed", "cancelled", "canceled", "timeout"}:
            time.sleep(3)
            continue
        if state in {"cancelled", "canceled"}:
            status = "cancelled"
        elif state != "succeeded":
            status = "failed"
        elif plan.kind == "dev":
            result = raw.get("result", {})
            command_ok = raw.get("command_ok")
            if command_ok is None:
                command_ok = type(result.get("exit_code")) is int and result["exit_code"] == 0
            status = "completed" if command_ok is True else "failed"
        else:
            status = "completed"
        return ExecutionReceipt(status, raw, previous_job)
