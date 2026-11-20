"""Task-bound command executor using the shared AKA execution contract."""
from __future__ import annotations
import shlex
from dataclasses import replace
import base64
from aka.light.execution.contract import BackendCapabilities, JobResult, ResourceIdentity
from aka.contracts.task import EvaluationPlan
from aka.light.tasks.content import tree
from .package import encode_files


class TaskExecutor:
    capabilities = BackendCapabilities(frozenset({"run"}), True)

    def __init__(self, *, target, directory, execution_id, previous_job, on_job, control):
        self.target, self.directory, self.execution_id = target, directory, execution_id
        self.previous_job, self.on_job, self.control = previous_job, on_job, control
        self.resource_identity = ResourceIdentity(target["backend"], previous_job or None)

    def run(self, command):
        self.capabilities.require(command)
        from aka.light.execution.task_transport import execute
        files = encode_files(tree(command.workspace))
        environment = {"PYTHONPYCACHEPREFIX": ".aka-cache/python", "TORCH_EXTENSIONS_DIR": ".aka-cache/torch",
            "TRITON_CACHE_DIR": ".aka-cache/triton", "CUDA_CACHE_PATH": ".aka-cache/cuda",
            **dict(command.environment)}
        if self.target["backend"] == "local":
            plan = EvaluationPlan("local", {"argv": list(command.command), "files": files,
                                            "environment": environment})
        else:
            # Dev's env_vars field is restricted to gateway configuration keys.
            # These variables belong to the executed process, just as with local.
            invocation = ["env", *(f"{key}={value}" for key, value in environment.items()), *command.command]
            plan = EvaluationPlan("dev", {"command": shlex.join(invocation), "files": files,
                "recycle": True,
                "oss_outputs": [{"path": path} for path in command.outputs],
                "dev_intent": "other", "dev_note": "AKA command execution"})
        receipt = execute(plan, {**self.target, "timeout_s": command.timeout}, self.directory,
                          self.execution_id, self.previous_job, self.on_job,
                          lambda: "cancel" if command.cancel_event is not None and command.cancel_event.is_set()
                          else self.control())
        result = receipt.payload.get("result", receipt.payload)
        if self.target["backend"] == "local" and receipt.status == "completed":
            from aka.light.tasks.content import relative
            artifacts = {name: {"base64": base64.b64encode((self.directory / relative(name)).read_bytes()).decode()}
                         for name in command.outputs if (self.directory / relative(name)).is_file()}
            receipt = replace(receipt, payload={**receipt.payload, "artifacts": artifacts})
        self.resource_identity = ResourceIdentity(self.target["backend"], receipt.job_id or None)
        return JobResult(0 if receipt.status == "completed" else 1,
                         result.get("stdout", ""), result.get("stderr", ""),
                         result.get("exit_code"), self.resource_identity,
                         details={"receipt": {"status": receipt.status, "payload": receipt.payload,
                                              "job_id": receipt.job_id}})
