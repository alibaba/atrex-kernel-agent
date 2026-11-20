"""Caller-owned harness adapter for arbitrary multi-file computational modules."""
from __future__ import annotations
import json
from aka.contracts.task import Assessment, EvaluationPlan, Metric
from ..content import tree
from aka.light.execution.package import encode_files


class Adapter:
    protocol_version = 1
    capabilities = frozenset({"acceptance", "diagnostic", "parallel_candidates"})

    def validate(self, contract, config):
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(s, str) and s for s in command):
            raise ValueError("Command contracts require a nonempty argv list")
        if not config.get("instructions"):
            raise ValueError("Command contracts require Agent instructions defining the module interface")
        if not config.get("delivery"):
            raise ValueError("Command contracts require delivery instructions")

    def instructions(self, contract, config):
        return config["instructions"]

    def prepare(self, context):
        files = {"contract/" + name: data for name, data in tree(context.contract).items()}
        files.update({"candidate/" + name: data for name, data in tree(context.candidate).items()})
        argv = [part.replace("{candidate}", "candidate").replace("{contract}", "contract").replace("{scope}", context.scope)
                for part in context.config["command"]]
        return EvaluationPlan("command", {"argv": argv, "files": encode_files(files),
            "environment": {"PYTHONPATH": "candidate", **context.config.get("environment", {})}})

    def interpret(self, context, receipt):
        result = receipt.payload.get("result", receipt.payload)
        stdout = result.get("stdout", "")
        prefix = context.config.get("result_prefix", "AKA_TASK_RESULT=")
        rows = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
        if len(rows) != 1:
            return Assessment("unknown", False, reason="Harness must emit exactly one result document")
        try:
            value = json.loads(rows[0])
            if set(value) - {"verdict", "complete", "metrics", "environment", "reason", "details", "failure_kind"}:
                raise ValueError("Unknown assessment fields")
            metrics = tuple(Metric(**item) for item in value.get("metrics", []))
            return Assessment(value["verdict"], value["complete"], metrics,
                              value.get("environment", ""), value.get("reason", ""), value.get("details", {}),
                              value.get("failure_kind", "correctness" if value["verdict"] == "fail" else ""))
        except (ValueError, KeyError, TypeError) as error:
            return Assessment("unknown", False, reason=f"Invalid harness assessment: {error}")

    def delivery(self, contract, config):
        return config["delivery"]
