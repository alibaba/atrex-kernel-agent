"""Construct the preserved application and adapt it to the neutral startup seam."""
from pathlib import Path

from aka.contracts.application import ApplicationRequest
from .entrypoint import create_legacy_application
from .processes import use_launch_environment, invocation_environment

name = "application"
provide = ("startup",)
inject = ("problem", "source", "candidate")
Config = {"type": "object", "properties": {"repo_root": {"type": "string", "minLength": 1}},
          "required": ["repo_root"], "additionalProperties": False}
interpolate = ("repo_root",)
identity_packages = ("aka.legacy.application", "aka.contracts", "aka.bootstrap")


class LegacyStartup:
    def __init__(self, application, repo_root):
        self._application = application
        self._repo_root = repo_root

    def run(self, invocation):
        try:
            from orchestrator.process_launch import optimizer_entrypoint
        except ModuleNotFoundError as exc:
            if exc.name not in {"orchestrator", "orchestrator.process_launch"}:
                raise
            raise RuntimeError("make the matching AKA checkout importable before invocation") from exc
        expected = invocation.bindings.get("optimizer_script")
        if expected is None or Path(expected).resolve() != optimizer_entrypoint():
            raise RuntimeError("startup selected a different optimizer entrypoint")
        with invocation_environment(invocation.environment), use_launch_environment(invocation.environment):
            return self._application.run(ApplicationRequest(self._repo_root, invocation.argv))


def apply(ctx, config):
    root = Path(config["repo_root"]).resolve()
    ctx.provide("startup", LegacyStartup(create_legacy_application(root, problem=ctx.problem, source=ctx.source, workspace=ctx.candidate), root))
