"""Task runtime options projected onto existing Agent settings and host profiles."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
from aka.agent.native import supports_same_session_resume


def resolve(config):
    values, environment = {}, {}
    if config.get("profile"):
        path = Path(os.environ.get("AKA_AGENT_PROFILES", Path.home()/".atrex/aka-agents.json"))
        profile = json.loads(path.read_text())[config["profile"]]
        values.update({k: v for k, v in profile.items() if k not in {"environment", "secret_environment"}})
        environment.update(profile.get("environment", {}))
        for target, source in profile.get("secret_environment", {}).items():
            if source not in os.environ:
                raise ValueError("Agent profile requires environment variable " + source)
            environment[target] = os.environ[source]
    values.update(config)
    values.setdefault("runtime", "claude")
    runtime = values["runtime"]
    from aka.agent.backends import DEFAULT_BACKEND_REGISTRY
    adapter = DEFAULT_BACKEND_REGISTRY.create(runtime)
    settings = json.loads(environment.get(adapter.settings_variable,
        os.environ.get(adapter.settings_variable, "{}")))
    if values.get("model"):
        settings["model"] = values["model"]
    if values.get("provider"):
        if runtime != "pi":
            raise ValueError("provider is a pi option; other provider connections use host profiles")
        settings["provider"] = values["provider"]
    if settings:
        environment[adapter.settings_variable] = json.dumps(settings)
    return values, environment


def preflight(config):
    values, environment = resolve(config)
    runtime = values["runtime"]
    if not supports_same_session_resume(runtime):
        raise ValueError("Task research requires a persistent runtime: " + runtime)
    if not shutil.which(runtime, path=environment.get("PATH", os.environ.get("PATH"))):
        raise ValueError("Agent executable is unavailable: " + runtime)
    return values, environment
