"""One isolated native Wiki invocation with a dedicated no-tools model policy.

No retrieval algorithm is replaced. Only the native bridge launch boundary is
adapted, and its final timing receipt is written by upstream query_nl itself.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from aka.light.workspace.files import publish
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.wiki.config import WikiConfig

MODEL_ENV = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
             "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")
AUTH_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")


def provider_environment(config: WikiConfig, inherited: dict) -> dict:
    """Read only explicitly selected private settings; never return them in receipts."""
    values = {}
    if config.settings_file:
        try:
            document = json.loads(read_regular_bytes(Path(config.settings_file), limit=1024 * 1024))
            settings = document["env"]
            endpoint = settings["ANTHROPIC_BASE_URL"].rstrip("/")
            if not config.base_url or endpoint != config.base_url.rstrip("/"):
                raise ValueError("endpoint mismatch")
            values = {name: settings[name] for name in AUTH_ENV
                      if isinstance(settings.get(name), str) and settings[name].strip()}
            if not values:
                raise ValueError("missing credentials")
            values["ANTHROPIC_BASE_URL"] = endpoint
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError("Wiki provider settings are invalid or do not match the configured endpoint") from error
    elif inherited.get("ANTHROPIC_BASE_URL"):
        # Preserve the trusted Supervisor's provider route, not Agent environment.
        values["ANTHROPIC_BASE_URL"] = inherited["ANTHROPIC_BASE_URL"]
    model = config.model or inherited.get("ANTHROPIC_MODEL", "")
    if model:
        values.update({name: model for name in MODEL_ENV})
    values["CLAUDE_CODE_EFFORT_LEVEL"] = config.effort
    return values


def main() -> int:
    request_path = Path(sys.argv[1])
    request = json.loads(read_regular_bytes(request_path, limit=512 * 1024))
    config = WikiConfig(**request["config"])
    root = Path(request["tools"])
    sys.path.insert(0, str(root))
    tool = importlib.import_module(request["tool"])
    children = []
    progress = {"phase": "starting", "bridge_attempts": 0, "bridge_latency_ms": 0.0,
                "retrieval_latency_ms": None}
    started = time.monotonic()

    def save():
        publish(request_path.parent, "phase.json", json.dumps(dict(progress,
            worker_latency_ms=round((time.monotonic() - started) * 1000, 3))).encode())

    def stop(*_):
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def terminate(*_):
        stop()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        save()
        if request["tool"] == "query_nl":
            bridge = tool.agent_launch
            overrides = provider_environment(config, dict(os.environ))
            original_environment = bridge.bridge_environment
            def bridge_environment(cli, source=None):
                result = original_environment(cli, source)
                if cli == "claude":
                    if config.settings_file:
                        for name in (*AUTH_ENV, "CLAUDE_CODE_OAUTH_TOKEN"):
                            result.pop(name, None)
                    result.update(overrides)
                return result
            bridge.bridge_environment = bridge_environment
            original_command = bridge.build_claude_json_command
            def command(prompt, session_id=None):
                result = original_command(prompt, session_id)
                result[result.index("--effort") + 1] = config.effort
                if config.model:
                    result[1:1] = ["--model", config.model]
                return result
            bridge.build_claude_json_command = command

            def start(command, cwd, environment):
                # Standalone parser, never the Campaign handoff/restart wrapper.
                child = subprocess.Popen(command, cwd=cwd, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True)
                children.append(child)
                return child
            bridge._start_bridge_process = start
            original_run = bridge.run_json
            def run(*args, **kwargs):
                progress.update(phase="bridge", bridge_attempts=progress["bridge_attempts"] + 1)
                save()
                before = time.monotonic()
                try:
                    return original_run(*args, **kwargs)
                finally:
                    progress["bridge_latency_ms"] += (time.monotonic() - before) * 1000
                    progress["phase"] = "retrieval_or_bridge_repair"
                    save()
            bridge.run_json = run
        return tool.main(request["argv"])
    except ValueError:
        # Settings and provider exceptions may include secrets/private paths.
        print("Wiki configuration or native query validation failed; inspect Supervisor diagnostics.", file=sys.stderr)
        return 2
    finally:
        stop()
        progress["phase"] = "finished"
        save()


if __name__ == "__main__":
    raise SystemExit(main())
