#!/usr/bin/env python3
"""Owned gateway process entry with bounded result-less transport recovery."""
import subprocess
import sys

from aka.legacy.execution.gateway.recovery import run_with_recovery, _kind, _record_gateway_failure


def run_transport(argv):
    return subprocess.run([sys.executable, "-m", "aka.legacy.execution.gateway.sandbox", *argv],
                          capture_output=True, text=True)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _kind(arguments) in {"run", "dev"}:
        result = run_with_recovery(arguments, runner=run_transport)
    else:
        result = run_transport(arguments)
    import json, os
    lines = (result.stdout or "").splitlines(keepends=True)
    for index, line in enumerate(lines):
        prefix = "[test_kernel] RESULT_JSON="
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix):])
            if isinstance(payload, dict):
                payload.setdefault("measurement_conditions", {})["gateway_warmup_iters"] = os.environ.get("AKA_GATEWAY_WARMUP_ITERS", "5")
                lines[index] = prefix + json.dumps(payload) + "\n"
    result.stdout = "".join(lines)
    _record_gateway_failure(arguments, result)
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
