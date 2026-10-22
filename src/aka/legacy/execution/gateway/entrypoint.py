#!/usr/bin/env python3
"""Owned gateway process entry with bounded result-less transport recovery."""
import subprocess
import sys

from aka.legacy.execution.gateway.recovery import run_with_recovery, _kind


def run_transport(argv):
    import os
    from aka.contracts.execution import ResourceIdentity
    from aka.legacy.execution.outcomes import FactChannel, emit_fact, captured_result, FACTS_FD
    from aka.legacy.execution.gateway.recovery import TransportCapture
    channel = FactChannel(forward=emit_fact)
    try:
        process = subprocess.run([sys.executable, "-m", "aka.legacy.execution.gateway.sandbox", *argv],
            capture_output=True, text=True, pass_fds=(channel.writer,),
            env={**os.environ, FACTS_FD: str(channel.writer)})
    finally:
        channel.close()
    outcome = captured_result(process, resource=ResourceIdentity("gateway"),
        records=channel.records, decode_output=False)
    return TransportCapture(process.args, process.returncode, process.stdout, process.stderr, outcome)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _kind(arguments) in {"run", "dev"}:
        result = run_with_recovery(arguments, runner=run_transport)
    else:
        result = run_transport(arguments)
    import os
    from aka.legacy.evaluation.gateway_measurement import with_measurement_conditions
    result.stdout = with_measurement_conditions(
        result.stdout or "", gateway_warmup_iters=os.environ.get("AKA_GATEWAY_WARMUP_ITERS", "5"))
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
