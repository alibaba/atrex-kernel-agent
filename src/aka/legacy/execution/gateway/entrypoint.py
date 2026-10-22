#!/usr/bin/env python3
"""Owned gateway process entry with bounded result-less transport recovery."""
import subprocess
import sys

from aka.legacy.execution.gateway.recovery import run_with_recovery, _kind


def run_transport(argv, *, launcher=None):
    import os
    from aka.contracts.execution import ResourceIdentity
    from aka.legacy.execution.outcomes import FactChannel, emit_fact, captured_result, FACTS_FD
    from aka.legacy.execution.gateway.recovery import TransportCapture
    launcher = launcher or (sys.executable, "-m", "aka.legacy.execution.gateway.sandbox")
    descriptors = (int(argv[argv.index("--prepared-request-fd") + 1]),) if "--prepared-request-fd" in argv else ()
    from .returned_files import FILES_FD
    if os.environ.get(FILES_FD):
        descriptors += (int(os.environ[FILES_FD]),)
    channel = FactChannel(forward=emit_fact)
    try:
        process = subprocess.run([*launcher, *argv],
            capture_output=True, text=True, pass_fds=(channel.writer, *descriptors),
            env={**os.environ, FACTS_FD: str(channel.writer)})
    finally:
        channel.close()
    outcome = captured_result(process, resource=ResourceIdentity("gateway"),
        records=channel.records, decode_output=False)
    return TransportCapture(process.args, process.returncode, process.stdout, process.stderr, outcome)


def main(argv=None, *, runner=run_transport, decorate_stdout=lambda value: value):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _kind(arguments) in {"run", "dev"}:
        result = run_with_recovery(arguments, runner=runner)
    else:
        result = runner(arguments)
    sys.stdout.write(decorate_stdout(result.stdout or ""))
    sys.stderr.write(result.stderr or "")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
