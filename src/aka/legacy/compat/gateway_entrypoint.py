"""Legacy Gateway launcher composition with Evaluation-owned output conditions."""
import os
import sys
from functools import partial
from aka.legacy.execution.gateway.entrypoint import main as execution_main, run_transport
from aka.legacy.evaluation.gateway_measurement import with_measurement_conditions


def main(argv=None):
    return execution_main(argv, runner=partial(run_transport,
        launcher=(sys.executable, "-m", "aka.legacy.compat.gateway")), decorate_stdout=lambda stdout: with_measurement_conditions(
        stdout, gateway_warmup_iters=os.environ.get("AKA_GATEWAY_WARMUP_ITERS", "5")))


if __name__ == "__main__":
    raise SystemExit(main())
