"""Generic launch command; business arguments follow an explicit boundary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from aka.contracts.startup import Invocation
from .continuation import DIGEST_ENV, SELECTION_ENV
from .host import check_profile, resume, run_profile


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(prog="aka run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", type=Path, help="start from the selected launch profile")
    mode.add_argument("--resume", action="store_true", help="reconstruct the recorded launch selection")
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--check", action="store_true", help="assemble and validate the profile without running the startup target")
    args = parser.parse_args(argv[:boundary])
    if boundary == len(argv):
        parser.error("separate startup arguments with --")
    if args.resume:
        if args.patch or args.var or args.check:
            parser.error("--resume does not accept --patch, --var or --check")
        return resume(Invocation(argv[boundary + 1:]))
    if SELECTION_ENV in os.environ or DIGEST_ENV in os.environ:
        parser.error("launch selection already supplied; use --resume or clear AKA_LAUNCH_SELECTION and AKA_LAUNCH_DIGEST")
    variables = {}
    for item in args.var:
        if "=" not in item:
            parser.error("--var requires NAME=VALUE")
        name, value = item.split("=", 1)
        if name in variables:
            parser.error(f"duplicate variable: {name}")
        variables[name] = value
    if args.check:
        if argv[boundary + 1:]:
            parser.error("--check does not accept startup arguments")
        return check_profile(args.profile, patch_files=args.patch, variables=variables)
    return run_profile(args.profile, Invocation(argv[boundary + 1:]),
                       patch_files=args.patch, variables=variables)


if __name__ == "__main__":
    raise SystemExit(main())
