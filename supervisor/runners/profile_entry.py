"""Remote-only command-profile adapter; the Agent calls --kind profile."""

import argparse
import re
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiler", choices=("ncu", "rocprofv3"))
    parser.add_argument("--kernel-name")
    parser.add_argument("--kernel-regex")
    parser.add_argument("--launch-skip", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    profiler = args.profiler or next(
        (name for name in ("ncu", "rocprofv3") if shutil.which(name)), None
    )
    if profiler is None:
        parser.error("GPU worker has neither ncu nor rocprofv3")
    command = [
        "bash",
        "tools/profile_nvidia.sh" if profiler == "ncu" else "tools/profile_kernel.sh",
        "profile_driver.py",
        "--output-dir",
        args.output_dir,
    ]
    if profiler == "ncu":
        selector = args.kernel_name or ("regex:" + args.kernel_regex if args.kernel_regex else None)
        if selector:
            command += ["--kernel-name", selector]
        for option, value in (
            ("--launch-skip", args.launch_skip),
            ("--launch-count", args.launch_count),
        ):
            if value is not None:
                command += [option, str(value)]
        if args.source:
            command.append("--source")
    else:
        if args.source:
            parser.error("source correlation requires the typed Profile route for rocprofv3")
        selector = (
            "^" + re.escape(args.kernel_name) + "$" if args.kernel_name else args.kernel_regex
        )
        if selector:
            command += ["--kernel-regex", selector]
        if args.launch_skip is not None or args.launch_count is not None:
            start = args.launch_skip if args.launch_skip is not None else 1
            count = args.launch_count if args.launch_count is not None else 1
            command += [
                "--iteration-range",
                "[" + ",".join(map(str, range(start, start + count))) + "]",
            ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
