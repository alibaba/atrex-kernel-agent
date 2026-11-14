"""Register retained physical endpoint, lifecycle and file transport CLI options."""
import argparse
import os
from .gateway.sandbox import (DEFAULT_SSH_HEALTH_COMMAND, DEFAULT_COMMAND_TIMEOUT, SSH_GPU_ENV)


def register_transport_options(parser):
    parser.add_argument("--num-gpus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--hardware",
        default=os.environ.get("ATREX_SANDBOX_GPU", ""),
        help="Remote GPU hardware token, e.g. REMOTE_GPU (default: ATREX_SANDBOX_GPU).",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "run", "profile", "dev", "check", "disassemble", "same_allocation_abba"),
        default="auto",
        help=(
            "Gateway interface to use. auto routes test_kernel.py to run, profiler "
            "wrappers to profile, and other commands to dev (default: auto). Typed "
            "jobs fall back to dev only when their source contract is unsupported."
        ),
    )
    parser.add_argument(
        "--gateway-profile",
        choices=("pre", "prod"),
        default=None,
        help="Gateway endpoint profile (default: ATREX_SANDBOX_PROFILE, then normal agate resolution).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Explicit gateway URL (default: ATREX_SANDBOX_URL; overrides environment profile/config).",
    )
    parser.add_argument(
        "--ssh",
        default=None,
        metavar="[USER@]HOST",
        help=(
            "OpenSSH target for Bubblewrap-isolated remote GPU execution "
            "(default: ATREX_SANDBOX_SSH). Reuses ~/.ssh/config and is "
            "mutually exclusive with gateway endpoint options."
        ),
    )
    parser.add_argument(
        "--ssh-init",
        default=os.environ.get("ATREX_SANDBOX_SSH_INIT", ""),
        metavar="COMMAND",
        help=(
            "Remote shell initialization run before commands and probes, e.g. a "
            "Conda activation (default: ATREX_SANDBOX_SSH_INIT)."
        ),
    )
    parser.add_argument(
        "--ssh-runtime-bind",
        action="append",
        default=None,
        metavar="REMOTE_PATH[=SANDBOX_PATH]",
        help=(
            "Read-only runtime directory exposed inside the SSH Bubblewrap sandbox "
            "(repeatable; default: ATREX_SANDBOX_SSH_RUNTIME_BINDS JSON array)."
        ),
    )
    parser.add_argument(
        "--ssh-gpu",
        default=None,
        metavar="INDEX",
        help=(
            "Physical GPU index assigned to an SSH job (required with --ssh; "
            f"default: {SSH_GPU_ENV}). MIG selectors are rejected until capability-node "
            "assignment is implemented."
        ),
    )
    parser.add_argument(
        "--health-command",
        default=os.environ.get(
            "ATREX_SANDBOX_HEALTH_COMMAND", DEFAULT_SSH_HEALTH_COMMAND
        ),
        metavar="COMMAND",
        help=(
            "Remote GPU health probe used to distinguish candidate failures from "
            "environment failures (default: ATREX_SANDBOX_HEALTH_COMMAND)."
        ),
    )
    parser.add_argument(
        "--runtime-health-command",
        default=os.environ.get("ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND", ""),
        metavar="COMMAND",
        help="Additional trusted evaluator/framework probe; replayed by recovery.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check SSH health before optimization and record failures for recovery.",
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Run only the configured SSH health probe; do not upload a workspace.",
    )
    parser.add_argument(
        "--workspace", default=".", help="Local workspace to upload (default: cwd)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(
            os.environ.get("ATREX_SANDBOX_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))
        ),
        help=(
            "Remote command execution timeout in positive seconds "
            "(default: 600; queue wait is budgeted separately)."
        ),
    )
    parser.add_argument(
        "--sync",
        action="append",
        default=[],
        metavar="PATH",
        help="Relative profile/result path to copy back (repeatable; default: profiles).",
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Do not copy any files back."
    )
    parser.add_argument(
        "--inline-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Return synchronized files through the legacy stdout archive instead of "
            "agate OSS (default: use OSS with standard agate gateway profiles; "
            "custom gateway URLs and config use inline output)."
        ),
    )
    parser.add_argument(
        "--include-raw-profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include raw .ncu-rep/ATT artifacts with legacy inline output "
            "(OSS output includes them by default)."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional required workspace file or directory for a non-evaluator command "
            "(repeatable). Arbitrary commands otherwise receive only paths referenced by "
            "their argv; the full workspace is never uploaded implicitly."
        ),
    )
    parser.add_argument(
        "--max-input-file-mb",
        type=int,
        default=16,
        help="Skip individual workspace input files larger than this (default: 16 MiB).",
    )
    parser.add_argument(
        "--max-output-file-mb",
        type=int,
        default=512,
        help="Skip individual returned artifacts larger than this (default: 512 MiB).",
    )
    parser.add_argument("--max-output-file-bytes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package and print the request summary only.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    return parser
