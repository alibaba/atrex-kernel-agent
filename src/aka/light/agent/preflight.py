"""Public, model-free checks for the Light Agent execution environment."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Mapping

ATREX_BENCH_VERSION_ENV = "ATREX_SANDBOX_ATREX_BENCH_VERSION"
MINIMUM_AGATE_VERSION = "0.14.30"
_RELEASE_VERSION = re.compile(
    r"^(?P<release>\d+(?:\.\d+)+)"
    r"(?P<suffix>(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?)"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)


def _agate_version() -> str:
    """Return the installed public Agate SDK version."""
    try:
        from atrex_gateway_client import __version__
    except ModuleNotFoundError as exc:
        raise RuntimeError("cannot import the Agate SDK") from exc
    if not isinstance(__version__, str) or not __version__.strip():
        raise RuntimeError("Agate SDK reported no package version")
    return __version__.strip()


def _minimum_version_key(value: str) -> tuple[tuple[int, ...], int]:
    """Parse the stable release ordering needed by the minimum SDK gate."""
    match = _RELEASE_VERSION.fullmatch(value)
    if match is None:
        raise RuntimeError(f"invalid Agate SDK version: {value!r}")
    release = tuple(int(component) for component in match.group("release").split("."))
    suffix = match.group("suffix")
    if not suffix or suffix.startswith(".post"):
        stage = 0
    else:
        stage = -1
    return release, stage


def _release_at_least(actual: str, minimum: str) -> bool:
    actual_release, actual_stage = _minimum_version_key(actual)
    minimum_release, minimum_stage = _minimum_version_key(minimum)
    width = max(len(actual_release), len(minimum_release))
    actual_release += (0,) * (width - len(actual_release))
    minimum_release += (0,) * (width - len(minimum_release))
    return (actual_release, actual_stage) >= (minimum_release, minimum_stage)


def check_minimum_agate_version(
    *,
    actual_version: str | None = None,
    minimum_version: str = MINIMUM_AGATE_VERSION,
) -> str:
    """Return the installed SDK version after enforcing Light's minimum."""
    actual_version = (
        _agate_version()
        if actual_version is None
        else actual_version.strip()
    )
    minimum_version = minimum_version.strip()
    if not actual_version:
        raise RuntimeError("Agate SDK reported no package version")
    if not minimum_version:
        raise ValueError("minimum Agate version must not be empty")
    if not _release_at_least(actual_version, minimum_version):
        raise RuntimeError(
            "Agate SDK version is too old: "
            f"requires >= {minimum_version}, found {actual_version}"
        )
    return actual_version


def check_required_versions(
    required_agate_version: str,
    required_atrex_bench_version: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Enforce the SDK minimum and exact deployment/evaluator pins."""
    required_agate_version = required_agate_version.strip()
    required_atrex_bench_version = required_atrex_bench_version.strip()
    if not required_agate_version:
        raise ValueError("required Agate version must not be empty")
    if not required_atrex_bench_version:
        raise ValueError("required Atrex-Bench version must not be empty")

    check_minimum_agate_version(actual_version=required_agate_version)
    actual_agate_version = check_minimum_agate_version()
    if actual_agate_version != required_agate_version:
        raise RuntimeError(
            "Agate version mismatch: "
            f"required {required_agate_version}, found {actual_agate_version}"
        )

    configured_atrex_bench_version = (os.environ if environment is None else environment).get(
        ATREX_BENCH_VERSION_ENV, ""
    ).strip()
    if not configured_atrex_bench_version:
        raise RuntimeError(
            f"{ATREX_BENCH_VERSION_ENV} is required; refusing an unpinned remote evaluator"
        )
    if configured_atrex_bench_version != required_atrex_bench_version:
        raise RuntimeError(
            "Atrex-Bench version mismatch: "
            f"required {required_atrex_bench_version}, "
            f"configured {configured_atrex_bench_version}"
        )
    return {
        "agate": actual_agate_version,
        "atrex_bench": configured_atrex_bench_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--require-agate-version")
    parser.add_argument("--require-atrex-bench-version")
    args = parser.parse_args()
    if bool(args.require_agate_version) != bool(args.require_atrex_bench_version):
        parser.error(
            "--require-agate-version and --require-atrex-bench-version must be supplied together"
        )
    versions = None
    if args.require_agate_version:
        try:
            versions = check_required_versions(
                args.require_agate_version,
                args.require_atrex_bench_version,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(
                f"Light installation version preflight failed ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            return 1
    result: dict[str, object] = {"check": "light_environment", "status": "ready"}
    if versions is not None:
        result["versions"] = versions
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
