#!/usr/bin/env bash
set -euo pipefail

deploy_root="${ATREX_LIGHT_DEPLOY_ROOT:-/workspace/deploy}"
deployment_config="${deploy_root}/deployment.json"

if [[ ! -f "${deployment_config}" ]]; then
  echo "Light installation requires ${deployment_config}" >&2
  exit 1
fi

IFS=$'\t' read -r required_agate_version required_atrex_bench_version < <(
  python3 - "${deployment_config}" <<'PY'
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
try:
    config = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read deployment version contract {path}: {exc}")

values = []
for key in ("agate_client", "atrex_bench_version"):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"deployment version contract requires non-empty {key!r}")
    value = value.strip()
    if any(character.isspace() for character in value):
        raise SystemExit(f"deployment version contract requires a whitespace-free {key!r}")
    values.append(value)
match = re.fullmatch(
    r"(?P<release>\d+(?:\.\d+)+)(?P<suffix>(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?)"
    r"(?:\+[0-9A-Za-z.-]+)?",
    values[0],
)
if match is None:
    raise SystemExit(
        f"deployment version contract has invalid 'agate_client': {values[0]!r}"
    )
release = tuple(int(component) for component in match.group("release").split("."))
release += (0,) * (3 - len(release))
stable = not match.group("suffix") or match.group("suffix").startswith(".post")
if release < (0, 14, 30) or (release == (0, 14, 30) and not stable):
    raise SystemExit(
        "deployment version contract requires agate_client>=0.14.30; "
        f"configured {values[0]}"
    )
print("\t".join(values))
PY
)

if [[ -z "${required_agate_version}" || -z "${required_atrex_bench_version}" ]]; then
  echo "deployment version contract did not yield both required versions" >&2
  exit 1
fi
readonly required_agate_version
readonly required_atrex_bench_version

agate_wheels=(
  "${deploy_root}"/wheels/atrex_gateway_client-"${required_agate_version}"-*.whl
)
if [[ ${#agate_wheels[@]} -ne 1 || ! -f "${agate_wheels[0]}" ]]; then
  echo "Light installation requires exactly one Agate SDK wheel for ${required_agate_version}" >&2
  exit 1
fi

python3 -m pip install --no-index --no-deps --force-reinstall \
  "${agate_wheels[0]}" \
  "${deploy_root}"/wheels/atrex_aka_framework-*.whl \
  "${deploy_root}"/wheels/atrex_aka_legacy-*.whl \
  "${deploy_root}"/wheels/atrex_aka_light-*.whl

command -v bwrap >/dev/null

export ATREX_SANDBOX_ATREX_BENCH_VERSION="${required_atrex_bench_version}"

aka optimize --help >/dev/null
python3 -m aka.light.agent.preflight \
  --require-agate-version "${required_agate_version}" \
  --require-atrex-bench-version "${required_atrex_bench_version}"
