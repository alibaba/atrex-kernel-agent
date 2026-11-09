#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_light.sh [OPTIONS] AKA_REPO_ROOT

Build and install matching atrex-aka-framework, atrex-aka-legacy, and
atrex-aka-light wheels from one AKA checkout.

Options:
  --python PATH                     Python interpreter (default: python3)
  --wheel-dir PATH                  Keep wheels in this directory
  --no-install-build-dependency     Fail instead of installing `build` if absent
  -h, --help                        Show this help
EOF
}

python_bin="python3"
wheel_dir=""
install_build_dependency=1

while (($#)); do
  case "$1" in
    --python)
      (($# >= 2)) || { echo "install_light.sh: --python requires a value" >&2; exit 2; }
      python_bin="$2"
      shift 2
      ;;
    --wheel-dir)
      (($# >= 2)) || { echo "install_light.sh: --wheel-dir requires a value" >&2; exit 2; }
      wheel_dir="$2"
      shift 2
      ;;
    --no-install-build-dependency)
      install_build_dependency=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "install_light.sh: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if (($# != 1)); then
  usage >&2
  exit 2
fi

repo_root="$1"
if [[ ! -d "$repo_root" ]]; then
  echo "install_light.sh: AKA checkout does not exist: $repo_root" >&2
  exit 2
fi
repo_root="$(cd "$repo_root" && pwd -P)"

if [[ ! -f "$repo_root/tools/build_distributions.py" ]]; then
  echo "install_light.sh: missing tools/build_distributions.py under $repo_root" >&2
  exit 2
fi
for package in atrex-aka-framework atrex-aka-legacy atrex-aka-light; do
  if [[ ! -f "$repo_root/packages/$package/pyproject.toml" ]]; then
    echo "install_light.sh: missing package metadata for $package" >&2
    exit 2
  fi
done

if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
  echo "install_light.sh: Python interpreter is not executable: $python_bin" >&2
  exit 2
fi
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "install_light.sh: Python 3.10 or newer is required" >&2
  exit 2
fi

temporary_wheels=0
if [[ -z "$wheel_dir" ]]; then
  wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/aka-light-wheels.XXXXXX")"
  temporary_wheels=1
else
  mkdir -p "$wheel_dir"
  wheel_dir="$(cd "$wheel_dir" && pwd -P)"
fi

cleanup() {
  if ((temporary_wheels)); then
    rm -rf -- "$wheel_dir"
  fi
}
trap cleanup EXIT

if ! "$python_bin" -c 'import build' >/dev/null 2>&1; then
  if ((install_build_dependency)); then
    "$python_bin" -m pip install build
  else
    echo "install_light.sh: Python package 'build' is required" >&2
    exit 1
  fi
fi

commit="not-a-git-checkout"
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit="$(git -C "$repo_root" rev-parse HEAD)"
fi

echo "Installing AKA Light from: $repo_root"
echo "Source commit: $commit"
echo "Wheel directory: $wheel_dir"

"$python_bin" "$repo_root/tools/build_distributions.py" \
  --output "$wheel_dir" \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light

"$python_bin" -m pip install --force-reinstall --no-index \
  --find-links "$wheel_dir" atrex-aka-light

"$python_bin" -m aka --help
"$python_bin" -m aka optimize --help >/dev/null

echo "AKA Light installation verified."
echo "Use this matching checkout at launch: --repo-root $repo_root"
if ((!temporary_wheels)); then
  echo "Built wheels retained at: $wheel_dir"
fi
