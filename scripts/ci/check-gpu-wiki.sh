#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

changed_files="$(
  if [ -n "${AONE_MERGE_TARGET_SHA:-}" ]; then
    git diff --name-only "${AONE_MERGE_TARGET_SHA}...HEAD"
  elif git rev-parse --verify origin/main >/dev/null 2>&1; then
    git diff --name-only origin/main...HEAD
  else
    git diff --name-only HEAD
  fi
)"

if ! printf '%s\n' "$changed_files" | grep -q '^gpu-wiki/'; then
  echo "No gpu-wiki changes detected; skipping gpu-wiki gate."
  exit 0
fi

python3 gpu-wiki/scripts/check-self-contained.py --root gpu-wiki
python3 -m unittest gpu-wiki/scripts/test_check_self_contained.py -v
git diff --check

if command -v openspec >/dev/null 2>&1; then
  openspec validate wikify-gpu-wiki-knowledge --strict
else
  echo "openspec not found; skipping optional OpenSpec validation."
fi
