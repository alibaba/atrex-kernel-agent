#!/usr/bin/env python3
"""Build the AKA distributions; wheels are built from their sdists.

Install the `build` frontend in a build environment first. Package directories
contain metadata, not duplicate source trees; invoke this script from any cwd.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNERS = {
    'atrex-aka-core': ('aka/__init__.py', 'aka/__main__.py', 'aka/cli.py', 'aka/core'),
    'atrex-aka-contracts': ('aka/contracts',),
    'atrex-aka-optimization': ('aka/legacy',),
    'atrex-aka-bootstrap': ('aka/bootstrap',),
    'atrex-aka-dashboard': ('aka/dashboard',),
}


def owned(distribution: str) -> tuple[Path, ...]:
    files = []
    suffixes = ('.py', '.json', '.html', '.woff2', '.txt') if distribution == 'atrex-aka-dashboard' else ('.py', '.json')
    for relative in OWNERS[distribution]:
        path = ROOT / 'src' / relative
        files.extend([path] if path.is_file() else
                     sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in suffixes))
    return tuple(files)


def build(distribution: str, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f'{distribution}-build-') as tmp:
        stage = Path(tmp)
        shutil.copy2(ROOT / 'packages' / distribution / 'pyproject.toml', stage / 'pyproject.toml')
        for legal in ('LICENSE', 'NOTICE'):
            shutil.copy2(ROOT / legal, stage / legal)
        for source in owned(distribution):
            target = stage / 'src' / source.relative_to(ROOT / 'src')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        # The default `build` operation deliberately builds a wheel from the sdist.
        subprocess.run([sys.executable, '-m', 'build', '--outdir', str(output.resolve())], cwd=stage, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('distributions', nargs='*')
    args = parser.parse_args()
    distributions = args.distributions or list(OWNERS)
    for distribution in distributions:
        if distribution not in OWNERS:
            parser.error(f'unknown distribution: {distribution}')
        build(distribution, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
