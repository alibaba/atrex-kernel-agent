#!/usr/bin/env python3
"""Build the AKA distributions; wheels are built from their sdists.

Install the `build` frontend in a build environment first. Package directories
contain metadata, not duplicate source trees; invoke this script from any cwd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNERS = {
    'atrex-aka-framework': ('aka/__init__.py', 'aka/__main__.py', 'aka/cli.py', 'aka/core', 'aka/contracts', 'aka/bootstrap'),
    'atrex-aka-legacy': ('aka/legacy',),
    'atrex-aka-dashboard': ('aka/dashboard',),
    'atrex-aka-observation-long-horizon': ('aka/observation/long_horizon',),
}


def owned(distribution: str) -> tuple[Path, ...]:
    files = []
    suffixes = ('.py', '.json', '.html', '.woff2', '.txt') if distribution == 'atrex-aka-dashboard' else ('.py', '.json', '.sh', '.md')
    for relative in OWNERS[distribution]:
        path = ROOT / 'src' / relative
        files.extend([path] if path.is_file() else
                     sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in suffixes))
    if distribution == 'atrex-aka-legacy':
        # Runtime content belongs to its installed module. Include binary and
        # extensionless attachments as well as prompts and Skills; never stage
        # checkout-level tools, resources or generated caches.
        files = sorted(p for p in (ROOT / 'src' / 'aka' / 'legacy').rglob('*')
            if p.is_file() and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.parts)
            and p.suffix not in {'.pyc', '.pyo'})
    return tuple(dict.fromkeys(files))


def build_wiki_resources(output: Path) -> None:
    """Export the default local plugins and the public Wiki data closure."""
    output.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with tarfile.open(output / 'atrex-aka-public-resources-0.1.0.tar.gz', 'w:gz') as archive:
        for relative in ('plugins/gpu-wiki', 'plugins/micro-analysis', 'gpu-wiki'):
            directory = ROOT / relative
            if not directory.is_dir():
                raise FileNotFoundError(f'missing selected public resource: {directory}')
            for path in sorted(directory.rglob('*')):
                if {'.git', '__pycache__', '.pytest_cache'}.intersection(path.parts) or path.suffix in {'.pyc', '.pyo'}:
                    continue
                if path.is_symlink():
                    raise ValueError(f'resource export requires materialized files: {path}')
                if path.is_file():
                    name = path.relative_to(ROOT).as_posix()
                    inventory[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                       'mode': path.stat().st_mode & 0o777}
                    archive.add(path, arcname=name, recursive=False)
    identity = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (output / 'public-wiki-resource.json').write_text(json.dumps({
        'schema_version': 1, 'resource': 'aka-public-resources', 'identity': identity, 'files': inventory,
    }, indent=2) + '\n')



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
    parser.add_argument('--with-wiki-resources', action='store_true',
                        help='also export the default public plugins and Wiki data closure as a separate artifact')
    parser.add_argument('distributions', nargs='*')
    args = parser.parse_args()
    distributions = args.distributions or list(OWNERS)
    for distribution in distributions:
        if distribution not in OWNERS:
            parser.error(f'unknown distribution: {distribution}')
        build(distribution, args.output)
    if args.with_wiki_resources:
        build_wiki_resources(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
