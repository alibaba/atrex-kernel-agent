"""Translate explicit runtime paths into Git excludes without changing tracked files."""
from pathlib import Path
import subprocess

from aka.contracts.workspace import WorkspaceExclusion

# Retained path spellings for the existing Task state adapter.
RUNTIME_DIR = '.atrex_long_horizon'
VERIFY_DIR = 'verification_artifacts/.atrex_long_horizon_verify'
LIVE_MEMORY_FILE = 'memory/live.json'
STALL_STATE_FILE = '.orchestrator_state.json'


def exclude_runtime_paths(workspace: Path, exclusions: tuple[WorkspaceExclusion, ...]) -> None:
    exclusions = tuple(exclusions)
    if any(not isinstance(item, WorkspaceExclusion) for item in exclusions):
        raise TypeError('workspace exclusions require explicit WorkspaceExclusion values')
    if not exclusions:
        return
    # A trailing space must be escaped in Git's literal pattern syntax.
    rules = tuple(dict.fromkeys('/' + item.path.rstrip(' ') +
        '\\ ' * (len(item.path) - len(item.path.rstrip(' '))) +
        ('/' if item.subtree else '') for item in exclusions))
    result = subprocess.run(
        ['git', 'rev-parse', '--git-path', 'info/exclude'],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f'cannot locate git exclude file: {result.stderr.strip()}')
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = workspace / path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    missing = [rule for rule in rules if rule not in text.splitlines()]
    if missing:
        suffix = ('' if not text or text.endswith('\n') else '\n') + '\n'.join(missing) + '\n'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(suffix)
