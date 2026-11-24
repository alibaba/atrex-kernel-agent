"""Read-only mechanism search over operator-configured reference source mounts.

Source hits are navigation aids, not admitted Wiki facts or evidence of applicability.
No shell, model call, or operator/architecture filtering is involved here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

EXTENSIONS = {'.py', '.cu', '.cuh', '.h', '.hpp', '.cpp', '.cc', '.c', '.ptx',
              '.md', '.rst', '.txt', '.mlir', '.td', '.inl', '.hip'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'build', 'dist', 'generated'}


def search_terms(intent):
    terms = []
    for mechanism in intent.get('mechanisms', []):
        terms.extend([mechanism['name'], *mechanism['aliases']])
    # A standard request may use the deterministic bridge and have no mechanism.
    if not terms:
        terms = intent.get('free_text_terms', []) + intent.get('operator_terms', [])
    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))


def pattern(term):
    # split-KV / split_kv / splitKV / split kv share a literal token pattern.
    tokens = re.findall(r'[^\W_]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', term), re.UNICODE)
    return re.compile(r'(?<![A-Za-z0-9])' + r'[\s_-]*'.join(map(re.escape, tokens))
                      + r'(?![A-Za-z0-9])', re.IGNORECASE) if tokens else None


def configured_roots(default_root):
    configured = os.environ.get('ATREX_WIKI_REFERENCE_PATHS')
    if configured is not None:
        # The Runtime sets this even when empty, overriding incidental local mounts.
        return {name: Path(path) for name, path in json.loads(configured).items()}
    if not default_root.is_dir():
        return {}
    return {p.name: p for p in sorted(default_root.iterdir()) if p.is_dir()}


def search(intent, default_root, *, limit=12, timeout=10.0):
    terms = search_terms(intent)
    patterns = [(term, pattern(term)) for term in terms]
    patterns = [(t, p) for t, p in patterns if p is not None]
    combined = re.compile('|'.join(p.pattern for _, p in patterns), re.IGNORECASE) if patterns else None
    primary = {m['name'] for m in intent.get('mechanisms', [])}
    roots = configured_roots(default_root)
    meta = {'terms': terms, 'status': 'complete', 'files_searched': 0,
            'files_skipped': 0, 'matches_omitted': 0,
            'interpretation': 'Source matches are unverified implementation references; check target applicability.'}
    result = {'source_search': meta, 'source_matches': []}
    if not roots or not patterns:
        meta['status'] = 'not_configured' if not roots else 'no_search_terms'
        return result
    deadline = time.monotonic() + timeout
    hits = []
    total = 0
    for name, configured in roots.items():
        root = configured.resolve()
        if not root.is_dir():
            meta['files_skipped'] += 1
            meta['status'] = 'partial'
            continue
        seen = set()
        for directory, dirs, files in os.walk(root, followlinks=True):
            if time.monotonic() >= deadline:
                meta['status'] = 'partial'
                meta['stop_reason'] = 'search_time_budget'
                break
            current = Path(directory)
            resolved = current.resolve()
            # Follow in-root links only, with cycle protection. Each configured
            # top-level mount can itself be a symlink to an external repository.
            if not resolved.is_relative_to(root) or resolved in seen:
                dirs[:] = []
                continue
            seen.add(resolved)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for filename in sorted(files):
                if time.monotonic() >= deadline:
                    meta['status'] = 'partial'
                    meta['stop_reason'] = 'search_time_budget'
                    break
                path = current / filename
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    if not path.resolve().is_relative_to(root):
                        meta['files_skipped'] += 1
                        continue
                    info = path.stat()
                    if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
                        meta['files_skipped'] += 1
                        continue
                    content = path.read_bytes()
                    if b'\0' in content:
                        continue
                    text = content.decode('utf-8')
                except (OSError, UnicodeError):
                    meta['files_skipped'] += 1
                    continue
                meta['files_searched'] += 1
                if not combined.search(text):
                    continue
                lines = text.splitlines()
                source_digest = hashlib.sha256(content).hexdigest()
                previous_end = -1
                file_hits = 0
                for index, line in enumerate(lines):
                    if not combined.search(line):
                        continue
                    matched = [t for t, p in patterns if p.search(line)]
                    if not matched or index < previous_end:
                        continue
                    start, end = max(0, index - 3), min(len(lines), index + 6)
                    previous_end = end
                    snippet = '\n'.join(lines[start:end])
                    total += 1
                    file_hits += 1
                    if file_hits > 2:
                        continue
                    hit = {'path': f'gpu-wiki/references/{name}/{path.relative_to(root).as_posix()}',
                           'start_line': start + 1, 'end_line': end, 'match_line': index + 1,
                           'matched_terms': matched, 'snippet': snippet[:6000],
                           'snippet_truncated': len(snippet) > 6000,
                           'source_sha256': source_digest}
                    # Keep memory bounded, but scan later files so early directory
                    # ordering does not automatically consume the result budget.
                    hits.append(hit)
                    hits.sort(key=lambda h: (-len(primary.intersection(h['matched_terms'])), -len(h['matched_terms']),
                        h['path'].endswith(('.md', '.rst', '.txt')), h['path'], h['match_line']))
                    del hits[limit:]
            if time.monotonic() >= deadline:
                break
    meta['matches_omitted'] = max(0, total - len(hits))
    if meta['files_skipped']:
        meta['status'] = 'partial'
    result['source_matches'] = hits
    return result
