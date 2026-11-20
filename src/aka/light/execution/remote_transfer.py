"""Crash-resumable downloads, bound to an immutable remote file digest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .remote_resilience import Failure


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def cache_file(directory, digest):
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise Failure('invalid_artifact_digest')
    root = Path(directory) / 'transfer-cache'
    if root.is_symlink():
        raise Failure('invalid_transfer_cache')
    root.mkdir(exist_ok=True)
    return root / (digest + '.complete')


def store(directory, data):
    digest = hashlib.sha256(data).hexdigest()
    target = cache_file(directory, digest)
    if target.is_symlink():
        raise Failure('invalid_transfer_cache')
    if target.is_file() and target.stat().st_size == len(data) and file_hash(target) == digest:
        return digest
    temporary = target.with_suffix('.store-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return digest


def read(directory, name, size, digest, fetch, *, chunk_bytes=500000):
    """Fetch(offset,size) must use an idempotent RPC; failed chunks are not appended.

    A changed remote digest selects a different checkpoint. Corrupt checkpoints
    are discarded, never returned or mixed with the next completed result.
    """
    if type(size) is not int or size < 0:
        raise Failure('invalid_artifact_size')
    complete = cache_file(directory, digest)
    partial = complete.with_suffix('.part')
    if any(p.is_symlink() for p in [complete, partial]):
        raise Failure('invalid_transfer_cache')
    if complete.is_file():
        if complete.stat().st_size == size and file_hash(complete) == digest:
            return complete.read_bytes()
        complete.unlink()
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    # Checkpoints are advanced only after a complete chunk has been durably
    # appended. A process death during a write can leave a short last chunk.
    if offset != size and offset % chunk_bytes:
        offset -= offset % chunk_bytes
        with partial.open('r+b') as stream:
            stream.truncate(offset)
    with partial.open('ab') as stream:
        while offset < size:
            wanted = min(chunk_bytes, size - offset)
            data = fetch(offset, wanted)
            if not isinstance(data, bytes) or len(data) != wanted:
                raise Failure('incomplete_artifact_chunk', transient=True)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            offset += wanted
    if file_hash(partial) != digest:
        partial.unlink()
        raise Failure('artifact_checksum_mismatch', transient=True)
    partial.replace(complete)
    return complete.read_bytes()
