"""Local archive transport and native Atrex-Bench evaluation.

Like the localhost gateway, this adapter runs trusted code as its own user.
Archives are checked before extraction; evaluation evidence comes from the
evaluator, including every native ABBA block.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import sys
import tarfile

MAX_BYTES = 24 * 1024 * 1024


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("source path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ValueError("source path must stay inside the source directory")
    return path


def validate_source(value):
    if isinstance(value, str) and value.strip():
        return len(value.encode())
    if not isinstance(value, dict) or set(value) != {"archive", "entry_point"}:
        raise ValueError("source must be Python text or an archive/entry_point object")
    relative_path(value["archive"])
    relative_path(value["entry_point"])
    if not value["entry_point"].endswith(".py"):
        raise ValueError("archive entry_point must be a Python file")
    return 0


class UploadStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, token, suffix):
        if not isinstance(token, str) or not re.fullmatch(r"lu_[0-9a-f]{32}", token):
            raise ValueError("invalid local upload reference")
        return self.root / (token + suffix)

    def reserve(self, payload):
        if not isinstance(payload, dict) or payload.get("kind") not in {"eval", "profile"}:
            raise ValueError("local uploads support Eval and profile source archives")
        files = payload.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 2:
            raise ValueError("expected one candidate archive and an optional baseline archive")
        names, total = set(), 0
        for item in files:
            relative_path(item["path"])
            size = item.get("bytes")
            if type(size) is not int or not 0 < size <= MAX_BYTES:
                raise ValueError("invalid source archive size")
            if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("invalid source archive digest")
            if item["path"] in names:
                raise ValueError("duplicate source archive path")
            names.add(item["path"])
            total += size
        if total > MAX_BYTES:
            raise ValueError("source archives exceed the 24 MiB limit")
        uploads = []
        for item in files:
            token = "lu_" + secrets.token_hex(16)
            metadata = dict(item, kind=payload["kind"], spec=payload.get("spec"))
            with self._path(token, ".json").open("x") as stream:
                os.chmod(stream.name, 0o600)
                json.dump(metadata, stream)
            uploads.append({"path": item["path"], "upload_ref": token})
        return {"job_id": "upload_" + secrets.token_hex(12), "uploads": uploads}

    def put(self, token, data):
        metadata = json.loads(self._path(token, ".json").read_text())
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError("source archive size or digest mismatch")
        path = self._path(token, ".data")
        try:
            with path.open("xb") as stream:
                os.chmod(stream.name, 0o600)
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("local upload is immutable") from None

    def read(self, token, name, kind, spec):
        metadata = json.loads(self._path(token, ".json").read_text())
        if (metadata["path"], metadata["kind"], metadata["spec"]) != (name, kind, spec):
            raise ValueError("source archive does not belong to this request")
        data = self._path(token, ".data").read_bytes()
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError("source archive size or digest mismatch")
        return data


def materialize_source(source, destination, request, uploads, kind):
    destination.mkdir(mode=0o700)
    if isinstance(source, str):
        target = destination / "kernel.py"
        target.write_text(source)
        return target
    matches = [item for item in request.get("oss_files", []) if item.get("path") == source["archive"]]
    if len(matches) != 1:
        raise ValueError("source archive requires exactly one upload reference")
    data = uploads.read(matches[0]["upload_ref"], source["archive"], kind, request["spec"])
    total, names = 0, set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        for member in bundle:
            path = relative_path(member.name)
            if not member.isfile() or path.as_posix() in names:
                raise ValueError("source archives must contain unique regular files")
            total += member.size
            if total > MAX_BYTES or len(names) >= 2000:
                raise ValueError("source archive exceeds extraction limits")
            names.add(path.as_posix())
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as stream:
                target.write_bytes(stream.read())
            target.chmod(0o600)
    target = destination.joinpath(*relative_path(source["entry_point"]).parts)
    if not target.is_file():
        raise ValueError("source archive entry point is missing")
    return target


def run(config_path, runner):
    """Run each complete ABBA block through the evaluator's native CLI."""
    config = json.loads(Path(config_path).read_text())
    root = Path(config.pop("local_output"))
    repeats = config.pop("local_abba_repeats", 0)
    blocks = []
    for index in range(repeats or 1):
        output = root / ("block-" + str(index))
        current = dict(config, output=str(output))
        path = root.parent / ("eval-config-" + str(index) + ".json")
        path.write_text(json.dumps(current))
        receipt = root.parent / ("eval-result-path-" + str(index) + ".txt")
        process = subprocess.run([sys.executable, str(runner), "--config", str(path),
                                  "--sdk-result-path-output", str(receipt)])
        if not receipt.is_file():
            raise RuntimeError(f"evaluator exited {process.returncode} without its result receipt")
        result_path = Path(receipt.read_text().strip()).resolve()
        result_path.relative_to(output.resolve())
        result = json.loads(result_path.read_text())
        if repeats:
            rows = result.get("abba", {}).get("runs", [])
            expected = [("baseline", "A", 0), ("candidate", "B", 0),
                        ("candidate", "B", 1), ("baseline", "A", 1)]
            if len(rows) != 4 or any(
                    (row.get("index"), row.get("revision"), row.get("label"), row.get("repeat"))
                    != (i, *identity) or not isinstance(row.get("result"), dict)
                    for i, (row, identity) in enumerate(zip(rows, expected))):
                raise RuntimeError("evaluator returned incomplete native ABBA evidence")
        blocks.append(result)
    # Keep the complete SDK evidence; never synthesize passing measurements.
    result = {"eval_mode": "abba", "abba": {"sdk_results": blocks}} if repeats else blocks[0]
    root.mkdir(exist_ok=True)
    (root / "local-result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
