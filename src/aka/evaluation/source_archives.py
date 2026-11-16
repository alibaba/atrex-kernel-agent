"""Freeze complete public source content into deterministic native archives."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

from aka.contracts.evaluation import MeasurementArtifact
from aka.contracts.execution import PreparedSourceArchive
from aka.contracts.workspace import RevisionContent


def _name(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or path == PurePosixPath(".") or "\\" in name
            or any(ord(char) < 32 for char in name)):
        raise ValueError("source archive member must be a safe relative path")
    return name


def source_artifact(source):
    if not isinstance(source, RevisionContent) or not any(isinstance(source.files.get(name), bytes) for name in ('kernel.py', 'solution.json')):
        raise ValueError("candidate source must contain a Python entry point or native manifest")
    if set(source.files) == {"kernel.py"}:
        return MeasurementArtifact("candidate.py", source.files["kernel.py"], "candidate-source", "text/x-python"), "raw-kernel/1"
    value = {"schema_version": 1, "files": {_name(name): None if data is None else base64.b64encode(data).decode()
        for name, data in sorted(source.files.items())}}
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return MeasurementArtifact("candidate-source.json", content, "candidate-source"), "source-map/1"


def prepare_source(source, *, role="candidate"):
    artifact, format_name = source_artifact(source)
    if format_name == "raw-kernel/1":
        return source.files["kernel.py"].decode(), None, artifact, format_name
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, content in sorted(source.files.items()):
                if content is None:
                    continue
                member = tarfile.TarInfo(_name(name))
                member.size, member.mode = len(content), 0o644
                archive.addfile(member, io.BytesIO(content))
    native = source.files.get('kernel.py') is None
    packed = PreparedSourceArchive(role, f"archives/{role}.tar.gz", "solution.json" if native else "kernel.py", output.getvalue())
    if native:
        manifest = json.loads(source.files['solution.json'])
        if (manifest.get('schema_version') != 1 or manifest.get('entry_point', {}).get('type') != 'cuda'
                or manifest.get('source', {}).get('type') != 'local'
                or set(manifest['source']['files']) != set(source.files) - {'solution.json'}):
            raise ValueError('native manifest must declare exactly its retained sources')
        manifest['source'] = {'type': 'oss_archive', 'archive': packed.path,
                              'sha256': hashlib.sha256(packed.content).hexdigest(),
                              'files': {name: hashlib.sha256(data).hexdigest() for name, data in source.files.items()}}
        return manifest, packed, artifact, format_name
    return {"archive": packed.path, "entry_point": packed.entry_point}, packed, artifact, format_name


def archive_source(archive):
    """Read the Evaluation-owned prepared archive, never external archive URLs."""
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as packed:
        for member in packed.getmembers():
            name = _name(member.name)
            if not member.isfile() or name in files or member.mode != 0o644:
                raise ValueError("prepared source archive contains an unsupported member")
            stream = packed.extractfile(member)
            if stream is None:
                raise ValueError("prepared source archive member is unavailable")
            files[name] = stream.read()
    if not files or archive.entry_point not in files:
        raise ValueError("prepared source archive has no declared entry point")
    return RevisionContent(files)


def source_parameters(source, *, identity=None):
    artifact, format_name = source_artifact(source)
    checksum = hashlib.sha256(artifact.content).hexdigest()
    value = {"candidate_source_sha256": checksum, "candidate_source_artifact_sha256": checksum,
             "candidate_source_format": format_name}
    if identity is not None:
        value["candidate_source_identity"] = identity
    return value


def request_sources(request, archives):
    """Validate exact descriptor bindings and expose frozen measurement source facts."""
    selected = {item.role: item for item in archives}
    candidate = request.get("candidate")
    parameters = {}
    artifact = None
    for role, value in (("candidate", candidate), ("baseline", request.get("abba", {}).get("baseline"))):
        archive = selected.get(role)
        if isinstance(value, dict):
            from aka.execution.gateway.source_archives import archive_matches
            if archive is None or not archive_matches(value, archive):
                raise ValueError("prepared source descriptor differs from its archive binding")
            source = archive_source(archive)
            if value.get("schema_version") == 1:
                expected, _, _, _ = prepare_source(source, role=role)
                if value != expected:
                    raise ValueError("native manifest differs from its retained source content")
        elif archive is not None:
            raise ValueError("prepared source archive lacks its descriptor")
        elif isinstance(value, str):
            source = RevisionContent({"kernel.py": value.encode()})
        elif value is None and role == "baseline":
            continue
        else:
            raise ValueError("prepared request lacks candidate source")
        facts = source_parameters(source)
        if role == "candidate":
            parameters.update(facts)
            artifact, _ = source_artifact(source)
        else:
            parameters["baseline_source_sha256"] = facts["candidate_source_sha256"]
        if archive is not None:
            parameters[role + "_source_archive_sha256"] = hashlib.sha256(archive.content).hexdigest()
    if set(selected) - ({"candidate", "baseline"} if request.get("abba") else {"candidate"}):
        raise ValueError("prepared request contains an unrelated source archive")
    return parameters, artifact
