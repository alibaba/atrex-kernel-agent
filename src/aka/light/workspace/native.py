"""Editable CUDA manifests, packaged as one content-addressed candidate.

AKA's local source list is resolved only by the Supervisor. Agate receives its
standard oss_archive manifest with exact file and archive digests.
"""
from __future__ import annotations
import json
from pathlib import Path

POLICY = {"kind": "agate_cuda", "package_root": ".", "immutable_files": {},
          "runtime_requirements": [], "editable_roots": []}
SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx"}


def manifest(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or "schema_version" not in value:
        return None
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("solution.json requires schema_version 1")
    if not isinstance(value.get("entry_point"), dict) or value["entry_point"].get("type") != "cuda":
        raise ValueError("Versioned AKA manifests currently require a CUDA entry_point")
    if set(value) - {"schema_version", "name", "version", "source", "entry_point", "dependencies", "targets"}:
        raise ValueError("Unknown solution.json fields")
    if any(not isinstance(value.get(key), str) or not value[key] for key in ("name", "version")):
        raise ValueError("solution.json requires name and version strings")
    source = value.get("source")
    if not isinstance(source, dict) or set(source) != {"type", "files"} or source["type"] != "local":
        raise ValueError('AKA editable manifests require source={"type":"local","files":["kernel.cu", ...]}')
    names = source["files"]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(names) != len(set(names)):
        raise ValueError("source.files must be a nonempty list of distinct local paths")
    from aka.light.workspace.sources import relative, source_name
    for name in names:
        relative(name)
        if not source_name(name) or Path(name).suffix not in SUFFIXES or name.startswith(".agate_dependencies/"):
            raise ValueError(f"Not an editable native source path: {name}")
    build = value["entry_point"].get("build", {})
    if (not isinstance(build, dict) or build.get("toolchain") != "torch_extension" or not isinstance(build.get("sources"), list)
            or not build["sources"] or any(not isinstance(n, str) for n in build["sources"])
            or not set(build["sources"]) <= set(names)
            or not any(n.endswith(".cu") for n in build["sources"])):
        raise ValueError("CUDA build.sources must name declared sources including a .cu; toolchain is torch_extension")
    return value


def read(workspace):
    from aka.light.agent.session_tail import read_regular_bytes
    try:
        raw = read_regular_bytes(workspace / "solution.json", limit=16 * 1024 * 1024 + 1)
    except FileNotFoundError:
        return None
    return manifest(raw)


def validate(files):
    value = manifest(files["solution.json"])
    if value is None or set(files) != {"solution.json", *value["source"]["files"]}:
        raise ValueError("Native snapshot must contain exactly the manifest and its declared files")
    if len(files) > 2001 or sum(len(v) for k, v in files.items() if k != "solution.json") > 4 * 1024 * 1024:
        raise ValueError("Native source exceeds 2000 source files / 4 MiB")
    for data in files.values():
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Native source file exceeds 16 MiB")
        data.decode("utf-8")
    return files


def collect(workspace):
    from aka.light.agent.session_tail import read_regular_bytes
    raw = read_regular_bytes(workspace / "solution.json", limit=16 * 1024 * 1024 + 1)
    value = manifest(raw)
    if value is None:
        raise ValueError("CUDA candidate requires solution.json")
    return validate({"solution.json": raw, **{name: read_regular_bytes(workspace / name, limit=16 * 1024 * 1024 + 1)
                                            for name in value["source"]["files"]}})


def editable(name):
    from aka.light.workspace.sources import source_name, relative
    relative(name)
    return name == "solution.json" or source_name(name) and Path(name).suffix in SUFFIXES
