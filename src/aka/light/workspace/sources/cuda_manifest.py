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
    from .common import relative, source_name
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
    from .common import source_name, relative
    relative(name)
    return name == "solution.json" or source_name(name) and Path(name).suffix in SUFFIXES


class CudaManifest:
    def seed_metadata(self, workspace, framework, name, *, native_eval):pass
    def prepare_source(self, request, source_provider, *, options, state_root, native_eval):
        if request.initial.kind == 'sol' or not native_eval:
            raise ValueError('CUDA manifests require native Atrex-Bench Eval, not SOL/SSH')
        # V0 is an evaluator-owned reference input; the candidate starts at V1.
        return source_provider.prepare(request)
    name = 'cuda_manifest'
    default_policy = POLICY
    def inspect_policy(self, get):
        raw = get('solution.json')
        return (True, POLICY) if raw is not None and manifest(raw) is not None else (False, None)
    def validate_restore(self, previous, target):pass
    from .common import Layout
    layout = Layout(None, 'cuda', True, ('solution.json',), archived=True)
    def validate_policy(self, policy):
        if policy != POLICY:raise ValueError('Invalid CUDA manifest source contract')
        return policy
    def validate(self, files, policy):return validate(files)
    def collect(self, workspace, policy, *, validate=True):return collect(workspace)
    def from_git(self, workspace, revision, policy):
        from .common import git_blob,git_files
        value=manifest(git_blob(workspace,revision,'solution.json'))
        if value is None:raise ValueError('CUDA candidate requires solution.json')
        return validate(git_files(workspace,revision,('solution.json',*value['source']['files'])))
    def from_inputs(self, inputs, policy, prefix):
        value=manifest(inputs[prefix+'solution.json'])
        return validate({name:inputs[prefix+name] for name in ('solution.json',*value['source']['files'])})
    def editable(self, policy, name):return editable(name)
    def instructions(self, policy):
        return '\n## Native CUDA candidate\nEdit the manifest and its declared native sources. Read the atrex-gpu-gateway Skill for file organization, JSON configuration and measurement usage.\n'
    def metadata(self, policy):return {}
    def obsolete(self, workspace, policy):
        value=read(workspace)
        return {'solution.json',*value['source']['files']} if value else set()
    def install(self, workspace, files, policy):
        from .common import publish
        (workspace/'kernel.py').unlink(missing_ok=True)
        for name,data in files.items():publish(workspace,name,data)
    def execution_source(self, files, policy, archive):
        import hashlib
        from .common import archive_bytes,ExecutionSource
        value=manifest(files['solution.json'])
        contents={name:files[name].decode() for name in value['source']['files']}
        value['source']={'type':'oss_archive','archive':archive,
            'sha256':hashlib.sha256(archive_bytes(contents)).hexdigest(),
            'files':{name:hashlib.sha256(files[name]).hexdigest() for name in contents}}
        return ExecutionSource(value,{archive:contents})
    def declared_paths(self, workspace, policy):return set(collect(workspace))
    def evidence_paths(self, workspace, policy):
        from .common import source_name
        return {'solution.json'}|{p.relative_to(workspace).as_posix() for p in workspace.rglob('*')
            if p.is_file() and source_name(p.relative_to(workspace).as_posix()) and p.suffix in SUFFIXES}

    def typed_limitation(self, workspace, policy):
        collect(workspace)
        return None
    def language_hints(self, workspace):return []
