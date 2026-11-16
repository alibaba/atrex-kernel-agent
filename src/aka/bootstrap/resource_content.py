"""Compare public retained content with Bootstrap's checked original identities."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from aka.contracts.agent_resources import (
    AgentResourceFile, ResourceDirectoryFact, ResourceRootFacts, ResourcesSnapshotFacts,
)
from aka.core.boot import token_table
from aka.core.declaration import declare
from aka.core.errors import CompositionError
from aka.core.keys import ServiceKey
from aka.core.loader import import_plugin
from aka.core.lock import package_dir, snapshot
from .recorded_launch import _record, inspect_recorded_launch


def _relative(value, *, dot=False):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(char) < 32 for char in value)):
        raise ValueError("invalid content member")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != value
            or (not dot and value == ".")):
        raise ValueError("invalid content member")
    return Path(value)


def _absolute(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("invalid content root")
    path = Path(value)
    if (not path.is_absolute() or path.as_posix() != str(value) or ".." in path.parts
            or "\\" in str(value) or any(ord(char) < 32 for char in str(value))):
        raise ValueError("invalid content root")
    return path


def _mode(value):
    if type(value) is not int or not 0 <= value <= 0o777:
        raise ValueError("invalid content mode")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid content digest")


def _file_map(value):
    if not isinstance(value, Mapping):
        raise ValueError("invalid content map")
    result = {}
    for name, digest in value.items():
        path = _relative(name, dot=True)
        _digest(digest)
        if path in result:
            raise ValueError("duplicate content member")
        result[path] = digest
    if any(parent in result for path in result for parent in path.parents):
        raise ValueError("conflicting content members")
    return result


def _facts(resources):
    if (not isinstance(resources, ResourcesSnapshotFacts)
            or not isinstance(resources.selection_lock, Mapping)
            or not isinstance(resources.roots, tuple)
            or not isinstance(resources.absent_optional_roots, tuple)):
        raise ValueError("invalid retained facts")
    _digest(resources.identity)
    roots = []
    for row in resources.roots:
        if not isinstance(row, ResourceRootFacts) or row.kind not in ("file", "directory"):
            raise ValueError("invalid retained root")
        root = _absolute(row.root)
        _mode(row.mode)
        if (not isinstance(row.files, tuple) or not isinstance(row.directories, tuple)
                or any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots)):
            raise ValueError("overlapping retained roots")
        roots.append(root)
        files, directories = {}, set()
        for entry in row.files:
            if not isinstance(entry, AgentResourceFile):
                raise ValueError("invalid retained file")
            member = _relative(entry.path, dot=row.kind == "file")
            _mode(entry.mode); _digest(entry.sha256)
            if member in files:
                raise ValueError("duplicate retained file")
            files[member] = entry.sha256
        for entry in row.directories:
            if not isinstance(entry, ResourceDirectoryFact):
                raise ValueError("invalid retained directory")
            member = _relative(entry.path)
            _mode(entry.mode)
            if member in directories or member in files:
                raise ValueError("conflicting retained directory")
            directories.add(member)
        if row.kind == "file":
            if set(files) != {Path(".")} or directories or row.files[0].mode != row.mode:
                raise ValueError("invalid retained file root")
        else:
            for member in (*files, *directories):
                if any(parent in files or (parent != Path(".") and parent not in directories)
                       for parent in member.parents):
                    raise ValueError("incomplete retained directories")
        if [entry.path for entry in row.files] != sorted(entry.path for entry in row.files):
            raise ValueError("unordered retained files")
        if [entry.path for entry in row.directories] != sorted(entry.path for entry in row.directories):
            raise ValueError("unordered retained directories")
    absent = tuple(_absolute(path) for path in resources.absent_optional_roots)
    if len(set(absent)) != len(absent):
        raise ValueError("duplicate absent roots")


def _visible(path, policy):
    # These are the existing proof-domain exclusions, not a new shared policy.
    if policy == "explicit":
        return True
    if ("__pycache__" in path.parts or path.name == ".DS_Store"
            or path.suffix in {".pyc", ".pyo"}):
        return False
    if policy == "data" and ".git" in path.parts:
        return False
    if policy == "dependency" and any(part.endswith((".dist-info", ".egg-info"))
                                       for part in path.parts):
        return False
    return True


@dataclass(frozen=True)
class _Domain:
    root: Path
    scope: Path
    files: Mapping[Path, str]
    policy: str
    file_only: bool = False

    def covers(self, row):
        return (row.root == self.scope and row.kind == "file" if self.file_only
                else row.root.is_relative_to(self.scope))

    def projection(self, row):
        expected = {}
        for member, digest in self.files.items():
            path = self.root / member
            if path.is_relative_to(row.root):
                relative = path.relative_to(row.root)
                if (row.kind == "file" and relative != Path(".")) or (
                        row.kind == "directory" and relative == Path(".")):
                    raise ValueError("incompatible recorded root kind")
                expected[relative] = digest
        observed = {Path(entry.path): entry.sha256 for entry in row.files
                    if _visible(row.root / entry.path, self.policy)}
        if not expected or expected != observed:
            raise ValueError("recorded content differs")
        return expected

    def check_partial(self, row):
        """Require overlapping known content to agree, without proving the root."""
        if not self.scope.is_relative_to(row.root):
            return
        expected = {self.root / member: digest for member, digest in self.files.items()
                    if (self.root / member).is_relative_to(self.scope)}
        observed = {row.root / entry.path: entry.sha256 for entry in row.files
                    if (row.root / entry.path).is_relative_to(self.scope)
                    and _visible(row.root / entry.path, self.policy)}
        if expected != observed:
            raise ValueError("overlapping recorded content differs")


def _data_domains(requirements):
    domains, pins = [], []
    for requirement in requirements:
        root = _absolute(requirement["root"])
        files = _file_map({name: value for name, value in requirement["identity"].items()
                           if not name.startswith("gitlink:")})
        pins.extend(root / _relative(name) for name in requirement["submodules"])
        scopes = [root / _relative(member, dot=True) for member in requirement["paths"]]
        if any(not any((root / member).is_relative_to(scope) for scope in scopes)
               for member in files):
            raise ValueError("content map is outside its declared scopes")
        for member in requirement["paths"]:
            scope = root / _relative(member, dot=True)
            if any(scope.is_relative_to(domain.scope) or domain.scope.is_relative_to(scope)
                   for domain in domains):
                raise ValueError("overlapping recorded data scopes")
            domains.append(_Domain(root, scope, files, "data", scope.relative_to(root) in files))
    return domains, pins


def _code_domain(module, files, *, policy, explicit=False):
    spec = importlib.util.find_spec(module)
    root = package_dir(module)
    if spec is None or root is None:
        raise ValueError("recorded package is unavailable")
    root = _absolute(root)
    if root.resolve(strict=True) != root:
        raise ValueError("package path is aliased")
    locations = tuple(spec.submodule_search_locations or ())
    if len(locations) > 1:
        raise ValueError("package path is ambiguous")
    if locations and _absolute(locations[0]) != root:
        raise ValueError("package path changed during inspection")
    mapping = _file_map(files)
    if explicit:
        return [_Domain(root, root / name, mapping, "explicit", True) for name in mapping]
    if not locations:
        origin = _absolute(spec.origin)
        if (origin.parent != root or origin.resolve(strict=True) != origin
                or set(mapping) != {Path(origin.name)}):
            raise ValueError("invalid module file scope")
        return [_Domain(root, origin, mapping, policy, True)]
    return [_Domain(root, root, mapping, policy)]


def _code_domains(value, launch):
    from .continuation import encode

    seams = token_table(launch.token_identity)
    declarations = {row.id: None if row.disabled else
                    declare(import_plugin(row.name), known_seams=seams).with_extra_inject(row.inject)
                    for row in launch.composition.rows}
    # Recheck current declarations/code before using their membership scopes.
    # This cannot activate recorded data, environment or availability callbacks.
    checked = replace(launch.composition, variables=dict(launch.composition.variables))
    current = snapshot(checked, declarations, launch.token_identity,
                       effective_configs={row.id: row.config for row in launch.composition.enabled})
    if encode(current) != encode(value["identity"]):
        raise ValueError("recorded code scope changed")
    domains = []
    for module, files in value["host"].items():
        domains.extend(_code_domain(module, files, policy="data"))
    for entry in value["identity"]["entries"]:
        declaration = declarations[entry["id"]]
        if declaration is None:
            continue
        if declaration.identity_files:
            primary = (declaration.module.rsplit(".", 1)[0]
                       if declaration.module.endswith(".plugin") else declaration.module)
        else:
            primary = declaration.module
        for module, files in entry["identity_files"].items():
            dependency = module in declaration.identity_packages
            domains.extend(_code_domain(module, files,
                policy="dependency" if dependency else "code",
                explicit=module == primary and bool(declaration.identity_files) and not dependency))
    return domains


def compare_recorded_resource_content(payload: str, resources: ResourcesSnapshotFacts, *,
                                      tokens: Sequence[ServiceKey] = ()) -> None:
    """Check original content correspondence, never selection binding or readiness.

    Matching installed packages must remain complete at their original Resources
    paths. Declared external data may be absent. Strict restore, runtime checks
    and all owner/checkpoint handoffs remain independent requirements.
    """
    try:
        launch = inspect_recorded_launch(payload, tokens=tokens)
        value, *_ = _record(payload)
        _facts(resources)
        data, pins = _data_domains(value["resources"])
        domains = [*data, *_code_domains(value, launch)]
        for row in resources.roots:
            if any(row.root.is_relative_to(pin) or pin.is_relative_to(row.root) for pin in pins):
                raise ValueError("Git proof is required")
            proofs = [domain.projection(row) for domain in domains if domain.covers(row)]
            if not proofs or any(proof != proofs[0] for proof in proofs[1:]):
                raise ValueError("complete content proof is unavailable")
            for domain in domains:
                if not domain.covers(row):
                    domain.check_partial(row)
    except Exception:
        raise CompositionError("recorded-resources", "recorded resource content cannot be verified") from None
