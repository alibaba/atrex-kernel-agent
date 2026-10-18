"""Persist explicit Problem roles; never infer roles from stored filenames."""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile

from aka.contracts.content import WorkspaceFile
from aka.contracts.problem import AgentProblem, EvaluationInput, InitialSource, LoadedProblem, ProblemSnapshot


SCHEMA = "aka.problem.snapshot.v2"


def _relative(root: Path, path: str) -> Path:
    candidate = root / path
    if Path(path).is_absolute() or ".." in Path(path).parts or candidate.resolve() != candidate.absolute():
        raise ValueError(f"unsafe Problem snapshot path: {path}")
    return candidate


def inspect(root: Path, *, identity: str | None = None) -> tuple[LoadedProblem, ProblemSnapshot]:
    """Validate all retained bytes and role mappings without modifying any path."""
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("unsafe or missing Problem snapshot directory")
    raw_manifest = (root / "manifest.json").read_bytes()
    manifest_digest = hashlib.sha256(raw_manifest).hexdigest()
    if manifest_digest != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Problem snapshot manifest changed")
    manifest = json.loads(raw_manifest)
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported Problem snapshot schema")
    evidence_files = manifest["files"]
    expected = set(evidence_files) | {"manifest.json", "manifest.sha256"}
    authored = root / "authored-contract.json"
    authored_digest = None
    authored_mode = None
    if authored.exists():
        record = json.loads(authored.read_bytes())
        if (set(record) != {"text", "sha256"}
                or hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["sha256"]):
            raise ValueError("saved authored public contract changed")
        authored_digest = hashlib.sha256(authored.read_bytes()).hexdigest()
        authored_mode = stat.S_IMODE(authored.stat().st_mode)
        expected.add("authored-contract.json")
    directories = manifest["directories"]
    for name, mode in directories.items():
        directory = _relative(root, name)
        if not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode) != mode:
            raise ValueError(f"Problem snapshot directory changed: {name}")
    actual = set()
    for entry in root.rglob("*"):
        name = entry.relative_to(root).as_posix()
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError(f"unsafe Problem snapshot entry: {name}")
        if entry.is_dir():
            if name not in directories:
                raise ValueError(f"untracked Problem snapshot directory: {name}")
        else:
            actual.add(name)
    if actual != expected:
        raise ValueError("Problem snapshot has missing or untracked files")
    for name, evidence in evidence_files.items():
        entry = _relative(root, name)
        if (hashlib.sha256(entry.read_bytes()).hexdigest() != evidence["sha256"]
                or stat.S_IMODE(entry.stat().st_mode) != evidence["mode"]):
            raise ValueError(f"Problem snapshot content changed: {name}")
    content_identity = hashlib.sha256(json.dumps(
        {"manifest": manifest_digest, "authored": authored_digest, "authored_mode": authored_mode}, sort_keys=True).encode()).hexdigest()
    if identity is not None and identity != content_identity:
        raise ValueError("Problem snapshot identity differs")

    def path(name):
        if name is None:
            return None
        value = _relative(root, name)
        if not value.exists():
            raise ValueError(f"Problem snapshot role is missing: {name}")
        if value.is_dir() and name not in directories:
            raise ValueError(f"Problem directory role has no content evidence: {name}")
        if value.is_file() and name not in manifest["files"]:
            raise ValueError(f"Problem role has no content evidence: {name}")
        return value

    def files(entries):
        for item in entries:
            _relative(root, item["path"])
        return tuple(WorkspaceFile(item["path"], source=path(item["source"]), copy=item["copy"])
                     for item in entries)

    def resource(name):
        record = manifest[name]
        return files([record])[0] if record is not None else None

    evaluation = EvaluationInput(manifest["adapter"], path(manifest["evaluation_directory"]),
                                 files(manifest["evaluation"]), path(manifest["runtime"]),
                                 path(manifest["private"]), tuple(manifest["case_ids"]),
                                 resource("harness"), resource("worker_requirements"))
    from .resources import validate_resources
    validate_resources(evaluation)
    problem = LoadedProblem(manifest["name"], evaluation.directory, path(manifest["reference"]),
                         evaluation.runtime_root, path(manifest["public_contract"]),
                         manifest["public_contract_source"], AgentProblem(files(manifest["public"])),
                         InitialSource(manifest["source_kind"], path(manifest["source_entry"]),
                                       files(manifest["initial"])), evaluation, root)
    return problem, ProblemSnapshot(root, content_identity)


def restore(root: Path, origin: Path, mode: str) -> LoadedProblem:
    problem, _ = inspect(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest["origin"] != str(origin) or manifest["mode"] != mode:
        raise ValueError("Problem snapshot belongs to a different source or optimization mode")
    return problem


def copy_snapshot(snapshot: ProblemSnapshot, destination: Path):
    inspect(snapshot.directory, identity=snapshot.identity)
    destination = Path(destination).absolute()
    if destination.resolve() != destination:
        raise ValueError("unsafe Problem snapshot destination")
    if destination.exists():
        return inspect(destination, identity=snapshot.identity)
    if snapshot.directory in destination.parents:
        raise ValueError("snapshot destination is inside its source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".problem-export-", dir=destination.parent) as raw:
        staging = Path(raw) / "snapshot"
        shutil.copytree(snapshot.directory, staging)
        inspect(staging, identity=snapshot.identity)
        try:
            staging.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
    return inspect(destination, identity=snapshot.identity)


def capture(problem: LoadedProblem, root: Path, origin: Path, mode: str) -> LoadedProblem:
    """Publish atomically. Existing snapshots are restored, never overwritten."""
    root = root.resolve()
    if root.exists():
        return restore(root, origin, mode)
    if problem.agent is None or problem.initial_source is None or problem.evaluation is None:
        raise TypeError("Problem snapshots require explicit Agent, initial source and evaluation roles")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".problem-", dir=root.parent) as raw:
        staging = Path(raw)

        def copy_files(entries, prefix):
            if len({item.path for item in entries}) != len(entries):
                raise ValueError("duplicate Problem destination paths")
            result = []
            for item in entries:
                _relative(staging, item.path)
                target = _relative(staging, f"{prefix}/{item.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.source is not None:
                    shutil.copy2(item.source, target)
                else:
                    target.write_text(item.text, encoding="utf-8")
                result.append(dict(path=item.path, source=target.relative_to(staging).as_posix(), copy=item.copy))
            return result

        public = copy_files(problem.agent.files, "public")
        initial = copy_files(problem.initial_source.files, "initial")
        evaluation = copy_files(problem.evaluation.files, "evaluation")
        (staging / "evaluation").mkdir(exist_ok=True)
        from .resources import validate_resources
        validate_resources(problem.evaluation)
        harness = (copy_files((problem.evaluation.harness,), "resources/harness")[0]
                   if problem.evaluation.harness is not None else None)
        requirements = copy_files((problem.evaluation.worker_requirements,), "resources/worker")[0]
        runtime = None
        if problem.evaluation.runtime_root is not None:
            runtime = "runtime"
            source = problem.evaluation.runtime_root
            (staging / runtime / "scripts").mkdir(parents=True)
            shutil.copy2(source / "scripts/run_eval.py", staging / runtime / "scripts/run_eval.py")
            shutil.copytree(source / "src/atrex_bench", staging / runtime / "src/atrex_bench",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

        def role_path(source, entries, stored):
            for item, record in zip(entries, stored):
                if item.source == source:
                    return record["source"]
            raise ValueError(f"Problem role is absent from its declared files: {source}")

        manifest = dict(schema=SCHEMA, origin=str(origin), mode=mode, name=problem.name,
                        reference=role_path(problem.reference, problem.agent.files, public),
                        public_contract=(role_path(problem.public_contract, problem.agent.files, public)
                                         if problem.public_contract else None),
                        public_contract_source=problem.public_contract_source,
                        public=public, initial=initial, evaluation=evaluation,
                        source_kind=problem.initial_source.kind,
                        source_entry=role_path(problem.initial_source.entry, problem.initial_source.files, initial),
                        evaluation_directory="evaluation", adapter=problem.evaluation.adapter,
                        private="evaluation" if problem.evaluation.private_reference_dir else None,
                        runtime=runtime, case_ids=problem.evaluation.case_ids,
                        harness=harness, worker_requirements=requirements)
        if manifest["private"] is not None:
            from .layout import validate_private_shapes, validate_agent_problem
            captured_shapes = staging / "evaluation/shapes.json"
            manifest["case_ids"] = tuple(str(value) for value in validate_private_shapes(captured_shapes))
            if manifest["public_contract"]:
                validate_agent_problem(staging / manifest["public_contract"], private_shapes_path=captured_shapes)
        manifest["directories"] = {p.relative_to(staging).as_posix(): stat.S_IMODE(p.stat().st_mode)
                                   for p in staging.rglob("*") if p.is_dir()}
        manifest["files"] = {p.relative_to(staging).as_posix():
                             dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  mode=stat.S_IMODE(p.stat().st_mode))
                             for p in staging.rglob("*") if p.is_file()}
        raw_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        (staging / "manifest.json").write_bytes(raw_manifest)
        (staging / "manifest.sha256").write_text(hashlib.sha256(raw_manifest).hexdigest() + "\n")
        inspect(staging)
        try:
            staging.rename(root)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another capture published first. Validate that snapshot below
            # after discarding our staging directory.
    return restore(root, origin, mode)
