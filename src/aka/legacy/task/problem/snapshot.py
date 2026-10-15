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
from aka.contracts.problem import AgentProblem, EvaluationInput, InitialSource, LoadedProblem


SCHEMA = "aka.problem.snapshot.v1"


def _relative(root: Path, path: str) -> Path:
    candidate = root / path
    if Path(path).is_absolute() or ".." in Path(path).parts or candidate.resolve() != candidate.absolute():
        raise ValueError(f"unsafe Problem snapshot path: {path}")
    return candidate


def restore(root: Path, origin: Path, mode: str) -> LoadedProblem:
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Problem snapshot manifest changed")
    manifest = json.loads(raw_manifest)
    if (manifest["schema"] != SCHEMA or manifest["origin"] != str(origin)
            or manifest["mode"] != mode):
        raise ValueError("Problem snapshot belongs to a different source or optimization mode")
    for name, evidence in manifest["files"].items():
        path = _relative(root, name)
        if (hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]
                or stat.S_IMODE(path.stat().st_mode) != evidence["mode"]):
            raise ValueError(f"Problem snapshot content changed: {name}")

    def path(name):
        if name is None:
            return None
        value = _relative(root, name)
        if not value.exists():
            raise ValueError(f"Problem snapshot role is missing: {name}")
        if value.is_file() and name not in manifest["files"]:
            raise ValueError(f"Problem role has no content evidence: {name}")
        return value

    def files(entries):
        return tuple(WorkspaceFile(item["path"], source=path(item["source"]), copy=item["copy"])
                     for item in entries)

    evaluation = EvaluationInput(manifest["adapter"], path(manifest["evaluation_directory"]),
                                 files(manifest["evaluation"]), path(manifest["runtime"]),
                                 path(manifest["private"]), tuple(manifest["case_ids"]))
    return LoadedProblem(manifest["name"], evaluation.directory, path(manifest["reference"]),
                         evaluation.runtime_root, path(manifest["public_contract"]),
                         manifest["public_contract_source"], AgentProblem(files(manifest["public"])),
                         InitialSource(manifest["source_kind"], path(manifest["source_entry"]),
                                       files(manifest["initial"])), evaluation, root)


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
                        runtime=runtime, case_ids=problem.evaluation.case_ids)
        if manifest["private"] is not None:
            from .layout import validate_private_shapes, validate_agent_problem
            captured_shapes = staging / "evaluation/shapes.json"
            manifest["case_ids"] = tuple(str(value) for value in validate_private_shapes(captured_shapes))
            if manifest["public_contract"]:
                validate_agent_problem(staging / manifest["public_contract"], private_shapes_path=captured_shapes)
        manifest["files"] = {p.relative_to(staging).as_posix():
                             dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  mode=stat.S_IMODE(p.stat().st_mode))
                             for p in staging.rglob("*") if p.is_file()}
        raw_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        (staging / "manifest.json").write_bytes(raw_manifest)
        (staging / "manifest.sha256").write_text(hashlib.sha256(raw_manifest).hexdigest() + "\n")
        try:
            staging.rename(root)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another capture published first. Validate that snapshot below
            # after discarding our staging directory.
    return restore(root, origin, mode)
