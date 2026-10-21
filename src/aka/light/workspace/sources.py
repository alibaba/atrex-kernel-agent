"""Whole-Kernel identities and task-owned source boundaries.

Single-file Kernels keep their original bytes/IDs. Tree Kernels use canonical
UTF-8 file maps (including the fixed adapter), never a generated import shim.
The controller pins the policy in Git; Agent copies are documentation only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time

from aka.light.orchestrator.session_tail import read_regular_bytes
from aka.light.supervisor.workspace import publish

POLICY = ".atrex_source.json"
MAGIC = b"ATREX-SOURCE-TREE-v1\n"
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_SNAPSHOT = 6 * MAX_TOTAL
CACHES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
PROTECTED = {"atrex_bench", "torch", "cutlass", "cuda", "triton", "site", "sitecustomize",
             "usercustomize", "importlib", "packaging", "json", "subprocess", "os", "sys"}
# Controller/public inputs and runtime state are not Candidate source files.
WORKSPACE_ONLY = {"tools", "skills", "scratch", "memory", "input", "profiles", "eval", "plans",
    "reference", "reference-projects", "gpu-wiki", "atrex-bench", "verification_artifacts",
    "README.md", "CLAUDE.md", "WORKSPACE.md", "agent_problem.json", "reference.py", "input.py",
    "shapes.json", "metadata.json", "roofline.json", "valid.py", "definition.json", "workload.jsonl",
    "solution.json", "config.json", "test_kernel.py", "profile_driver.py", "evaluations.jsonl",
    "numerical_probe.py", "probe-request.json", "policy_review_request.json",
    ".gitignore", ".claude", ".codex", ".qoder", ".agents", ".pi", ".orchestrator_mode.json",
    ".gpu_wiki_profile", ".atrex_long_horizon", ".atrex_versions", ".atrex_plugins",
    ".atrex_stall_state.json", ".orchestrator_state.json", "framework_baseline.json",
    "baseline_report.md", "trace-retention-manifest.json", ".atrex_environment", ".qodersec", ".venv", "node_modules", POLICY}


def relative(value: str, *, dot=False) -> str:
    if dot and value == ".":
        return value
    if (not isinstance(value, str) or not value or value.startswith(("/", "-"))
            or any(c in value for c in ("\\", "\x00", "\n", "\r"))
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"Unsafe source path: {value!r}")
    return value


def validate_import_options(options):
    if (not isinstance(options, dict)
            or set(options) not in ({"manifest", "repository"}, {"manifest", "directory"})
            or any(not isinstance(value, str) or not value for value in options.values())):
        raise ValueError("source_tree requires manifest and exactly one of repository or directory")
    return options


def ignored(name):
    return any(p in CACHES for p in Path(name).parts) or Path(name).suffix in {".pyc", ".pyo"}


def source_name(name):
    first = name.split("/")[0]
    return not (first in WORKSPACE_ONLY or ignored(name))


def editable(policy, name):
    return any(name == root or name.startswith(root + "/") for root in policy["editable_roots"])


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {
            "schema_version", "source_revision", "seed_digest", "package_root", "editable_roots",
            "immutable_files", "runtime_requirements"} or policy["schema_version"] != 1:
        raise ValueError("Invalid source contract")
    relative(policy["package_root"], dot=True)
    revision = policy["source_revision"]
    seed = policy["seed_digest"]
    valid_revision = revision is None or (isinstance(revision, str) and len(revision) in {40, 64}
                                         and all(c in "0123456789abcdef" for c in revision))
    if (not valid_revision
            or not isinstance(seed, str) or not seed.startswith("sha256:") or len(seed) != 71
            or any(c not in "0123456789abcdef" for c in seed[7:])):
        raise ValueError("Source contract needs a seed digest and an exact Git revision or null for a directory snapshot")
    if not isinstance(policy["editable_roots"], list) or not policy["editable_roots"]:
        raise ValueError("Source contract requires editable_roots")
    for root in policy["editable_roots"]:
        relative(root)
        if not source_name(root) or root == "kernel.py":
            raise ValueError("Editable root collides with workspace infrastructure or the fixed adapter")
    if not isinstance(policy["immutable_files"], dict) or "kernel.py" not in policy["immutable_files"]:
        raise ValueError("Source contract must pin its kernel.py adapter")
    for name, digest in policy["immutable_files"].items():
        relative(name)
        if not source_name(name) or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid fixed source identity or reserved workspace path")
    requirements = policy["runtime_requirements"]
    if not isinstance(requirements, list):
        raise ValueError("runtime_requirements must be a list")
    for item in requirements:
        if (not isinstance(item, dict) or set(item) - {"distribution", "import", "version"}
                or not item.get("distribution") or not item.get("import")
                or any(not isinstance(v, str) or "\n" in v or "\r" in v for v in item.values())):
            raise ValueError("Invalid runtime requirement")
    return policy


def policy_at(workspace: Path, *, committed=False):
    if committed:
        result = subprocess.run(["git", "show", f"HEAD:{POLICY}"], cwd=workspace, capture_output=True)
        if result.returncode:
            if (workspace / POLICY).exists():
                raise ValueError("Source policy is not pinned in the controller commit")
            return None
        raw = result.stdout
    else:
        try:
            raw = read_regular_bytes(workspace / POLICY, limit=MAX_FILE + 1)
        except FileNotFoundError:
            return None
    if len(raw) > MAX_FILE:
        raise ValueError("Source policy is too large")
    return validate_policy(json.loads(raw))


def validate_files(files, policy):
    validate_policy(policy)
    if len(files) > 4096 or sum(len(data) for data in files.values()) > MAX_TOTAL:
        raise ValueError("Source tree exceeds 4096 files / 64 MiB")
    for name, data in files.items():
        relative(name)
        if not source_name(name):
            raise ValueError(f"Source path collides with workspace infrastructure: {name}")
        if len(data) > MAX_FILE:
            raise ValueError(f"Source file exceeds 16 MiB: {name}")
        data.decode("utf-8")
        package = Path(name)
        if policy["package_root"] != "." and package.is_relative_to(policy["package_root"]):
            package = package.relative_to(policy["package_root"])
        if package.parts[0].removesuffix(".py") in PROTECTED:
            raise ValueError(f"Source shadows a protected import: {name}")
        expected = policy["immutable_files"].get(name)
        if expected is not None:
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Fixed source file was modified: {name}")
        elif not editable(policy, name):
            raise ValueError(f"Source file is outside editable_roots: {name}")
        elif Path(name).suffix in {".so", ".a", ".o", ".cubin", ".whl", ".pth"}:
            raise ValueError(f"Prebuilt/loading artifact is not permitted: {name}")
    missing = policy["immutable_files"].keys() - files.keys()
    if missing:
        raise ValueError("Fixed source files were removed: " + ", ".join(sorted(missing)))
    return files


def pack(files, policy):
    validate_files(files, policy)
    return MAGIC + json.dumps({"contract": policy, "files": {k: v.decode("utf-8") for k, v in sorted(files.items())}},
                              sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def from_inputs(inputs, *, prefix=""):
    """Use the already frozen measurement bytes, never reread a mutable draft."""
    policy = inputs.get(prefix + POLICY)
    if policy is None:
        return inputs[prefix + "kernel.py"]
    return pack({name[len(prefix):]: data for name, data in inputs.items()
                 if name.startswith(prefix) and source_name(name[len(prefix):])},
                json.loads(policy))


def unpack(source):
    if not source.startswith(MAGIC):
        return None, {"kernel.py": source}
    if len(source) > MAX_SNAPSHOT:
        raise ValueError("Source snapshot exceeds the bounded limit")
    value = json.loads(source[len(MAGIC):])
    files = {name: data.encode() for name, data in value["files"].items()}
    validate_files(files, value["contract"])
    if pack(files, value["contract"]) != source:
        raise ValueError("Source snapshot is not canonical")
    return value["contract"], files


def collect(workspace, policy, *, validate=True):
    files = {}
    for parent, dirs, names in os.walk(workspace, followlinks=False):
        rel = Path(parent).relative_to(workspace)
        for name in (*dirs, *names):
            key = (rel / name).as_posix()
            if (key.split("/")[0] not in WORKSPACE_ONLY | {".git"}
                    and (Path(parent) / name).is_symlink()):
                raise ValueError("Source tree cannot contain symlinks")
        dirs[:] = [n for n in dirs if source_name((rel / n).as_posix())]
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Source tree cannot contain symlink directories")
        for name in names:
            relative_name = (rel / name).as_posix()
            if not source_name(relative_name):
                continue
            files[relative_name] = read_regular_bytes(Path(parent) / name, limit=MAX_FILE + 1)
            if len(files) > 4096 or sum(map(len, files.values())) > MAX_TOTAL:
                raise ValueError("Source tree exceeds 4096 files / 64 MiB")
    return validate_files(files, policy) if validate else files


_DISCOVER = object()


def snapshot(workspace, *, policy=_DISCOVER):
    policy = policy_at(workspace) if policy is _DISCOVER else policy
    if policy is None:
        if (workspace / POLICY).exists() or (workspace / POLICY).is_symlink():
            raise ValueError("This single-file task cannot introduce a source-tree policy")
        return read_regular_bytes(workspace / "kernel.py", limit=MAX_FILE + 1)
    return pack(collect(workspace, policy), policy)


def from_git(workspace, revision):
    from aka.light.long_horizon.git_episode import git_blob
    result = subprocess.run(["git", "show", f"{revision}:{POLICY}"], cwd=workspace, capture_output=True)
    if result.returncode:
        return git_blob(workspace, revision, "kernel.py")
    policy = validate_policy(json.loads(result.stdout))
    listing = subprocess.run(["git", "ls-tree", "-r", "-z", revision], cwd=workspace, capture_output=True, check=True).stdout
    files = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        name = name.decode()
        if source_name(name):
            if meta.split()[0] not in {b"100644", b"100755"}:
                raise ValueError("Committed source contains a link or special object")
            files[name] = git_blob(workspace, revision, name)
    return pack(files, policy)


def install(workspace, source, *, replace=True):
    policy, files = unpack(source)
    if policy is not None:
        # Only remove prior source files after a complete new snapshot validates.
        # Keep a recoverable private copy at the caller's transaction boundary.
        previous = policy_at(workspace)
        if previous is not None and previous != policy:
            raise ValueError("Source contract cannot change within a Campaign")
        if replace and previous:
            # Recovery may be replacing a partially published tree. Validate
            # the new snapshot, not the completeness of the obsolete one.
            for name in collect(workspace, previous, validate=False).keys() - files.keys():
                from aka.light.orchestrator.agent_home import open_private_directory
                with open_private_directory((workspace / name).parent) as parent:
                    os.unlink(Path(name).name, dir_fd=parent)
        publish(workspace, POLICY, (json.dumps(policy, sort_keys=True) + "\n").encode())
    for name, data in files.items():
        publish(workspace, name, data)


def instructions(policy):
    if policy is None:
        return ""
    return ("\n## Kernel source tree\nThe fixed kernel.py adapter and bundled source form one Kernel. "
            "Edit/add/delete source only inside editable_roots: " + json.dumps(policy["editable_roots"]) +
            ". All other imported files are fixed; do not change, remove or rename them. "
            "Do not rewrite the adapter or replace the package with a single-file implementation. "
            "Keep notes/tools and temporary builds in tools/, skills/ and scratch/. "
            "Evaluate, Profile, Check, Disassemble, sealed reports and broadcasts bind the entire tree. "
            "kernel-read writes a directory for tree Kernels; pass that directory to --baseline-path.\n")


def allowed_changes(paths, policy):
    return all(editable(policy, name) and name not in policy["immutable_files"] for name in paths)


def commit_tree_candidate(handle, source):
    """Seal measured additions/modifications/deletions with a private Git index."""
    from aka.light.long_horizon.git_episode import (
        git_head, git_text, changed_paths, TIMELINE_PROBE_MARKERS, protected_violation, working_changes,
    )
    import tempfile
    policy, files = unpack(source)
    baseline_policy, _ = unpack(from_git(handle.path, handle.base_commit))
    if policy != baseline_policy:
        raise ValueError("Candidate source contract differs from its Episode baseline")
    if any(marker.encode() in data for data in files.values() for marker in TIMELINE_PROBE_MARKERS):
        raise ValueError("Candidate source still contains timeline profiling probes")
    head = git_head(handle.path)
    violation = protected_violation(working_changes(handle.path))
    if violation:
        raise ValueError(violation)
    if git_text(handle.path, "symbolic-ref", "--quiet", "--short", "HEAD") != handle.branch:
        raise RuntimeError("Candidate branch differs from its Episode")
    subprocess.run(["git", "merge-base", "--is-ancestor", handle.base_commit, head], cwd=handle.path, check=True)
    staged = [p for p in git_text(handle.path, "diff", "--cached", "--name-only", "--no-renames", "-z").split("\0") if p]
    if not allowed_changes(changed_paths(handle.path, handle.base_commit, head) + staged, policy):
        raise ValueError("Candidate may change only declared editable source paths")
    previous = from_git(handle.path, head)
    _, old_files = unpack(previous)
    if source != previous:
        with tempfile.TemporaryDirectory(prefix="aka-tree-index-") as temporary:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"),
                       GIT_AUTHOR_NAME="AKA Supervisor", GIT_COMMITTER_NAME="AKA Supervisor",
                       GIT_AUTHOR_EMAIL="supervisor@atrex.local", GIT_COMMITTER_EMAIL="supervisor@atrex.local")
            def run(*args, data=None):
                return subprocess.run(["git", *args], cwd=handle.path, env=env, input=data,
                                      capture_output=True, check=True).stdout.decode().strip()
            run("read-tree", head)
            for name in old_files.keys() - files.keys():
                run("update-index", "--force-remove", "--", name)
            for name, data in files.items():
                if old_files.get(name) != data:
                    blob = run("hash-object", "-w", "--stdin", data=data)
                    run("update-index", "--add", "--cacheinfo", "100644", blob, name)
            tree = run("write-tree")
            commit = run("-c", "commit.gpgSign=false", "commit-tree", tree, "-p", head,
                         "-m", f"Episode {handle.episode}: selected source tree")
            run("update-ref", "HEAD", commit, head)
            head = commit
    install(handle.path, source)
    subprocess.run(["git", "read-tree", head], cwd=handle.path, check=True)
    return head


def _git_sources(repository, revision, paths):
    if not isinstance(revision, str) or len(revision) not in {40, 64} or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Source revision must be an exact Git commit hash")
    kind = subprocess.run(["git", "cat-file", "-t", revision], cwd=repository,
                          capture_output=True, timeout=30)
    if kind.returncode or kind.stdout.strip() != b"commit":
        raise ValueError("Source revision must identify an existing Git commit")
    # Bound disk usage and elapsed time even if git never closes stdout.
    import tempfile
    with tempfile.TemporaryFile() as archive:
        process = subprocess.Popen(["git", "archive", "--format=tar", revision, "--", *paths], cwd=repository, stdout=archive, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None:
                if os.fstat(archive.fileno()).st_size > 2 * MAX_TOTAL:
                    raise ValueError("Source archive exceeds its bounded limit")
                if time.monotonic() >= deadline:
                    raise ValueError("Pinned source archive timed out")
                time.sleep(0.02)
            if process.returncode:
                raise ValueError("Cannot archive the pinned source revision")
            if os.fstat(archive.fileno()).st_size > 2 * MAX_TOTAL:
                raise ValueError("Source archive exceeds its bounded limit")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        archive.seek(0)
        files = {}
        with tarfile.open(fileobj=archive) as tar:
            for member in tar:
                if member.isdir():
                    relative(member.name.rstrip("/"))
                    continue
                name = relative(member.name)
                if not member.isfile() or name in files or member.size > MAX_FILE:
                    raise ValueError("Source archive contains an unsafe/duplicate/oversized entry")
                data = tar.extractfile(member).read(MAX_FILE + 1)
                if not ignored(name):
                    files[name] = data
                if len(files) > 4096 or sum(map(len, files.values())) > MAX_TOTAL:
                    raise ValueError("Source archive exceeds 4096 files / 64 MiB")
    return files


def _directory_sources(directory, paths):
    """Read selected local bytes without Git, links, special files or silent omissions."""
    directory = Path(directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Source directory must be a directory")
    files, total, entries = {}, 0, 0

    def add(path):
        nonlocal total
        name = path.relative_to(directory).as_posix()
        if name in files or ignored(name):
            return
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source entry must be a regular non-symlink file: {name}")
        if before.st_size > MAX_FILE:
            raise ValueError(f"Source file exceeds 16 MiB: {name}")
        data = read_regular_bytes(path, limit=MAX_FILE + 1)
        if len(data) > MAX_FILE:
            raise ValueError(f"Source file exceeds 16 MiB: {name}")
        after = path.lstat()
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        if identity(before) != identity(after) or len(data) != after.st_size:
            raise ValueError(f"Source file changed while taking the initial snapshot: {name}")
        total += len(data)
        files[name] = data
        if len(files) > 4096 or total > MAX_TOTAL:
            raise ValueError("Source directory exceeds 4096 files / 64 MiB")

    def failed(error):
        raise error

    for selected in paths:
        path = directory / selected
        # Validate every selected component, including directories that might
        # otherwise resolve through a link before os.walk starts.
        for parent in (path, *path.parents):
            if parent == directory:
                break
            if parent.is_symlink():
                raise ValueError(f"Source tree cannot contain symlinks: {selected}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            for current, dirs, names in os.walk(path, followlinks=False, onerror=failed):
                entries += len(dirs) + len(names) + 1
                if entries > 16384:
                    raise ValueError("Source directory exceeds its traversal limit")
                for name in dirs:
                    if (Path(current) / name).is_symlink():
                        raise ValueError("Source tree cannot contain symlink directories")
                dirs[:] = [name for name in dirs if not ignored((Path(current) / name).relative_to(directory).as_posix())]
                for name in names:
                    add(Path(current) / name)
        else:
            add(path)
    return files


def import_tree(manifest_path, repository=None, *, directory=None):
    if (repository is None) == (directory is None):
        raise ValueError("Choose exactly one source repository or directory")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(read_regular_bytes(manifest_path, limit=MAX_FILE))
    if manifest.get("schema_version", 1) != 1 or manifest.get("runtime_support") or manifest.get("repository_search", {}).get("mode", "snapshot") != "snapshot":
        raise ValueError("Only snapshot source manifests without runtime_support are supported")
    spec = manifest["source"]
    paths = [relative(p) for p in spec["archive_paths"]]
    if not paths:
        raise ValueError("Source archive_paths cannot be empty")
    # Directory mode intentionally snapshots working bytes, including dirty and
    # untracked files. A manifest revision cannot describe these bytes.
    revision = spec.get("revision") if repository is not None else None
    files = (_git_sources(Path(repository).expanduser().resolve(), revision, paths)
             if repository is not None else _directory_sources(directory, paths))
    if "kernel.py" in files:
        raise ValueError("Imported source collides with the fixed adapter")
    files["kernel.py"] = read_regular_bytes(manifest_path.parent / relative(manifest["adapter"]), limit=MAX_FILE + 1)
    roots = [relative(p) for p in manifest["editable_roots"]]
    for root in roots:
        if not any(name.startswith(root + "/") for name in files):
            raise ValueError("editable_roots must name imported source directories")
    policy = {"schema_version": 1, "source_revision": revision, "seed_digest": "sha256:" + hashlib.sha256(
        json.dumps({k: v.decode() for k, v in sorted(files.items())}, sort_keys=True).encode()).hexdigest(),
        "package_root": relative(spec.get("package_root", "."), dot=True), "editable_roots": roots,
        "immutable_files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()
                            if not any(k.startswith(p + "/") for p in roots)},
        "runtime_requirements": manifest.get("runtime_requirements", [])}
    if policy["package_root"] != ".":
        raise ValueError("Native Eval archives require package_root='.', matching Runtime; no Dev fallback")
    return pack(files, policy)


def initial_snapshot(workspace, options):
    """Seal once before V0; failed bootstrap/restarts never reread mutable inputs."""
    import fcntl
    from aka.light.orchestrator.agent_home import open_private_directory
    from aka.light.orchestrator.durable_state import durable_write_json
    from aka.light.runtime.paths import supervisor_campaign_root
    validate_import_options(options)
    configuration = {key: os.path.abspath(Path(value).expanduser()) for key, value in options.items()}
    root = supervisor_campaign_root(workspace) / "initial-source"
    path = root / "snapshot.json"
    with open_private_directory(root) as parent:
        try:
            lock = os.open("lock", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        except FileExistsError:
            lock = os.open("lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=parent)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                raw = read_regular_bytes(path, limit=MAX_SNAPSHOT + MAX_FILE + 1)
            except FileNotFoundError:
                source = import_tree(configuration["manifest"], configuration.get("repository"),
                                     directory=configuration.get("directory"))
                value = json.loads(source[len(MAGIC):])
                durable_write_json(path, {"schema_version": 1, "configuration": configuration,
                    "snapshot_digest": hashlib.sha256(source).hexdigest(), **value})
                return source
            if len(raw) > MAX_SNAPSHOT + MAX_FILE:
                raise ValueError("Initial source snapshot exceeds its size limit")
            value = json.loads(raw)
            if value.get("schema_version") != 1 or value.get("configuration") != configuration:
                raise ValueError("Initial source configuration changed; use a new Campaign workspace")
            source = pack({name: text.encode() for name, text in value["files"].items()}, value["contract"])
            if hashlib.sha256(source).hexdigest() != value.get("snapshot_digest"):
                raise ValueError("Initial source snapshot failed integrity verification")
            return source
        finally:
            os.close(lock)
