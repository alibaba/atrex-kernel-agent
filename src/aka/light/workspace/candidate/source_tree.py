"""Pinned adapter plus editable source directories; independent file-system implementation."""
from __future__ import annotations
import hashlib,json,os,stat,subprocess,tarfile,time
from pathlib import Path
from .common import (POLICY, MAGIC, MAX_FILE, MAX_TOTAL, MAX_SNAPSHOT, PROTECTED, WORKSPACE_ONLY,
                     relative, ignored, source_name, read_regular_bytes, publish, git_files, Layout, ExecutionSource)


def pack(files, policy):
    from .storage import pack as encode
    return encode(files, policy)

def validate_import_options(options):
    if (not isinstance(options, dict)
            or set(options) not in ({"manifest", "repository"}, {"manifest", "directory"})
            or any(not isinstance(value, str) or not value for value in options.values())):
        raise ValueError("source_tree requires manifest and exactly one of repository or directory")
    return options

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

def instructions(policy):
    return ("\n## Kernel source tree\nThe fixed kernel.py adapter and bundled source form one Kernel. "
            "Edit/add/delete source only inside editable_roots: " + json.dumps(policy["editable_roots"]) +
            ". All other imported files are fixed; do not change, remove or rename them. "
            "Do not rewrite the adapter or replace the package with a single-file implementation. "
            "Keep notes/tools and temporary builds in tools/, skills/ and scratch/. "
            "Evaluate, Profile, Check, Disassemble, sealed reports and broadcasts bind the entire tree. "
            "kernel-read writes a directory for tree Kernels; pass that directory to --baseline-path.\n")

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

def initial_snapshot(options, *, state_root, seed=None):
    """Seal once before V0; failed bootstrap/restarts never reread mutable inputs."""
    import fcntl
    from aka.light.agent.home import open_private_directory
    from aka.legacy.task.persistence import durable_write_json
    if options is not None:
        validate_import_options(options)
    configuration = ({key: os.path.abspath(Path(value).expanduser()) for key, value in options.items()}
                     if options is not None else {"source": "provider"})
    root = state_root
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
                source = (seed() if options is None else
                          import_tree(configuration["manifest"], configuration.get("repository"),
                                      directory=configuration.get("directory")))
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


class SourceTree:
    def seed_metadata(self, workspace, framework, name, *, native_eval):pass
    def prepare_source(self, request, source_provider, *, options, state_root, native_eval):
        if request.initial.kind == 'sol' or not native_eval:
            raise ValueError('Source trees require native Atrex-Bench Eval; select single_file explicitly for SOL/SSH')
        from aka.contracts.content import SourceContent, WorkspaceFile
        from .artifacts import Artifact
        def seed():
            content = source_provider.prepare(request)
            if len(content.files) != 1 or content.files[0].path != 'kernel.py':
                raise ValueError('Default source-tree initialization requires one kernel.py; use source_tree import configuration for a package')
            entry = content.files[0]
            files = {'impl/model.py': entry.source.read_bytes() if entry.source is not None else entry.text.encode(),
                     'impl/__init__.py': b'', 'kernel.py': b'from impl.model import Model\n'}
            policy = {'schema_version': 1, 'source_revision': None,
                      'seed_digest': 'sha256:' + hashlib.sha256(json.dumps(
                          {k: v.decode() for k, v in sorted(files.items())}, sort_keys=True).encode()).hexdigest(),
                      'package_root': '.', 'editable_roots': ['impl'],
                      'immutable_files': {'kernel.py': hashlib.sha256(files['kernel.py']).hexdigest()},
                      'runtime_requirements': []}
            return pack(files, policy)
        artifact = Artifact(initial_snapshot(options, state_root=state_root, seed=seed))
        return SourceContent(tuple(WorkspaceFile(name, text=data.decode())
                                   for name, data in artifact.materialized_files.items()))
    name = 'source_tree'
    default_policy = None
    def inspect_policy(self, get):
        raw = get(POLICY)
        return (True, json.loads(raw)) if raw is not None else (False, None)
    def validate_restore(self, previous, target):
        if previous != target:raise ValueError('Source contract cannot change within a Campaign')
    layout = Layout('kernel.py', replaceable_baseline=False, archived=True, auxiliary_metadata=('solution.json',))
    def validate_policy(self, policy):return validate_policy(policy)
    def validate(self, files, policy):return validate_files(files, policy)
    def collect(self, workspace, policy, *, validate=True):return collect(workspace, policy, validate=validate)
    def from_git(self, workspace, revision, policy):return validate_files(git_files(workspace, revision), policy)
    def from_inputs(self, inputs, policy, prefix):
        return validate_files({name[len(prefix):]:data for name,data in inputs.items()
                              if name.startswith(prefix) and source_name(name[len(prefix):])}, policy)
    def editable(self, policy, name):return editable(policy, name)
    def instructions(self, policy):return instructions(policy)
    def metadata(self, policy):return {POLICY:(json.dumps(policy, sort_keys=True)+'\n').encode()}
    def obsolete(self, workspace, policy):return set(collect(workspace, policy, validate=False))
    def install(self, workspace, files, policy):
        for name,data in {**files, **self.metadata(policy)}.items():publish(workspace,name,data)
    def execution_source(self, files, policy, archive):
        if policy['package_root'] != '.':raise ValueError("Eval archives require package_root='.'; no Dev fallback")
        requirements=tuple(v['distribution']+v.get('version','') for v in policy['runtime_requirements'])
        return ExecutionSource({'archive':archive,'entry_point':'kernel.py'},
                               {archive:{k:v.decode() for k,v in files.items()}},requirements)
    def declared_paths(self, workspace, policy):return set(collect(workspace,policy))|{POLICY}
    def evidence_paths(self, workspace, policy):return self.declared_paths(workspace,policy)

    def typed_limitation(self, workspace, policy):return None
    def language_hints(self, workspace):
        path = workspace/'solution.json'
        value = json.loads(path.read_text()) if path.is_file() else {}
        hints = value.get('languages', [])
        return [str(v) for v in hints] if isinstance(hints,list) else []
