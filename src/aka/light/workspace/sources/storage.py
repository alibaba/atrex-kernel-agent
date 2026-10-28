"""Shared snapshot codec and Git transaction; format behavior belongs to selected implementations."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
from .common import (POLICY, MAGIC, MAX_FILE, MAX_TOTAL, MAX_SNAPSHOT, WORKSPACE_ONLY,
                     SELECTION, relative, ignored, source_name, remove_files, publish)
from .registry import for_policy,resolve
from .source_tree import validate_import_options,import_tree,initial_snapshot


def validate_policy(policy):return for_policy(policy).validate_policy(policy)
def validate_files(files,policy):
    system = for_policy(policy)
    system.validate_policy(policy)
    return system.validate(files,policy)
def editable(policy,name):return for_policy(policy).editable(policy,name)
def instructions(policy):return for_policy(policy).instructions(policy)
def allowed_changes(paths,policy):
    system=for_policy(policy)
    fixed=policy.get('immutable_files',{}) if policy else {}
    return all(system.editable(policy,name) and name not in fixed for name in paths)
def policy_at(workspace,*,committed=False):return resolve(workspace,revision='HEAD' if committed else None)[1]
def collect(workspace,policy,*,validate=True):return for_policy(policy).collect(workspace,policy,validate=validate)

def pack(files,policy):
    validate_files(files,policy)
    if policy is None:return next(iter(files.values()))
    return MAGIC+json.dumps({'contract':policy,'files':{k:v.decode('utf-8') for k,v in sorted(files.items())}},
        sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()

def unpack(source):
    if not source.startswith(MAGIC):
        return None, {for_policy(None).layout.python_entrypoint: source}
    if len(source) > MAX_SNAPSHOT:
        raise ValueError("Source snapshot exceeds the bounded limit")
    value = json.loads(source[len(MAGIC):])
    files = {name: data.encode() for name, data in value["files"].items()}
    validate_files(files, value["contract"])
    if pack(files, value["contract"]) != source:
        raise ValueError("Source snapshot is not canonical")
    return value["contract"], files


def from_inputs(inputs,*,prefix=''):
    from .registry import from_inputs as select_inputs
    system,policy=select_inputs(inputs,prefix)
    return pack(system.from_inputs(inputs,policy,prefix),policy)

_DISCOVER=object()
def snapshot(workspace,*,policy=_DISCOVER):
    if policy is _DISCOVER:system,policy=resolve(workspace)
    else:system=for_policy(policy)
    return pack(system.collect(workspace,policy),policy)

def from_git(workspace,revision):
    system,policy=resolve(workspace,revision=revision)
    return pack(system.from_git(workspace,revision,policy),policy)

def install(workspace,source,*,replace=True,bind=False):
    policy,files=unpack(source);system=for_policy(policy)
    try:
        previous,previous_policy=resolve(workspace)
        previous.validate_restore(previous_policy, policy)
        obsolete=previous.obsolete(workspace,previous_policy)
    except (ValueError,UnicodeError):
        # A broken draft must not prevent restoration of a sealed snapshot.
        if (workspace/POLICY).exists():raise
        obsolete=set()
    if replace:remove_files(workspace,obsolete-files.keys())
    system.install(workspace,files,policy)
    if bind or (workspace/SELECTION).is_file():
        publish(workspace,SELECTION,(json.dumps({'name':system.name},sort_keys=True)+'\n').encode())

def commit_candidate(handle, source):
    """Seal measured additions/modifications/deletions with a private Git index."""
    from aka.legacy.task.git import git_head, git_text, changed_paths
    import tempfile
    policy, files = unpack(source)
    baseline_policy, _ = unpack(from_git(handle.path, handle.base_commit))
    if source == from_git(handle.path, handle.base_commit):
        raise ValueError("Candidate has no source change relative to its baseline")
    if policy != baseline_policy:
        raise ValueError("Candidate source contract differs from its Episode baseline")
    head = git_head(handle.path)
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
                         "-m", f"Episode {handle.episode}: selected candidate")
            run("update-ref", "HEAD", commit, head)
            head = commit
    install(handle.path, source)
    subprocess.run(["git", "read-tree", head], cwd=handle.path, check=True)
    return head
