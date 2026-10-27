"""Workflow provenance, marker and stall decisions over selected Workspace facts."""
import json
import re

from aka.contracts.content import WorkspaceFile
from aka.legacy.task.candidate_workspace.filenames import FRAMEWORK_BASELINE_FILE
from aka.legacy.task.candidate_workspace.excludes import STALL_STATE_FILE


def _record_revision(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    return str(value).strip()


class WorkspaceState:
    def __init__(self, workspace, candidate_workspace):
        self.workspace = workspace
        self.candidate_workspace = candidate_workspace

    def for_workspace(self, workspace):
        return type(self)(workspace, self.candidate_workspace)

    def current_revision(self):
        return self.candidate_workspace.current_revision(self.workspace)

    def history(self, *, revision=None, paths=(), changed_path=None):
        return self.candidate_workspace.revision_history(self.workspace, revision=revision,
            paths=paths, changed_path=changed_path)

    def fact(self, revision=None, *, paths=()):
        history = self.history(revision=revision, paths=paths)
        return history[0] if history else None

    def committed_file(self, path, *, revision=None):
        revision = self.current_revision() if revision is None else revision
        if not revision:
            return None
        return self.candidate_workspace.read_revision(self.workspace, revision, [path]).files[path]

    def content_identity(self, revision, path):
        if not revision:
            return ""
        fact = self.fact(revision, paths=(path,))
        return fact.content_identities[path] or "" if fact is not None else ""

    def committed_identity(self, path="kernel.py"):
        return self.content_identity(self.current_revision(), path)

    def working_identity(self, path="kernel.py"):
        return self.candidate_workspace.working_facts(self.workspace,
            paths=(path,)).content_identities[path] or ""

    def working_changes(self):
        return self.candidate_workspace.working_facts(self.workspace).changed_paths

    def initial_kernel_revision(self):
        history = self.history(changed_path="kernel.py")
        return history[-1].revision if history else ""

    def latest_kernel_revision(self):
        history = self.history(changed_path="kernel.py")
        return history[0].revision if history else ""

    def kernel_is_initial_baseline(self):
        baseline = self.initial_kernel_revision()
        identity = self.content_identity(baseline, "kernel.py")
        return bool(identity and identity == self.committed_identity())

    def is_ancestor(self, ancestor, *, revision=None):
        fact = self.fact(ancestor)
        return fact is not None and any(row.revision == fact.revision
                                       for row in self.history(revision=revision))

    def framework_baseline_revision(self):
        content = self.committed_file(FRAMEWORK_BASELINE_FILE)
        if content is None:
            return "", 0
        try:
            marker = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} is not valid JSON: {exc}") from exc
        if not isinstance(marker, dict):
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} must contain a JSON object")
        revision = _record_revision(marker.get("commit"))
        if not revision:
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has no usable revision identity")
        try:
            fact = self.fact(revision, paths=("kernel.py",))
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} points at missing revision {revision}") from exc
        if fact is None or not self.is_ancestor(fact.revision):
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} revision {revision} is not an ancestor of current revision")
        identity = fact.content_identities["kernel.py"]
        if not identity or identity != marker.get("kernel_blob"):
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} kernel content identity does not match revision {revision}")
        match = re.fullmatch(r"v(\d+)", str(marker.get("version") or ""))
        if match is None:
            raise RuntimeError(f"{FRAMEWORK_BASELINE_FILE} has an unusable version {marker.get('version')!r}")
        # Preserve the recorded identity (including old short/numeric Git IDs).
        return revision, int(match.group(1))

    def restored_stall(self):
        try:
            content = self.candidate_workspace.read_working(self.workspace, (STALL_STATE_FILE,)).files[STALL_STATE_FILE]
            value = json.loads(content).get("stall") if content is not None else None
            if isinstance(value, int):
                return int(value)
        except (OSError, ValueError, TypeError):
            pass
        history = self.history(paths=("kernel.py",))
        identities = {fact.revision: fact.content_identities["kernel.py"] for fact in history}
        trailing = 0
        for fact in history:
            identity = fact.content_identities["kernel.py"]
            if (not fact.parents and identity is not None) or (
                    fact.parents and all(identity != identities[parent] for parent in fact.parents)):
                break
            trailing += 1
        return trailing

    def save_stall(self, value):
        try:
            self.candidate_workspace.install_files(self.workspace,
                (WorkspaceFile(STALL_STATE_FILE, text=json.dumps({"stall": int(value)}, indent=2)),))
        except OSError:
            pass

    def restore(self, revision, *, retain_working_files, expected_revision=None):
        return self.candidate_workspace.restore_revision(self.workspace,
            expected_revision=self.current_revision() if expected_revision is None else expected_revision, revision=revision,
            retain_working_files=retain_working_files)
