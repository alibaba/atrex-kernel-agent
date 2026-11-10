"""Selected workspace implementation using the original candidate mechanics."""
from aka.contracts.workspace import CandidateHandle
from .legacy import EpisodeWorktree, promote_candidate
from .snapshot import GitSnapshots


def _legacy(handle):
    return EpisodeWorktree(handle.episode, handle.base_commit, handle.branch, handle.path)


class GitCandidateWorkspace(GitSnapshots):
    def prepare_draft(self, workspace):
        from .draft import prepare_draft
        return prepare_draft(workspace)

    def exclude_runtime_paths(self, workspace, exclusions):
        from .excludes import exclude_runtime_paths
        return exclude_runtime_paths(workspace, exclusions)

    def current_revision(self, workspace):
        from .records import current_revision
        return current_revision(workspace)

    def read_working(self, workspace, paths):
        from .records import read_working
        return read_working(workspace, paths)

    def read_source(self, workspace, *, revision=None):
        from .source import read_source
        return read_source(workspace, revision=revision)

    def install_source(self, workspace, source):
        from .source import install_source
        return install_source(workspace, source)

    def validate_candidate_source(self, handle, source):
        from .source import validate_candidate_source
        return validate_candidate_source(handle, source)

    def seal_candidate(self, handle, source):
        from .source import seal_candidate
        return seal_candidate(handle, source)

    def source_identity(self, source):
        from .source import source_identity
        return source_identity(source)

    def fork_revision(self, workspace, revision, destination):
        from .retention import fork_revision
        return fork_revision(workspace, revision, destination)

    def retain_revision(self, workspace, revision, *, retention_key):
        from .retention import retain_revision
        return retain_revision(workspace, revision, retention_key=retention_key)

    def import_revision(self, workspace, source_workspace, revision, *, retention_key):
        from .retention import import_revision
        return import_revision(workspace, source_workspace, revision, retention_key=retention_key)

    def revision_history(self, workspace, *, revision=None, paths=(), changed_path=None):
        from .history import revision_history
        return revision_history(workspace, revision=revision, paths=paths, changed_path=changed_path)

    def revision_changes(self, workspace, base_revision, revision):
        from .history import revision_changes
        return revision_changes(workspace, base_revision, revision)

    def working_facts(self, workspace, *, paths=(), include_untracked=True):
        from .history import working_facts
        return working_facts(workspace, paths=paths, include_untracked=include_untracked)

    def restore_revision(self, workspace, *, expected_revision, revision, retain_working_files):
        from .restore import restore_revision
        return restore_revision(workspace, expected_revision=expected_revision,
                                revision=revision, retain_working_files=retain_working_files)

    def record(self, workspace, *, expected_revision, files, message, deleted_paths=()):
        from .records import record
        return record(workspace, expected_revision=expected_revision,
                      files=files, message=message, deleted_paths=deleted_paths)

    def snapshot_files(self, workspace, destination, *, directories, exclude_paths=()):
        from .files import snapshot_files
        return snapshot_files(workspace, destination, directories=directories,
                              exclude_paths=exclude_paths)

    def restore_files(self, snapshot, workspace, *, exclude_paths=()):
        from .files import restore_files
        return restore_files(snapshot, workspace, exclude_paths=exclude_paths)

    def clear_paths(self, workspace, paths):
        from .files import clear_paths
        return clear_paths(workspace, paths)

    def filesystem_access(self, workspace, *, include_git=False):
        from .access import filesystem_access
        return filesystem_access(workspace, include_git=include_git)

    def export_checkpoint(self, workspace, destination, *, candidates=()):
        from .checkpoint import export_checkpoint
        return export_checkpoint(workspace, destination, candidates=candidates)

    def validate_checkpoint(self, snapshot, *, check_destinations=False):
        from .checkpoint import validate_checkpoint
        return validate_checkpoint(snapshot, check_destinations=check_destinations)

    def checkpoint_files(self, snapshot, paths):
        from .checkpoint import checkpoint_files
        return checkpoint_files(snapshot, paths)

    def restore_checkpoint(self, snapshot):
        from .checkpoint import restore_checkpoint
        return restore_checkpoint(snapshot)

    def baseline_changes(self, workspace, baseline_commit, *, required_paths=()):
        from .baseline import baseline_changes
        return baseline_changes(workspace, baseline_commit, required_paths=required_paths)

    def restore_baseline(self, workspace, baseline_commit):
        from .baseline import restore_baseline
        return restore_baseline(workspace, baseline_commit)

    def install_files(self, workspace, files, *, immutable=False):
        from .content import install_files
        return install_files(workspace, files, immutable=immutable)

    def plan(self, workspace, episode, base_commit, root=None):
        handle = EpisodeWorktree.plan(workspace, episode, base_commit, root)
        return CandidateHandle(handle.episode, handle.base_commit, handle.branch, handle.path)

    def materialize(self, handle, workspace):
        _legacy(handle).materialize(workspace)

    def inspect_candidate(self, workspace, handle):
        from .binding import inspect_candidate
        return inspect_candidate(workspace, handle)

    def validate_candidate(self, handle, candidate_commit):
        from .source import validate_candidate
        return validate_candidate(handle, candidate_commit)

    def restore_sealed_candidate(self, handle, candidate_commit, source):
        from .source import restore_sealed_candidate
        return restore_sealed_candidate(handle, candidate_commit, source)

    def archive(self, handle, destination, candidate_commit="HEAD"):
        return _legacy(handle).archive(destination, candidate_commit)

    def remove(self, handle, workspace):
        _legacy(handle).remove(workspace)

    def promote(self, workspace, **options):
        return promote_candidate(workspace, **options)


    def initialize_kernel(self, workspace, content, *, runtime_files, context_directories=()):
        from .initial import initialize_kernel
        return initialize_kernel(workspace, content, runtime_files=runtime_files,
                                 context_directories=context_directories)

    def prepare_sol(self, workspace, content, *, problem_files, harness, runtime_files,
                    context_directories=()):
        from .initial import prepare_sol
        return prepare_sol(workspace, content, problem_files=problem_files,
                           harness=harness, runtime_files=runtime_files,
                           context_directories=context_directories)
