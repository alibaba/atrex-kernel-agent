"""Source reads, initialization and peer import for Agent-owned repositories."""
from .snapshot import GitSnapshots


class GitCandidateWorkspace(GitSnapshots):
    def bind_source_contract(self, namespace, *, file_system, contract):
        from aka.task.source.formats.private_state import bind_namespace
        return bind_namespace(namespace, file_system=file_system, contract=contract)


    def initialize_agent(self, workspace, content):
        from .persistent import initialize_agent
        return initialize_agent(workspace, content)


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


    def install_files(self, workspace, files, *, immutable=False):
        from .content import install_files
        return install_files(workspace, files, immutable=immutable)
