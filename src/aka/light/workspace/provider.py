"""Task provider owns draft reset, snapshots and sealed Git promotion."""
from pathlib import Path

from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.contracts.workspace import CandidateHandle


class ManagedCandidateWorkspace(GitCandidateWorkspace):
    managed_drafts = True

    def filesystem_access(self, workspace, *, include_git=False):
        from aka.contracts.filesystem import FilesystemAccess
        from aka.light.orchestrator.episode_workspace import PUBLIC_FILES
        from .sources import POLICY
        if include_git:
            raise ValueError("Managed candidates retain Git under the Supervisor")
        workspace = Path(workspace).resolve(strict=True)
        protected = tuple(workspace / name for name in (*PUBLIC_FILES, "memory", "input", POLICY)
                          if (workspace / name).exists())
        return FilesystemAccess(hidden_paths=(workspace / ".git",), read_only_paths=protected)

    def __init__(self, *, inherit_state: bool = True):
        if type(inherit_state) is not bool:
            raise ValueError("inherit_state must be boolean")
        self.inherit_state = inherit_state

    def install_runtime(self, workspace, atrex_bench_root=None, **options):
        from aka.light.orchestrator.workspace_runtime import link_runtime
        return link_runtime(workspace, atrex_bench_root, **options)

    def open_draft(self, workspace, root, **options):
        from aka.light.orchestrator.episode_workspace import EpisodeWorkspace
        return EpisodeWorkspace(workspace, root, inherit_state=self.inherit_state, **options)

    def plan(self, workspace, episode, base_commit, root=None):
        from aka.light.long_horizon.git_episode import EpisodeWorktree
        handle = EpisodeWorktree.plan(workspace, episode, base_commit, root)
        return CandidateHandle(handle.episode, handle.base_commit, handle.branch, handle.path)

    def restore_handle(self, episode, base_commit, branch, path):
        return CandidateHandle(episode, base_commit, branch, path)

    @staticmethod
    def _implementation(handle):
        from aka.light.long_horizon.git_episode import EpisodeWorktree
        return EpisodeWorktree(handle.episode, handle.base_commit, handle.branch, handle.path)

    def validate_candidate(self, handle, candidate_commit):
        return self._implementation(handle).validate_candidate(candidate_commit)

    def restore_sealed_candidate(self, handle, candidate_commit, source):
        return self._implementation(handle).restore_sealed_candidate(candidate_commit, source)

    def reset_scratch(self, handle):
        self._implementation(handle).reset_scratch()

    def materialize(self, handle, workspace):
        self._implementation(handle).materialize(workspace)

    def archive(self, handle, destination, candidate_commit="HEAD"):
        return self._implementation(handle).archive(destination, candidate_commit)

    def remove(self, handle, workspace):
        self._implementation(handle).remove(workspace)

    def promote(self, workspace, **options):
        from aka.light.long_horizon.git_episode import promote_candidate
        return promote_candidate(workspace, **options)
