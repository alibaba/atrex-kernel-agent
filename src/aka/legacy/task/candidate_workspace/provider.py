"""Selected workspace implementation using the original candidate mechanics."""
from aka.contracts.workspace import CandidateHandle
from .legacy import EpisodeWorktree, promote_candidate
from .snapshot import GitSnapshots


def _legacy(handle):
    return EpisodeWorktree(handle.episode, handle.base_commit, handle.branch, handle.path)


class GitCandidateWorkspace(GitSnapshots):
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

    def baseline_changes(self, workspace, baseline_commit):
        from .baseline import baseline_changes
        return baseline_changes(workspace, baseline_commit)

    def restore_baseline(self, workspace, baseline_commit):
        from .baseline import restore_baseline
        return restore_baseline(workspace, baseline_commit)

    def install_files(self, workspace, files):
        from .content import install_files
        return install_files(workspace, files)

    def plan(self, workspace, episode, base_commit, root=None):
        handle = EpisodeWorktree.plan(workspace, episode, base_commit, root)
        return CandidateHandle(handle.episode, handle.base_commit, handle.branch, handle.path)

    def materialize(self, handle, workspace):
        _legacy(handle).materialize(workspace)

    def validate_candidate(self, handle, candidate_commit):
        return _legacy(handle).validate_candidate(candidate_commit)

    def archive(self, handle, destination, candidate_commit="HEAD"):
        return _legacy(handle).archive(destination, candidate_commit)

    def remove(self, handle, workspace):
        _legacy(handle).remove(workspace)

    def promote(self, workspace, **options):
        return promote_candidate(workspace, **options)


    def initialize_kernel(self, reference_dir, entry, arguments, *, content, working_directory=None):
        from .initial import initialize_kernel
        return initialize_kernel(reference_dir, entry, arguments, content=content,
                                 working_directory=working_directory)

    def prepare_sol(self, workspace, content, *, problem_files, reference_dir, harness):
        from .initial import prepare_sol
        return prepare_sol(workspace, content, problem_files=problem_files, reference_dir=reference_dir, harness=harness)


    def install_runtime(self, workspace, atrex_bench_root=None, **options):
        from .runtime import link_runtime
        return link_runtime(workspace, atrex_bench_root, **options)
