"""Task provider owns draft reset, snapshots and sealed Git promotion."""
from pathlib import Path
from dataclasses import replace
import subprocess

from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.legacy.evaluation.resources import profile_driver_script
from aka.contracts.workspace import CandidateHandle
from aka.contracts.content import WorkspaceFile
from aka.light.workspace.files import publish
from aka.light.workspace.resource_catalog import resource


class ManagedCandidateWorkspace(GitCandidateWorkspace):
    managed_drafts = True

    def __init__(self, *, inherit_state: bool = True, file_system: str | None = None, source_tree: dict | None = None):
        if type(inherit_state) is not bool:
            raise ValueError("inherit_state must be boolean")
        self.inherit_state = inherit_state
        if file_system is not None:
            from .sources import select
            select(file_system)
        self.file_system = file_system
        self.source_tree = source_tree
        if source_tree is not None:
            from .sources import validate_import_options
            validate_import_options(source_tree)

    def source_options(self, legacy_options=None):
        if self.source_tree is not None and legacy_options is not None and self.source_tree != legacy_options:
            raise ValueError('Conflicting workspace and legacy workflow source-tree configuration')
        return self.source_tree if self.source_tree is not None else legacy_options

    def prepare_source(self, system, request, source_provider, *, options, state_root, native_eval):
        return system.prepare_source(request, source_provider, options=options,
                                     state_root=state_root, native_eval=native_eval)

    def select_file_system(self, *, framework, source_tree=False):
        from .sources import task_selection
        return task_selection(self.file_system, framework=framework, source_tree=source_tree)

    def activate_file_system(self, workspace, system):
        from .sources.registry import resolve
        from .sources.common import SELECTION
        import json
        current, _ = resolve(workspace, revision='HEAD')
        if current.name != system.name:
            raise ValueError(f'Selected {system.name} file system requires a matching candidate; found {current.name}')
        data = (json.dumps({'name':system.name}, sort_keys=True)+'\n').encode()
        path = workspace / SELECTION
        if path.is_file() and path.read_bytes() != data:
            raise ValueError('File system selection cannot change when resuming a Campaign')
        publish(workspace, SELECTION, data)
        subprocess.run(['git','add','--',SELECTION],cwd=workspace,check=True)
        changed = subprocess.run(['git','diff','--cached','--quiet','--',SELECTION],cwd=workspace)
        if changed.returncode:
            subprocess.run(['git','commit','-q','-m','Pin candidate file system','--',SELECTION],cwd=workspace,check=True)

    def prepare_sol(self, workspace, content, *, problem_files):
        # SourceProvider owns implementation bytes, not the selected Workflow's
        # tool/Git instructions. Do not inherit the legacy direct-run README.
        readme = (
            "# Managed SOL optimization workspace\n\n"
            "Optimize `kernel.py` while preserving the definition's destination-passing "
            "interface. Maximize the arithmetic mean of workload speedups against the "
            "reference while every workload passes its correctness checks.\n\n"
            "Use `python3 tools/sandbox.py --kind run --no-sync`. The Supervisor owns "
            "the evaluator, its configuration, solution metadata and Git. Do not run "
            "the evaluator locally or edit those files. Submit `episode-report` through "
            "the Runtime; read the installed gpu-measurement and runtime-records Skills "
            "for request examples. Put temporary probes and requests under `scratch/`.\n"
        )
        # The shared initializer now seals its explicit contributions itself.
        # Select Light guidance before that commit, not as a later dirty edit.
        return super().prepare_sol(workspace, replace(content, readme=readme),
            problem_files=problem_files,
            harness=WorkspaceFile("test_kernel.py", source=resource("reference/test_kernel.py")),
            runtime_files=(
                WorkspaceFile("CLAUDE.md", source=resource("reference/CLAUDE.md")),
                WorkspaceFile("profile_driver.py", source=profile_driver_script()),
            ))

    def initialize_at(self, workspace, content):
        workspace.mkdir(parents=True, exist_ok=True)
        self.install_files(workspace, content.files)
        for name in ("CLAUDE.md", "test_kernel.py"):
            publish(workspace, name, resource(f"reference/{name}").read_bytes())
        publish(workspace, "profile_driver.py", profile_driver_script().read_bytes())
        publish(workspace, "README.md", b"# Kernel optimization workspace\n")
        publish(workspace, ".gitignore", b"__pycache__/\n*.pyc\n/scratch\n")
        (workspace / "memory").mkdir(exist_ok=True)
        if not (workspace / ".git").exists():
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.name", "GPU Kernel Optimizer"], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.email", "gpu-kernel-optimizer@local"], check=True)

    def install_runtime(self, workspace, atrex_bench_root=None, **options):
        from aka.light.workspace.assets import link_runtime
        return link_runtime(workspace, atrex_bench_root, **options)

    def open_draft(self, workspace, root, **options):
        from aka.light.workspace.episode import EpisodeWorkspace
        return EpisodeWorkspace(workspace, root, inherit_state=self.inherit_state, **options)

    def plan(self, workspace, episode, base_commit, root=None):
        from aka.light.workspace.git import EpisodeWorktree
        handle = EpisodeWorktree.plan(workspace, episode, base_commit, root)
        return CandidateHandle(handle.episode, handle.base_commit, handle.branch, handle.path)

    def restore_handle(self, episode, base_commit, branch, path):
        return CandidateHandle(episode, base_commit, branch, path)

    @staticmethod
    def _implementation(handle):
        from aka.light.workspace.git import EpisodeWorktree
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
        from aka.light.workspace.git import promote_candidate
        return promote_candidate(workspace, **options)
