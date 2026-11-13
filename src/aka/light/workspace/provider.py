"""Task provider owns source setup and the persistent Agent workspace.

Legacy worktree operations remain inherited only for the separate Legacy
workflow contract. AKA Light does not call them: every audit segment reuses the
same trajectory repository.
"""
from dataclasses import replace
import subprocess

from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.legacy.evaluation.resources import profile_driver_script
from aka.contracts.content import WorkspaceFile
from aka.light.workspace.files import publish
from aka.light.workspace.resource_catalog import resource


class ManagedCandidateWorkspace(GitCandidateWorkspace):
    managed_drafts = True

    def __init__(
        self,
        *,
        inherit_state: bool | None = None,
        file_system: str | None = None,
        source_tree: dict | None = None,
    ):
        # Read old Light compositions that explicitly stored the former default,
        # but do not retain the retired per-Episode workspace mode.
        if inherit_state is not None and type(inherit_state) is not bool:
            raise ValueError("inherit_state must be a boolean when reading an old config")
        if inherit_state is False:
            raise ValueError(
                "inherit_state=false is retired; Light always uses one persistent "
                "trajectory workspace"
            )
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
        from .sources.private_state import bind
        from .sources.registry import resolve

        current, contract = resolve(workspace, revision='HEAD')
        if current.name != system.name:
            raise ValueError(f'Selected {system.name} file system requires a matching candidate; found {current.name}')
        bind(workspace, file_system=system.name, contract=contract)

    def prepare_sol(self, workspace, content, *, problem_files):
        # SourceProvider owns implementation bytes, not the selected Workflow's
        # tool/Git instructions. Do not inherit the legacy direct-run README.
        readme = (
            "# Managed SOL optimization workspace\n\n"
            "Optimize `kernel.py` while preserving the definition's destination-passing "
            "interface. Maximize the arithmetic mean of workload speedups against the "
            "reference while every workload passes its correctness checks.\n\n"
            "Use `python3 tools/sandbox.py --kind run --no-sync`. The Supervisor owns "
            "the evaluator, its configuration and solution metadata. Manage the trajectory's "
            "local Git history, and commit a clean Kernel HEAD before Runtime requests. Do not run "
            "the evaluator locally or edit those files. Submit an exact reachable commit with "
            "`submit-kernel` after a standard full Evaluate passes for that exact commit; read the installed "
            "gpu-measurement and runtime-records Skills "
            "for request examples. Put temporary probes and requests under `scratch/`.\n"
        )
        # The shared initializer now seals its explicit contributions itself.
        # Select Light guidance before that commit, not as a later dirty edit.
        result = super().prepare_sol(workspace, replace(content, readme=readme),
            problem_files=problem_files,
            harness=WorkspaceFile("test_kernel.py", source=resource("reference/test_kernel.py")),
            runtime_files=(
                WorkspaceFile("CLAUDE.md", source=resource("reference/CLAUDE.md")),
                WorkspaceFile("profile_driver.py", source=profile_driver_script()),
            ))
        from aka.light.workspace.auxiliary import ensure_workspace_scratch
        ensure_workspace_scratch(workspace)
        return result

    def initialize_at(self, workspace, content):
        workspace.mkdir(parents=True, exist_ok=True)
        from aka.light.workspace.auxiliary import ensure_workspace_scratch
        ensure_workspace_scratch(workspace)
        self.install_files(workspace, content.files)
        for name in ("CLAUDE.md", "test_kernel.py"):
            publish(workspace, name, resource(f"reference/{name}").read_bytes())
        publish(workspace, "profile_driver.py", profile_driver_script().read_bytes())
        publish(workspace, "README.md", b"# Kernel optimization workspace\n")
        from aka.light.workspace.agent_git import initialize_gitignore
        initialize_gitignore(workspace)
        if not (workspace / ".git").exists():
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.name", "GPU Kernel Optimizer"], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.email", "gpu-kernel-optimizer@local"], check=True)

    def install_runtime(self, workspace, atrex_bench_root=None, **options):
        from aka.light.workspace.assets import link_runtime
        return link_runtime(workspace, atrex_bench_root, **options)

    def open_draft(self, workspace, root, **options):
        from aka.light.workspace.episode import EpisodeWorkspace
        return EpisodeWorkspace(workspace, root, **options)
