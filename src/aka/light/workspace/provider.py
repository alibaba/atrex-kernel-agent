"""Task provider owns source setup and the persistent Agent workspace.

Legacy worktree operations remain inherited only for the separate Legacy
workflow contract. AKA Light does not call them: every audit segment reuses the
same trajectory repository.
"""
import subprocess

from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
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
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "test_kernel.py").unlink(missing_ok=True)
        repository = workspace / ".git"
        if repository.is_dir():
            history = subprocess.run(
                ["git", "rev-list", "--reverse", "HEAD", "--", "kernel.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            if history:
                source_commit = history[0]
                expected = next(
                    item for item in content.files if item.path == "kernel.py"
                )
                expected_bytes = (
                    expected.source.read_bytes()
                    if expected.source is not None
                    else expected.text.encode("utf-8")
                )
                committed = subprocess.run(
                    ["git", "show", f"{source_commit}:kernel.py"],
                    cwd=workspace,
                    capture_output=True,
                    check=True,
                ).stdout
                if committed != expected_bytes:
                    raise ValueError("V0 selected SOL Kernel source changed")
                return source_commit

        # Public task inputs, evaluator metadata and guidance remain in the
        # workspace but outside Agent Git. Only Kernel source enters V0.
        self.install_files(workspace, problem_files)
        from aka.legacy.task.candidate_workspace.initial import SOL_CONFIG
        (workspace / "config.json").write_text(SOL_CONFIG, encoding="utf-8")
        self.install_files(workspace, content.files)
        (workspace / "test_kernel.py").unlink(missing_ok=True)
        publish(
            workspace,
            "CLAUDE.md",
            resource("reference/CLAUDE.md").read_bytes(),
        )
        (workspace / "README.md").write_text(readme, encoding="utf-8")
        from aka.light.workspace.agent_git import AgentGit, initialize_gitignore
        initialize_gitignore(workspace)
        agent_git = AgentGit(workspace)
        agent_git.prepare()
        result = agent_git.head()
        if result is None:
            raise RuntimeError("SOL V0 did not produce an Agent Git commit")
        from aka.light.workspace.auxiliary import ensure_workspace_scratch
        ensure_workspace_scratch(workspace)
        return result

    def initialize_at(self, workspace, content):
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "test_kernel.py").unlink(missing_ok=True)
        from aka.light.workspace.auxiliary import ensure_workspace_scratch
        ensure_workspace_scratch(workspace)
        self.install_files(workspace, content.files)
        (workspace / "test_kernel.py").unlink(missing_ok=True)
        publish(
            workspace,
            "CLAUDE.md",
            resource("reference/CLAUDE.md").read_bytes(),
        )
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
