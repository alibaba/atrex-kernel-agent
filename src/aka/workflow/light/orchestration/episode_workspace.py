"""Runtime-managed projections over a persistent Agent-owned Git repository."""
from pathlib import Path
import stat
from aka.core.filesystem import open_private_directory
from aka.task.source.formats import instructions as source_instructions, policy_at
from aka.task.candidate_workspace.agent_git import AgentGit
from aka.resources.bundles import resource_path

EPISODE_WORKSPACE_ENV = "ATREX_EPISODE_WORKSPACE"

class EpisodeWorkspace:
    def __init__(self, source_workspace, directory=None, *, is_ppu=False,
                 research_records=False, research_modules=(), plugin_registry=None,
                 resources_registry=None, workspace_policy="", persistent_root=None):
        self.source_workspace = Path(source_workspace).resolve(strict=True)
        self.root = self.source_workspace
        if persistent_root is not None and Path(persistent_root).resolve() != self.root:
            raise ValueError("An Episode must use its persistent Agent workspace directly")
        self.persistent = True
        self.is_ppu = is_ppu
        self.research_records = research_records
        self.research_modules = tuple(research_modules)
        self.plugin_registry = resources_registry or plugin_registry
        if self.plugin_registry is None:
            raise TypeError("Episode resources must be supplied by the selected Resources owner")
        self.workspace_policy = workspace_policy
        self.agent_git = AgentGit(self.root)
        self.reserved_paths = ()

    def prepare(self):
        if not (self.root / ".git").is_dir():
            raise ValueError("Campaign must initialize Agent Git before opening a Session")
        self.prepare_session()
        return self.root

    def prepare_session(self):
        from aka.workflow.light.host.workspace import publish
        from aka.resources.managed_assets import ensure_readmes
        # Install only declared resources. Never copy/restore source or touch Git.
        self.plugin_registry.install(self.root)
        self.plugin_registry.refresh_custom_skills(self.root)
        self.reserved_paths = tuple(self.plugin_registry.installed_paths(self.root))
        ensure_readmes(self.root)
        for name in ("records", "research-library", "scratch"):
            with open_private_directory(self.root / name):
                pass
        progress = self.root / "PROGRESS.md"
        if not progress.exists():
            publish(self.root, "PROGRESS.md", b"# Current state\n\nNo measurements yet.\n")
        if progress.is_symlink() or not stat.S_ISREG(progress.lstat().st_mode):
            raise ValueError("PROGRESS.md must be a regular file")
        instructions = resource_path("reference/CLAUDE.md").read_text()
        publish(self.root, "CLAUDE.md", (instructions + "\n" + self.workspace_policy).encode())
        self.reserved_paths += ("CLAUDE.md", "PROGRESS.md", "records", "research-library")
        persistent = True
        policy = "This trajectory keeps one persistent Agent workspace and Git history across Episodes.\n"
        locations = ["root PROGRESS.md", "records/<gateway_record_id>.json"]
        if self.research_modules:
            if "directions" in self.research_modules:
                locations.append(
                    "research-library/directions/<direction_id>/vN/direction.json"
                )
            if "experiments" in self.research_modules:
                locations.append(
                    "research-library/experiments/<experiment_id>/vN/experiment.json "
                    "and its archived references"
                )
        research_projection = (
            "Runtime projects current research state and per-Shape historical winners into "
            + " and ".join(locations)
            + ". `PROGRESS.md`, `records/`, and `research-library/` are disposable Runtime "
              "projections. These files are ignored by Kernel Git and regenerated after "
              "native-mode edits; Bubblewrap mounts them read-only. Change research state "
              "only through documented Runtime operations.\n"
        )
        from aka.workflow.light.host.workspace import publish
        source_policy = policy_at(self.source_workspace)
        git_lifetime = (
            "Local Git history persists across Episodes and Epochs. "
            if persistent
            else "Local Git history belongs to this Episode and its resumed Sessions only. "
        )
        publish(self.root, "WORKSPACE.md", (
            "# Workspace contract\n\n" + policy + source_instructions(source_policy) + "\n"
            "## Local state\n\n"
            + ("`scratch/` is trajectory-local and persists across Episodes; keep it bounded. "
               if persistent else
               "`scratch/` survives a resumed Session but starts empty in a new Episode. ")
            + "Custom `skills/` and `tools/` follow the inheritance policy above; update their "
              "README when contents change. Never store credentials here.\n\n"
            "Use this trajectory-private Git repository for checkpoints and rollback. Remotes, "
            "credentials, hooks, submodules and LFS are disabled. Commit the complete candidate and "
            "keep the worktree clean before a Kernel-bound Runtime request or `submit-kernel`; "
            "Runtime never commits Agent work on your behalf. HEAD is `kernel_commit_id`; all Kernel "
            "commit IDs shown here come from local or imported peer Git, never Supervisor Git. "
            "`submit-kernel` requires saved passing full-Evaluate evidence for one exact commit and "
            "applies final-source policy without launching another evaluation, promoting, rolling "
            "back, or changing HEAD. "
            + git_lifetime + "Imported peers use remote-tracking refs such as "
            "`atrex/T2`; their measurement indexes are under "
            "`refs/notes/atrex-measurements`.\n\n"
            "## Runtime projections\n\n" + research_projection +
            "\nSupervisor-private Session archives are retained for audit and token accounting but are "
            "not projected into the Agent workspace. Use Git, PROGRESS.md, research records and Gateway "
            "Records for durable optimization state.\n"
        ).encode())
        from aka.task.source.formats import pinned_paths
        fixed = pinned_paths(self.source_workspace)
        return tuple(name for name in self.reserved_paths if (self.root / name).exists()) + ("WORKSPACE.md", *fixed)

    def publish(self):
        """The working tree already is the authoritative Agent workspace."""
        return None
