"""Persistent Agent workspace projected across Episode audit segments."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from pathlib import Path

from aka.legacy.task.persistence import durable_unlink, durable_write_json
from aka.light.agent.home import open_private_directory
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.workspace.sources import (
    install as install_source,
    instructions as source_instructions,
    policy_at,
    snapshot as source_snapshot,
)

EPISODE_WORKSPACE_ENV = "ATREX_EPISODE_WORKSPACE"
PUBLIC_FILES = (
    "README.md", "CLAUDE.md", "agent_problem.json", "reference.py", "input.py",
    "definition.json", "workload.jsonl", "solution.json", "shapes.json", "metadata.json", "roofline.json",
    "valid.py",
)
DIAGNOSTIC_TREES = ("scratch",)
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def _read(path: Path) -> bytes:
    try:
        value = read_regular_bytes(path, limit=MAX_FILE + 1)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"Episode diagnostic file must be a regular non-symlink file: {path.name}") from error
    if len(value) > MAX_FILE:
        raise ValueError(f"Episode file exceeds 16 MiB: {path.name}")
    return value


def _claude_instructions(workspace_policy: str) -> bytes:
    """Combine static Agent rules with the Supervisor-owned campaign policy."""
    from aka.light.workspace.resource_catalog import resource

    base = _read(resource("reference/CLAUDE.md")).rstrip()
    policy = workspace_policy.strip().encode()
    return base + (b"\n\n" + policy if policy else b"") + b"\n"


def bounded_tree_entries(root: Path):
    """Never follow mutable symlinks; enforce aggregate limits before publication."""
    count = total = 0
    if root.is_symlink():
        raise ValueError("Episode diagnostics cannot be symlinks")
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Episode diagnostic directories cannot be symlinks")
        for name in names:
            path = Path(parent) / name
            data = _read(path)
            count += 1
            total += len(data)
            if count > 4096 or total > MAX_TOTAL:
                raise ValueError("Episode diagnostics exceed 4096 files / 64 MiB")
            yield path.relative_to(root).as_posix(), data


class EpisodeWorkspace:
    def __init__(self, source_workspace: Path, directory: Path, *, is_ppu: bool = False,
                 research_records: bool = True, research_modules=("directions", "experiments"), plugin_registry=None,
                 workspace_policy: str = "",
                 persistent_root: Path | None = None):
        if plugin_registry is None:
            from aka.light.runtime.catalog import PluginRegistry
            plugin_registry = PluginRegistry()
        self.source_workspace = source_workspace
        self.persistent = persistent_root is not None
        # ``directory`` remains Episode-scoped evidence storage.  With the
        # default inheritance policy the Agent repository itself is instead
        # trajectory-scoped and survives every Episode.
        self.root = (
            persistent_root.resolve()
            if persistent_root is not None
            else directory / "workspace"
        )
        self.private_root = self.root.parent if self.persistent else directory
        # A persistent Agent repository is its own recovery authority. Its
        # lineage state is written independently by SupervisorRuntime; workspace
        # preparation must not add a second mutable state document beside it.
        self.state_path = None if self.persistent else directory / "state.json"
        self.is_ppu = is_ppu
        self.research_records = research_records
        self.research_modules = tuple(research_modules)
        self.plugin_registry = plugin_registry
        if not isinstance(workspace_policy, str):
            raise TypeError("workspace_policy must be text")
        self.workspace_policy = workspace_policy
        from aka.light.workspace.managed_assets import reserved_paths
        self.reserved_paths = reserved_paths(plugin_registry)
        from aka.light.workspace.agent_git import AgentGit
        # The real trajectory repository is the recovery authority. Light no
        # longer snapshots it into a per-Episode bundle.
        self.agent_git = AgentGit(self.root)

    def _retire_memory_tree(self) -> None:
        """Move the retired Agent-visible memory tree into private storage."""
        root = self.root / "memory"
        try:
            mode = root.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(mode) or root.is_symlink():
            raise ValueError("Agent memory path must be a real directory")
        backup = (
            self.private_root
            / "retired-agent-memory"
            / uuid.uuid4().hex
            / "memory"
        )
        with open_private_directory(self.root) as source, open_private_directory(
            backup.parent
        ) as destination:
            os.rename("memory", backup.name, src_dir_fd=source, dst_dir_fd=destination)

    def _retire_workspace_state(self) -> None:
        """Delete the superseded workspace checkpoint after Git is durable."""
        if not self.persistent:
            return
        legacy = self.private_root / "agent-workspace-state.json"
        try:
            metadata = legacy.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode) or legacy.is_symlink():
            raise ValueError(
                "Legacy Agent workspace state must be a regular non-symlink file"
            )
        durable_unlink(legacy)

    def prepare(self) -> Path:
        from aka.light.workspace.files import publish
        persistent = self.persistent
        fresh = (
            not (self.root / ".git").is_dir()
            if persistent
            else not self.state_path.exists()
        )
        state = (
            {}
            if fresh or self.state_path is None
            else json.loads(_read(self.state_path))
        )
        workspace_mode = "persistent" if persistent else "episode"
        if (
            not persistent
            and not fresh
            and state.get("workspace_mode", workspace_mode) != workspace_mode
        ):
            raise ValueError("Cannot change Agent workspace persistence while resuming")
        if (
            not persistent
            and not fresh
            and state.get("research_records", True) != self.research_records
        ):
            raise ValueError("Cannot change research_records for an existing Episode draft")
        if (
            not persistent
            and not fresh
            and tuple(
                state.get("research_modules", ("directions", "experiments"))
            )
            != self.research_modules
        ):
            raise ValueError("Cannot change research_modules for an existing Episode draft")
        with open_private_directory(self.root):
            pass
        # Rebind a missing private source contract before decoding historical
        # commits; the contract itself never enters Agent Git.
        from aka.light.workspace.sources.private_state import (
            bind as bind_source_state,
            read as read_source_state,
        )

        source_state = read_source_state(self.source_workspace)
        if source_state is None:
            agent_source_state = read_source_state(self.root)
            if agent_source_state is not None:
                file_system, contract = agent_source_state
                bind_source_state(
                    self.source_workspace,
                    file_system=file_system,
                    contract=contract,
                )
        source_policy = policy_at(self.source_workspace, committed=True)
        source = source_snapshot(self.source_workspace, policy=source_policy)
        digest = hashlib.sha256(source).hexdigest()
        # An Episode-scoped view follows the controller source exactly. A
        # trajectory-persistent repository is initialized only once and then
        # keeps rejected candidates, branches and WIP across Episode boundaries.
        # Source contracts are Campaign identities and cannot change in place.
        contract_changed = False
        if persistent and not fresh:
            try:
                contract_changed = policy_at(self.root) != source_policy
            except (OSError, ValueError):
                contract_changed = True
        if contract_changed:
            raise ValueError(
                "Persistent Agent workspace source contract differs from the Campaign; "
                "start a new Campaign instead of replacing trajectory history"
            )
        if fresh or (not persistent and digest != state.get("kernel_digest")):
            install_source(self.root, source, bind=True)
        for name in PUBLIC_FILES:
            # Manifest bytes belong to the draft/snapshot, not immutable public inputs.
            from aka.light.workspace.sources import layout_at
            if layout_at(self.source_workspace, committed=True).owns(name) or layout_at(self.root).owns(name):
                continue
            if name == "CLAUDE.md":
                # Instructions follow the running Campaign, not an old V0's
                # retired Setup/Fast workflow.
                content = _claude_instructions(self.workspace_policy)
                publish(self.root, name, content)
                continue
            try:
                content = _read(self.source_workspace / name)
            except FileNotFoundError:
                continue
            publish(self.root, name, content)
        # Establish the trajectory-local Agent history before Runtime binds the
        # workspace.  Historical controller memory is deliberately not copied:
        # Git, PROGRESS.md, research projections and Gateway Records are the
        # only durable Agent-facing optimization state.
        self._retire_memory_tree()
        self.agent_git.prepare()
        self._retire_workspace_state()
        for name in DIAGNOSTIC_TREES:
            with open_private_directory(self.root / name):
                pass
            if fresh:
                for relative, content in bounded_tree_entries(self.source_workspace / name):
                    publish(self.root, f"{name}/{relative}", content)
        # These are Supervisor-created asset links, never candidate source.
        # Do not install the evaluator or copy .git / control checkpoints.
        # Refresh only reserved assets; custom Skills/tools survive Session resumes.
        self.prepare_session()
        if self.state_path is not None:
            durable_write_json(
                self.state_path,
                dict(
                    state,
                    kernel_digest=digest,
                    controller_kernel_digest=digest,
                    workspace_mode=workspace_mode,
                    research_records=self.research_records,
                    research_modules=list(self.research_modules),
                    runtime_plugins=self.plugin_registry.snapshot(),
                ),
            )
        return self.root

    def prepare_session(self) -> tuple[str, ...]:
        """Restore managed clients/Skills and refresh discovery, never user files."""
        # The Campaign Library is projected as ordinary files so native and
        # Bubblewrap profiles share one layout.  Bubblewrap overlays this
        # stable directory read-only; the Supervisor can still refresh its
        # contents from outside the namespace without restarting the Session.
        if self.research_modules:
            with open_private_directory(self.root / "research-library"):
                pass
        # Releases before the persistent-Session design copied finalized prior
        # Episode conversations into a derived input subtree. That view is no
        # longer an Agent input: one native conversation continues across
        # audit segments, while complete captures stay Supervisor-private.
        input_root = self.root / "input"
        try:
            input_mode = input_root.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            input_mode = None
        if input_mode is not None:
            if not stat.S_ISDIR(input_mode):
                raise ValueError("Managed workspace input path must be a directory")
            with tempfile.TemporaryDirectory(
                prefix="retired-history-projection-", dir=self.private_root
            ) as temporary:
                with open_private_directory(input_root) as source, open_private_directory(
                    Path(temporary)
                ) as destination:
                    try:
                        os.rename(
                            "evidence",
                            "evidence",
                            src_dir_fd=source,
                            dst_dir_fd=destination,
                        )
                    except FileNotFoundError:
                        pass
        from aka.light.workspace.assets import link_custom_skills, link_runtime
        from aka.light.workspace.managed_assets import ensure_readmes
        link_runtime(self.root, is_ppu=self.is_ppu, research_records=self.research_records,
                     research_modules=self.research_modules,
                     plugin_registry=self.plugin_registry)
        ensure_readmes(self.root)
        link_custom_skills(self.root, self.reserved_paths)
        persistent = self.persistent
        policy = (
            "This trajectory uses one stable Agent repository across Episodes. Kernel WIP, "
            "local Git branches/tags, scratch/, custom skills/ and tools/ remain in place; "
            "commit review never rewrites or promotes that history."
        )
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
            + ". `research-library/` is a disposable, visibility-filtered copy of the "
              "Campaign Library. These files are ignored by Kernel Git and regenerated after "
              "native-mode edits; Bubblewrap mounts the tree read-only. Change research state "
              "only through documented Runtime operations.\n"
        )
        from aka.light.workspace.files import publish
        source_policy = policy_at(self.source_workspace, committed=True)
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
        from aka.light.workspace.sources import pinned_paths
        fixed = pinned_paths(self.source_workspace)
        return tuple(name for name in self.reserved_paths if (self.root / name).exists()) + ("WORKSPACE.md", *fixed)

    def publish(self) -> None:
        from aka.light.workspace.files import publish

        state = (
            json.loads(_read(self.state_path))
            if self.state_path is not None
            else None
        )
        if state is not None:
            state["agent_git"] = self.agent_git.capture()
        content = source_snapshot(
            self.root, policy=policy_at(self.source_workspace, committed=True)
        )
        # New Light uses the persistent Agent repository directly. The copy-back
        # branch remains only for legacy callers that still provide a separate
        # source workspace.
        if self.source_workspace.resolve() != self.root.resolve():
            install_source(self.source_workspace, content)
            for name in DIAGNOSTIC_TREES:
                for relative, data in bounded_tree_entries(self.root / name):
                    publish(self.source_workspace, f"{name}/{relative}", data)
        if self.state_path is not None and state is not None:
            durable_write_json(
                self.state_path,
                dict(
                    state,
                    kernel_digest=hashlib.sha256(content).hexdigest(),
                ),
            )
