"""Persistent Episode drafts with an isolated Agent-local Git repository."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from aka.light.agent.home import open_private_directory
from aka.legacy.task.persistence import durable_write_json
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.workspace.sources import (
    snapshot as source_snapshot,
    install as install_source,
    policy_at,
    instructions as source_instructions,
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
    def __init__(self, worktree: Path, directory: Path, *, is_ppu: bool = False,
                 research_records: bool = True, research_modules=("directions", "experiments"), plugin_registry=None,
                 inherit_state: bool = True, adaptive_root: Path | None = None,
                 episode: int | None = None, workspace_policy: str = "",
                 persistent_root: Path | None = None):
        if plugin_registry is None:
            from aka.light.runtime.catalog import PluginRegistry
            plugin_registry = PluginRegistry()
        self.worktree = worktree
        self.persistent = persistent_root is not None
        # ``directory`` remains Episode-scoped evidence storage.  With the
        # default inheritance policy the Agent repository itself is instead
        # trajectory-scoped and survives every Episode.
        self.root = (
            persistent_root.resolve()
            if persistent_root is not None
            else directory / "workspace"
        )
        self.state_path = (
            self.root.parent / "agent-workspace-state.json"
            if persistent_root is not None
            else directory / "state.json"
        )
        self.is_ppu = is_ppu
        self.research_records = research_records
        self.research_modules = tuple(research_modules)
        self.plugin_registry = plugin_registry
        self.inherit_state = inherit_state
        if not isinstance(workspace_policy, str):
            raise TypeError("workspace_policy must be text")
        self.workspace_policy = workspace_policy
        self.adaptive_root = adaptive_root
        from aka.light.workspace.adaptive import AdaptiveState, reserved_paths
        self.reserved_paths = reserved_paths(plugin_registry)
        self.adaptive = (AdaptiveState(adaptive_root, episode, self.reserved_paths)
                         if adaptive_root is not None else None)
        from aka.light.workspace.agent_git import AgentGit
        git_state_root = adaptive_root or self.state_path.parent
        self.agent_git = AgentGit(self.root, git_state_root / "agent-git.bundle")

    def prepare(self) -> Path:
        from aka.light.workspace.files import publish
        fresh = not self.state_path.exists()
        state = {} if fresh else json.loads(_read(self.state_path))
        persistent = self.persistent
        workspace_mode = "persistent" if persistent else "episode"
        if not fresh and state.get("workspace_mode", workspace_mode) != workspace_mode:
            raise ValueError("Cannot change Agent workspace persistence while resuming")
        if not fresh and state.get("research_records", True) != self.research_records:
            raise ValueError("Cannot change research_records for an existing Episode draft")
        if not fresh and tuple(state.get("research_modules", ("directions", "experiments"))) != self.research_modules:
            raise ValueError("Cannot change research_modules for an existing Episode draft")
        if "inherit_state" in state and state["inherit_state"] != self.inherit_state:
            raise ValueError("Cannot change inherit_state while resuming an existing Episode")
        with open_private_directory(self.root):
            pass
        if fresh and self.adaptive is not None:
            # Pin the seed before copying. A restart during preparation must not
            # choose a newer snapshot or overlay it on a half-materialized tree.
            seed_path = self.state_path.parent / "adaptive-seed.json"
            try:
                seed = json.loads(_read(seed_path))
                if seed["inherit_state"] != self.inherit_state:
                    raise ValueError("Cannot change inherit_state during draft preparation")
            except FileNotFoundError:
                seed = {"inherit_state": self.inherit_state,
                        "snapshot": self.adaptive.previous() if self.inherit_state else None}
                durable_write_json(seed_path, seed)
            # These are incomplete initialization leftovers, not a running draft.
            # Move them out of the Agent view rather than deleting user data.
            import uuid
            for tree in ("skills", "tools"):
                with open_private_directory(self.root) as parent:
                    try:
                        os.stat(tree, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    backup = self.state_path.parent / "preparation-backups" / uuid.uuid4().hex
                    with open_private_directory(backup) as target:
                        os.rename(tree, tree, src_dir_fd=parent, dst_dir_fd=target)
            if seed["snapshot"] is not None:
                self.adaptive.restore(self.root, seed["snapshot"])
            state["adaptive_seed"] = seed["snapshot"]
        controller_policy = policy_at(self.worktree, committed=True)
        source = source_snapshot(self.worktree, policy=controller_policy)
        digest = hashlib.sha256(source).hexdigest()
        # An Episode-scoped view follows the controller source exactly. A
        # trajectory-persistent repository is initialized only once and then
        # keeps rejected candidates, branches and WIP across Episode boundaries.
        # Source contracts are Campaign identities and cannot change in place.
        contract_changed = False
        if persistent and not fresh:
            try:
                contract_changed = policy_at(self.root) != controller_policy
            except (OSError, ValueError):
                contract_changed = True
        if contract_changed:
            raise ValueError(
                "Persistent Agent workspace source contract differs from the controller; "
                "start a new Campaign instead of replacing trajectory history"
            )
        if fresh or (not persistent and digest != state.get("kernel_digest")):
            install_source(self.root, source, bind=True)
        for name in PUBLIC_FILES:
            # Manifest bytes belong to the draft/snapshot, not immutable public inputs.
            from aka.light.workspace.sources import layout_at
            if layout_at(self.worktree, committed=True).owns(name) or layout_at(self.root).owns(name):
                continue
            if name == "CLAUDE.md":
                # Instructions follow the running controller, not an old V0's
                # retired Setup/Fast workflow. Do not rewrite the committed V0.
                content = _claude_instructions(self.workspace_policy)
                publish(self.root, name, content)
                continue
            try:
                content = _read(self.worktree / name)
            except FileNotFoundError:
                continue
            publish(self.root, name, content)
        # Establish Agent history before projecting controller-authored memory,
        # so every Git identity in that projection can be translated into the
        # local/peer Agent namespace.
        self.agent_git.prepare()
        with open_private_directory(self.root / "memory"):
            pass
        for path in (self.worktree / "memory").glob("v*.json"):
            if path.stem[1:].isascii() and path.stem[1:].isdigit():
                content = _read(path)
                try:
                    value = json.loads(content)
                except (UnicodeError, json.JSONDecodeError):
                    value = None
                if isinstance(value, dict):
                    from aka.light.workspace.agent_git import GIT_COMMIT_ID
                    from aka.light.workspace.sources import from_git

                    def project_commit_ids(item, key=""):
                        if isinstance(item, dict):
                            return {
                                name: project_commit_ids(child, name)
                                for name, child in item.items()
                            }
                        if isinstance(item, list):
                            return [project_commit_ids(child, key) for child in item]
                        if (
                            isinstance(item, str)
                            and "commit" in key
                            and GIT_COMMIT_ID.fullmatch(item) is not None
                        ):
                            if self.agent_git.commit_reachable(item):
                                return item
                            try:
                                source_at_commit = from_git(self.worktree, item)
                            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                                return None
                            return self.agent_git.find_kernel_commit(source_at_commit)
                        return item

                    content = (
                        json.dumps(
                            project_commit_ids(value),
                            indent=2,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode()
                publish(self.root, f"memory/{path.name}", content)
        for name in DIAGNOSTIC_TREES:
            with open_private_directory(self.root / name):
                pass
            if fresh:
                for relative, content in bounded_tree_entries(self.worktree / name):
                    publish(self.root, f"{name}/{relative}", content)
        # These are controller-created asset links, never copied from Agent files.
        # Do not install the evaluator or copy .git / control checkpoints.
        plugin_snapshot = self.plugin_registry.snapshot()
        # Refresh only reserved assets; custom Skills/tools survive Session resumes.
        self.prepare_session()
        durable_write_json(self.state_path, dict(state, kernel_digest=digest,
                                               controller_kernel_digest=digest,
                                               workspace_mode=workspace_mode,
                                               research_records=self.research_records,
                                               research_modules=list(self.research_modules),
                                               inherit_state=self.inherit_state,
                                               runtime_plugins=plugin_snapshot))
        return self.root

    def prepare_session(self) -> tuple[str, ...]:
        """Restore managed clients/Skills and refresh discovery, never user files."""
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
                prefix="retired-history-projection-", dir=self.state_path.parent
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
        from aka.light.workspace.assets import link_runtime, link_custom_skills
        from aka.light.workspace.adaptive import ensure_readmes
        link_runtime(self.root, is_ppu=self.is_ppu, research_records=self.research_records,
                     research_modules=self.research_modules,
                     plugin_registry=self.plugin_registry)
        ensure_readmes(self.root)
        link_custom_skills(self.root, self.reserved_paths)
        persistent = self.persistent
        policy = ("This trajectory uses one stable Agent repository across Episodes. Kernel WIP, "
                  "local Git branches/tags, scratch/, custom skills/ and tools/ remain in place; "
                  "commit review never rewrites or promotes that history." if persistent else
                  "Custom skills/ and tools/ are saved after each Session and inherited by the next "
                  "Episode, regardless of Kernel promotion." if self.inherit_state else
                  "Cross-Episode inheritance is disabled. Custom skills/ and tools/ are retained "
                  "during this Episode only; the next Episode starts from packaged defaults.")
        locations = ["root PROGRESS.md", "records/<gateway_record_id>.json"]
        if self.research_modules:
            if "directions" in self.research_modules:
                locations.append("directions/<status>/<direction_id>.json")
            if "experiments" in self.research_modules:
                locations.append(
                    "experiments/<experiment_id>/experiment.json and archived references"
                )
        research_projection = (
            "Runtime projects current research state and per-Shape historical winners into "
            + " and ".join(locations)
            + ". These read-only files are regenerated from private Runtime state and ignored by "
              "Kernel Git; change research state only through documented Runtime operations.\n"
        )
        from aka.light.workspace.files import publish
        source_policy = policy_at(self.worktree, committed=True)
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
            + git_lifetime + "Imported peers are under "
            "`refs/atrex/shared/` and their measurement indexes under "
            "`refs/notes/atrex-measurements`.\n\n"
            "## Runtime projections\n\n" + research_projection +
            "\nSupervisor-private Session archives are retained for audit and token accounting but are "
            "not projected into the Agent workspace. Use Git, PROGRESS.md, research records and Gateway "
            "Records for durable optimization state.\n"
        ).encode())
        from aka.light.workspace.sources import pinned_paths
        fixed = pinned_paths(self.worktree)
        return tuple(name for name in self.reserved_paths if (self.root / name).exists()) + ("WORKSPACE.md", *fixed)

    def publish(self) -> None:
        from aka.light.workspace.files import publish

        state = json.loads(_read(self.state_path))
        state["agent_git"] = self.agent_git.capture()
        state["agent_git_checkpoint"] = self.agent_git.persist()
        if self.adaptive is not None:
            # Checkpoint even when this Session fails or its candidate is rejected.
            # Do not import/execute these files in the Supervisor.
            state["adaptive_snapshot"] = self.adaptive.save(self.root)
            durable_write_json(self.state_path, state)
        content = source_snapshot(self.root, policy=policy_at(self.worktree, committed=True))
        install_source(self.worktree, content)
        for name in DIAGNOSTIC_TREES:
            for relative, data in bounded_tree_entries(self.root / name):
                publish(self.worktree, f"{name}/{relative}", data)
        durable_write_json(self.state_path, dict(state, kernel_digest=hashlib.sha256(content).hexdigest()))
