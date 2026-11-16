"""Private Campaign storage layout, independent of the HTTP server lifecycle."""

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path


SESSION_SHELL_GUARD = Path(__file__).resolve().parents[3] / "execution" / "runtime_tools" / "session_shell_guard.sh"
CAMPAIGN_MANIFEST = "campaign.json"
CAMPAIGN_LOCK = "campaign.lock"
CAMPAIGN_LAYOUT = "atrex.light.campaign.v3"
LEGACY_CAMPAIGN_LAYOUT = "atrex.light.campaign.v2"


def _v0_configuration(value: object) -> dict[str, str] | None:
    """Validate the immutable public identity of the Campaign's V0 source."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "kernel_commit_id",
        "kernel_artifact_digest",
    }:
        raise ValueError(
            "AKA Light Campaign V0 must contain kernel_commit_id and "
            "kernel_artifact_digest"
        )
    commit = value.get("kernel_commit_id")
    digest = value.get("kernel_artifact_digest")
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("AKA Light Campaign V0 commit must be a complete Git commit ID")
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError("AKA Light Campaign V0 digest must be a SHA-256 identity")
    return {
        "kernel_commit_id": commit,
        "kernel_artifact_digest": digest,
    }


@contextmanager
def _campaign_lock(root: Path):
    """Serialize all writes to the single Campaign manifest."""
    path = root / CAMPAIGN_LOCK
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("AKA Light Campaign lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _source_configuration(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the Campaign-wide immutable source binding."""
    if not isinstance(value, Mapping) or set(value) != {"file_system", "contract"}:
        raise ValueError(
            "AKA Light Campaign source must contain file_system and contract"
        )
    file_system = value.get("file_system")
    contract = value.get("contract")
    if not isinstance(file_system, str) or not file_system:
        raise ValueError("AKA Light Campaign source file_system must be non-empty text")
    if contract is not None and not isinstance(contract, Mapping):
        raise ValueError("AKA Light Campaign source contract must be an object or null")
    from aka.task.source.formats.registry import select

    normalized_contract = (
        None
        if contract is None
        else json.loads(json.dumps(dict(contract), allow_nan=False))
    )
    select(file_system).validate_policy(normalized_contract)
    return {"file_system": file_system, "contract": normalized_contract}


def _campaign_manifest(
    *,
    name: str,
    launch_composition: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    from aka.core.lock import validate_snapshot

    if not isinstance(name, str) or not name:
        raise ValueError("AKA Light Campaign name must be non-empty text")
    validate_snapshot(
        launch_composition,
        source=f"{CAMPAIGN_MANIFEST}:launch_composition",
    )
    return {
        "schema_version": 3,
        "layout": CAMPAIGN_LAYOUT,
        "name": name,
        "launch_composition": json.loads(
            json.dumps(launch_composition, allow_nan=False)
        ),
        "source": _source_configuration(source),
        "v0": None,
        "runtime_configuration": None,
    }


def _read_campaign_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("AKA Light Campaign manifest is unreadable") from error
    common = {
        "schema_version",
        "layout",
        "name",
        "launch_composition",
    }
    schema = value.get("schema_version") if isinstance(value, dict) else None
    layout = value.get("layout") if isinstance(value, dict) else None
    required = common | ({"source"} if schema == 3 else set())
    optional = {"runtime_configuration", "v0", "policy"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - optional
        or (schema, layout)
        not in {(2, LEGACY_CAMPAIGN_LAYOUT), (3, CAMPAIGN_LAYOUT)}
        or not isinstance(value.get("name"), str)
        or not value["name"]
        or not isinstance(value.get("launch_composition"), dict)
        or (
            value.get("runtime_configuration") is not None
            and not isinstance(value.get("runtime_configuration"), dict)
        )
    ):
        raise ValueError("AKA Light Campaign manifest has an unsupported schema")
    from aka.core.lock import validate_snapshot

    validate_snapshot(
        value["launch_composition"],
        source=f"{path}:launch_composition",
    )
    if schema == 3:
        value["source"] = _source_configuration(value["source"])
    value["v0"] = _v0_configuration(value.get("v0"))
    value.setdefault("runtime_configuration", None)
    return value


def _legacy_source_binding(directory: Path) -> dict[str, object] | None:
    selected_path = directory / "file-system.json"
    contract_path = directory / "source-contract.json"
    if not selected_path.exists() and not contract_path.exists():
        return None
    if not selected_path.is_file() or not contract_path.is_file():
        raise ValueError("Legacy private source state is incomplete")
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Legacy private source state is unreadable") from error
    if (
        not isinstance(selected, dict)
        or selected.get("schema_version") != 1
        or not isinstance(selected.get("name"), str)
        or not isinstance(contract, dict)
        or set(contract) != {"schema_version", "contract"}
        or contract.get("schema_version") != 1
    ):
        raise ValueError("Legacy private source state is invalid")
    return _source_configuration(
        {"file_system": selected["name"], "contract": contract["contract"]}
    )


def _legacy_source_directories(root: Path) -> tuple[Path, ...]:
    directories = [root / "source"]
    lineages = root / "lineages"
    if lineages.is_dir():
        directories.extend(
            path / "source"
            for path in sorted(lineages.iterdir())
            if path.is_dir() and path.name.startswith("t")
        )
    return tuple(directories)


def _retire_legacy_source_directories(root: Path) -> None:
    for directory in _legacy_source_directories(root):
        for name in ("file-system.json", "source-contract.json"):
            (directory / name).unlink(missing_ok=True)
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # Never remove unrelated historical evidence from a non-empty path.
            pass


def initialize_campaign_root(
    root: Path,
    *,
    name: str,
    launch_composition: Mapping[str, object],
    source: Mapping[str, object],
) -> Path:
    """Create or validate the non-Git container selected by ``--workspace``.

    A Light Campaign is an orchestration/evidence directory, not a candidate
    repository.  Only ``lineages/tNNNN/agent-workspace`` owns Git history.
    Keeping this marker at the root makes path resolution explicit without a
    parent-directory hash or a hidden wrapper directory.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / ".git").exists():
        raise ValueError(
            "AKA Light --workspace must be a Campaign container, not a Git "
            "repository; choose a new empty directory"
        )
    detached_composition = root / ".atrex_plugins" / "composition.json"
    if detached_composition.exists():
        raise ValueError(
            "AKA Light Campaign launch composition must live only in "
            "campaign.json; remove the detached .atrex_plugins/composition.json"
        )
    manifest = root / CAMPAIGN_MANIFEST
    expected = _campaign_manifest(
        name=name,
        launch_composition=launch_composition,
        source=source,
    )
    with _campaign_lock(root):
        if manifest.exists():
            current = _read_campaign_manifest(manifest)
            if current["name"] != name:
                raise ValueError(
                    "AKA Light Campaign name differs from this launch"
                )
            from aka.core.lock import reconcile_snapshots

            reconcile_snapshots(
                current["launch_composition"],
                expected["launch_composition"],
                source=f"{manifest}:launch_composition",
            )
            if current.get("schema_version") == 3:
                if current["source"] != expected["source"]:
                    raise ValueError(
                        "AKA Light Campaign source contract differs from this launch"
                    )
                return root

            # One-time migration from the former detached source/ state. Every
            # legacy lineage must agree with the newly prepared Campaign binding.
            for directory in _legacy_source_directories(root):
                legacy = _legacy_source_binding(directory)
                if legacy is not None and legacy != expected["source"]:
                    raise ValueError(
                        "Legacy Campaign source contract differs from this launch"
                    )
            expected["runtime_configuration"] = current.get("runtime_configuration")
            expected["v0"] = current.get("v0")
            from aka.core.filesystem import durable_write_json

            durable_write_json(manifest, expected)
            _retire_legacy_source_directories(root)
            return root
        existing = {path.name for path in root.iterdir()} - {CAMPAIGN_LOCK}
        if existing:
            raise ValueError(
                "A new AKA Light Campaign workspace must be empty; composition state "
                "is written atomically inside campaign.json"
            )
        from aka.core.filesystem import durable_write_json

        durable_write_json(manifest, expected)
    return root


def campaign_source_configuration(
    workspace: Path,
) -> tuple[str, dict | None] | None:
    """Return the v3 Campaign source binding without consulting Agent files."""
    root = supervisor_campaign_base_root(Path(workspace))
    manifest = root / CAMPAIGN_MANIFEST
    if not manifest.is_file():
        return None
    value = _read_campaign_manifest(manifest)
    source = value.get("source")
    if not isinstance(source, dict):
        return None
    contract = source["contract"]
    return source["file_system"], contract


def _direct_campaign_identity(workspace: Path) -> tuple[Path, int | None] | None:
    """Resolve the Campaign layout from its root or an Agent workspace."""
    workspace = workspace.resolve()
    if (workspace / CAMPAIGN_MANIFEST).is_file():
        _read_campaign_manifest(workspace / CAMPAIGN_MANIFEST)
        return workspace, None
    # Resolve an Agent workspace or any lineage-private path to the same
    # namespace; never redirect from metadata stored in Agent files.
    for lineage in (workspace, *workspace.parents):
        if (
            lineage.parent.name == "lineages"
            and lineage.name.startswith("t")
        ):
            root = lineage.parent.parent
            suffix = lineage.name[1:]
        else:
            continue
        if (root / CAMPAIGN_MANIFEST).is_file():
            _read_campaign_manifest(root / CAMPAIGN_MANIFEST)
            if suffix.isdigit() and int(suffix) > 0:
                return root, int(suffix)
    return None


def legacy_supervisor_campaign_root(workspace: Path) -> Path:
    """Return the pre-pooled-layout root derived only from this workspace."""
    workspace = workspace.resolve()
    scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return workspace.parent / ".atrex-supervisor-runtime" / scope


def _pooled_campaign_identity(workspace: Path) -> tuple[Path, int] | None:
    """Return the validated pooled Campaign root and trajectory identity."""
    workspace = workspace.resolve()
    marker = workspace / ".git" / "atrex-search-origin.json"
    try:
        metadata = marker.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024:
            raise ValueError("Pooled-search origin marker is not a bounded regular file")
        value = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("Pooled-search origin marker is unreadable") from error
    campaign_value = value.get("campaign") if isinstance(value, dict) else None
    trajectory = value.get("trajectory") if isinstance(value, dict) else None
    if (
        not isinstance(campaign_value, str)
        or not campaign_value
        or type(trajectory) is not int
        or trajectory <= 0
    ):
        raise ValueError("Pooled-search origin marker has an invalid Campaign identity")
    campaign = Path(campaign_value).expanduser().resolve()
    campaign_root = legacy_supervisor_campaign_root(campaign)
    expected = campaign_root / "search" / "trajectories" / f"t{trajectory:04d}"
    if workspace != expected.resolve():
        raise ValueError("Pooled-search origin marker does not match its trajectory workspace")
    return campaign_root, trajectory


def supervisor_campaign_base_root(workspace: Path) -> Path:
    """Return the single private root owned by the whole Campaign."""
    direct = _direct_campaign_identity(workspace)
    if direct is not None:
        return direct[0]
    identity = _pooled_campaign_identity(workspace)
    return identity[0] if identity is not None else legacy_supervisor_campaign_root(workspace)


def supervisor_campaign_root(workspace: Path) -> Path:
    """Return the Campaign-private lineage namespace for this workspace.

    Pooled controller workspaces carry a trusted origin marker written before
    the Agent repository is created.  Validate that the workspace is exactly
    the trajectory path below the named Campaign before using the central
    Campaign root; arbitrary repositories cannot redirect private state.
    """
    workspace = workspace.resolve()
    direct = _direct_campaign_identity(workspace)
    if direct is not None:
        campaign_root, trajectory = direct
        return (
            campaign_root
            if trajectory is None
            else campaign_root / "lineages" / f"t{trajectory:04d}"
        )
    identity = _pooled_campaign_identity(workspace)
    if identity is None:
        return legacy_supervisor_campaign_root(workspace)
    campaign_root, trajectory = identity
    return campaign_root / "lineages" / f"t{trajectory:04d}"


def record_campaign_runtime_configuration(
    workspace: Path,
    configuration: Mapping[str, object],
) -> Path | None:
    """Atomically attach the effective Runtime snapshot to ``campaign.json``.

    Compatibility workspaces without the current Campaign manifest remain
    readable, but Runtime never recreates their detached ``runtime-config.json``.
    """
    if not isinstance(configuration, Mapping):
        raise TypeError("Runtime configuration snapshot must be a mapping")
    normalized = json.loads(json.dumps(dict(configuration), allow_nan=False))
    if not isinstance(normalized, dict):
        raise TypeError("Runtime configuration snapshot must be a JSON object")
    root = supervisor_campaign_base_root(workspace)
    manifest = root / CAMPAIGN_MANIFEST
    if not manifest.is_file():
        return None
    with _campaign_lock(root):
        current = _read_campaign_manifest(manifest)
        current["runtime_configuration"] = normalized
        from aka.core.filesystem import durable_write_json

        durable_write_json(manifest, current)
    return manifest


def record_campaign_v0(
    workspace: Path,
    *,
    kernel_commit_id: str,
    kernel_artifact_digest: str,
) -> Path | None:
    """Attach the archived V0 source identity to ``campaign.json`` once.

    The source bytes remain authoritative in
    ``measurements/kernels/<kernel_commit_id>/kernel.py``.  The Campaign
    manifest stores only the stable Agent commit and its canonical source-tree
    digest, avoiding a second initial-source snapshot.
    """
    identity = _v0_configuration(
        {
            "kernel_commit_id": kernel_commit_id,
            "kernel_artifact_digest": kernel_artifact_digest,
        }
    )
    assert identity is not None
    root = supervisor_campaign_base_root(workspace)
    manifest = root / CAMPAIGN_MANIFEST
    if not manifest.is_file():
        return None

    from aka.task.source.formats import MAX_SNAPSHOT
    from aka.core.filesystem import read_regular_bytes

    artifact = (
        supervisor_measurement_root(workspace)
        / "kernels"
        / kernel_commit_id
        / "kernel.py"
    )
    try:
        contents = read_regular_bytes(artifact, limit=MAX_SNAPSHOT + 1)
    except FileNotFoundError as error:
        raise ValueError(
            "Campaign V0 source must be archived in MeasurementStore before "
            "recording its manifest identity"
        ) from error
    if "sha256:" + hashlib.sha256(contents).hexdigest() != kernel_artifact_digest:
        raise ValueError("Campaign V0 MeasurementStore archive has the wrong digest")

    with _campaign_lock(root):
        current = _read_campaign_manifest(manifest)
        recorded = current.get("v0")
        if recorded is not None and recorded != identity:
            raise ValueError("AKA Light Campaign V0 identity cannot change")
        current["v0"] = identity
        from aka.core.filesystem import durable_write_json

        durable_write_json(manifest, current)
    return manifest


def campaign_runtime_configuration(workspace: Path) -> dict[str, object] | None:
    """Read the effective Runtime snapshot, including old detached evidence."""
    manifest = supervisor_campaign_base_root(workspace) / CAMPAIGN_MANIFEST
    if manifest.is_file():
        current = _read_campaign_manifest(manifest)
        value = current.get("runtime_configuration")
        if isinstance(value, dict):
            return value

    # Read-only compatibility for Campaigns created before the snapshot was
    # folded into campaign.json. Current Runtime never writes this path.
    legacy = supervisor_campaign_root(workspace) / "runtime-config.json"
    try:
        metadata = legacy.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            return None
        value = json.loads(legacy.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def supervisor_measurement_root(workspace: Path) -> Path:
    """Return the one canonical Measurement Store for the Campaign."""
    return supervisor_campaign_base_root(workspace) / "measurements"


def supervisor_submission_root(workspace: Path) -> Path:
    """Return the one immutable submit-kernel audit store for the Campaign."""
    return supervisor_campaign_base_root(workspace) / "submissions"


def supervisor_review_root(workspace: Path) -> Path:
    """Return the Campaign staging root for reviews before object attachment.

    Completed reviews live beside the Direction/Experiment revision or Kernel
    submission they reviewed.  This root contains only short-lived staging
    directories and legacy migration inputs, never an independent review index.
    """
    return supervisor_campaign_base_root(workspace) / ".review-staging"


def supervisor_runtime_audit_root(workspace: Path) -> Path:
    """Return the lineage-private Runtime request audit root."""
    return supervisor_campaign_root(workspace) / "runtime-audit"


def runtime_audit_files(lineage_root: Path) -> list[Path]:
    """Return current request/plugin receipts and legacy flat receipts."""
    paths = [
        *lineage_root.glob("runtime-audit/requests/request-*.json"),
        *lineage_root.glob("runtime-audit/plugin-calls/request-*.json"),
        *lineage_root.glob("request-*.json"),
    ]
    return sorted({path.resolve(): path for path in paths}.values())


def supervisor_research_catalog_root(workspace: Path) -> Path:
    """Return append-only research ordering/audit metadata for the Campaign.

    Per-lineage Research Stores continue to own local queue order, focus and
    leases.  Version content lives in the Campaign Research Library; this log
    supplies stable sequence/watermark metadata for synchronous visibility and
    private mutation audit.
    """
    return supervisor_campaign_base_root(workspace) / "research-catalog"


def supervisor_research_library_root(workspace: Path) -> Path:
    """Return the one canonical Direction/Experiment library for the Campaign.

    Agent workspaces contain disposable, visibility-filtered copies of this
    tree.  The canonical bytes live once at Campaign scope; the Catalog keeps
    append-only audit/order metadata and is not an Agent-facing projection.
    """
    return supervisor_campaign_base_root(workspace) / "research-library"


def supervisor_agent_control_root(workspace: Path) -> Path:
    """Return Session-control storage outside the Agent's Kernel repository.

    Hooks need a small writable rendezvous for active-time reminders and
    output-limit recovery.  Keeping it below the Kernel workspace used to make
    Supervisor state Agent-visible and dirty Git.  The control directory is
    mounted explicitly for managed bwrap Sessions; it is never candidate
    source or projected research evidence.
    """
    return supervisor_campaign_root(workspace) / "control"


def legacy_resource_root() -> Path:
    """Return the profile-selected public checkout used by legacy resource adapters.

    Installed launches bind this value through the launch profile. The source-tree
    fallback keeps direct developer imports usable without recreating the deleted
    ``aka.legacy.application.paths`` compatibility module.
    """
    configured = os.environ.get("AKA_LEGACY_REPO_ROOT")
    root = (Path(configured).expanduser().resolve() if configured
            else Path(__file__).resolve().parents[4])
    if not (root / "src" / "aka").is_dir():
        raise RuntimeError(
            "AKA_LEGACY_REPO_ROOT must name the matching AKA checkout selected by the profile"
        )
    return root
