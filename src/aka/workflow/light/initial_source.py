"""Initial source identity without an implicit GPU measurement."""
import hashlib

from aka.experiment.measurement_store import MeasurementStore
from aka.task.source.formats import Artifact

from .support.paths import (
    _read_campaign_manifest, record_campaign_v0,
    supervisor_campaign_base_root, supervisor_measurement_root,
)


def initialize_source(campaign) -> dict:
    """Archive the initial Agent commit; no evaluator or executor is invoked."""
    workspace = campaign.workspace
    source = Artifact(campaign._selected_initial_source)
    commit = campaign.candidate_workspace.initialize_agent(
        workspace, campaign._prepared_initial_content)
    committed = campaign.candidate_workspace.read_revision(workspace, commit, list(source.files))
    if dict(committed.files) != source.files:
        raise ValueError("Initial Agent commit differs from the selected source")
    store = MeasurementStore(supervisor_measurement_root(workspace))
    identity = store.kernel(source.snapshot, commit)
    record_campaign_v0(workspace, kernel_commit_id=commit,
                       kernel_artifact_digest=identity["kernel_artifact_digest"])
    # Archiving source must not fabricate an Evaluate Record or reference score.
    return identity


def prepare_campaign(campaign) -> None:
    """Continue the existing Agent repository, or initialize its source once."""
    root = supervisor_campaign_base_root(campaign.workspace)
    manifest = _read_campaign_manifest(root / "campaign.json")
    campaign.candidate_workspace.bind_source_contract(root, **manifest["source"])
    if campaign.candidate_workspace.current_revision(campaign.workspace):
        campaign._link_runtime()
    else:
        campaign.initialize_source()


def initial_kernel(campaign) -> dict:
    """Describe the shared Git root without treating it as a measured winner."""
    manifest = _read_campaign_manifest(
        supervisor_campaign_base_root(campaign.workspace) / "campaign.json")
    identity = manifest.get("v0")
    if identity is None:
        raise ValueError("Campaign initial source has not been archived")
    commit = identity["kernel_commit_id"]
    source = MeasurementStore(supervisor_measurement_root(campaign.workspace)).read_kernel(commit)
    checksum = hashlib.sha256(source).hexdigest()
    if "sha256:" + checksum != identity["kernel_artifact_digest"]:
        raise ValueError("Campaign initial source archive digest differs")
    expected = Artifact(source).files
    if dict(campaign.candidate_workspace.read_revision(campaign.workspace, commit, list(expected)).files) != expected:
        raise ValueError("Campaign initial source commit is missing or differs from its archive")
    return {"workspace": str(campaign.workspace), "commit": commit, "kernel_id": commit,
            "sha256": checksum, "record": {}, "trajectory": 0, "episode": 0,
            "latency_us_geomean": None}
