"""Candidate file system: the only owner of source layouts and snapshot encoding.

Workflows consume opaque snapshots and Artifact operations. Format implementations,
file ownership, archive descriptors and compatibility live in this package.
"""
from .storage import (
    POLICY, MAGIC, MAX_FILE, MAX_TOTAL, MAX_SNAPSHOT, WORKSPACE_ONLY,
    relative, ignored, source_name, editable, validate_policy, validate_files,
    validate_import_options, policy_at, pack, unpack, collect, snapshot, from_git,
    from_inputs, install, instructions, allowed_changes, commit_candidate,
    import_tree, initial_snapshot,
)
from .artifacts import (
    Artifact, ExecutionSource, Layout, workspace_identity, baseline_marker, read, read_path, revision_identity, revision_changed, from_revision, layout, layout_at, required_layout,
    archive_bytes, review_paths, commit_paths, pinned_paths, freeze_for_execution,
    stage_frozen, declared_paths, evidence_paths, typed_limitation, capture_paths, restore_capture,
    seed_baseline_metadata, measurement_paths, measurement_artifacts, language_hints,
)
