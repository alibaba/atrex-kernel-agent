"""Global object IDs, independent of an Episode's physical evidence store."""

from __future__ import annotations

import re
import uuid

KERNEL_ARTIFACT_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
KERNEL_RECORD_ID_RE = re.compile(r"kernel-[0-9a-f]{32}")
GATEWAY_RECORD_ID_RE = re.compile(r"gateway-[0-9a-f]{32}")
DIRECTION_ID_RE = re.compile(r"direction_[0-9a-f]{32}")
EXPERIMENT_ID_RE = re.compile(r"experiment_[0-9a-f]{32}")


def kernel_id_for_digest(digest: str) -> str:
    """Return the same opaque UUID everywhere for the same exact source digest."""
    if KERNEL_ARTIFACT_DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("private Kernel Artifact Digest is invalid")
    # This namespace/name contract must remain stable across releases and hosts.
    return "kernel-" + uuid.uuid5(uuid.NAMESPACE_URL, f"urn:atrex:kernel-source:{digest}").hex


def validate_kernel_identity(identity: dict, digest: str) -> dict[str, str]:
    """Require the global content identity; no Episode-local IDs or aliases."""
    canonical = kernel_id_for_digest(digest)
    stored = identity.get("kernel_id")
    if identity.get("kernel_artifact_digest") != digest or stored != canonical:
        raise ValueError("Kernel identity record is inconsistent")
    return {"kernel_id": canonical, "kernel_artifact_digest": digest}
