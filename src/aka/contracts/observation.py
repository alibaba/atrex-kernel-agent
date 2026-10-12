"""Versioned, read-only observations shared by workflow adapters and dashboards.

Adapters own workspace discovery, native file formats, stage names and settlement
semantics. Readers return JSON-compatible dictionaries; the HTTP/UI layer does
not inspect workflow files. Missing usage stays null and never means zero.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypedDict, Union

OBSERVATION_VERSION = 1
Measurement = Literal["exact", "partial", "unavailable"]


class ObservationProtocolError(ValueError):
    def __init__(self, actual: Any):
        self.actual = actual
        super().__init__(f"Unsupported observation protocol {actual!r}; expected {OBSERVATION_VERSION}")


def check_observation_version(actual: Any) -> None:
    if type(actual) is not int or actual != OBSERVATION_VERSION:
        raise ObservationProtocolError(actual)


class Stage(TypedDict):
    id: str
    label: str


class TokenUsage(TypedDict):
    """Backend totals include its own cache accounting; do not sum token types."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    cache_write_tokens: Optional[int]
    total_tokens: Optional[int]
    measurement: Measurement


class TokenTotal(TypedDict):
    total: Optional[int]
    measurement: Measurement
    scope: str


class CampaignTokens(TokenTotal):
    """One settled ledger, never the sum of overlapping receipts and telemetry."""

    control_total: Optional[int]
    receipt_count: int
    reason_codes: list[str]
    pending: bool


class EpisodeTokens(TypedDict):
    total: Optional[int]
    control_tokens: Optional[int]
    raw_control_tokens: Optional[int]
    measurement: Measurement
    terminal_usage: TokenUsage
    attribution_measurement: Measurement
    reason_codes: list[str]


class TerminalRecord(TypedDict):
    status: Optional[str]
    updated_at: Optional[str]
    stale: bool


class _IssueDetail(TypedDict, total=False):
    detail: str
    expected: Any
    actual: Any


class Issue(_IssueDetail):
    file: str
    code: str


class Progress(TypedDict):
    phase: str
    phase_label: str
    stage_id: Optional[str]
    episode: Optional[int]
    version: Optional[int]
    mode: Optional[str]
    experiment_count: Optional[int]
    latest_experiment: dict[str, Any]
    updated_at: Optional[str]
    record_updating: bool
    pending_tokens: bool
    last_terminal: TerminalRecord


class PhaseUsage(TypedDict):
    phase: str
    label: str
    usage: TokenUsage


class Counts(TypedDict):
    episodes: Optional[int]
    accepted: Optional[int]
    rejected: Optional[int]


class Incumbent(TypedDict):
    version: int
    value: float
    unit: str
    label: str
    objective: str
    series: Union[str, int]
    source: str


class PerformanceRecord(TypedDict):
    version: int
    promoted: bool
    episode: Optional[int]
    incumbent: Optional[Incumbent]


class Canonical(TypedDict):
    latest_version: Optional[int]
    incumbent: Optional[Incumbent]
    history: list[PerformanceRecord]


class EpisodeSummary(TypedDict):
    episode: int
    version: Optional[int]
    mode: Optional[str]
    status: Optional[str]
    accepted: bool
    summary: Optional[str]
    violation: Optional[str]
    verification_gate: Optional[str]
    tokens: EpisodeTokens


class Artifact(TypedDict):
    """Opaque IDs use letters, digits, underscores and hyphens; names are basenames."""

    id: str
    label: str
    filename: str
    content_type: str
    kind: Literal["file", "bundle"]
    size_bytes: Optional[int]


class _CampaignArtifacts(TypedDict, total=False):
    artifacts: list[Artifact]
    artifact_scope: str


class CampaignSummary(TypedDict):
    id: str
    name: str
    display_name: str
    framework: Optional[str]
    platform: Optional[str]
    updated_at: Optional[str]
    progress: Progress
    tokens: CampaignTokens
    counts: Counts


class Campaign(CampaignSummary, _CampaignArtifacts):
    observation_version: int
    workspace: str
    arch: Optional[str]
    stages: list[Stage]
    canonical: Canonical
    episodes: list[EpisodeSummary]
    issues: list[Issue]


class Campaigns(TypedDict):
    observation_version: int
    campaigns: list[CampaignSummary]
    tokens: TokenTotal
    empty_message: str


class Episode(TypedDict):
    observation_version: int
    episode: int
    version: Optional[int]
    mode: Optional[str]
    status: str
    accepted: Optional[bool]
    summary: Optional[str]
    violation: Optional[str]
    verification: dict[str, Any]
    tokens: EpisodeTokens
    phases: list[PhaseUsage]
    experiments: list[dict[str, Any]]
    latest_experiment: Optional[dict[str, Any]]
    detail_available: bool
    issues: list[Issue]
    evidence: list[str]


class ObservationReader(Protocol):
    """One workspace session; calls may overlap and must not mutate workflow data.

    Campaign IDs use letters, digits, underscores and hyphens. Unknown IDs or
    episodes raise KeyError. Stages and phase usage rows are ordered by the
    adapter, with display labels included. ``canonical.incumbent``
    carries value, unit, label and objective; history uses adapter-defined series
    boundaries. Token totals retain their documented scope and measurement.
    """

    def campaigns(self) -> Campaigns: ...
    def campaign(self, identifier: str) -> Campaign: ...
    def episode(self, identifier: str, number: int) -> Episode: ...


class ArtifactReader(Protocol):
    """Optional v1 capability for readers declaring campaign.artifacts.

    Only declared IDs may be downloaded. Missing or removed outputs raise
    KeyError; reading must not write workflow files. The source owns file access
    and packaging. Dashboard never accepts a client-supplied filesystem path.
    Existing v1 readers may omit artifacts and this capability entirely.
    """

    def artifact(self, identifier: str, artifact_id: str) -> bytes: ...


class ObservationSource(Protocol):
    """Injectable factory; constructing it does not scan or bind resources.

    Preview files may only be written to the temporary directory supplied by
    the dashboard. No Context/Root is retained by a source or reader.
    """

    protocol_version: int

    def open(self, workspace: Path) -> ObservationReader: ...
    def create_preview(self, temporary_directory: Path) -> Path: ...
