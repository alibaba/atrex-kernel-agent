---
name: atrex-aka-light-observability
description: Inspect and export AKA Light campaign evidence with the eight supported observability tools. Use for lineage or trajectory status, Direction/Experiment state, token usage, evaluated or submitted Kernel history, accepted winners, Runtime and Agate call statistics, Gateway-to-Job tracing, raw evidence export, or diagnosing Light progress without mutating the campaign. Do not use for starting, stopping, restarting, cleaning, or modifying a campaign.
---

# AKA Light observability

Inspect current AKA Light Campaigns from durable evidence without changing Runtime, Agate,
Agent Git, Research Store, or scheduler state.

Pooled search has one Campaign-private state tree and one canonical Measurement Store. Treat
`lineages/tNNNN/` as lineage namespaces for Journals, Research, Sessions, and Runtime audits—not
as independent Campaign stores. Gateway Record bytes exist once and are attributed through the
originating Runtime audit.

## Establish the authority

Identify:

- a checkout compatible with the Campaign data format (`AKA_REPO`);
- the top-level Campaign workspace (`CAMPAIGN`), not a trajectory subdirectory.

Before a nontrivial inspection, read the complete authoritative command and field guide matching
the user's language:

```text
Chinese: $AKA_REPO/observability/USAGE.md
English: $AKA_REPO/observability/USAGE.en.md
```

For another language, use the English guide unless the user prefers Chinese. The two guides define
the same commands, options, fields, examples, and evidence semantics. When any observable contract
changes, update both guides together.

Use each script's `--help` as the final CLI authority. Do not reconstruct unsupported options
from older scripts or documentation.

## Select the smallest sufficient tool

| Question | Tool |
|---|---|
| Is each lineage alive; what is its Session, Direction/Experiment and token state? | `light_lineage_status.py` |
| Which evaluated Kernels exist, which submit-kernel decisions belong to each, and at what token cost? | `light_kernel_history.py` |
| Which verified accepted commit is best by GeoMean or individual Shape? | `light_kernel_best.py` |
| How many Runtime calls succeeded/failed/pended and how long did they take? | `light_runtime_calls.py` |
| How many physical Agate submissions succeeded/failed/pended and what were queue/execution times? | `light_agate_calls.py` |
| Export an exact Agent commit, its source tree and every bound measurement | `export_light_kernel.py` |
| Expand one logical Gateway Record into all physical submissions | `light_gateway_jobs.py` |
| Export Runtime request, all raw Agate attempts and Agent-visible response for one logical Gateway Record | `export_light_gateway_record.py` |

Start broad status requests with lineage status. Add Kernel history/best only when performance or
accepted output matters. Use Runtime/Agate detail filters only when diagnosing calls; their
summaries remain Campaign-wide.

## Preserve identity boundaries

Keep these separate:

```text
Agent Git commit
→ Runtime request ID
→ Gateway Record ID
→ primary Agate Job ID
→ optional Ray/backend Job ID
```

A Gateway Record is one logical measurement and can contain several Shape batches, retries, and
physical Jobs. A backend ID embedded in an outer Ray response does not imply that the backend's
independent raw response is available.

Public observability JSON uses `runtime_request_id`, `agate_job_id`, and
`backend_agate_job_ids` consistently. Preserve upstream `request_id`, `job_id`, or
`backend_job_ids` only inside explicitly raw payloads.

## Interpret state conservatively

- `writer_active` is current lineage execution authority. A persisted scheduler checkpoint alone
  never proves a process is alive.
- `pending` means known nonterminal evidence; `unknown` means insufficient evidence. Neither is a
  Kernel correctness failure.
- Runtime `duration_seconds` is request end-to-end time.
- Agate `queue_seconds` is submitted-to-started only; `execution_seconds` is started-to-finished;
  `total_seconds` is submitted-to-terminal.
- Timing averages use only persisted samples. Always retain sample count and coverage.
- Review tokens remain separate from main-Agent tokens. Reasoning tokens are a breakdown, not an
  amount to add to total tokens again.
- `evidence_health.complete=false` is an audit-chain warning, not automatically a candidate failure.

## Winner authority

Current winners require all three:

```text
accepted submit-kernel receipt for exact Agent commit
∩ passing standard full Evaluate for that commit
∩ complete per-Shape performance evidence
```

Do not use old Framework Baseline, handoff, promotion, canonical `vN`, or Episode
`attempt.json.accepted` fields as winner authority. Per-Shape best rows may come from different
commits and do not form a synthetic checkout-able Kernel.

## Filter and bound output

Use `--json` for analysis and `--compact` for line-oriented collection. Runtime and Agate call
tools return summary only unless detail selectors such as `--kind`, `--status`, `--lineage`,
`--since`, `--until`, or `--last` are supplied. Direction/Experiment detail is bounded by default;
use `--all-research` only when the complete set is genuinely necessary.

Read only the relevant bounded fields from large Records. Do not print complete stdout, NCU
counter catalogs, per-Shape payloads, or source trees into the conversation unless requested.

## Diagnose without mutating

1. Establish lineage writer liveness and newest durable activity.
2. Identify the Runtime request and its normalized status/error.
3. If a Gateway Record exists, expand or export that logical Record instead of resubmitting or
   trying to reconstruct the request from one physical Job ID.
4. Separate Runtime duration, Agate queue time, execution time, retry history, and transport error.
5. Export evidence only when raw payloads or exact source are needed.
6. Report the original error class and any material evidence-health gap.

Status-only requests do not authorize retry, restart, cancellation, cleanup, checkout changes, or
private-state edits. Export tools may create only a new user-selected output directory and refuse
to overwrite an existing path.

## Report clearly

Include the Campaign workspace, observation time, lineage, exact identities, current liveness,
normalized outcome, distinct timing components, token measurement quality, and material evidence
gaps. State explicitly when a conclusion is unavailable instead of guessing from timestamps.
