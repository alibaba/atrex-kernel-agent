# AKA Light observability

Language: [简体中文完整手册](USAGE.md) | [Complete English guide](USAGE.en.md)

This directory exposes exactly eight supported operator commands. They inspect durable Campaign
evidence without submitting, retrying, cancelling, or mutating Runtime/Agate work.

Pooled search uses one Campaign-private state tree. Lineage-specific Research, Session,
and Runtime-audit data are namespaced below `lineages/tNNNN/`; immutable Gateway Records are
stored once in the Campaign Measurement Store. Observability attributes central Records to their
originating lineage from Runtime audits instead of treating shared evidence as copied local data.

| Operator question | Command |
|---|---|
| What is each lineage doing, including Direction/Experiment state and tokens? | `light_lineage_status.py` |
| Which evaluated Kernels, observed implementation technologies, and nested submit-kernel decisions did each lineage produce? | `light_kernel_history.py` |
| Which accepted commit is best by GeoMean and by individual Shape? | `light_kernel_best.py` |
| How often does each Runtime kind succeed/fail and how long does it take? | `light_runtime_calls.py` |
| How often does each physical Agate kind succeed/fail and what are its timings? | `light_agate_calls.py` |
| Export one commit's source and all bound measurements | `export_light_kernel.py` |
| Expand one Gateway Record into every physical Agate submission | `light_gateway_jobs.py` |
| Export one Gateway Record's Runtime request, raw Agate attempts, and Agent receipt | `export_light_gateway_record.py` |

Implementation shared by these commands lives under `_internal/` and is not a supported CLI.

## Quick start

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign

python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" --json
```

Filter recent Runtime or Agate calls only when details are needed:

```bash
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" \
  --kind evaluate --status failed --last 20 --json
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" \
  --status failed --status pending --last 20 --json
```

Trace one logical measurement:

```bash
RECORD=gateway-0123456789abcdef0123456789abcdef
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id "$RECORD" --json
python3 "$AKA_REPO/observability/export_light_gateway_record.py" "$CAMPAIGN" \
  --gateway-record-id "$RECORD" --output /path/to/new-gateway-evidence
```

Export one exact Agent commit and its bound measurements:

```bash
python3 "$AKA_REPO/observability/export_light_kernel.py" "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-kernel-evidence
```

Export paths must not exist. The six status/query tools support human-readable output, `--json`,
and `--compact`; use each command's `--help` as the final CLI authority.

## Identity and evidence authority

Keep the layers distinct:

```text
Agent Git commit
→ Runtime request ID
→ Gateway Record ID
→ primary Agate Job ID
→ optional Ray/backend Job ID
```

Use the Gateway Record ID to retrieve a complete logical request. A Record can contain multiple
physical Jobs because of Shape batches, retries, or Ray Data execution. A backend Job ID alone may
not have an independently persisted raw response.

Public JSON uses `runtime_request_id`, `agate_job_id`, and `backend_agate_job_ids`. Upstream
`request_id`, `job_id`, and `backend_job_ids` appear only inside explicitly raw evidence files.

Current winners require an accepted `submit-kernel` receipt for the exact Agent commit, a passing
standard full Evaluate for that commit, and complete per-Shape performance evidence. Historical
Framework Baseline, handoff, promotion, canonical `vN`, or Episode acceptance fields are not
current winner authority.

For all options, JSON schemas, field semantics, timing definitions, token accounting, examples,
and audit workflows, read [USAGE.md](USAGE.md) or [USAGE.en.md](USAGE.en.md).
