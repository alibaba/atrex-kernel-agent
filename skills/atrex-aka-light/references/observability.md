# AKA Light observability

Use this file for live status, diagnosis, metrics and reproducibility. Run read-only tools from a checkout compatible with the campaign. Never submit, cancel or mutate Agate work during a status-only request.

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign-workspace
```

## First pass

```bash
python3 "$AKA_REPO/observability/light_runtime_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_submission_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
python3 "$AKA_REPO/observability/light_token_time.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_research_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_tool_usage.py" "$CAMPAIGN" --json
```

Use `--help` before optional filters or watch flags. Status scripts are convenience projections; immutable Gateway Records, Runtime audit, private Journal `kernel_submissions`, MeasurementStore, Agent Git, search checkpoint, and current `PROGRESS.md`/`frontier.json` are the authority for their respective facts.

## Current winner model

Current Light has no Supervisor promotion state. A winner candidate must satisfy:

```text
accepted submit-kernel receipt for exact Agent commit
∩ passing standard full Evaluate for that commit
∩ complete per-Shape performance evidence
```

- `PROGRESS.md` shows the compact Agent-facing state and best accepted submission for its lineage.
- `frontier.json` contains the accepted-submission Pareto/frontier projection.
- Multi-trajectory search may additionally show a global best accepted submission after sharing; a single trajectory need not repeat a separate global row.
- An unsubmitted full Evaluate remains valid measurement evidence in `records/`, but it is not a lineage/global winner and is not broadcast as one.
- `submit-kernel` acceptance should refresh the local winner/frontier promptly; cross-trajectory global state follows the configured synchronization point.

Do not use old fields named `promoted`, `incumbent`, canonical `vN`, or Episode `attempt.json.accepted` as current winner authority. `light_submission_status.py` joins private `kernel_submissions` to exact MeasurementStore evidence; `light_acceptance.py` is only its compatibility alias.

## State boundaries

Keep these independent:

| Layer | Meaning |
|---|---|
| Agent Session | Thinking, shell/file tool, Runtime tool, or waiting state |
| Episode | Audit segment of one persistent trajectory Session; not a candidate version |
| Agent Git | Public Kernel identity and engineering history |
| Runtime request | Evaluate, explicit ABBA, Profile, Check, Disassemble, Env, Dev, Command, Wiki or research mutation |
| Gateway Record | Immutable normalized request/result and Agent-visible receipt |
| Physical Agate job | Queue/execution terminal state; one Gateway Record may reference multiple jobs/batches |
| Kernel submission | Review receipt for one exact evaluated Agent Git commit |
| Search | Trajectory/Epoch scheduling, peer publication and accepted-submission winners |

`light_gateway_jobs.py` expands one logical Gateway Record into every durably
captured physical submission, including failed attempts that were later
retried. Each row reports the Job ID, normalized status, error class/reason,
Shape IDs, checkpoint, and submission number. Physical Job IDs remain
Supervisor-private and are not projected into the Agent workspace.

`waiting_agate` does not prove the remote job is currently running. Correlate the logical Runtime call with saved physical job records. A transport timeout, queue timeout, Ray failure or missing terminal response is infrastructure evidence, not candidate correctness failure.

## Diagnose slow or failed progress

1. Identify scope, trajectory, Epoch, Episode audit segment, Session, pending tool and newest durable timestamp.
2. If Runtime is pending, inspect the request receipt and whether the Agent was paused during it.
3. If Agate is involved, separate queue time, execution time, polling/transport failure and retry delay.
4. Classify as `infrastructure`, `candidate`, `policy`, `protocol`, or `unknown`; preserve the original error and do not relabel infrastructure as correctness.
5. Inspect `shape_ids`, outer batch identity and reuse fields before claiming all Shapes reran.
6. For a rejected submission, distinguish missing/failed Evaluate evidence from final-source review rejection.
7. For research friction, inspect Direction/Experiment request, inline Review result and private audit; do not infer that a Review state changed the Kernel.
8. For pooled search, inspect each lineage independently and the last completed sharing boundary.

Preserve `null` and explicit availability reasons. Do not estimate missing token splits, active time, queue time or retry sleep from wall-clock timestamps alone.

## Research and sharing

`PROGRESS.md` is a compact projection, not the complete Research Store. Use `light_research_status.py` and private events to inspect:

- local Direction queue, focus and lifecycle;
- shared discovery pool and source trajectory;
- Experiment records and associated Gateway Record IDs;
- claim-review quality and evidence eligibility;
- peer Git refs and sharing boundary;
- accepted-submission lineage/global winners.

Eligible shared records are read-only evidence. A trajectory must explicitly absorb or derive peer work; sharing never overwrites local HEAD. Conversations and scratch are not shared.

## Runtime output and Records

Runtime terminal output should be a compact receipt: outcome/correctness, aggregate latency, actionable error summary, Gateway Record ID/path, and bounded stdout/stderr preview where applicable. Complete results live in `records/<gateway-id>.json`.

For truncated Dev/Command output, the receipt should identify both path and field, for example `result.stdout`. Read only the required bounded field instead of printing the complete Record into the conversation. Env is intentionally returned without the Dev/Command preview truncation when the current checkout defines it that way.

Gateway Record IDs use the public `gateway-...` identity. Physical `ev_...` job IDs remain private transport evidence and must not be substituted into Agent-facing research records.

## Token and tool analysis

Use `light_token_time.py` for main-Agent Session/trajectory/Episode totals and for a separate
per-kind independent-Review ledger. Review calls and tokens are grouped as final-source policy,
Direction claim, Experiment claim, and supplemental-numerical review; they are never merged into main-Agent totals.
Every managed main-Agent or Review invocation also archives a complete `conversation.jsonl` and
`token-usage.json` under Supervisor-private Campaign state. These raw Review transcripts are not
projected into the optimizer workspace; only bounded verdicts and explicitly selected historical
main-Agent evidence may be Agent-visible.
Input, output, cache, total, and reasoning tokens are shown separately. When all Provider counters
are absent, finalized conversation text supplies a best-effort `estimated` input/output/reasoning
total while cache fields remain null. A missing reasoning counter is not inferred when other native
Provider counters are available. Use `light_tool_usage.py` for
Runtime audit counts, outcomes and durations. Do not join independent logs by timestamp when no
stable request identity exists.
The JSON `all_paths_totals` object adds main-Agent and every Review total, while keeping reasoning
as a non-additive breakdown. Its measurement distinguishes `exact`, `estimated`, `partial`, and
`unavailable` coverage.

Long provider calls can cross the shell frontend's 600-second foreground boundary while the actual Light/Reviewer process continues. Treat backgrounding or `exit 143` as a separate frontend/process issue; inspect the captured Session and private Review receipt before declaring the Review failed.

## Export source and evidence

Export a complete evidence bundle:

```bash
python3 "$AKA_REPO/observability/export_light_evidence.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-evidence-directory
```

Export source by Agent Git commit or Gateway Record:

```bash
python3 "$AKA_REPO/observability/export_light_source.py" "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-source-directory

python3 "$AKA_REPO/observability/export_light_source.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-source-directory
```

The output path must not exist. Export before destructive cleanup or when comparing raw Agate output with the Agent-visible projection.

## Reporting checklist

Report only supported facts:

- checkout commit and campaign workspace;
- per-lineage Epoch/Episode segment, health, Session phase and staleness;
- best accepted submission and best measured passing full Evaluate, clearly distinguished;
- exact Agent commit and Gateway Record identities;
- pending/failed Agate batches, error class, retries and reuse;
- token totals requested by the user;
- missing/incomplete fields and their availability reason.

Anchor all conclusions to the newest durable timestamp. A read-only inspection does not authorize restart, retry, cleanup or deletion.
