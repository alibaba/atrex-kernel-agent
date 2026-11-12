# AKA Light observability

Read this file for live status, diagnosis, metrics, or reproducibility. Use the `observability/` directory from a checkout compatible with the campaign. These scripts are read-only and must not submit, cancel, or mutate Agate work.

Set paths once:

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign-workspace
```

Status commands accept a top-level campaign workspace, automatically discover pooled-search trajectory scopes, print tables by default, and support `--json` or `--compact`.

## First-pass status

Run these in order for a general “how is it running?” request:

```bash
python3 "$AKA_REPO/observability/light_runtime_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_performance.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_token_time.py" "$CAMPAIGN" --json
```

Summarize per scope/trajectory: current stage and Episode, health and active Session phase, accepted and measured best latency, candidate state, pending/failed Agate jobs, token totals, and newest durable timestamp.

## Command map

| Question | Command |
|---|---|
| Campaign, trajectory, Episode, Session and Agent phase | `light_runtime_status.py` |
| Performance, correctness, baselines, per-Episode and per-Shape results | `light_performance.py` |
| Accepted best versus best passing full Evaluate | `light_latency_status.py` |
| Correctness gates, rejections, promotion and evidence | `light_acceptance.py` |
| Agate calls/jobs, retries, batches, timeouts and error classes | `light_agate_status.py` |
| Session/Episode tokens, wall time and token efficiency | `light_token_time.py` |
| Runtime Tool calls, outcomes, durations and attribution | `light_tool_usage.py` |
| GPU Wiki queries, results, latency and adoption | `light_wiki_usage.py` |
| Exact measured source plus raw/Agent-visible evidence | `export_light_evidence.py` |
| Source package by Git commit, Kernel ID or Gateway Record ID | `export_light_source.py` |

Use `--help` on the selected script before relying on optional filters or watch flags.

## Interpret runtime state

`light_runtime_status.py` reports Workflow phase and health separately from active Agent Sessions. Common Agent phases are:

- `thinking`: no pending tool result;
- `executing_tool`: a normal shell/file tool is pending;
- `executing_runtime_tool`: a local Runtime Tool is pending;
- `waiting_agate`: an Evaluate, ABBA, Profile, Check, Disassemble, or Dev request is pending;
- `waiting_wiki`: a GPU Wiki request is pending.

`waiting_agate` does not prove the remote job is running; correlate its Job ID with `light_agate_status.py`. `stalled` is based on durable update age and threshold, not proof of deadlock.

## Interpret performance and promotion

- `accepted_best` is the lowest latency already accepted by Workflow state.
- `best_passing_full_eval` is the lowest passing standard full Evaluate, even if it was never promoted.
- A passing or faster ordinary Evaluate does not replace the incumbent without terminal verification and promotion.
- Read `State` together with `Promoted`: `pass`, `correctness_failure`, `rejected`, `infrastructure_error`, `interrupted`, and `unmeasured` describe candidate Workflow outcome, not Agent or Agate state.
- Use `light_acceptance.py` for `decision_reason`, incumbent/candidate comparison, and evidence identities.

Do not combine per-Shape winners into a synthetic Kernel. Use a single measured source identity for any reported whole-workload latency.

## Diagnose slow or failed progress

1. Use runtime status to identify the exact scope, stage, Episode, Session, Agent phase, pending tool, and staleness.
2. If waiting on Agate, inspect logical calls and physical jobs. Report queue, execution and total time separately when persisted.
3. Classify failures as `infrastructure`, `candidate`, `policy`, `protocol`, or `unknown`. Do not count infrastructure failure as correctness failure.
4. Inspect batch fields before claiming a retry reran everything; Evaluate and ABBA may reuse completed Shape batches.
5. If the Agent is thinking or repeatedly restarting, inspect Session token/incidents and the corresponding conversation trace without exposing secrets.
6. If Campaign state is inactive or stalled, inspect the owning Supervisor process and logs; do not restart until the failure boundary is understood.

Current main does not durably persist every fact. Exact reasoning-token splits, Agent active time, Agate wait time, retry sleep duration, live lock/slot occupancy, and lossless audit-to-Session joins may be unavailable. Preserve `null` and the availability reason rather than estimating.

## Token and tool analysis

Use `light_token_time.py` for Session, startup, trajectory and Episode totals, average tokens per Attempt, incidents, and latency improvement per million tokens. Bootstrap and Framework Baseline belong to startup totals and should not be folded into ordinary Attempt averages.

Use `light_tool_usage.py` for authoritative Runtime audit counts and durations. Session attribution is a separate source; do not join it to audit receipts by timestamp when the checkout lacks a stable identity.

## Continuous latency sampling

```bash
python3 "$AKA_REPO/observability/light_latency_status.py" \
  "$CAMPAIGN" --watch 30 --compact
```

Use a normal one-shot `--json` query for reports. Watch mode is for external collection and does not mutate the Campaign.

## Export exact source and evidence

Export a complete measurement evidence bundle:

```bash
python3 "$AKA_REPO/observability/export_light_evidence.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-evidence-directory
```

The output contains the exact source package, physical Agate Job IDs and raw responses, the response Light exposed to the Agent, and a provenance manifest. The output path must not already exist.

Export only source by accepted Git revision or measured identity:

```bash
python3 "$AKA_REPO/observability/export_light_source.py" "$CAMPAIGN" \
  --git-commit 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-source-directory

python3 "$AKA_REPO/observability/export_light_source.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-source-directory
```

Use evidence export before destructive cleanup or when comparing Agate's raw response with Agent-visible projection.

## Reporting checklist

For a status response, include only supported facts:

- checkout commit and campaign workspace;
- per-scope stage, Episode, health, Session phase and staleness;
- accepted best and best passing full-Eval latency with source/Record identity;
- promotion/rejection state and reason;
- pending or failed Agate jobs, error class, retries and batch reuse;
- token totals and per-Attempt figures requested by the user;
- whether fields are unavailable, incomplete, or derived from a fallback.

Anchor conclusions to the newest durable timestamps. Do not describe an old snapshot as current.
