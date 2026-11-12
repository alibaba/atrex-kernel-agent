# AKA observability scripts

This directory contains read-only operator scripts for inspecting running AKA
Campaigns. Scripts must not submit, cancel, or mutate Agate work.

Use Python 3.10 or newer. Every status command accepts a top-level Light
Campaign workspace, prints a compact human-readable table by default, and
supports `--json` or `--compact` for machine-readable output.

## Commands

按可观测性计划逐项说明命令、字段和预期输出的中文操作手册见
[`USAGE.md`](USAGE.md)。

| Area | Command |
|---|---|
| Campaign, trajectory, Episode and Session status | `light_runtime_status.py` |
| Performance, correctness and per-Shape winners | `light_performance.py` |
| Accepted and best passing full-Eval latency | `light_latency_status.py` |
| `submit-kernel` decisions, rejection reasons and exact Evaluate evidence | `light_submission_status.py` (`light_acceptance.py` is a compatibility alias) |
| Agate call overview, current checkpoints, historical retries and errors | `light_agate_status.py` |
| All physical Agate attempts for one Gateway Record | `light_gateway_jobs.py` |
| Main-Agent Session/Episode tokens, per-kind Review usage, and Kernel milestone checkpoints | `light_token_time.py` |
| Runtime Tool counts, outcomes, duration and attribution | `light_tool_usage.py` |
| Research Board, Direction/Experiment queues, sharing and Review audit | `light_research_status.py` |
| Kernel source, raw Agate jobs and Agent-visible response | `export_light_evidence.py` |
| GPU Wiki queries, latency, results and adoption | `light_wiki_usage.py` |

Examples:

```bash
python3 observability/light_runtime_status.py /path/to/campaign
python3 observability/light_performance.py /path/to/campaign --json
python3 observability/light_submission_status.py /path/to/campaign
python3 observability/light_agate_status.py /path/to/campaign --json
python3 observability/light_gateway_jobs.py /path/to/campaign \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
python3 observability/light_token_time.py /path/to/campaign
python3 observability/light_tool_usage.py /path/to/campaign
python3 observability/light_research_status.py /path/to/campaign
python3 observability/light_wiki_usage.py /path/to/campaign
```

The commands automatically include pooled-search trajectory workspaces. Token usage may use the
structured `estimated` fallback recorded from a finalized Session conversation when every Provider
counter is absent. Other facts that are not durably persisted remain unavailable rather than being
inferred from unstructured logs.

Main-Agent and independent-Review totals are intentionally separate. Review usage is grouped by
`final_source_policy_review`, `direction_claim_review`, `experiment_claim_review`, and
`supplemental_numerical_review`, including invocation counts. Reasoning tokens are reported as a
separate field when the provider exposes a native counter. If all Provider counters are absent,
reasoning text may instead be estimated from explicitly captured reasoning blocks; it is still a
subset of output and is not added to total tokens twice.
`all_paths_totals` provides the combined main-Agent plus all-Review bill; provider-native
reasoning remains a separate breakdown and is not added to `total_tokens` twice.
Each passing standard full Evaluate and accepted `submit-kernel` also records an immutable
Supervisor-private checkpoint of cumulative main-Agent, per-kind Review, and all-path usage.
`light_token_time.py` exposes these as `passing_full_evaluate_token_checkpoints` and
`accepted_submission_token_checkpoints`. A live Provider counter that has not settled is retained
with `measurement=partial` rather than being mislabeled exact. Exact Agent active time,
exact Agate-wait time, actual retry sleep intervals,
Runtime slot/queue occupancy, or a lossless audit-receipt-to-Session join. The
corresponding commands return `null` plus an availability explanation for
those fields; all other reported values come from durable Campaign records.

## Export one measurement and its evidence

```bash
python3 observability/export_light_evidence.py /path/to/campaign \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-export-directory
```

The new directory contains:

- `kernel/`: the exact measured source package;
- `agate-jobs.json`: raw physical Agate responses and Job IDs;
- `agent-response.json`: the response exposed to the Agent by Light;
- `manifest.json`: identities and provenance for the export.

## Light latency status

`light_latency_status.py` reports two distinct facts:

- `accepted_best`: the lowest GeoMean latency among verified accepted `submit-kernel` decisions;
- `best_passing_full_eval`: the lowest GeoMean latency observed in a passing,
  standard full Agate Eval, whether or not that Kernel was submitted.

Current Light uses the Agent's clean Git HEAD as the public Kernel identity. Both
accepted state and Measurement Records expose it as `kernel_commit_id`; the
Supervisor's separate archive commit remains private audit data. The script
resolves each accepted private-Journal `kernel_submissions[]` entry, checks that
its full-Evaluate Gateway Record names the same Agent commit, and returns the
immutable Record identity and verified source-artifact digest. Episode
`attempt.json` is execution telemetry only and is never an acceptance source.

The exact state and Measurement Record sources are documented in
[`light-latency-data-sources.md`](light-latency-data-sources.md).

Run it from any directory with Python 3.10 or newer:

```bash
python3 /path/to/atrex-kernel-agent/observability/light_latency_status.py \
  /path/to/light-campaign-workspace
```

Continuously refresh every 30 seconds:

```bash
python3 /path/to/atrex-kernel-agent/observability/light_latency_status.py \
  /path/to/light-campaign-workspace --watch 30
```

For line-oriented collection:

```bash
python3 /path/to/atrex-kernel-agent/observability/light_latency_status.py \
  /path/to/light-campaign-workspace --watch 30 --compact
```

The command automatically discovers all durable measurement scopes belonging
to the supplied top-level Campaign workspace. No trajectory or search-mode
selection is required.

Only completed, cacheable, untruncated standard full Eval records with passing
correctness are eligible for `best_passing_full_eval`. Correctness-only,
single-Shape/subset, custom-input, ABBA, Profile, Dev, failed, and running work
is excluded. Per-Shape Pareto minima are deliberately not combined into an
unmeasured synthetic Kernel.

## Export a reported Kernel source package

Export an accepted or measured result by its Agent Kernel commit:

```bash
python3 /path/to/atrex-kernel-agent/observability/export_light_source.py \
  /path/to/light-campaign-workspace \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/exported-source
```

Export a measured result by its Gateway Record:

```bash
python3 /path/to/atrex-kernel-agent/observability/export_light_source.py \
  /path/to/light-campaign-workspace \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/exported-source
```

The latency status JSON can be consumed directly. The exporter uses
`kernel_commit_id` for either result, and prefers the evidence-bound
`gateway_record_id` for `best_passing_full_eval`:

```bash
python3 /path/to/atrex-kernel-agent/observability/export_light_source.py \
  /path/to/light-campaign-workspace \
  --status-json /path/to/status.json \
  --select best_passing_full_eval \
  --output /path/to/exported-source
```

The output path must not already exist. Every source format is materialized as
a directory, preserving the single-file, source-tree, or CUDA manifest layout.
