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
| Candidate gates, rejection reasons and promotions | `light_acceptance.py` |
| Agate jobs, retries, batches, timeouts and errors | `light_agate_status.py` |
| Session/Episode tokens, wall time and token efficiency | `light_token_time.py` |
| Runtime Tool counts, outcomes, duration and attribution | `light_tool_usage.py` |
| Research Board, Direction/Experiment queues, sharing and Review audit | `light_research_status.py` |
| Kernel source, raw Agate jobs and Agent-visible response | `export_light_evidence.py` |
| GPU Wiki queries, latency, results and adoption | `light_wiki_usage.py` |

Examples:

```bash
python3 observability/light_runtime_status.py /path/to/campaign
python3 observability/light_performance.py /path/to/campaign --json
python3 observability/light_acceptance.py /path/to/campaign
python3 observability/light_agate_status.py /path/to/campaign --json
python3 observability/light_token_time.py /path/to/campaign
python3 observability/light_tool_usage.py /path/to/campaign
python3 observability/light_research_status.py /path/to/campaign
python3 observability/light_wiki_usage.py /path/to/campaign
```

The commands automatically include pooled-search trajectory workspaces. If a
fact is not durably persisted by current `main`, the JSON explicitly reports it
as unavailable instead of inferring a value from unstructured logs.

Current `main` does not durably persist a separate reasoning-token counter,
exact Agent active time, exact Agate-wait time, actual retry sleep intervals,
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

- `accepted_best`: the lowest GeoMean latency already accepted by Light;
- `best_passing_full_eval`: the lowest GeoMean latency observed in a passing,
  standard full Agate Eval, whether or not that Kernel was later promoted.

Current Light uses the Agent's clean Git HEAD as the public Kernel identity. Both
accepted state and Measurement Records expose it as `kernel_commit_id`; the
Supervisor's separate archive commit remains private audit data. The script
resolves each accepted `attempt.json`, checks that its terminal Gateway Record
names the same Agent commit, and returns the immutable Record identity and
verified source-artifact digest.

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
