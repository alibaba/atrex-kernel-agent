---
name: atrex-aka-light-observability
description: Inspect and export AKA Light campaign evidence with the eight supported observability tools. Use for lineage or trajectory status, Direction/Experiment state, token usage, evaluated or submitted Kernel history, accepted winners, Runtime and Agate call statistics, Gateway-to-Job tracing, raw evidence export, or diagnosing Light progress without mutating the campaign. Do not use for starting, stopping, restarting, cleaning, or modifying a campaign.
---

# AKA Light observability

Inspect durable AKA Light evidence without mutating the Campaign. Identify a compatible AKA
checkout (`AKA_REPO`) and the top-level Campaign workspace (`CAMPAIGN`), not a trajectory
subdirectory. Read only the reference for the user's current use case; each script's `--help`
remains the final CLI authority.

Use the merged Light checkout (`aka.workflow.light` and public shared modules). These scripts do
not require Legacy or its export adapter. Skill files and scripts must be updated together;
installing the Python wheels alone does not update `observability/` or this Skill.

## Route by use case

| User intent | Read | Primary tool |
|---|---|---|
| “状态如何”, “跑得怎么样”, lineage/Session/Direction/Experiment/token progress | [Lineage status](references/light-lineage-status.md) | `light_lineage_status.py` |
| “最近在优化什么”, inspect current Agent work or Session friction | Run lineage status, then follow its [bounded conversation-tail procedure](references/light-lineage-status.md#inspect-what-the-agent-is-optimizing-now) | `light_lineage_status.py` + last 100 events from its `conversation_path` |
| Evaluated Kernels, submission decisions, latency and token cost over time | `observability/USAGE.md` section 4 | `light_kernel_history.py` |
| Accepted global or per-Shape winner | `observability/USAGE.md` section 5 | `light_kernel_best.py` |
| Runtime tool success/failure/pending counts, latency, or recent calls | `observability/USAGE.md` section 6 | `light_runtime_calls.py` |
| Physical Agate submissions, queue/execution time, or recent failures | `observability/USAGE.md` section 7 | `light_agate_calls.py` |
| Export one Agent commit and all bound evidence | `observability/USAGE.md` section 8 | `export_light_kernel.py` |
| Expand one Gateway Record into physical Agate Jobs | `observability/USAGE.md` section 9 | `light_gateway_jobs.py` |
| Export one logical Runtime request, raw Agate attempts, and Agent receipt | `observability/USAGE.md` section 10 | `export_light_gateway_record.py` |
| Generate/open a local Dashboard or save a snapshot | `atrex-aka-light-dashboard` Skill; fallback: `docs/optimization-dashboard.md` in the checkout | `light_dashboard.py` |

For a Dashboard over a remote Campaign, run `light_dashboard.py CAMPAIGN --capture FILE` in the
remote execution environment, download `FILE`, and serve that JSON locally. Do not download the
Campaign and recompute liveness: captured liveness is authoritative at `captured_at`.

The remaining seven script guides are still in the bilingual monolithic manual and should be
split into one reference per script as they are revised. Do not load the complete manual for a
status-only request.

## Shared boundaries

- Status requests are read-only and do not authorize retry, restart, cancellation, cleanup,
  checkout changes, or private-state edits.
- Keep Agent commit, Runtime request, Gateway Record, primary Agate Job, and backend Job IDs
  distinct.
- `pending` and `unknown` are not Kernel failures. Report missing evidence instead of inferring it.
- Use bounded JSON fields; do not print complete stdout, NCU catalogs, source trees, or Session
  histories unless the selected use case requires them.
- Export commands create only a new user-selected destination and must not overwrite it.
- The dashboard binds to `127.0.0.1`, exposes only read-only GET APIs, and must not grow workflow
  controls. Its summaries are the same observability snapshots used by the CLIs.
