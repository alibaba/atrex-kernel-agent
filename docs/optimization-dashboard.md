# AKA Light read-only dashboard

The dashboard reads the same Campaign snapshots as the eight supported observability commands.
It shows lineage activity, Direction/Experiment state, tokens, passing full Evaluates, submission
reviews, accepted winners, Runtime calls, Agate attempts, and Gateway evidence. It does not launch
jobs, submit Kernels, retry, cancel, or change Campaign state.

The [Dashboard Skill](../skills/atrex-aka-light-dashboard/SKILL.md) provides a local-user workflow,
including a Vector Add example, background startup, verification and captured snapshots in
[Chinese](../skills/atrex-aka-light-dashboard/references/local-dashboard.md) and
[English](../skills/atrex-aka-light-dashboard/references/local-dashboard.en.md).

## Run locally

Use the merged AKA checkout and the Python environment used by Light:

```bash
python3 /path/to/aka/observability/light_dashboard.py /path/to/campaign --port 8765
```

Open the printed loopback URL. The script uses the checkout's `src/` and `observability/` modules,
including the shared HTTP server and bundled fonts; it does not require the removed Legacy
workflow, observation adapter, or an additional Dashboard wheel. The Campaign must contain
`campaign.json`. A lineage's `agent-workspace/` is not the Campaign root.

From an installed release, use the same viewer through its module entry point:

```bash
python -m pip install atrex-aka-dashboard==0.3.0
python -m aka.dashboard /path/to/campaign --port 8765
```

The Dashboard package installs matching `0.3.0` Framework and Light packages,
including the observability modules, HTML and fonts. No Dashboard plugin,
`observation_source` provider, or separate startup profile is needed.

| Option | Default | Meaning |
|---|---|---|
| `source` | Required | Campaign root, or a previously captured dashboard JSON |
| `--host` | `127.0.0.1` | Loopback only; other hosts are rejected |
| `--port` | `8765` | HTTP port; `0` chooses a free port |
| `--refresh-ms` | `5000` | Positive browser refresh interval in milliseconds |
| `--capture PATH` | None | Capture once to JSON and exit without serving HTTP |

## Observe a remote Campaign

Capture on the machine or Sandbox actually running the Campaign, then download only that JSON:

```bash
python3 /path/to/aka/observability/light_dashboard.py /path/to/campaign \
  --capture /tmp/light-dashboard.json
# After downloading the snapshot:
python3 /path/to/aka/observability/light_dashboard.py /local/light-dashboard.json
```

A capture is a point-in-time observation (`captured_at`), not a live connection. The local viewer
preserves the captured remote liveness instead of inspecting local PIDs or locks. Capture again
to obtain fresh remote state. Private evidence and raw responses may be present in the JSON;
keep it with the Campaign's other audit artifacts.

## Sources and interpretation

- `light_lineage_status.py`: current lineage/Session/Research activity and token attribution.
  Process liveness comes from the actual Campaign writer lease, combined with lineage state and
  active Supervisor requests. A stale checkpoint alone is not proof of a running process.
- `light_kernel_history.py`: passing full Evaluates and nested submission revisions, bound to
  exact Agent Git commits, with performance and cumulative token checkpoints.
- `light_kernel_best.py`: best accepted submissions by GeoMean and individual Shape.
- `light_runtime_calls.py` and `light_agate_calls.py`: logical Runtime calls and physical Agate
  attempts, with explicit success/failure/pending/unknown states and observed timing coverage.
- `export_light_gateway_record.py`: the Gateway Record's request, Runtime audits, raw Agate
  attempts, and Agent-facing response. A Record can be consumed by multiple lineages while its
  canonical evidence stays in the Campaign Measurement Store.

Missing evidence remains unavailable; missing token usage is not zero. Review time is not GPU
queue time. The viewer does not reinterpret an ordinary Episode stop as promotion or rejection.

The HTTP API accepts read-only GET requests. Runtime/Agate detail filters support `kind`,
`status`, `lineage`, and bounded `last` (1–500). Gateway details use a complete Gateway Record ID;
Runtime details use the audited Runtime request ID. POST/PUT/PATCH/DELETE are rejected.

See [中文工具手册](../observability/USAGE.md), [English tool guide](../observability/USAGE.en.md),
and the [observability Skill](../skills/atrex-aka-light-observability/SKILL.md).
