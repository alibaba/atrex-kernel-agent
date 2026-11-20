# Local AKA Light Dashboard

## Example: inspect an existing local Vector Add run

User: `Make a dashboard for runs/vector-add showing performance, tokens and call errors.`

Reuse the existing AKA checkout, Python environment and Campaign. Replace the example paths:

```bash
AKA_REPO=/path/to/atrex-kernel-agent
AKA_PYTHON=/path/to/aka-venv/bin/python
CAMPAIGN=/path/to/runs/vector-add

test -f "$CAMPAIGN/campaign.json"
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" --help
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --port 0
```

Open the actual printed URL, for example `http://127.0.0.1:49152/`; the example port is not fixed.
The default refresh interval is five seconds. Observability reads the local Campaign directly;
it does not contact Agate or measure Kernels again. No new Agate credentials, GPU, mode, DSL or
Atrex-Bench path is required for inspection.

If only a parent directory is known, discover `campaign.json` within that bounded directory,
not the entire home directory. Each containing directory is a Campaign; neither its parent nor
an Agent workspace is a valid substitute.

## Background startup and verification

Use a new output directory outside the Campaign for the Dashboard process log and PID:

```bash
DASHBOARD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aka-light-dashboard.XXXXXX")
nohup "$AKA_PYTHON" -u "$AKA_REPO/observability/light_dashboard.py" \
  "$CAMPAIGN" --port 0 \
  > "$DASHBOARD_DIR/server.log" 2>&1 < /dev/null &
dashboard_pid=$!
printf '%s\n' "$dashboard_pid" > "$DASHBOARD_DIR/server.pid"
```

Read the URL from `server.log` and check that the process has not exited. On failure, inspect the
original error and correct the path, environment or port. A PID file alone is not proof of success.

Check the actual URL:

```bash
# Replace with the address printed in server.log.
DASHBOARD_URL=http://127.0.0.1:49152
curl --fail --silent --show-error "$DASHBOARD_URL/" > /dev/null
curl --fail --silent --show-error "$DASHBOARD_URL/api/summary" \
  > "$DASHBOARD_DIR/verified-summary.json"
```

Inspect `workspace`, `captured_at`, `source_mode` and relevant counters without printing the entire
JSON into the conversation. Collection time depends on Campaign size; a pending response is not
empty data. An initialized Campaign with no Evaluates can legitimately have empty results; label
that state instead of inventing performance points.

Return the URL, Campaign path and live/captured mode, plus the log path for a background server.
Stopping the Dashboard affects only its HTTP process, not the Campaign. Operate only on the
process started for this task after confirming its identity.

## Options

| Option | Default | Meaning |
|---|---|---|
| Positional `source` | Required | Local Campaign root or captured Dashboard JSON |
| `--host` | `127.0.0.1` | Loopback only |
| `--port` | `8765` | HTTP port; `0` allocates a free port |
| `--refresh-ms` | `5000` | Positive browser refresh interval in milliseconds |
| `--capture PATH` | Unset | Capture once to JSON and exit without serving HTTP |

## Save a point-in-time view

For an archive or offline view, capture directly from the local Campaign:

```bash
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" \
  --capture "$DASHBOARD_DIR/snapshot.json"
"$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" \
  "$DASHBOARD_DIR/snapshot.json" --port 0
```

Choose a new destination; the CLI replaces an existing file, so do not overwrite an earlier
capture unless requested. `captured_observability` describes state at `captured_at`. Browser
refresh keeps reading that capture; collect again for new data.
The current capture retains up to the latest 500 Runtime calls and 500 Agate calls with associated
details, while call summaries remain Campaign-wide. Kernel/Research lists also use their tools'
default scope; this is not a complete history backup. Captures can contain private source,
parameters and responses; keep them with local audit data.

## Inspect details as needed

The UI uses these GET endpoints, which also support verifying a particular view:

| Endpoint | Content |
|---|---|
| `/api/summary` | Overall snapshot and capture timestamp |
| `/api/lineages` | Lineages, current work, Direction/Experiment and tokens |
| `/api/kernels` | Full Evaluates and associated Submission history |
| `/api/winners` | Best accepted submissions |
| `/api/runtime-calls?kind=evaluate&status=failed&last=10` | Recent failed Evaluate Runtime calls |
| `/api/agate-calls?status=pending&last=10` | Recent nonterminal Agate calls |
| `/api/gateway-records/gateway-…` | Evidence for a full Gateway Record ID |
| `/api/runtime-requests/…` | Audit for a full Runtime request ID |

Call filters accept `kind`, `status`, `lineage` and `last`. Status is
`succeeded/failed/pending/unknown`; `last` is 1–500. Filters select details; call summaries remain
Campaign-wide. Captured mode can filter only retained details. A 404 for an uncaptured detail
does not imply lost evidence in the original Campaign.

For multiple Campaigns, default to separate instances and links. If a combined page is requested,
keep using the same observability snapshots, preserve Campaign identities and capture times, and
do not reparse logs, combine Shape identities across operators or redefine token accounting.
