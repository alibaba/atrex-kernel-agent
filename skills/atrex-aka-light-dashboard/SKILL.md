---
name: atrex-aka-light-dashboard
description: Open or capture a local read-only AKA Light dashboard from existing Campaigns using the observability tools. Use for requests to generate a dashboard, visualize Kernel latency and token history, or browse lineage, Research, submission, Runtime and Agate evidence. This inspects existing runs; it does not launch optimization campaigns.
---

# AKA Light Dashboard

Assume the user runs AKA locally. Read the existing Campaign directly through the checkout's
observability tools and serve the existing Dashboard. No Sandbox, download, GPU job or new
optimization run is needed. Dashboard inspection does not require the launch Skill's intake for
credentials, GPU, mode, DSL or Agent backend.

## Open a local Dashboard

1. Resolve `AKA_REPO`, the compatible AKA checkout, and `CAMPAIGN`, the requested top-level
   directory containing `campaign.json`. Use paths already supplied in context. If the user gives
   a runs directory, discover its Campaign roots and select the requested scope; do not pass the
   parent directory or `lineages/tNNNN/agent-workspace` as a Campaign. Ask only for a path or scope
   that remains ambiguous.
2. Use the Python environment that can run the checkout's observability scripts. Read
   `observability/light_dashboard.py --help` before constructing the command. Existing AKA users
   do not need a separate Dashboard package. Do not install or upgrade the optimization stack
   merely to view a Campaign.
3. Start the read-only server against the Campaign directory. Prefer `--port 0` unless the user
   requests a port. When starting it in the background, retain its PID and log in a new output
   directory outside the Campaign. Read the actual URL from stdout; do not assume a port.

   ```bash
   "$AKA_PYTHON" "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --port 0
   ```

4. Verify the printed URL and `/api/summary` both return successfully. Check that `workspace`
   matches the requested Campaign, `source_mode` is `live_observability`, and `captured_at` is
   recent. A successful HTML response alone does not verify data collection. Inspect one relevant
   populated view, such as Kernel history or a recent Runtime call, when present.
5. Return the clickable local URL, the Campaign it represents and whether it is live or captured.
   If collection fails, report the concrete error; do not substitute empty data or report success.

For background startup, a concrete Vector Add example, options and saved snapshots, read
[本地使用说明](references/local-dashboard.md) or [English local guide](references/local-dashboard.en.md).
Read only the language needed for the task.

## Data contract

`light_dashboard.py` calls the same snapshot functions as the supported observability CLIs.
Reuse these functions for any requested custom presentation; do not recreate status, token or
winner calculations by scraping logs, sessions or `PROGRESS.md`.

| Dashboard content | Observability source |
|---|---|
| Lineage activity, Direction/Experiment state, main/review token usage | `light_lineage_status.py` |
| Passing full Evaluates, submission revisions, commit IDs, latency and token checkpoints | `light_kernel_history.py` |
| Best accepted submissions overall and per Shape | `light_kernel_best.py` |
| Logical Runtime calls, result states and elapsed time | `light_runtime_calls.py` |
| Physical Agate attempts, states and observed timing | `light_agate_calls.py` |
| Original request, Agate responses and Agent receipt for a Gateway Record | `export_light_gateway_record.py` |

Preserve these distinctions:

- A passing full Evaluate is not necessarily an accepted submission. Show both without treating
  an unsubmitted result as the accepted winner.
- `partial` or unavailable token checkpoints are incomplete observations, not zero or exact totals.
- `pending` and `unknown` are not proof of a failed Kernel. Review time, GPU queue time, Agate
  end-to-end time and Runtime duration are separate quantities.
- For call details, distinguish Agent → Runtime, Runtime → Service, Service → Runtime and
  Runtime → Agent. Service → Runtime means the archived service response; launcher logs must not
  replace missing Agate JSON.
- A saved snapshot reports state at `captured_at`; refreshing its page does not obtain fresh
  Campaign data. Captures retain bounded call details and are not complete Campaign backups.

## Scope

The native server accepts one Campaign or one captured JSON. For several Campaigns, use separate
instances/links unless the user requests a combined view. Keep each Campaign's identity and
capture timestamp in a combined view; do not pool different operators or Shape identities.

Run live observation in the Supervisor's environment so process/lock checks are meaningful. If
AKA runs in a local VM, collect there; a copied Campaign on the host is not live process evidence.
Use captured JSON when only an offline view is available.

Keep the server on `127.0.0.1`. The Dashboard must remain read-only; do not start, stop, retry,
cancel, clean or modify optimization work. Do not terminate an unrelated process to free a port.
The Skill and `observability/` must come from a compatible checkout. Never substitute synthetic
measurements when a real Campaign has no results yet.
