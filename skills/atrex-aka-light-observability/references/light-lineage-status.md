# Lineage status: “状态如何” / “跑得怎么样”

Use this reference when the user wants a factual current-progress report for one or more AKA Light
Campaigns. The primary tool is `observability/light_lineage_status.py`. This use case is read-only.

## Desired outcome

Report, for every selected lineage:

- whether a writer currently owns execution;
- the current Episode/phase and completed Episode count;
- the active Agent or Supervisor operation and its elapsed time;
- recent activity or silence;
- Direction/Experiment counts and focus;
- main-Agent, Review, and total token usage with measurement quality;
- concrete failures or evidence gaps that materially affect progress.

Do not run all eight observability tools by default. Do not scan full `conversation.jsonl` files for
an ordinary status request.

## Locate the scope

Use a checkout compatible with the Campaign format:

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign
```

`CAMPAIGN` is the directory containing `campaign.json`, `coordinator.json` for pooled search, or
the equivalent serial Campaign state. Do not pass `lineages/t0001/agent-workspace`.

For a matrix root containing multiple Campaigns, first enumerate bounded `campaign.json` paths,
then run the same status command once per Campaign. Report the matrix total and identify each
Campaign by its relative path. Do not merge identically named lineages across Campaigns.

When the Campaign is inside an Aone Sandbox, use the standalone Sandbox skill's read-only shell
workflow. `akfctl sandbox shell` requires a TTY; execute the observability command in the Sandbox
without downloading or modifying the Campaign.

## Run the smallest sufficient query

Human-readable overview:

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN"
```

Structured analysis:

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" --json
```

One or more lineages:

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" \
  --lineage t0001 --lineage t0003 --json
```

Use the interpreter/environment that can import the selected checkout. If bare `python3` cannot
import `aka`, use the checkout's installed environment or `PYTHONPATH="$AKA_REPO/src"`; do not
change Campaign files to work around an import error.

## Options

| Option | Default | Meaning |
|---|---:|---|
| `--stale-after SECONDS` | `600` | Classify a lock-owning lineage with no active Supervisor request and no recent activity as `running_no_recent_activity`; it does not stop anything |
| `--lineage NAME` | all | Select a lineage such as `t0001`; repeatable |
| `--research-limit N` | `50` | Maximum Direction and Experiment detail rows per lineage |
| `--all-research` | off | Return all Direction/Experiment rows; use only when necessary |
| `--json` | off | Emit indented structured JSON |
| `--compact` | off | Emit one-line JSON and imply `--json` |

Use `--help` as the final option authority.

## Interpret liveness

Use these fields together:

| Field | Interpretation |
|---|---|
| `runtime.writer_active` | Current lineage execution authority; a scheduler checkpoint alone does not prove liveness |
| `runtime.liveness` | `running_recent_activity`, `running_supervisor_activity`, `running_no_recent_activity`, or `not_running` |
| `runtime.latest_activity_at` / `stagnation_seconds` | Age of the newest durable activity |
| `runtime.main_session_silence_seconds` | Main Agent silence even when Supervisor work is active |
| `runtime.active_sessions` | Current Agent phase/tool, when captured |
| `runtime.supervisor_activity` | Current or latest authoritative Runtime operation, including Claim Review |

A long Claim Review with `supervisor_activity.state=running` is active progress, not a stalled
lineage. A quiet main Session plus no active Supervisor operation is only suspicious after the
configured stale threshold. `pending` means known nonterminal evidence; `unknown` means evidence
is insufficient. Neither proves a Kernel failure.

## Interpret research and tokens

- `research.enabled_modules` confirms whether Direction and/or Experiment recording is enabled.
- `direction_counts`, `active_direction_ids`, and `focus_direction_id` describe local work state.
- `local_direction_queue` and `shared_direction_queue` are preferences, not proof that peer code
  was adopted.
- `last_shared_queue_sync` and `peer_git_refs` show publication/import state.
- `claim_reviews.quality_counts` describes Review outcomes; Review acceptance is not Kernel
  submission acceptance.
- `tokens.main_agent`, `tokens.reviews`, and `tokens.all_paths` are separate. Reasoning tokens are
  a breakdown already included in total tokens. Always report `measurement` as `exact`,
  `estimated`, `partial`, or `unavailable`.

## Decide whether to go deeper

Stop after lineage status when it answers the user's question and all lineages show coherent
progress or completion.

Continue only when evidence requires it:

- User asks for latency, evaluated commits, or accepted outputs: use Kernel history/best.
- A Runtime operation failed or repeats: use Runtime calls, filtered by kind/status/lineage/last.
- A Gateway operation is pending or failed: use Agate calls, then expand its Gateway Record.
- User explicitly asks to inspect Session friction, or status shows unexplained inactivity: read
  only the affected lineage's recent Session tail and relevant Runtime receipts. Do not scan every
  Session in the matrix.
- Campaign process died but activity still says running: report stale observability and inspect
  process/OOM evidence; do not describe the orphan request as healthy progress.

## Report format

Lead with the overall outcome, then provide a compact row per lineage or configuration:

| Campaign / lineage | Liveness | Progress | Current work | Research | Tokens | Issue |
|---|---|---|---|---|---:|---|

Include the Campaign path and observation time. Distinguish active work, completed work, true
failure, known pending work, unknown evidence, and stale state. Mention only errors that materially
affect progress; do not treat ordinary failed probes or rejected candidate implementations as
Light framework failures without supporting evidence.
