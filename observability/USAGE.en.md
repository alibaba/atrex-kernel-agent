# AKA Light Observability: Complete Guide

Language: [简体中文](USAGE.md) | **English**

This guide documents the eight supported AKA Light observability entry points: how to select a
command, every option, the expected JSON structure, field semantics, evidence authority, and
common interpretation errors. Each script's `--help` output is the final authority for CLI syntax;
this guide explains what the options and results mean.

## 1. Scope and safety boundary

These tools read the merged Light Campaign layout and use public owners such as `aka.agent`,
`aka.evaluation`, and `aka.execution`, with orchestration under `aka.workflow.light`. Legacy is
not required. Skills and scripts are checkout resources; update them from the same checkout as
the Python packages.

Always pass the top-level Campaign workspace:

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign
```

The commands discover serial Campaign state and pooled-search trajectories such as `t0001` and
`t0002` automatically. Do not pass a trajectory directory unless a command explicitly asks for it.

Pooled search has one Campaign-private state tree. Lineage-specific Research, Sessions,
and Runtime audits are namespaced below `lineages/tNNNN/`; immutable Gateway Records are stored
once in the Campaign Measurement Store. Observability attributes central Records to their
originating lineage from Runtime audits instead of counting shared evidence as local copies. When
a peer lineage reuses a central Record, its original `runtime_request_id` is resolved across all
Campaign audit namespaces; the consumer needs only its own audit that references the Record and
does not duplicate the creator's audit.

- Six status/query commands are read-only. They never submit, retry, cancel, or mutate Runtime or
  Agate work.
- The two `export_*` commands only create a new local output directory and refuse to overwrite an
  existing path.
- A status inspection does not authorize restarting a Campaign, resubmitting Agate work, cleaning
  history, or editing Supervisor-private state.
- Run the tools from an AKA checkout compatible with the Campaign's persisted format.

The eight supported entry points are:

| Operator question | Command | Primary authority |
|---|---|---|
| Is each lineage alive, what is it doing, and what are its research/token states? | `light_lineage_status.py` | writer lock, Session capture, Research Store, token ledger |
| Which passing full Evaluates and accepted submissions did each lineage produce? | `light_kernel_history.py` | Measurement Store, Campaign Submission Store, token checkpoints |
| Which accepted Kernel is best by GeoMean and by individual Shape? | `light_kernel_best.py` | verified accepted submissions |
| How often do Runtime tools succeed/fail, and how long do calls take? | `light_runtime_calls.py` | Runtime lifecycle audit receipts |
| How often do physical Agate submissions succeed/fail, and what are their timings? | `light_agate_calls.py` | Gateway Job checkpoints |
| Export one Agent commit and every measurement bound to it | `export_light_kernel.py` | Agent Git, Measurement Store, Job checkpoints |
| Expand one Gateway Record into every physical Agate submission | `light_gateway_jobs.py` | `submission-*.json` and current checkpoints |
| Export Runtime request, every raw Agate attempt, and Agent receipt for one Gateway Record | `export_light_gateway_record.py` | Runtime audit, Gateway Record, Job checkpoints |

Only these eight supported commands remain at the top level. Shared implementation lives under
`_internal/` and is not a supported CLI.

## 2. Shared concepts

### 2.1 Scope, trajectory, and lineage

Typical `scope`/`lineage` values are:

- `campaign`: a serial Campaign or top-level controller scope;
- `t0001`, `t0002`: pooled-search trajectories;
- `trajectory`: the corresponding integer, such as `1` or `2`; the top-level campaign is usually
  `null`.

`--lineage` accepts the scope label, for example `--lineage t0001`. It does not accept `T1`, an
Episode number, or a Session ID. Repeat the option to select a union:

```bash
--lineage t0001 --lineage t0003
```

### 2.2 The five identity layers

Keep these identities distinct:

```text
Agent Git commit ID
  └─ Runtime request ID
       └─ Gateway Record ID
            └─ primary Agate Job ID
                 └─ optional Ray/backend Job ID
```

| Identity | Meaning |
|---|---|
| `kernel_commit_id` | Public identity of the complete source tree in Agent Git |
| `runtime_request_id` | Stable audit identity of one Runtime HTTP request |
| `gateway_record_id` | Immutable normalized logical measurement; may contain shards, retries, and multiple Jobs |
| `agate_job_id` | Public observability field for one physical Agate submission |
| `backend_agate_job_ids` | Internal IDs returned by an outer Ray Data Job; an independent raw response may not be available |

All public observability JSON uses these canonical names. Only immutable raw evidence may retain
Runtime's upstream `request_id`, or Agate's upstream `job_id` and `backend_job_ids` fields.

One Gateway Record can reference multiple Agate Jobs because of Shape batching or infrastructure
retries. One Agent commit can also have multiple Evaluate, Profile, Check, Dev, or Command Records.

### 2.3 Output modes

The six status/query commands share these output options:

| Option | Behavior |
|---|---|
| no output option | Human-readable table |
| `--json` | Indented structured JSON |
| `--compact` | Single-line JSON; implies `--json` |

Structured status results generally contain:

```json
{
  "schema_version": 1,
  "observed_at": "2026-11-13T02:00:00+00:00",
  "campaign_workspace": "/workspace/campaign"
}
```

- `schema_version` is the output schema version of that script, not a Campaign or Agate version.
- `observed_at` is when the read-only snapshot was generated.
- `campaign_workspace` is the resolved top-level workspace.

### 2.4 Normalized outcomes

Runtime and Agate detail views normalize outcomes as follows:

| Status | Meaning |
|---|---|
| `succeeded` | A durable successful terminal outcome exists |
| `failed` | A durable failure/rejection/timeout terminal outcome or structured error exists |
| `pending` | The persisted state is known to be running, queued, submitted, or otherwise nonterminal |
| `unknown` | Durable evidence is insufficient; this is neither failure nor proof that work is still running |

Infrastructure, transport, and queue failures are not candidate correctness failures. Never turn
`unknown` into a candidate failure or blindly resubmit it.

### 2.5 Timing fields

| Field | Exact definition |
|---|---|
| Runtime `duration_seconds` | Runtime request receipt-to-terminal end-to-end time |
| Agate `queue_seconds` | Agate `created_at → started_at`; a persisted running-phase boundary may be used when native `started_at` is absent |
| Agate `execution_seconds` | Agate `started_at → finished_at`; uses the same phase boundary as `queue_seconds` |
| Agate `total_seconds` | Agate `created_at → finished_at`; submission-to-terminal time |
| `pending_age_seconds` | Age from the known start/submission timestamp to this observation |

`waiting_seconds` and `average_waiting_seconds` are compatibility aliases for `queue_seconds`.
They are not Runtime end-to-end wait times.

Each timing field has a companion `*_source`; summaries expose `*_source_counts`:

- `agate_started_at`: native Agate `started_at`;
- `agate_phase_elapsed`: phase boundary derived from persisted `updated_at - phase_elapsed_ms`;
- `agate_running_updated_at`: Agate `updated_at` from the first observed running snapshot;
- `runtime_observed_running`: local first-running observation, whose precision is bounded by the poll interval;
- `agate_created_finished`: native Agate `created_at → finished_at`.

New SDK requests privately persist queued/running/terminal observations so phase timing remains
available when the terminal Agate payload omits `started_at`. Historical checkpoints without those
facts keep `queue_seconds` and `execution_seconds` as `null`; the tool never invents a split from
`total_seconds`.

Aggregates only use persisted timing samples:

```json
{
  "average_duration_seconds": 12.4,
  "p50_duration_seconds": 8.0,
  "p95_duration_seconds": 31.2,
  "max_duration_seconds": 35.0,
  "duration_seconds_samples": 9,
  "duration_seconds_coverage": 0.9
}
```

Coverage of `0.9` means nine of ten calls have a usable timing sample. It is not a 90% success
rate.

### 2.6 `evidence_health`

Kernel history, winner, and Kernel export results include an evidence-health projection:

```json
{
  "complete": false,
  "issue_counts": {
    "incomplete_gateway_records": 1,
    "unreadable_gateway_records": 0,
    "accepted_submissions_missing_evidence": 0,
    "missing_token_checkpoints": 1,
    "unresolvable_kernel_commits": 0,
    "unlinked_runtime_audits": 0,
    "gateway_records_missing_runtime_audit": 0
  },
  "incomplete_gateway_records": [
    {"scope": "t0001", "gateway_record_id": "gateway-..."}
  ]
}
```

`complete=false` means the audit chain has a gap; it does not automatically mean the Kernel or
measurement failed.

- `incomplete_gateway_records`: a logical measurement has not produced a complete `record.json`.
- `unreadable_gateway_records`: a Record exists but integrity or parsing failed.
- `accepted_submissions_missing_evidence`: an accepted submission cannot be joined to an eligible
  full Evaluate.
- `missing_token_checkpoints`: the event exists, but its cumulative token checkpoint is absent.
- `unresolvable_kernel_commits`: an accepted commit cannot be resolved from saved Agent Git/source
  artifacts.
- `unlinked_runtime_audits`: a Runtime receipt references a missing Gateway Record.
- `gateway_records_missing_runtime_audit`: a new-format Gateway Record has an original Runtime
  request ID, but no creator receipt exists in any Campaign lineage. Each issue includes the
  `gateway_record_id` and `visible_scopes`; peer reuse alone is not a missing-audit condition.

Old Campaigns may not contain stable request IDs or token checkpoints. The tools report those gaps
instead of joining unrelated records by timestamp.

## 3. `light_lineage_status.py`

### Purpose

Use this command first to answer whether each search line is running, what its Agent Session is
doing, what Research records exist, and how many tokens each path consumed.

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" [options]
```

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--stale-after SECONDS` | `600` | Classify a lock-owning lineage with no active Supervisor request as `running_no_recent_activity` after this age; does not stop it |
| `--lineage NAME` | all | Return only this scope; repeatable |
| `--research-limit N` | `50` | Maximum Direction rows and Experiment rows per lineage; positive integer |
| `--all-research` | off | Return every Direction and Experiment; overrides the limit |
| `--json` | off | Indented JSON |
| `--compact` | off | Single-line JSON |

### Representative result

```json
{
  "campaign_configuration": {
    "search": {"trajectories": 3, "synchronization": "asynchronous"},
    "search_binding_digest": "...",
    "token_budget": 0
  },
  "lineages": [
    {
      "lineage": "t0001",
      "trajectory": 1,
      "workspace": "/workspace/.../t0001",
      "runtime": {
        "phase": "exploring",
        "writer_active": true,
        "liveness": "running_recent_activity",
        "current_episode": 2,
        "completed_episodes": 1,
        "latest_activity_at": "...",
        "stagnation_seconds": 14.2,
        "main_session_silence_seconds": 14.2,
        "active_session_count": 1,
        "active_sessions": [
          {"phase": "waiting_agate", "tool": "evaluate", "activity_state": "recent_activity", "conversation_path": "/campaign/lineages/t0001/segments/episodes/e00000002/sessions/run-00000001/conversation.jsonl"}
        ],
        "latest_session": {
          "session_id": "session-example",
          "conversation_path": "/campaign/lineages/t0001/segments/episodes/e00000002/sessions/run-00000001/conversation.jsonl"
        },
        "active_supervisor_activity_count": 1,
        "supervisor_activity": {
          "phase": "claim_review",
          "operation": "record-experiment",
          "state": "running",
          "elapsed_seconds": 1842.0
        }
      },
      "runtime_configuration": {"optimization_mode": "ultra", "hardware": "L20D"},
      "research": {
        "enabled_modules": ["directions", "experiments"],
        "direction_counts": {"in_progress": 1, "deferred": 2},
        "directions": [{"direction_id": "direction_...", "status": "in_progress", "local_rank": 1}],
        "experiment_count": 3,
        "experiments": [{"experiment_id": "experiment_...", "gateway_record_ids": ["gateway-..."]}],
        "focus_direction_id": "direction_...",
        "active_direction_ids": ["direction_..."],
        "local_direction_queue": ["direction_..."],
        "shared_direction_queue": ["direction_peer_..."],
        "last_shared_queue_sync": {"sequence": 27, "origin": {"epoch": 2}},
        "peer_git_refs": {"available": true, "count": 2, "refs": []},
        "claim_reviews": {"count": 4, "quality_counts": {"supported": 3}}
      },
      "tokens": {
        "main_agent": {"total_tokens": 120000, "reasoning_tokens": 20000, "measurement": "exact"},
        "reviews": {"total_tokens": 18000, "measurement": "exact"},
        "all_paths": {"total_tokens": 138000, "measurement": "exact"}
      },
      "search_lineage": {"epoch": 2, "status": "running"}
    }
  ]
}
```

### Important fields

- `writer_active` is the direct current-execution authority. If it is false, the trajectory
  must not be reported as running.
- `liveness` is `running_recent_activity`, `running_supervisor_activity`,
  `running_no_recent_activity`, or `not_running`. `running_supervisor_activity` means the main
  Session may be quiet while an authoritative Runtime request is still executing, so a long Claim
  Review is not misreported as lineage stagnation.
- Campaign `phase` describes the controller. Session phase is typically `thinking`,
  `executing_tool`, `executing_runtime_tool`, `waiting_agate`, or `waiting_wiki`. A former
  `thinking` phase with no Provider event beyond the stale threshold is reported as
  `provider_wait_unknown` rather than claiming continuous reasoning.
- `main_session_silence_seconds` preserves the main Agent Session's time since its last event even
  while the Supervisor is reviewing, separating Session silence from whole-lineage inactivity.
- `active_sessions[].conversation_path` and `latest_session.conversation_path` are the authoritative
  main-Agent capture paths. To explain current optimization work, read only the last 100 JSONL
  events by default.
- `supervisor_activity` is the latest or current authoritative Runtime request. Active entries carry
  the operation, state, and elapsed time; long `record-experiment` / `update-direction` requests are
  projected as `claim_review`. A completed background Runtime request also advances the lineage's
  newest durable activity.
- `runtime_configuration` is the actual saved Runtime configuration, including mode, hardware,
  timeouts, and Shape batching. Prefer it over reconstructing settings from a launch command.
  Current Campaigns read it from top-level `campaign.json`; detached `runtime-config.json` is a
  read-only compatibility input for older Campaigns.
- `details_truncated` tells whether Research details were bounded.
- `local_direction_queue` and `shared_direction_queue` show local preference and peer publication;
  they do not prove peer work was adopted.
- `peer_git_refs` are imported read-only peer refs; they do not modify local HEAD.
- `tokens.main_agent`, `tokens.reviews`, and `tokens.all_paths` separate the accounting paths.
  Reasoning tokens are a breakdown and must not be added to total tokens again.

## 4. `light_kernel_history.py`

### Purpose and options

Use this command to identify the cumulative token point at which each lineage produced a passing
full Evaluate and which commits later passed `submit-kernel`.

```bash
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" [options]
```

| Option | Meaning |
|---|---|
| `--lineage NAME` | Select a lineage; repeatable |
| `--kernel-commit-id SHA` | Select an exact full Agent commit; repeatable |
| `--last N` | Keep the newest N `kernel_results` per lineage, not N globally |
| `--json` / `--compact` | Structured output mode |

### Representative result

```json
{
  "lineages": [
    {
      "lineage": "t0001",
      "trajectory": 1,
      "kernel_results": [
        {
          "kernel_commit_id": "012345...",
          "gateway_record_id": "gateway-...",
          "evaluated_at": "...",
          "performance": {
            "latency_us_geomean": 4.096,
            "latency_us_arith_mean": 4.21,
            "shape_count": 10,
            "latency_us_by_shape": {"0": 2.1, "1": 3.4}
          },
          "evaluate_token_usage": {
            "checkpoint_id": "...",
            "all_paths": {"total_tokens": 820000},
            "main_agent": {"total_tokens": 760000},
            "reviews": {"total_tokens": 60000}
          },
          "evaluate_token_usage_availability": "available",
          "submissions": [
            {
              "submission_id": "submission-...",
              "submission_revision": 2,
              "accepted": true,
              "evidence_status": "verified",
              "source_contract": "pass",
              "review": {
                "schema_version": 2,
                "verdict": "allow",
                "checks": [{
                  "id": "compute_provenance",
                  "decision": "allow",
                  "category": "self_authored_compute",
                  "reason": "The candidate source implements the core computation.",
                  "evidence": ["candidate/kernel.py:12"]
                }],
                "dependencies": [],
                "summary": "The final source satisfies the active mode."
              },
              "implementation_review": {
                "availability": "available",
                "dsl_policy": "guided",
                "configured_dsl": null,
                "required_phase": "open_dsl"
              },
              "diagnostics": [],
              "submit_token_usage": {
                "checkpoint_id": "...",
                "all_paths": {"total_tokens": 850000}
              },
              "submit_token_usage_availability": "available"
            }
          ]
        }
      ]
    }
  ],
  "evidence_health": {"complete": true, "issue_counts": {}},
  "detail_selection": {
    "lineages": [], "kernel_commit_ids": [], "last_per_lineage": null
  }
}
```

Any `kernel_results[]` row with `performance` must be a standard full Evaluate with no Shape subset
or custom inputs, a non-truncated zero-exit response, `all_pass=true`, positive latency for every
Shape, and an exact candidate Agent commit identity.

Each `submissions[]` entry is nested under the exact lineage, commit, and Gateway Record it cites,
so performance is not duplicated. A submission without matching eligible Evaluate evidence remains
visible with `performance=null`; the tool never guesses the association from time. `accepted=true`
is the original submission decision, while only `evidence_status=verified` is winner eligible.

`implementation_review.technologies` contains the implementation technologies actually observed
by the independent final-source reviewer, such as `cuda`, `triton`, `cutedsl`, or a mixed list. It
is not inferred from the Campaign mode or configured DSL. `availability=unavailable` means an old
submission or missing persisted review facts; the tool does not guess. Human-readable output shows
the values in the `Implementation` column.

`evaluate_token_usage` and `submit_token_usage` are separate immutable cumulative checkpoints at
the two events, not values inferred from the final Campaign total. A `missing_checkpoint` keeps the
event but means its exact token cost is unavailable. Rejected submissions currently have no
submission milestone and report `submit_token_usage_availability=not_recorded`.

A submission may reuse a peer's full Evaluate from the central Measurement Store. Its verified
performance is shown, but the consuming lineage has no local Evaluate token checkpoint:
`evaluate_token_usage=null`, with availability `shared_evidence_no_local_evaluate`. The tool does
not attribute the peer's Evaluate tokens to the consumer.

## 5. `light_kernel_best.py`

This command selects Campaign winners from verified accepted submissions. It has no filters:

```bash
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
```

Only the shared `--json` and `--compact` options are available.

```json
{
  "accepted_submission_count": 4,
  "geomean_best": {
    "scope": "t0002",
    "kernel_commit_id": "89abcd...",
    "gateway_record_id": "gateway-...",
    "latency_us_geomean": 3.91,
    "shape_count": 10
  },
  "shape_best": {
    "0": {
      "shape_id": "0",
      "latency_us": 1.84,
      "kernel_commit_id": "012345...",
      "gateway_record_id": "gateway-...",
      "lineage": "t0001"
    }
  },
  "evidence_health": {"complete": true, "issue_counts": {}}
}
```

- `geomean_best` is one real accepted commit; it is `null` if none is eligible.
- Each `shape_best` row is selected independently and may come from a different commit. The set is
  a Pareto observation, not a synthetic Kernel that can be checked out.
- A faster but unsubmitted passing full Evaluate is intentionally excluded.

## 6. `light_runtime_calls.py`

### Purpose and options

Use this command to count Runtime tool outcomes, measure Runtime end-to-end duration, and inspect
recent failed or pending requests.

```bash
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" [options]
```

| Option | Meaning |
|---|---|
| `--kind KIND` | Runtime tool kind such as `evaluate`, `profile`, `check`, `dev`, `command`, or `kernel_submit`; repeatable |
| `--status STATUS` | `succeeded`, `failed`, `pending`, or `unknown`; repeatable |
| `--lineage NAME` | Restrict detail rows to a scope; repeatable |
| `--since ISO_TIME` | Inclusive lower time bound for details |
| `--until ISO_TIME` | Inclusive upper time bound for details |
| `--last N` | Newest N details after all filters |
| `--json` / `--compact` | Structured output mode |

ISO-8601 values such as `2026-11-13T00:00:00Z` are accepted. An inverted interval or nonpositive
`--last` is rejected.

With no detail selector, `calls=[]` to avoid dumping large audit history. `summary` always covers
the entire Campaign; filters affect only `calls[]`.

```json
{
  "summary": [
    {
      "kind": "evaluate",
      "calls": 12,
      "succeeded": 9,
      "failed": 2,
      "pending": 1,
      "unknown": 0,
      "error_counts": {"job_deadline_exceeded": 2},
      "average_duration_seconds": 48.2,
      "p50_duration_seconds": 31.0,
      "p95_duration_seconds": 141.0,
      "max_duration_seconds": 156.0,
      "duration_seconds_samples": 11,
      "duration_seconds_coverage": 0.9167,
      "oldest_pending_seconds": 82.0
    }
  ],
  "calls": [
    {
      "scope": "t0001",
      "runtime_request_id": "4f...",
      "kind": "evaluate",
      "operation_family": "gateway",
      "status": "failed",
      "success": false,
      "success_basis": "request_lifecycle",
      "created_at": "...",
      "finished_at": "...",
      "duration_seconds": 156.0,
      "http_status": 503,
      "gateway_record_id": "gateway-...",
      "error": {"code": "infrastructure_unavailable", "error_class": "infra"}
    }
  ],
  "detail_selection": {
    "kinds": ["evaluate"],
    "statuses": ["failed"],
    "last": 20,
    "returned": 1,
    "campaign_total": 12,
    "summary_scope": "campaign"
  }
}
```

`success_basis=request_lifecycle` uses a new lifecycle receipt. `audit_exit_code` means a legacy
audit only supplied a process exit code. `unavailable` means success could not be determined.

## 7. `light_agate_calls.py`

This command has the same filters as Runtime calls, but its unit is one physical Agate
submission:

```bash
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" \
  --kind evaluate --status failed --lineage t0001 --last 20 --json
```

| Option | Meaning |
|---|---|
| `--kind KIND` | Agate operation kind; repeatable |
| `--status STATUS` | `succeeded/failed/pending/unknown`; repeatable |
| `--lineage NAME` | Scope selector; repeatable |
| `--since/--until ISO_TIME` | Inclusive Job-checkpoint update-time bounds |
| `--last N` | Newest N physical submissions after filtering |
| `--json` / `--compact` | Structured output mode |

With no selector, `calls=[]`; `summary` remains Campaign-wide.

```json
{
  "summary": [
    {
      "kind": "evaluate",
      "calls": 18,
      "succeeded": 14,
      "failed": 3,
      "pending": 1,
      "unknown": 0,
      "error_counts": {"queue_timeout": 1, "job_deadline_exceeded": 2},
      "average_queue_seconds": 12.0,
      "p95_queue_seconds": 44.0,
      "queue_seconds_samples": 15,
      "queue_seconds_coverage": 0.8333,
      "queue_seconds_source_counts": {"agate_phase_elapsed": 15},
      "average_execution_seconds": 60.0,
      "execution_seconds_source_counts": {"agate_phase_elapsed": 15},
      "average_total_seconds": 72.0,
      "total_seconds_source_counts": {"agate_created_finished": 18},
      "oldest_pending_seconds": 91.0,
      "average_waiting_seconds": 12.0
    }
  ],
  "calls": [
    {
      "scope": "t0001",
      "operation": "evaluate",
      "status": "failed",
      "agate_job_id": "ev_...",
      "gateway_record_id": "gateway-...",
      "checkpoint_path": "batch-0001/submission-0002.json",
      "submission": 2,
      "runtime_request_ids": ["4f..."],
      "queue_seconds": 5.1,
      "queue_seconds_source": "agate_phase_elapsed",
      "execution_seconds": 600.0,
      "execution_seconds_source": "agate_phase_elapsed",
      "total_seconds": 605.1,
      "total_seconds_source": "agate_created_finished",
      "error": {"reason": "job deadline exceeded", "category": "timeout"}
    }
  ]
}
```

`calls=18` counts physical submissions, not logical Evaluate calls. Shape batching and retries can
both increase the count.

## 8. `export_light_kernel.py`

Export one complete Agent commit and every measurement bound to that exact commit:

```bash
python3 "$AKA_REPO/observability/export_light_kernel.py" "$CAMPAIGN" \
  --kernel-commit-id 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-kernel-evidence
```

| Option | Requirement |
|---|---|
| `--kernel-commit-id SHA` | Full Agent Git commit resolvable from Campaign Git/source artifacts |
| `--output PATH` | Must not exist; never merged or overwritten |

This command always prints a JSON manifest summary; it has no `--json` option.

```text
new-kernel-evidence/
├── manifest.json
├── kernel/                         # complete source tree and original layout
└── measurements/
    └── gateway-.../
        ├── record.json             # normalized complete Gateway Record
        ├── request.json            # normalized Runtime request
        ├── agent-response.json     # exact projection returned to the Agent
        └── agate-jobs.json         # every physical submission and raw process/response
```

Representative manifest fields:

```json
{
  "kernel_commit_id": "012345...",
  "source_manifest": {"output_directory": "kernel", "files": ["kernel.py", "impl/a.cu"]},
  "measurement_count": 3,
  "measurements": [
    {
      "scope": "t0001",
      "gateway_record_id": "gateway-...",
      "operation": "evaluate",
      "passing_standard_full_evaluate": true,
      "runtime_request_id": "4f...",
      "result_summary": {
        "all_pass": true,
        "latency_us_geomean": 4.096,
        "shape_count": 10,
        "performance_objective": "geomean"
      },
      "agate_job_ids": ["ev_..."],
      "backend_agate_job_ids": []
    }
  ],
  "kernel_submissions": [],
  "evidence_health": {"complete": true},
  "output_directory": "/path/to/new-kernel-evidence"
}
```

An empty `kernel_submissions` array means the commit has no submission receipt; it does not mean
measurements are absent.

## 9. `light_gateway_jobs.py`

Expand one logical Gateway Record into all successful, failed, pending, reused, and retried
physical submissions:

```bash
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
```

| Option | Requirement |
|---|---|
| `--gateway-record-id ID` | Required complete `gateway-...` ID |
| `--json` / `--compact` | Structured output mode |

```json
{
  "gateway_record_id": "gateway-...",
  "summary": {
    "record_location_count": 1,
    "submission_attempt_count": 3,
    "agate_job_id_count": 3,
    "backend_agate_job_id_count": 1,
    "reused_attempt_count": 1,
    "status_counts": {"failed": 1, "succeeded": 2}
  },
  "locations": [
    {
      "scope": "t0001",
      "operation": "evaluate",
      "complete": true,
      "runtime_request_id": "4f...",
      "submission_attempt_count": 3,
      "record_directory": "/private/.../gateway-..."
    }
  ],
  "attempts": [
    {
      "scope": "t0001",
      "checkpoint_path": "batch-0001",
      "submission": 2,
      "checkpoint_phase": "terminal",
      "runtime_request_ids": ["4f..."],
      "agate_job_id": "ev_...",
      "status": "succeeded",
      "reused": false,
      "shape_ids": ["0", "1", "2"],
      "backend_agate_job_ids": ["ray_..."],
      "retry_counters": {"infra": 1},
      "error": null,
      "snapshot_path": "/private/.../submission-0002.json"
    }
  ]
}
```

- `submission_attempt_count` includes historical attempts, not only the current checkpoint.
- `agate_job_id_count` and `backend_agate_job_id_count` separately deduplicate outer Agate Jobs
  and Ray/backend IDs.
- `reused=true` means an existing batch checkpoint was reused, not that a new Job was submitted.
- `shape_ids` identifies the Shape subset of that physical submission.
- `error` preserves the structured category/reason/message.

## 10. `export_light_gateway_record.py`

Export the complete correlation chain for one logical Gateway Record. The export includes every
Shape shard, retry, primary Job, and persisted Ray/backend ID under that Record:

```bash
python3 "$AKA_REPO/observability/export_light_gateway_record.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-gateway-evidence
```

| Option | Requirement |
|---|---|
| `--gateway-record-id ID` | Required complete `gateway-...` ID |
| `--output PATH` | Must not exist |

The command always prints a JSON manifest summary. If the Record exists in both local and shared
trajectory stores, the exporter first verifies that the request, Gateway Record, and Agent receipt
are identical, then deduplicates and merges Runtime audits and Agate attempts. The export directory
contains one canonical copy; physical provenance appears only in
`manifest.json.source_locations`. Conflicting canonical content under the same Gateway ID fails
with `gateway_record_integrity_conflict` instead of selecting a copy.

```text
new-gateway-evidence/
├── manifest.json
├── runtime-request.json
├── runtime-audits.json
├── agate-attempts.json
├── agent-response.json
└── gateway-record.json
```

```json
{
  "gateway_record_id": "gateway-...",
  "source_location_count": 2,
  "runtime_request_ids": ["4f..."],
  "agate_job_ids": ["ev_..."],
  "backend_agate_job_ids": ["ray_..."],
  "runtime_audit_count": 1,
  "agate_attempt_count": 3,
  "agent_response_available": true,
  "source_locations": [
    {
      "scope": "t0001",
      "runtime_request_ids": ["4f..."],
      "runtime_audit_count": 1,
      "agate_attempt_count": 3
    }
  ],
  "output_directory": "/path/to/new-gateway-evidence"
}
```

`agate-attempts.json` preserves each physical attempt's `agate_job_id`,
`backend_agate_job_ids`,
status, retry counters, complete outer Agate response, and raw process. When a backend ID only
appears inside an outer Ray response, do not describe that outer response as the backend Job's
independent raw response. `agent-response.json` is the Agent-visible JSON body deterministically
reconstructed from the immutable Gateway Record under the current Runtime HTTP contract; it is not
the complete private execution response stored in the Record. Duplicate Runtime audits and Agate
attempts are exported once, while `source_audit_paths` and `source_snapshot_paths` retain physical
provenance.

## 11. Common diagnosis workflows

### Overall status

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" --last 5 --json
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
```

Use `writer_active/liveness` before inspecting Session phase. Do not infer process liveness from a
scheduler checkpoint.

### Recent Runtime failures

```bash
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" \
  --status failed --last 20 --json
```

Inspect `error.code/reason/error_class`, `http_status`, `runtime_request_id`, and
`gateway_record_id`. If a
Gateway Record exists, continue through that Record rather than resubmitting.

### Recent Agate failures and queueing

```bash
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" \
  --status failed --status pending --last 20 --json
```

- high queue time with normal execution indicates queue pressure;
- normal queue time with execution near the deadline indicates slow execution or Shape timeout;
- if total time cannot be reconciled, inspect sample coverage and missing timestamps rather than
  inventing values;
- multiple failed Jobs under one Gateway Record should be expanded to distinguish retries from
  Shape shards.

### Trace a Kernel to raw Agate evidence

```text
light_kernel_history.py
  → kernel_commit_id + gateway_record_id
export_light_kernel.py
  → complete source + measurements/*/agate-jobs.json
light_gateway_jobs.py
  → all agate_job_id/backend_agate_job_ids
export_light_gateway_record.py
  → Runtime request + every Agate attempt + exact Agent receipt for the logical request
```

### Check Direction/Experiment behavior

Use `light_lineage_status.py` to verify:

1. `enabled_modules` matches the launch configuration;
2. Direction lifecycle and local/shared rank change over time;
3. Experiments reference Gateway Records and candidate commits where appropriate;
4. claim-review quality is not permanently stuck at partially supported/unreviewed;
5. `last_shared_queue_sync` and `peer_git_refs` update at synchronization boundaries.

A supported Review does not mean a Kernel was accepted. A supported Experiment does not imply its
Direction should be closed.

## 12. Token semantics

Usage objects can contain:

```json
{
  "input_tokens": 100000,
  "output_tokens": 20000,
  "reasoning_tokens": 8000,
  "cache_read_tokens": 50000,
  "cache_write_tokens": 1000,
  "total_tokens": 120000,
  "measurement": "exact"
}
```

- `total_tokens` follows the Provider/Session total convention.
- `reasoning_tokens` is a breakdown and must not be added to total tokens again.
- `main_agent` and `reviews` remain separate; `all_paths` is their combined accounting path.
- Review usage can be split into `final_source_policy_review`, `direction_claim_review`,
  `experiment_claim_review`, and `supplemental_numerical_review`.
- `measurement=exact` means all counters are provider-native and complete.
- `estimated` means finalized conversation evidence supplied a best-effort estimate because the
  Provider did not return counters.
- `partial` means only some calls or fields have counters.
- `unavailable` means no verifiable counter exists.

Do not combine exact and estimated token figures without reporting measurement quality.

## 13. Errors and exit codes

- Success exits with code `0`.
- Missing workspaces, malformed IDs, unavailable Records/Jobs/commits, and invalid time intervals
  exit with code `1` and write a message to stderr.
- Missing required CLI arguments or argparse syntax errors normally exit with code `2`.
- Export commands refuse an existing output path.

An error is itself bounded evidence. For example, “Job ID is not available in this Campaign” does
not prove that the Job never existed; it only says the persisted Campaign does not contain it.

## 14. Reporting requirements

An evidence-backed operator report should preserve:

- Campaign workspace and observation time;
- lineage/trajectory;
- the correct Agent commit, Runtime request, Gateway Record, and Agate Job identities;
- writer liveness and newest durable activity;
- original succeeded/failed/pending/unknown classification;
- the distinction between queue, execution, Agate total, and Runtime duration;
- token measurement quality (`exact`, `estimated`, `partial`, or `unavailable`);
- material `evidence_health` gaps that could change the conclusion.

Do not use old Framework Baseline, handoff, promotion, canonical `vN`, or Episode
`attempt.json.accepted` fields as current winner authority. A current winner requires:

```text
accepted submit-kernel receipt for exact Agent commit
∩ passing standard full Evaluate for that commit
∩ complete per-Shape performance evidence
```

## 15. Read-only dashboard

`light_dashboard.py` renders the same snapshots as the eight tools: status, Research, tokens, Kernels, calls, and evidence.

```bash
python3 "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --port 8765
python3 "$AKA_REPO/observability/light_dashboard.py" "$CAMPAIGN" --capture /tmp/dashboard.json
python3 "$AKA_REPO/observability/light_dashboard.py" /tmp/dashboard.json --port 8765
```

The default address is `127.0.0.1:8765`; `--port 0` selects a free port and `--refresh-ms` defaults
to 5000 milliseconds. For remote Campaigns, run `--capture` remotely and download the JSON.
The local viewer preserves liveness at `captured_at` without rechecking local PIDs or locks.
There are no launch, retry, cancellation, or submission controls. See the
[Dashboard documentation](../docs/optimization-dashboard.md) for options and HTTP behavior.

### Claude token accounting

Main-Agent and independent-review usage accumulates deduplicated response counters throughout the invocation, including after completion. `result.usage` is retained as `terminal` audit evidence and never replaces the accumulated total. Input, output and cache counters remain separate; reasoning is a separately reported subset, not added twice. Missing response counters remain partial or unavailable; conversation-text estimates remain labelled estimated. Existing immutable checkpoints are not automatically rebuilt.
