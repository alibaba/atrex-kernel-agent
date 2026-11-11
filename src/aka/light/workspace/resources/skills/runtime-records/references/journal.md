# Direction and Experiment requests

All commands use the Session-scoped Runtime client. Put mutation requests below `scratch/`.
Requests must fit the documented 2 MiB envelope.

## Inspect progress and research files

Runtime automatically maintains:

```text
directions/<status>/<direction_id>.json
experiments/<experiment_id>/experiment.json
records/<gateway_record_id>.json
PROGRESS.md
```

Direction files contain their full decision-useful state, immutable `proposed_by` lineage, `access`,
the Agent-maintained plan, and either `local_rank` or `shared_rank`. Runtime does not track completed
plan steps or select the next action. Experiments identify their
Direction through `direction_id`; Runtime derives the reverse association when needed instead of
persisting a second index. `supporting_experiment_ids` is the reviewed evidence subset selected when
closing the Direction. `owner` identifies the trajectory
currently executing an in-progress Direction. `experiment.json` contains research semantics,
Gateway Record references, immutable `recorded_by` lineage, and `access`. Root
`records/` contains every currently visible standardized public result, flat and without per-
Experiment copies. Neither contains private requests, input hashes or executor streams. Kernel
commit IDs link Records to exact Git source. Search these files with normal file tools. They are
regenerated from private Runtime state and must not be edited.

Root `PROGRESS.md` is the bounded research projection and includes a capped per-Shape Pareto
summary, an independently ordered Local work queue, paused and terminal sections, and an advisory Shared direction pool
when Direction recording is enabled. Global aggregation orders only the shared pool and never
overwrites local priorities. Read root `frontier.json` for the complete eligible frontier. In
Experiment-only mode, `experiments/index.json` provides a compact catalog; when Directions are
enabled, navigate Experiments from their `direction_id` and filesystem paths instead of maintaining
a second durable index.
Peer Directions may appear as `in_progress(Tn)` in `PROGRESS.md`. Choose executable work from the
Local work queue. Use the Shared pool to absorb compatible mechanisms, evidence, and applicability
into one local Direction; cite every source ID in `derived_from_direction_ids`. Explicitly
`not_reviewed`, `partially_supported`, or `unreviewed` Directions and Experiments remain
trajectory-local until a reviewed mutation replaces them.

Claim-bearing mutations return `review` once. Its workflow fields are deliberately distinct:

- `archive_status`: whether Runtime stored the submitted claim;
- `claim_review_quality`: `not_reviewed`, `supported`, `partially_supported`, `unsupported`, or `unreviewed`;
- `evidence_eligible`: whether the claim may support terminal conclusions, peer publication, or handoff;
- `runtime_action`: how to repair or continue the Runtime workflow;
- `scientific_next_check`: the smallest evidence-producing test suggested by the reviewer.

The projected record retains only `claim_review_quality`, `scope_safe_interpretation`, and
`scientific_next_check`; full Review receipts and reviewer conversations remain private.

## Propose and select a Direction

A Direction is a durable optimization program organized around a bottleneck, workload regime, or
architecture objective and should normally span multiple candidates and Experiments. A single
implementation, parameter sweep, capability probe, or mechanism test is an Experiment under an
existing Direction, not a new Direction. Generic exploration, Runtime/evaluator maintenance,
infrastructure operation, and unrelated documentation do not belong in the Direction queue. `change_scope` is `incremental` or
`major_redesign` and describes the breadth of the intended Kernel architecture work.

```json
{
  "action": "propose",
  "name": "Decode parallelism and KV-traffic architecture",
  "change_scope": "major_redesign",
  "objective": "Remove repeated KV work and expose enough parallelism across decode regimes",
  "applicability": "Short-query, long-KV and GQA-dominated workloads",
  "expected_benefit": "Reduce decode-side full-workload latency",
  "plan": [
    "Measure GQA packing",
    "Explore split-KV and combine families",
    "Compare inline decode with KV materialization",
    "Determine workload-regime dispatch"
  ],
  "stop_conditions": [
    "All viable architecture families are measured",
    "Remaining variants have insufficient expected full-workload impact"
  ]
}
```

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/direction.json
```

`objective` states the bottleneck or architecture outcome that unifies the work. `applicability`
states the workload, Shape regime, semantic case or hardware condition covered by the Direction.
`expected_benefit` states the predicted observable Kernel improvement; avoid unsupported precision,
but name the affected metric, bottleneck, or branch in the implementation strategy. `stop_conditions`
is an optional array of non-empty strings, not one string.
`relationship`, `derived_from_direction_ids`, and
`derived_from_experiment_ids` may describe `refinement`, `correction`, `combination`, `alternative`,
`reimplementation`, or `contradiction`. Reuse the existing ID while objective and applicability are unchanged. Propose a
new idea only after refreshing the catalog and checking likely matches. These fields record
genealogy only. A derived Direction receives its own lifecycle, plan, owner and queue position; it
does not inherit parent status, Experiments, Gateway evidence, Kernel source or priority. Materialize
and switch to an imported parent Git ref separately when the new work actually continues that source.
Runtime validates visible parents, parent counts and cycles, but not whether the chosen relationship
label is scientifically correct.

When Experiment recording is disabled, a Direction may cite its measurement evidence directly with
`gateway_record_ids`. Include relevant IDs on `propose`, `revise`, or a terminal action; Runtime
preserves draft citations and supplies them to the admission or terminal Review. When Experiment
recording is enabled, keep measurement citations in Experiments: a derived Direction cites
`derived_from_experiment_ids`, while a terminal conclusion cites `supporting_experiment_ids`.

Correct an existing open local Direction without inventing a replacement identity:

```json
{
  "action": "revise",
  "direction_id": "direction_11111111111111111111111111111111",
  "name": "Decode parallelism and KV-traffic architecture",
  "change_scope": "major_redesign",
  "objective": "Remove repeated KV work and expose enough parallelism across decode regimes",
  "applicability": "Short-query, long-KV and GQA-dominated workloads",
  "expected_benefit": "Reduce decode-side full-workload latency",
  "plan": ["Measure GQA packing", "Evaluate KSPLIT=1/2/4", "Compare combine families"]
}
```

`revise` keeps `direction_id`, lifecycle status and genealogy and increments `revision`. Revising a
proposed draft remains unreviewed; revising an active Direction reruns Review. It replaces the objective,
scope and plan in the latest projection. Only `proposed` or
`in_progress` local Directions can be revised; restart a deferred or blocked Direction first.
Completed and abandoned Directions are terminal, so continued work uses a new derived Direction.
Shared peer Directions remain read-only. Prior submissions and Reviews remain in
the private Supervisor audit. The reviewer resumes the latest valid Review Session for this
Direction so it can compare the correction with its previous critique; if that native Session is
unavailable, Runtime retries once in a fresh Session.

Start selected work:

```json
{
  "action": "start",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "Highest expected impact; the first probe is bounded and reversible"
}
```

`start` runs the first independent Review of the effective proposal before changing its lifecycle.
Supported and partially supported starts may proceed locally; unsupported starts are rejected, while
partial or reviewer-unavailable state remains ineligible for terminal evidence and peer publication.
The lifecycle is `proposed --start--> in_progress`. Only `in_progress` may `complete`, `abandon`,
`defer`, or `block`; restart `deferred` or `blocked` work with `start`. `completed` and `abandoned`
are terminal and continued work requires a derived Direction.

A trajectory may own multiple local active Directions. The most recently started Direction is the
focus; queue reordering does not change it. Runtime infers an omitted Experiment `direction_id` only
when exactly one local Direction is active; with zero or multiple active Directions, include it.
Keep concurrent work intentional, address updates by Direction ID, and leave new ideas proposed until work actually starts. Continue each
selected redesign through bounded edit/measurement cycles instead of reporting its first promotable checkpoint.

Reorder the durable queue:

```json
{
  "direction_id": "direction_11111111111111111111111111111111",
  "before_direction_id": "direction_22222222222222222222222222222222",
  "reason": "The profile makes this the highest-information structural experiment"
}
```

```bash
python3 tools/sandbox.py --kind prioritize-direction --request-file scratch/priority.json
```

Use `after_direction_id` instead of `before_direction_id`, or omit both to move the Direction to the
end. Routine completion of plan steps does not require a Runtime mutation. If evidence materially
changes the claim or remaining plan, submit a complete `action=revise` Direction instead.

## Record an Experiment

Record a research conclusion when it becomes useful. Cite exact Gateway Record IDs and supporting
workspace material when they exist:

```json
{
  "name": "KSPLIT=2 complete-workload measurement",
  "intent": "Determine whether split KV pays for its reduction cost",
  "gateway_record_ids": ["gateway-0123456789abcdef0123456789abcdef"],
  "interpretation": "Long-KV regimes improved, while short-KV remains unresolved",
  "references": ["scratch/ksplit-analysis.md", "profiles/ksplit2"]
}
```

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

When Directions are enabled, Runtime infers `direction_id` only when exactly one local Direction is
active. With zero or multiple active Directions, include the ID explicitly. When Directions are
disabled, omit it.

Runtime validates every cited ID; its standardized public result is available under root `records/`.
Correctness, latency, failure, operation and Kernel commit identity live only in those records. Use
Git to inspect source changes between the cited commits. Do not copy these facts into
`experiment.json` or `interpretation`.

`gateway_record_ids` and `references` are independent optional arrays. Either may be omitted or
empty. An Experiment without Gateway Records remains a useful research note, but cannot by itself
support a measured Direction lifecycle decision or candidate promotion.

Revise a `local_writable` Experiment by resubmitting its required fields plus the `experiment_id` shown in
`experiments/<experiment_id>/experiment.json`. Runtime keeps that identity, increments `revision`,
reruns independent Review, and updates the projected latest version. Omitted optional evidence is
preserved; explicit empty arrays clear it. `direction_id` is immutable. Extend a `peer_read_only`
Experiment with a new local record rather than revising it. Prior submissions, Review receipts, conversations, and referenced bytes
remain available to the Supervisor's private audit even though the Agent sees only the latest revision.
The reviewer resumes that Experiment's latest valid Review Session; if native resume is unavailable,
Runtime falls back to one fresh Review Session without discarding the earlier audit records.

Every non-empty `references` entry must already exist and name a regular file or directory inside the Agent
workspace. Paths may not overlap or traverse links. Runtime snapshots at most 32 roots, 256 files,
16 MiB per file and 64 MiB total. It rejects the Experiment without mutation when a path is missing
or unsafe. On success, `experiment.json` contains rewritten paths such as
`experiments/<experiment_id>/references/scratch/ksplit-analysis.md`; these archived bytes are
independent of later edits to the originals and are shared with the Experiment at the configured
peer-publication boundary (a synchronous Epoch barrier or an asynchronous terminal-Episode publication).

Optional Wiki attribution retains `wiki_usage_status`, `wiki_query_ids`, and `wiki_usage`. Each usage
row has `query_id`, `disposition`, and the returned `wiki_id` or `cite`. Do not invent identifiers.

## Close, defer, or preserve a Direction

Close only when the research state is truly terminal:

```json
{
  "action": "complete",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "The planned KSPLIT family was implemented and measured",
  "supporting_experiment_ids": ["experiment_22222222222222222222222222222222"]
}
```

`complete`, `abandon`, `block`, and `defer` use the same lifecycle fields. There is no
`hypothesis_status`: a Direction is an optimization program, not one supported/refuted proposition.
`supporting_experiment_ids` optionally identifies measured work that informs the lifecycle decision.
Direction-only mode always omits `supporting_experiment_ids`.
Use `defer` for a deliberate strategic pause or lower-priority path that can be resumed by the Agent;
use `block` only for a concrete technical or external dependency that currently prevents progress.
`defer` or `block` also requires a non-empty `reopen_condition` describing what evidence or
capability should reactivate it. `abandon` and `complete` are terminal. Active Directions may cross an Episode boundary with
an ordinary `direction_terminal` handoff; close them only when their own objective and remaining
variants are genuinely exhausted. In pooled search, `cross_trajectory_share` specifically publishes
a reusable measured checkpoint to peers. For a blocked handoff, close all local active Directions
with `block` first.

## Continue a measured peer Kernel

Epoch sharing imports exact peer Git history under refs such as
`refs/atrex/shared/T2/e0001/head`. Keep or commit local work before switching. Inspect or continue the
peer source with normal Git:

```bash
git log --oneline --decorate refs/atrex/shared/T2/e0001/head
git diff HEAD refs/atrex/shared/T2/e0001/head
git switch -c explore/t2-e0001 refs/atrex/shared/T2/e0001/head
```

The checked-out commit is the Kernel ID. Switching source does not assert correctness, performance,
or promotion; evaluate the resulting clean HEAD normally. Legacy evidence whose original Git object
is unavailable stays private to Supervisor recovery and is not projected as reusable Agent evidence.

## Submit an Episode report

Candidate:

```json
{
  "status": "candidate_ready",
  "handoff_reason": "direction_terminal",
  "summary": "A measured checkpoint is ready for independent gates",
  "selected_experiment_id": "experiment_22222222222222222222222222222222"
}
```

With Experiments disabled, use `gateway_record_ids` instead. The selected evidence must include a
passing standard full Evaluate for the exact current source. Subset Evaluate, Profile, Dev, Check,
Disassemble, and ABBA alone do not satisfy nomination.

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

`handoff_reason` is required. With both modules enabled, `candidate_ready` plus
`direction_terminal` is the ordinary handoff value and may preserve local `in_progress` Directions.
For `candidate_ready`, the selected Experiment may belong to an `in_progress`, completed, or
abandoned visible Direction. A pivot with `direction_terminal` has no selected Experiment. In pooled
search, `cross_trajectory_share` with `status=candidate_ready` requires the selected Experiment to
belong to an `in_progress` Direction and publishes a reusable checkpoint backed by a passing standard
full Evaluate of the exact committed source. With Directions disabled,
the supported visible Experiment and exact-HEAD Evaluate requirements still apply to `candidate_ready`
without a Direction relation. A historical Experiment may be reused when its Review and measurement
remain eligible. Use `blocked` only with `status=blocked` and a non-empty `blocker`. Close every
local active Direction with `block` first. For example, pivot is
`{"status":"pivot","handoff_reason":"direction_terminal","summary":"..."}`. A repairable rejection
leaves the Episode open; follow `error.next_action` and resubmit only when the stated handoff condition
is satisfied. Acceptance is a handoff, not a promotion. Stop after acceptance.
