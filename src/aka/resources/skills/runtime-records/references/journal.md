# Direction and Experiment requests

All commands use the Session-scoped Runtime client. Put mutation requests below `scratch/`.
Requests must fit the documented 2 MiB envelope.

## Inspect progress and research files

Runtime automatically maintains:

```text
research-library/directions/<direction_id>/vN/direction.json
research-library/experiments/<experiment_id>/vN/experiment.json
records/<gateway_record_id>.json
PROGRESS.md
```

Direction and Experiment versions start at immutable `v1`. Each archived update creates the next
per-identity catalog version containing the claim and its compact Review state. A Direction `start`
retains the prior Review without running another one. The greatest visible `vN` is current for this lineage. Direction files contain their full decision-useful state, immutable `owner_lineage`,
`access`, and the Agent-maintained plan. Runtime does not track completed plan steps or select the
next action. Experiments identify their
Direction through `direction_id`; Runtime derives the reverse association when needed instead of
persisting a second index. `supporting_experiment_ids` is the reviewed evidence subset selected when
closing the Direction. `owner` identifies the trajectory
currently executing an in-progress Direction. Each version's `experiment.json` contains research semantics,
Gateway Record references, immutable `owner_lineage`, and `access`. Root
`records/` contains every currently visible standardized public result, flat and without per-
Experiment copies. Neither contains private requests, input hashes or executor streams. Kernel
commit IDs link Records to exact Git source. Search these files with normal file tools. They are
visibility-filtered copies of the Campaign Library and must not be edited. Bubblewrap mounts the
tree read-only; native-mode edits are discarded on the next Runtime refresh.

Root `PROGRESS.md` is the bounded research projection and includes a capped accepted-submission per-Shape Pareto
summary, an independently ordered Local work queue, paused and terminal sections, and an advisory Shared direction pool
when Direction recording is enabled. Each listed Direction includes its current assessment (`analysis`)
when recorded, alongside its objective and expected benefit. Global aggregation orders only the shared pool and never
overwrites local priorities. Read root `frontier.json` for the complete eligible accepted-submission frontier.
Peer Directions may appear as `in_progress(Tn)` in `PROGRESS.md`. Choose executable work from the
Local work queue. Use the Shared pool to absorb compatible mechanisms, evidence, and applicability
into one local Direction; cite every source ID in `derived_from_direction_ids`. Archived Direction
and Experiment versions enter the central Campaign Research Library. Both are reviewed on first
submission and start at immutable `v1`. `research-library/` is the
read-only, visibility-filtered workspace copy. Synchronous search exposes peer revisions only through the
watermark sealed at the last Epoch boundary; later revisions wait for the next Epoch. Asynchronous
search exposes the current catalog head immediately. Quality fields distinguish context from
established evidence and never change ownership or visibility.

## Cross-trajectory measurement visibility

Each lineage sees Records cited by the Direction/Experiment revisions currently visible in that
lineage's projection, plus passing standard full Evaluates. Accepted submissions and winners therefore
publish their complete Evaluate evidence automatically. Review quality controls whether a claim is
trusted, not whether its cited Record can be inspected. There is no Agent operation for publishing
arbitrary local evidence. Synchronous search uses the lineage's current Epoch watermark; asynchronous
search uses its live visible revision set. Every immutable Record remains stored once in the Campaign Store.

Claim-bearing mutations return `review` once. Its workflow fields are deliberately distinct:

- `archive_status`: whether Runtime stored the submitted claim;
- `claim_review_quality`: `supported`, `partially_supported`, `unsupported`, or `review_unavailable`;
- `evidence_eligible`: whether the claim may support Direction lifecycle conclusions; it does not control visibility;
- `runtime_action`: how to repair or continue the Runtime workflow;
- `scientific_next_check`: an advisory decision-relevant check, or `No additional check needed for the next decision.`

The projected record retains only `claim_review_quality`, `scope_safe_interpretation`, and
`scientific_next_check`; full Review receipts and reviewer conversations remain private.

Partial reviews do not require immediate repair or resubmission. Use observations within
`scope_safe_interpretation` for further candidate trials; unresolved claims remain uncertain.
Perform the suggested check or revise only when the uncertainty affects the next implementation
choice or lifecycle decision. Do not repeat measurements or submissions merely to obtain `supported`.
An unknown mechanism can coexist with a supported observation and next trial; excluding high-upside
paths, generalizing beyond measured workloads, or declaring a program exhausted needs stronger support.

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
  "plan": ["Measure GQA packing", "Evaluate KSPLIT=1/2/4", "Compare combine families"],
  "analysis": "Packing improved the measured long-KV regimes; split and combine variants remain open."
}
```

`revise` keeps `direction_id`, lifecycle status and genealogy and increments `revision`. Optional `analysis`
updates the current assessment or interim findings; omitting it preserves the previous value.
Supply non-empty text when updating it. The assessment is included in the revision,
Review, shared projection and `PROGRESS.md`; no close/restart is needed. Every revision is reviewed,
including revisions while still proposed. It replaces the objective,
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
  "direction_id": "direction_11111111111111111111111111111111"
}
```

`propose` runs admission Review of the optimization objective, applicability, and strategy space;
it requires no Experiment or prior measurement. Supported and partially supported proposals are
archived; unsupported proposals are rejected before mutation. `start` changes lifecycle without
another Review and retains the prior review quality. Optional `analysis` replaces the current
assessment; omitting it preserves that assessment. Partial or reviewer-unavailable state remains
shared context rather than established supporting evidence.
The lifecycle is `proposed --start--> in_progress`. Only `in_progress` may `complete`, `abandon`,
`defer`, or `block`; restart `deferred` or `blocked` work with `start`. `completed` and `abandoned`
are terminal and continued work requires a derived Direction.

`start` accepts a partially supported Direction without another Review. `defer` and `block`
also accept partially supported Review and require `reopen_condition`. Only `complete` and
`abandon` require a `supported` Review of that submission. A partial or unavailable terminal
Review leaves the Direction `in_progress`, returns `archive_status=not_archived`, and preserves
the attempted update and Review in the private audit. Prior supported Reviews do not satisfy
this terminal gate; proposing and starting work still require no Experiment.

A trajectory may own multiple local active Directions. The most recently started Direction is the
focus; queue reordering does not change it. Runtime infers an omitted Experiment `direction_id` only
when exactly one local Direction is active; with zero or multiple active Directions, include it.
An explicitly named local proposed Direction may also own an Experiment. It is optional:
proposing and starting work do not require one. A later `revise` Review receives the visible
Experiments already bound to that Direction when assessing its interim findings.
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

Record an Experiment at a decision boundary: when you use measured results to keep or discard a
candidate, select a parameter or workload strategy, rule out a branch, or switch paths. Record the
observations and chosen next action before pursuing that action. Combine completed trials that
inform this same decision; once the decision is made, do not postpone recording for the next probe,
a complete implementation, full Evaluate, mechanism proof, or Direction closure. Revise the
Experiment when later evidence updates the same conclusion; use a new Experiment for a different
question. Not every Command or Evaluate needs an Experiment.
Record observations, their measured scope, and the
next action; the mechanism may remain unknown. Cite all relevant Records or archived references,
including regressions that limit the conclusion. Runtime retains each measurement and its source
identity. A disposable probe that changes no decision need not be recorded:

```json
{
  "name": "KSPLIT=2 complete-workload measurement",
  "intent": "Determine whether split KV pays for its reduction cost",
  "gateway_record_ids": ["gateway-0123456789abcdef0123456789abcdef"],
  "interpretation": "The measured long-KV regimes improved while short-KV regressed. The mechanism is unknown; next test workload dispatch and measure its full-workload effect.",
  "references": ["scratch/ksplit-analysis.md", "profiles/ksplit2"]
}
```

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

A hardware capability probe also merits an Experiment when its result selects an implementation
path. For example, after checking an MMA instruction's output and throughput, record the bounded
observation before starting the Kernel that uses it; no full Evaluate is needed for this probe record:

```json
{
  "name": "FP8 MMA probe selects the next implementation path",
  "intent": "Determine whether FP8 MMA is usable for the next GEMM candidate",
  "gateway_record_ids": ["gateway-<actual-probe-record-id>"],
  "interpretation": "The tested FP8 MMA configuration passed the probe's output check and ran faster than its BF16 counterpart. Next implement an FP8 MMA candidate. This establishes only the tested probe behavior, not full-Kernel correctness, workload latency, or an architectural peak.",
  "references": ["scratch/probe_mma.py"]
}
```

Use the actual Record ID and observations. With Directions enabled, add the appropriate
`direction_id` as described below. A follow-up probe may revise this conclusion; it need not delay
recording the decision already made.

When Directions are enabled, Runtime infers `direction_id` only when exactly one local Direction is
active. With zero or multiple active Directions, include the ID explicitly. An explicitly named
local proposed Direction is valid, but is never selected implicitly. When Directions are disabled,
omit `direction_id`.

Runtime validates every cited ID; its standardized public result is available under root `records/`.
Direction and Experiment citations may include Records without a Kernel binding, including Env
and self-contained probes. Review assesses the evidence within its actual scope: an Env Record
can establish environment facts, but does not establish a candidate's correctness or latency.
Correctness, latency, failure, operation and Kernel commit identity remain authoritative in those
records. Use Git to inspect source changes between cited commits. Summarize only the observations
needed for the decision in `interpretation`, rather than duplicating complete result tables.

`gateway_record_ids` and `references` are independent optional arrays. Either may be omitted or
empty. An Experiment without Gateway Records remains a useful research note, but cannot by itself
establish a measured Direction lifecycle conclusion. `submit-kernel` independently checks the
exact commit's full Evaluate and final-source policy; it does not require an Experiment.

An Experiment does not require an immediate Direction revision. Continue the highest-value branch;
revise the Direction when its assessment or remaining plan changes materially.

Revise a `local_writable` Experiment by resubmitting its required fields plus the `experiment_id` shown in
the greatest visible `research-library/experiments/<experiment_id>/vN/experiment.json`. Runtime
keeps that identity, increments `revision`, reruns independent Review, and writes a new immutable
version. Omitted optional evidence is
preserved; explicit empty arrays clear it. `direction_id` is immutable. Extend a `peer_read_only`
Experiment with a new local record rather than revising it. Prior submissions, Review receipts, conversations, and referenced bytes
remain available to the Supervisor's private audit even though the Agent sees only the latest revision.
The reviewer resumes that Experiment's latest valid Review Session; if native resume is unavailable,
Runtime falls back to one fresh Review Session without discarding the earlier audit records.

Every non-empty `references` entry must already exist and name a regular file or directory inside the Agent
workspace. Paths may not overlap or traverse links. Runtime snapshots at most 32 roots, 256 files,
16 MiB per file and 64 MiB total. It rejects the Experiment without mutation when a path is missing
or unsafe. On success, that version's `experiment.json` contains rewritten paths such as
`research-library/experiments/<experiment_id>/vN/references/scratch/ksplit-analysis.md`; these archived bytes are
independent of later edits to the originals and are versioned once in the Campaign Library. Every
lineage receives a visibility-filtered copy; synchronous peers see the revision at the next Epoch watermark and asynchronous peers
see it with the live revision.

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
capability should reactivate it. `abandon` and `complete` are terminal. Active Directions may cross
an Episode boundary; close them only when their own objective and remaining variants are genuinely
exhausted. The Research Catalog publishes every archived revision according to the configured
visibility mode; Epoch sharing still imports peer Git refs and seals synchronous watermarks.

## Continue a measured peer Kernel

Epoch sharing imports exact peer Git history through remote-tracking refs such as
`atrex/T2`. Keep or commit local work before switching. Inspect or continue the
peer source with normal Git:

```bash
git log --oneline --decorate atrex/T2
git diff HEAD atrex/T2
git switch -c explore/t2 atrex/T2
```

The checked-out commit is the Kernel ID. Switching source does not assert correctness, performance,
or acceptance; evaluate the resulting clean HEAD normally. Legacy evidence whose original Git object
is unavailable stays private to Supervisor recovery and is not projected as reusable Agent evidence.

## Review an exact Kernel commit

```json
{"kernel_commit_id":"0123456789abcdef0123456789abcdef01234567"}
```

```bash
python3 tools/sandbox.py --kind submit-kernel --request-file scratch/kernel-submit.json
```

The commit may be current HEAD or another commit reachable from local/imported Agent refs. Run a
standard full Evaluate for that exact commit first. The response contains the stable
`submission_id`, immutable `revision`, `accepted`, `evaluation_evidence`, per-policy status and
actionable diagnostics. Submission reuses the saved result and launches no new evaluation.

The first call creates `v1`. Rejection does not alter Git or roll back source. Repair and fully
evaluate a new commit, then revise the same Submission with:

```json
{
  "submission_id": "submission-0123456789abcdef0123456789abcdef",
  "kernel_commit_id": "fedcba9876543210fedcba9876543210fedcba98"
}
```

Each revision receives a fresh immutable `vN`; the independent Reviewer resumes its prior Session
while rechecking the complete current candidate. An accepted older revision remains accepted if a
later revision fails. Replaying the same commit with unchanged evidence and decision is idempotent.
Omit `submission_id` only to create a distinct Submission. The Supervisor retains every revision,
Review conversation and token receipt privately for audit. A normal Session exit ends the Episode.
