# Direction and Experiment requests

All commands use the Session-scoped Runtime client. Put mutation requests below `scratch/`.
Requests must fit the documented 2 MiB envelope.

## Inspect progress and research files

Read `PROGRESS.md` and relevant `records/<gateway_record_id>.json` files. Enabled research modules
project immutable versions under `research-library/`; the highest visible `vN` is current.
Use file reads for inspection and Runtime mutations for updates. Each version retains its
`owner_lineage`, access, claim, and compact Review. Full receipts remain Supervisor-private.
Read `frontier.json` only when the complete accepted-submission Pareto is needed.

## Cross-trajectory measurement visibility

Each lineage sees Records cited by the Direction/Experiment revisions currently visible in that
lineage's projection, plus passing standard full Evaluates. Accepted submissions and winners therefore
publish their complete Evaluate evidence automatically. Review quality controls whether a claim is
trusted, not whether its cited Record can be inspected. There is no Agent operation for publishing
arbitrary local evidence. Synchronous search uses the lineage's current Epoch watermark; asynchronous
search uses its live visible revision set. Every immutable Record remains stored once in the Campaign Store.

Claim Review is disabled by default unless explicitly enabled for the Campaign.
When disabled, mutations that normally request it return
`archive_status: archived`, `review_skipped: true`, `claim_review_quality: review_not_requested`,
and `evidence_eligible: false`. This means recorded without independent review, not supported.
Recording, revisions, lifecycle operations and sharing remain available; attachment validation still applies.

Claim-bearing mutations return `review` once. Its workflow fields are deliberately distinct:

- `archive_status`: whether Runtime stored the submitted claim;
- `claim_review_quality`: `supported`, `partially_supported`, `unsupported`, or `review_unavailable`;
- `evidence_eligible`: whether the claim has established evidential support; it does not control visibility;
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

Establish or reuse a Direction once the high-level idea is clear. Exploratory probes and a minimal
implementation may precede proposal or start. Research bookkeeping does not gate implementation,
measurement, or submission. Update the Direction only when its high-level assessment, remaining
branches, or actual work state changes.

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
`reimplementation`, or `contradiction`. Keep the same Direction while the core optimization idea and applicable problem remain unchanged. Its revisions update the current assessment and remaining branches. Create a new Direction when the main optimization principle or target problem changes, not merely when a new implementation is tried. A successful Experiment adds a viable branch; it does not by itself complete the Direction. Propose a
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
when exactly one local Direction is active; with zero or multiple active Directions, it remains unassociated.
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

A Direction answers why a high-level optimization idea is worth exploring. Its Experiments
record concrete investigations, not just complete implementations. Illustrative example only:

```text
Direction: reduce output-write inefficiency through coalescing
  Experiment A: profile the current output stores to localize the suspected bottleneck
  Experiment B: microbenchmark two store layouts to test their throughput difference
  Experiment C: implement a coalesced output path and measure end-to-end latency
  Experiment D: test correctness on boundary shapes and non-contiguous inputs
```

Repeating B with more samples or repairing its timing script revises B. A materially different
probe method creates a new Experiment. Fixing a bug within C's same approach or sweeping its tile
sizes can revise C; replacing its key dataflow creates a new Experiment. D is a separate correctness
question even if it tests the same commit as C. No minimum number of Experiments is required.
All four inform the same Direction; a positive microbenchmark does not prove end-to-end speedup,
and a successful implementation does not automatically complete the Direction. Update the
Direction's assessment or remaining branches when these findings materially change them.

Record an Experiment when a concrete implementation, probe, or test has enough evidence to
assess the attempt. Combine its related debugging, parameter trials, and measurements. As you
conclude an attempt and prepare to switch to a materially different approach, promptly save a
short record of the method, expectations, observations, remaining uncertainty, and next choice;
do not defer that record until the next attempt produces results. Recording is not a prerequisite
for continuing implementation, measurement, or Kernel submission. Do not add verification solely
to finish the record, wait for mechanism proof, or revise a Direction merely because an Experiment
was recorded.

An Experiment is one concrete investigative attempt: a candidate implementation, diagnostic probe, microbenchmark, correctness test, profiling experiment, or controlled comparison. It records what was tried and what the observations establish. Keep its identity while the specific question and experimental approach remain unchanged. Use revision for repeat measurements, corrections, explanation, or repairs within that attempt. Create a new Experiment when the experimental question or method, implementation strategy, key dataflow, or combination changes materially, even within the same Direction. Several commits, parameter trials, and diagnostic Records may belong to one attempt; one commit or tool call does not require one Experiment. A complete Kernel or full Evaluate is not required. Preserve earlier attempts rather than replacing them with the latest conclusion. Not every Command or Evaluate needs an Experiment.
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

### Preserve predictions and observed outcomes

Use optional `implementation` to describe the concrete implementation, probe, or test method
and its difference from prior attempts. Use optional `expectations` for scoped comparisons:
each row contains non-empty `scope`, `prediction`, `observation`, and `assessment`
(`met`, `missed`, or `unresolved`). Keep `intent` for the experimental question and
`interpretation` for remaining hypotheses and the next action.
Distinguish expectations that were met, missed, or remain unresolved; mixed Shape outcomes stay
separate. Keep observations separate from causal explanations: an observed regression can be
established while its cause remains unknown. Label suspected causes as hypotheses, not findings.
Compilation or infrastructure failure leaves a performance prediction unresolved, not missed.
Do not run extra mechanism probes merely to fill in an explanation. Revisions retain the original
prediction and earlier outcome; append updated evidence and assessment rather than rewriting the
prediction to match the result. Cite the relevant Records or archived notes. For verifiable advance predictions, pass `--predictions '[{"scope":"long sequences","prediction":"Lower latency"}]'` to Evaluate or Command (before the command `--` separator). The optional request field `predictions` is a JSON array of `{"scope":"long sequences","prediction":"Lower latency"}`. Runtime freezes it before new execution and returns `frozen_predictions` with the Gateway Record. Cite the Record in `gateway_record_ids`; Runtime automatically brings in all its frozen predictions. Do not supply an index or repeat the forecast. Missing observations remain `unresolved`; explain the actual outcome in `interpretation`. Optional expectation rows with unambiguous matching scope/prediction may update observations and assessments. Frozen forecasts are retained on revision; Runtime never infers results or matches ambiguous identical forecasts across Records. Reused Evaluate results retain only their original forecasts; this call's new predictions are not registered. Local notes alone cannot establish execution order. Unbound expectations are marked `retrospective`. An exploratory
test may have no directional prediction. These optional fields require no extra Runtime call and add no Review or lifecycle gate.
Omitting them on revision retains the previous values. An explicit `expectations` array replaces
the rows in the new version (`[]` clears only unbound rows); frozen predictions must remain, and earlier immutable versions stay archived.
Preserve the original predictions when updating observations and assessments. Source identity
and measured performance remain in Gateway Records; do not duplicate their complete tables.

Illustrative structured request (not measured results from a real run):

```json
{
  "name": "Parallel partition with independent input reads",
  "intent": "Determine where parallel partitioning improves latency",
  "gateway_record_ids": [
    "gateway-<actual-record-id>"
  ],
  "references": [
    "scratch/partition-plan-before-eval.md"
  ],
  "interpretation": "Retain the parallel partitioning idea; compare an input-reuse variant and workload dispatch. Extra traffic is only a suspected cause. These results do not exclude other implementations.",
  "implementation": "Partition work while each partition reads its own copy of shared inputs",
  "expectations": [
    {
      "scope": "Long sequences",
      "prediction": "Lower latency (recorded in the archived pre-measurement plan)",
      "observation": "Latency improved on the measured long sequences",
      "assessment": "met"
    },
    {
      "scope": "Medium sequences",
      "prediction": "Added traffic will not erase the gain (recorded in the same plan)",
      "observation": "Latency regressed; its cause has not been established",
      "assessment": "missed"
    },
    {
      "scope": "Short sequences",
      "prediction": "No latency regression (recorded in the same plan)",
      "observation": "Latency regressed on the measured short sequences",
      "assessment": "missed"
    }
  ]
}
```

Use actual evidence, scopes, and files; do not invent a pre-measurement plan. If only latency was
measured, do not present an explanation about bandwidth, occupancy, or traffic as proven. Keep
unknown causes explicit and continue with the next informative implementation or test.

A hardware capability probe also merits an Experiment when its result selects an implementation
path. For example, after checking an MMA instruction's output and throughput, record the bounded
observation once the probe supports a decision; starting the Kernel need not wait for recording; no full Evaluate is needed for this probe record:

```json
{
  "name": "FP8 MMA probe selects the next implementation path",
  "intent": "Determine whether FP8 MMA is usable for the next GEMM candidate",
  "gateway_record_ids": ["gateway-<actual-probe-record-id>"],
  "interpretation": "The tested FP8 MMA configuration passed the probe's output check and ran faster than its BF16 counterpart. Next implement an FP8 MMA candidate. This establishes only the tested probe behavior, not full-Kernel correctness, workload latency, or an architectural peak.",
  "references": ["scratch/probe_mma.py"]
}
```

Use the actual Record ID and observations. A follow-up probe or repair within this implementation attempt may revise its evidence or conclusion;
it need not delay recording the decision already made.

<!-- directions:start -->
When Directions are enabled, Runtime infers `direction_id` only when exactly one local Direction is
active. With zero or multiple active Directions, omission records an unassociated Experiment. An explicitly named
local proposed Direction is valid, but is never selected implicitly. When Directions are disabled,
omit `direction_id`.

<!-- directions:end -->

Runtime validates every cited ID; its standardized public result is available under root `records/`.
Direction and Experiment citations may include Records without a Kernel binding, including Env
and self-contained probes. Review assesses the evidence within its actual scope: an Env Record
can establish environment facts, but does not establish a candidate's correctness or latency.
Correctness, latency, failure, operation and Kernel commit identity remain authoritative in those
records. Use Git to inspect source changes between cited commits. Summarize only the observations
needed for the decision in `interpretation`, rather than duplicating complete result tables.

`gateway_record_ids` and `references` are independent optional arrays. Either may be omitted or
empty. An Experiment without Gateway Records remains a useful research note, but cannot by itself
establish measured correctness or performance. `submit-kernel` independently checks the
exact commit's full Evaluate and final-source policy; it does not require an Experiment.

<!-- directions:start -->
An Experiment does not require an immediate Direction revision. Continue the highest-value branch;
revise the Direction when its assessment or remaining plan changes materially.
<!-- directions:end -->

Revise a `local_writable` Experiment by resubmitting its required fields plus the `experiment_id` shown in
the greatest visible `research-library/experiments/<experiment_id>/vN/experiment.json`. Runtime
keeps that identity, increments `revision`, reruns independent Review, and writes a new immutable
version. Omitted optional evidence is
preserved; explicit empty arrays clear it.
<!-- directions:start -->
A revision may add a local `direction_id` to an unassociated Experiment; an existing association is immutable.
<!-- directions:end -->
Extend a `peer_read_only`
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

<!-- experiments:start -->
Use optional Experiment `next_step` to record the chosen next action separately from the interpretation:
```json
"next_step": {
  "action": "Implement V-split while preserving input reuse",
  "reason": "The previous trial repeated input reads; partitioning with reuse remains untested"
}
```
`action` says what to do; `reason` explains why this is the next useful attempt. Both are nonempty strings when provided. Omit `next_step` on revision to preserve it; use `null` to clear an obsolete plan. This is an editable plan, not a new tool, mandatory report, Review requirement, or execution gate. A concrete next attempt normally stays under the existing Direction; create another Direction only when the high-level idea changes. The Direction attempt view displays action and reason separately; Runtime does not infer them from `interpretation` or execute them automatically.
<!-- experiments:end -->

<!-- directions:start -->
### Where to record the next step

- Another implementation, parameter choice, dataflow variant, or probe under the same high-level idea: keep the current Direction. Once the attempt has useful results, record an Experiment; do not create a Direction for every next action.
- Additional measurements or a correction to the same attempt: revise its Experiment. A materially different approach is a new Experiment under the same Direction when the high-level idea remains unchanged.
- A different primary optimization principle or target problem: create a new Direction.
- A distinct high-level idea worth exploring later: record a proposed Direction (or defer it using the supported lifecycle operation), so it does not survive only in conversation or another record's future-work text. Do not start it merely to document it.

When Experiment recording is enabled, use `next_step.action` and `next_step.reason` for the concrete selected action and rationale. Keep remaining same-idea branches in the Direction's plan or assessment. These are recording guidelines, not prerequisites for implementation or evaluation; do not turn each parameter trial into a separate record.
<!-- directions:end -->
