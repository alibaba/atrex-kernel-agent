## Direction records

You are responsible for both choices: which measured implementation to retain in your Git,
and which implementation or investigation to explore next. Runtime's accepted best is a measured
reference, not a command to keep modifying that candidate. After a meaningful attempt yields
results, a key expectation is missed, or you are about to switch paths, compare continuing an
untested branch, revisiting an earlier branch under a changed implementation or measurement,
and exploring another promising idea. Use the relevant accumulated evidence and its limits.
Choose for plausible full-workload benefit and what the trial can resolve. A `supported` review
label describes evidential support, not speed, importance, or search priority; do not rank by it.
A high-upside but uncertain rewrite may merit the smallest end-to-end trial without proving its
benefit first. Complexity or risk alone is not evidence that it cannot help.

Briefly state the next attempt, the evidence behind the choice, what remains untested, and what
the attempt will distinguish in your normal next-action explanation. Skip alternatives that are
irrelevant or already resolved; reuse the assessment until evidence changes. Do not perform this
comparison after every Command. It requires no separate selector, report, Runtime call, Review,
or research lifecycle transition. New probes or implementations of the same high-level idea remain within the same Direction.


Read the relevant visible Directions, including deferred or abandoned related work, using their analysis, remaining plan, and cited measurements. No Experiment is required in this mode.

When choosing the next implementation, revisit the visible evidence for the same bottleneck, workload regime, or architecture objective, including related attempts under other names. Use the current visible versions; consult older versions or bounded Gateway Record fields only to resolve a relevant ambiguity. In the normal next-action explanation, briefly connect: **Tried** (implementation and measured scope), **Ruled out** (only what those measurements actually exclude), and **Still untested** (remaining variants, regimes, or combinations). Cite the relevant record IDs and explain how the next candidate differs from earlier attempts. Reuse this assessment while it remains current; do not reread the library after every tool call.

A failed candidate does not rule out its whole strategy. Separate correctness or infrastructure failures from measured performance regressions, and distinguish cost-model estimates from end-to-end measurements. Missing evidence means untested, not disproven. Compare promising untested branches before repeating a measured variant; a repeat should address a changed implementation, setting, or unresolved measurement. This is a decision aid, not a new report: no extra record, revision, Review, or lifecycle transition is required merely to summarize prior evidence.

Direction recording is enabled; Experiment recording is disabled. Read `PROGRESS.md` and only relevant `research-library/directions/<direction_id>/vN/direction.json` files; the greatest visible `vN` is the current version. Reuse a Direction while its objective and applicability are unchanged, and check for duplicates before proposing another. A Direction is a durable optimization program around a bottleneck, workload regime, or architecture objective and should span multiple candidates or probes. A single implementation, parameter choice, capability probe, or mechanism test remains a bounded step inside an existing Direction; do not create a Direction for it. Runtime maintenance and infrastructure work are not Directions. Genealogy is descriptive and does not inherit source, evidence, status, or priority.

Keep the same Direction while the core optimization idea and applicable problem remain unchanged. Its revisions update the current assessment and remaining branches. Create a new Direction when the main optimization principle or target problem changes, not merely when a new implementation is tried. A successful Experiment adds a viable branch; it does not by itself complete the Direction.

Establish or reuse a Direction once the high-level idea is clear. Exploratory probes and a minimal implementation may precede proposal or start; research bookkeeping does not gate implementation or measurement. Update the Direction only when its high-level assessment, remaining branches, or actual work state changes. Lifecycle is explicit: `propose` creates `proposed`; call `start` with `direction_id` to enter `in_progress` (`analysis` is optional; omitting it preserves the current assessment); only an `in_progress` Direction may `complete`, `abandon`, `defer`, or `block`. Restart `deferred` or `blocked` work with `start`. `completed` and `abandoned` are terminal, so continued work needs a derived Direction.

`start`, `defer`, and `block` allow `partially_supported`. Only `complete` and `abandon` require `supported` Review of that terminal decision; otherwise the Direction remains `in_progress` and the Review is returned for correction.

Multiple local Directions may be `in_progress`; keep that set intentional and update each Direction by ID. The most recently started active Direction is the focus; queue reordering does not change it. Choose executable work from the Local work queue; paused Directions are not next work until explicitly restarted. The Shared direction pool is discovery-only: absorb compatible peer ideas into one local derivative citing every source ID. Experiment recording is disabled, so omit `supporting_experiment_ids`; closure evidence uses `gateway_record_ids`. Follow the installed Skill for exact mutation fields.

Continue related candidate trials without a revision after each measurement. Revise when the assessment or remaining plan changes materially, recording observations, their measured scope, and the next action. A mechanism may remain unknown when the evidence still justifies the next trial. Stronger support is needed to exclude a high-upside path, generalize beyond measured workloads, or declare the program exhausted.

Records cited by a Direction currently visible to this lineage and passing standard full-Evaluate Records are visible here; accepted submissions and winners therefore carry their complete Evaluate evidence. Other local Records remain private.

`propose` runs independent admission Review and writes immutable `v1`; no prior measurement is required. Every `revise` and closure is reviewed. `start` changes lifecycle without Review and retains the prior review quality. Each archived update writes the next immutable `vN`. Accepted versions enter the Campaign Research Library and are copied into each workspace according to its visibility boundary. Synchronous peers see the version frozen at the last Epoch boundary; asynchronous peers see the latest version immediately. Peer identities are read-only and must be continued through a new local derived Direction. Review labels evidence strength, not visibility. Partial records may guide further trials within `scope_safe_interpretation`; unresolved claims remain uncertain. `review_unavailable` records remain visible; Runtime retries an unavailable Reviewer once. Closure Review assesses whether the lifecycle decision is justified. Unsupported proposals, revisions, and closures are rejected before mutation and have no `archive_path`. Runtime returns the full Review once, then projects compact fields; full receipts and conversations remain Supervisor-private. `scientific_next_check` is advisory and may say no additional check is needed. Perform it or revise only when the uncertainty affects the next implementation choice or lifecycle decision; do not repeat submissions merely to obtain `supported`. Restart paused Directions first and continue terminal work through a derived Direction.

Keep implementation variants, parameter choices, dataflow trials and probes under the current Direction while the high-level idea stays the same; update its plan or assessment when needed. Create a new Direction when the main optimization principle or target problem changes. Record a distinct promising idea for later as proposed (or defer it through the supported lifecycle operation), rather than leaving it only in conversation. Do not start every future idea or create a Direction for each next action. Recording is not a prerequisite for further implementation or evaluation.
