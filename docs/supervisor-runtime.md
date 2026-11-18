# Supervisor GPU/Wiki Runtime

Light's private Tool Runtime serves Agent GPU, Wiki, research-record, and exact-commit
submission requests.
It is created by the selected Workflow behind `aka optimize`; no separate server command
is needed. The agent receives thin `tools/sandbox.py` and `tools/plugin.py` clients,
while credentials, private evaluator inputs, source snapshots, and records stay in the Supervisor.

## Lifecycle and authority

The Workflow composes public Execution, Experiment and Acceptance services with its Research
and Wiki services. A Session capability authorizes the exact draft and allowed operations. Every
request checks authorization, workspace binding, deadlines, and revocation.

| Owner | Implementation |
| --- | --- |
| Scoped tool transport | `aka.agent.tools` (Unix socket ToolHost) |
| Runtime orchestration | `aka.workflow.light.orchestration.supervisor_runtime` |
| Agent clients and resources | `aka.resources.tools` |
| Measurement records and reuse | `aka.experiment` |
| Native Eval/ABBA and public results | `aka.evaluation` |
| Physical dispatch and recovery | `aka.execution` |
| Research and submit-kernel validation | `aka.workflow.light.host.journal` |
| Submission audit | `aka.workflow.light.long_horizon.submissions` |

Direction and Experiment validation uses an in-memory snapshot of the Campaign Research Store.
The Runtime does not materialize a compatibility Episode Journal or
`research-transaction.json`; central Research, Submission, and Measurement stores are the only
durable authorities.

Research records support the next implementation decision. Related candidate trials may be grouped
into one Experiment at a decision checkpoint, path switch, or sharing point; each Runtime measurement
and its source identity is still archived independently. An Experiment summarizes observations,
their measured scope, and the next action, with an unknown mechanism allowed. A supported Experiment
does not require closing or immediately revising its Direction.

Claim Review distinguishes evidence for trying a candidate from evidence for establishing a mechanism.
An uncertain explanation does not lower the verdict if the observations still justify the proposed
trial. Excluding high-upside paths, generalizing outside measured workloads, or declaring a Direction
exhausted requires stronger evidence. Partial receipts retain `evidence_eligible=false` for the full
claim but permit trials using `scope_safe_interpretation`; they do not demand resubmission. The advisory
`scientific_next_check` may state that no additional check is needed. Perform a check or revise only
when the uncertainty affects the next implementation choice or a lifecycle decision. Correctness,
measurement validity, source identity, and final-source submission checks remain unchanged.

The Runtime stages the complete candidate selected by CandidateWorkspace, restores trusted
evaluator inputs, and delegates through these services. It does not trust a candidate-provided
evaluator or allow the agent to choose Supervisor storage paths. Known jobs and exact measurements
are durable across requests; Session tokens are not durable recovery credentials.

The Campaign root has one authoritative `campaign.json`, serialized by `campaign.lock`. Schema v3
contains the immutable launch composition, the Campaign-wide SourceProvider file-system/contract
binding, the latest effective `runtime_configuration`, and—after initial source preparation—the V0 Agent commit
and canonical source-tree digest. Current Runtime writes neither a detached `source/` tree, a
detached `runtime-config.json`, nor a persistent `initial-source/` snapshot. Source preparation uses
temporary staging; the only durable complete V0 source is archived at
`measurements/kernels/<V0-commit>/kernel.py`. Every trajectory resolves the same binding from the
manifest, and recovery rejects a different source contract. A persistent Agent workspace is
recovered directly from its real Git repository and that binding, so it has no separate
`agent-workspace-state.json`. Lineage `state.json` records only current segment/controller state.
Runtime request receipts live under `runtime-audit/requests/`, while plugin calls live under
`runtime-audit/plugin-calls/`. Detached Runtime config and flat request receipts are read-only
compatibility inputs for older Campaigns; retired workspace-state files are ignored.

Pooled Campaign coordination is also flat: `coordinator.json` and `coordinator.lock` live at the
Campaign root, while each immutable synchronization event lives under
`publications/pNNNNNNNN/`. Its global metadata is `publication.json`; producer Git bundles and
receipts live in `sources/tNNNN/`, while consumer import receipts live in `imports/tNNNN.json`. The
coordinator checkpoint does not duplicate per-lineage HEAD, Session, Episode or winner facts;
those remain in `lineages/tNNNN/state.json`. The common Agent V0 Git root is archived once under
`bootstrap/agent-v0.{bundle,json}`. The Campaign has no separate `peer-git/` or `git-sharing/`
namespace.

## GPU operations

These commands run **inside an active agent draft**:

```bash
python3 tools/sandbox.py --kind evaluate --no-sync
python3 tools/sandbox.py --kind evaluate --mode correctness_only --no-sync
python3 tools/sandbox.py --kind profile --profile-shape-id SHAPE_ID --no-sync
python3 tools/sandbox.py --kind dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py
python3 tools/sandbox.py --kind command -- python3 -m scratch.experiment
python3 tools/sandbox.py --kind env
```

Use `python3 -m scratch.<module>` for Python Command probes below `scratch/`. Directly running
`python3 scratch/<module>.py` makes `scratch/` rather than the uploaded workspace root the first
module-search directory, so imports such as `import kernel` may fail even though `kernel.py` was
uploaded correctly.

| Operation | Meaning |
| --- | --- |
| Evaluate (`evaluate`) | Full or correctness-only evaluator request; native candidate checks use five cases per Shape |
| ABBA (`evaluate --baseline-commit ...`) | Native comparison against an exact reachable baseline Git commit; Runtime exports its source privately |
| Profile | Profiler evidence for a selected shape; not correctness or promotion evidence |
| Dev | Bounded exploratory commands; a source-bound record does not prove the command measured that source |
| Check / Disassemble | Supported source diagnostics, subject to layout and worker capabilities |
| Env | Full worker capability response, archived in the central Measurement Store without Kernel performance |

Profile uses `--profile-shape-id`, not Evaluate's `--shape-id`.
For hidden-shape production inputs, the Supervisor selects the requested opaque case without
publishing its dimensions. Native Eval does not silently fall back to Dev after failure.

Source-tree Evaluate/ABBA use native source archives; tree diagnostics use the supported
trusted diagnostic route. Native CUDA manifests support Evaluate, Profile, and ABBA, but
not Agate Check/Disassemble or SOL. The [SSH backend](ssh-backend.md) supports
native Evaluate/ABBA and worker-supported diagnostics, including native manifests.
See [source-layout compatibility](native-cuda.md) before selecting an operation.

During exploration the Agent may choose any reachable Git commit with `--baseline-commit` for a
research comparison. `submit-kernel` does not launch ABBA or end the Session; it reviews one exact
commit that already has a passing, complete full Evaluate. An existing Record is reused only when
every request identity matches.

## Public results and original GPU errors

GPU calls and projected `records/<gateway_record_id>.json` contain `operation`,
`status`, `result`, `gateway_record_id`, and `kernel_commit_id` when source-bound.
ABBA is an Evaluate result with `result.comparison` and `baseline_kernel_commit_id`.
Dev output lives inside `result.stdout`, `result.stderr`, and its exit code.

Inspect each projected result's status and correctness. Initial failed operations retain nonzero
CLI outcomes. Legacy marker streams are private executor evidence, not the Agent parsing contract.

Feedback retains compiler/runner errors, rank-local checks, output identity, numerical errors,
tolerances, and failing-case seeds. Private dimensions and credentials are redacted.
The terminal receives the complete canonical `operation/status/result` receipt with its Record path.
The same redacted result is saved in the Record, without serialized-length clipping or stream previews.
Env uses this contract and archives its complete observation; no GPU Job is created.

Evaluate includes `result.correctness_by_shape` alongside `performance_by_shape`.
For each requested Shape it retains Agate's correctness status and all `cases`,
including passing and failing `outputs`, `mutated_inputs`, and `unexpected_mutations`.
Each output keeps its original name, pass flag, error, numerical error metrics,
mismatch/element counts and `tolerance` (including null thresholds). Input values,
generators and case seeds are not copied into this field. Batched Shape IDs remain
unchanged. With repeated measurements, a Shape contains `runs: [...]`, retaining
each measurement's status and cases instead of overwriting earlier checks. These
details are also returned by correctness-only Evaluate; no performance is required.

Malformed Light requests return field-level repair guidance. Agate HTTP rejections and
job errors retain the original `error` value and HTTP status, including unfamiliar fields.
Runtime does not rename `reason` to `code`, rewrite classifications, or add `next_action`.
Only Light-origin failures (such as local request validation or its wait-budget expiry)
receive a Light-generated error. Compile, correctness and performance failures retain
Agate's diagnostics for the affected Shapes; they are not replaced by an infrastructure
error or a generic missing-evidence summary.
Missing private records, failed credentials/configuration, and uncertain remote outcomes
require operator repair; they are not reasons to alter the kernel or bypass validation.

## Wiki

Wiki is enabled by default and is independently configurable through the `wiki` provider
and its selected configuration. The bundled catalog and schemas are package-owned;
the local corpus comes from the selected checkout and is pinned on first query.
A replacement provider can supply a remote knowledge service.

Agent tools use the mounted plugin client:

```bash
python3 tools/plugin.py list
python3 tools/plugin.py call gpu-wiki.query --input scratch/wiki-request.json
```

A provider with a compatible CLI root also supports `wiki-query`, `wiki-search`,
and `wiki-hardware` through the sandbox client. Wiki query caching, concurrency,
timeouts, and failures are separate from GPU jobs.
See [Wiki providers](optional-light-modules.md#independent-wiki-execution).

## Deadlines, isolation and recovery

The selected Workflow, Evaluation and Execution configuration controls measurement repetitions,
queue/request timeouts and Wiki options. Runtime requests do not pause the Agent by default.
`agent.config.pause_agent_during_requests=false` keeps its process tree runnable during GPU,
Wiki and Review calls. The client still waits for its response, but the Agent can use its backend's
background-task facility to continue independent work. Request time counts toward any configured
Agent time limit; Agent and Review limits remain unset by default. Setting the option to `true`
restores process suspension and excludes admitted request waits from Agent active time.
The Session Prompt and Campaign configuration audit reflect the effective Agent setting.
Cancellation and capability revocation terminate owned work rather than granting more authority.

The default queue timeout is 60 seconds per queue stage. An omitted request timeout
uses the controller's derived queue/execution policy; an explicit positive value is the
whole-request limit. GPU allocation deadlines and Wiki execution budgets remain separate.
See [module options](optional-light-modules.md#research-and-parallel-search) and
[active-time accounting](optional-light-modules.md#agent-active-time-limit-and-blocking-runtime-calls).

All Agate control-plane calls use the public SDK. Accepted jobs are short-polled by ID within
the request budget. An uncertain submission without an accepted ID is not replayed unless
the transport provides a persistent idempotency guarantee.
The current [measurement recovery policy](measurement-records.md#failure-and-recovery-policy)
defines which failed jobs can be replaced and how completed ABBA batches are retained.

Native Light is cooperative same-UID execution. `--profile light-sandbox` enforces
the agent filesystem view on Linux; it does not change the GPU transport or add
network isolation. The controller and Core own service teardown.
See [workspace isolation](agent-workspace-isolation.md).

## Operator interfaces

The Agent uses `python3 tools/sandbox.py --kind KIND --help` for operation-specific arguments.
The client requires a live Session capability. Operator inspection uses the eight supported
observability scripts; direct Agate SDK probes are outside the Campaign measurement history.

For evidence and limitations, see [Light verification](optional-light-modules.md#research-and-parallel-search).
