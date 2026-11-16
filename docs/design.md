# AKA Light architecture

AKA runs the Light workflow. `aka optimize` builds a public `WorkflowLaunch` from
`--problem`, `--workspace-base`, settings and runtime options. There is no second
Legacy workflow or `--` argument boundary. The `light` profile runs a native Agent;
`light-sandbox` uses Bubblewrap. Explicit SSH and DSH remain available.

## Public owners

| Owner | Responsibility |
|---|---|
| `aka.agent` | Provider Sessions, continuation, tool host, capture, process isolation |
| `aka.task` | Problem inputs, source contracts and candidate workspace initialization |
| `aka.evaluation` | Typed request preparation, Shape batching, interpretation, objective aggregation |
| `aka.execution` | Agate SDK or explicit SSH, transport recovery and job collection |
| `aka.experiment` | Durable execution attempts, measurements and source evidence |
| `aka.resources` | Runtime clients, Skills, prompt resources and plugin resources |
| `aka.workflow.light` | Campaign, lineage coordination, Research Library, submissions and policy |

Light orchestrates these public services; it does not carry a second Agent or Gateway
implementation. Agate control-plane operations use the SDK. Source archives upload through
SDK-issued upload slots. Explicit SSH uses the public SSH execution modules.

## Persistent Campaign

The Campaign root is not a Git repository. `campaign.json` contains launch identity and
source contract. Each `lineages/tNNNN/agent-workspace` contains a persistent Agent-owned Git
repository. Source initialization creates the common starting commit without evaluating it.
The Agent chooses when to measure. A segment ending does not reseed, checkout, rollback or
rewrite the Agent repository; the next segment resumes the Session and workspace.

All GPU measurements, failed attempts and source evidence enter the central
`measurements/` store. Lineages record visible IDs; Agent `records/` files are projections.
`research-library/` stores immutable Direction and Experiment versions, with attachments
and review evidence. `submissions/` stores versioned final-source reviews. Runtime requests
and Agent/reviewer conversation and token usage remain in Campaign-private audit storage.

The common initial Git root is used by every trajectory. At a publication, peer history is
imported as `refs/remotes/atrex/Tn`, pointing to that peer's published HEAD. It does not move
the consumer's HEAD. Synchronous research visibility stops at the last Epoch watermark;
asynchronous visibility follows publication updates. Revision remains owner-only.

## Optimization and submissions

A Direction describes a durable optimization program; Experiments describe specific tests,
measurements and interpretation. A supported Experiment does not complete its Direction.
Research annotations are optional. The default serial launch disables both modules.

`submit-kernel` reviews an exact Agent commit. It requires a passing complete full Evaluate
with parseable per-Shape performance, but no performance improvement. Fast skips independent
final-source review; Performance requires self-authored core computation; Ultra additionally
requires the explicitly configured DSL. Explicit ABBA remains an optional measurement.

Only accepted submissions enter lineage winners, global winners and Pareto views. Scores
follow the problem's declared objective after all Shape batches are aggregated. Runtime
refreshes local views after accepted submissions and segment completion; peer visibility
follows the configured publication schedule. There is no Framework Baseline, handoff,
episode-report, promotion or Supervisor-owned Git state machine.

## Measurement defaults

Defaults are 10 warmups, 20 benchmark iterations and a 60-second per-step timeout, configurable
with `--warmup-iters`, `--bench-iters`, and `--evaluation-step-timeout`. Explicit request values
are preserved or rejected with a concrete explanation, not silently clipped. Agent Sessions
and Reviews have no active-time deadline.

Outer batches contain at most 10 Shapes. Ray Data uses `shape_shards=1`. Ordinary Evaluate
retains the full reference and selects batch IDs. The removable ABBA compatibility adapter
constructs a complete batch-local reference and omits Shape selectors. Requests ask for
best-effort clock locking without prescribing MHz. Structured SDK rejection and candidate
failure diagnostics remain in both audit evidence and actionable Agent receipts.
