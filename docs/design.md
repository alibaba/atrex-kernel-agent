# Architecture Design

AKA's default `light` profile composes a coding Agent, managed CandidateWorkspace,
private HTTP Tool Runtime, and Supervisor-owned audit workflow. The Agent owns implementation
history and optimization decisions; the Supervisor owns evaluator inputs, request authorization,
private evidence, source-policy review, budgets, and recovery.

[Launch and configuration](quickstart.md) · [Detailed module reference](optional-light-modules.md)

## Entry point and composition

`aka optimize --repo-root /path/to/aka -- ...` converts optimization arguments into a typed
Workflow request, then launches through Bootstrap and Core. A launch manifest selects a
composition and startup target. Core validates configuration, assembles dependencies, and
disposes owned providers when the invocation ends.

| Profile | Workflow | Agent |
| --- | --- | --- |
| `light` | Managed drafts and unified Episodes | Native with capture |
| `light-sandbox` | Same Light workflow | Bubblewrap with capture |
| `workflow` | Explicit Legacy Fast/Full/goal workflow | Legacy |
| `observed` / `sandbox` | Legacy workflow with an Agent replacement | Capture-enabled native / Bubblewrap |

Light requires the framework, legacy, and light distributions from a matching revision.
The legacy package also supplies shared business implementations used by Light; installing it
does not select the Legacy workflow. Recorded continuations restore their frozen selection
rather than adopting new-launch defaults.

## Project structure

```text
src/aka/
├── core/                  # Composition, dependency injection, lifecycle
├── contracts/             # Shared service interfaces
├── bootstrap/             # Launch selection and continuation identity
├── light/
│   ├── agent/             # Sessions, capture, isolation, active-time accounting
│   ├── workspace/         # Candidate layouts, drafts, snapshots, private Git
│   ├── runtime/           # Authenticated HTTP tools and thin agent clients
│   ├── evaluation/        # Native Eval/ABBA, profiling, result interpretation
│   ├── experiment/        # Measurement identity, records, deduplication
│   ├── execution/         # GPU dispatch, job checkpoints, recovery
│   ├── wiki/              # Knowledge-provider execution and caching
│   ├── acceptance/        # Correctness, numerical checks, verification, audits
│   └── workflow/          # Baselines, Episodes, reports, budgets, search
├── legacy/                # Shared providers and explicit Legacy workflow
├── observation/           # Workflow observation adapter
└── dashboard/             # Optional viewer
```

`packages/` owns distribution metadata. `plugins/`, `gpu-wiki/`, and selected
`reference-projects/` supply checkout resources. Runtime prompts, Skills, clients, and
profiling helpers belong to their packages and are published into campaign workspaces.

## Authority boundaries

| Owner | Authority |
| --- | --- |
| Problem / Source | Public problem contract, private evaluator roles, frozen initial source and runtime requirements |
| CandidateWorkspace | Candidate layout, allowed edits, persistent Agent workspace and audit snapshots |
| Agent | CLI sessions, process supervision, usage observations, optional filesystem isolation |
| Tool Runtime | Session capability, workspace authorization, request routing, deadlines and revocation |
| Evaluation / Experiment / Execution | Measurement plan and result semantics / durable facts and reuse / physical GPU jobs |
| Wiki provider | Knowledge source and public query operations; Runtime retains authorization and caching |
| Journal / Review | Optional research records and exact-commit source/DSL policy review |
| Workflow | Persistent Session, audit segments, budgets, recovery and optional peer sharing |

The Agent does not choose private storage paths or rewrite the trusted evaluator. It identifies
Kernel source with its own full Git commit IDs. Native mode separates work cooperatively under the same OS user.
Bubblewrap adds filesystem enforcement; it does not add a network allowlist or change
the remote GPU executor.

## Campaign lifecycle

1. **Freeze inputs.** Load the operator, initial source, evaluator contract, runtime pins,
   and selected candidate format. Shapes are exposed by default. Production uses a
   generalized public contract only when `--no-expose-evaluator-shapes` successfully stages
   `agent_problem.json`; the persisted visibility records that actual projection. If the requested
   hidden projection is unsupported, startup fails rather than falling back to exposed inputs.
2. **Establish V0.** Measure the selected initial source and create one canonical Agent V0 Git
   root. Pooled trajectories receive the same root commit; there is no separate Framework
   Baseline or accepted V1 descendant.
3. **Explore.** Start or resume the one persistent Agent Session in the trajectory's stable Git repository.
   Uncommitted work, local branches, tags, `scratch/`, tools, and Skills survive Episode
   boundaries. Runtime capabilities, public inputs, and shared evidence are refreshed for
   the new Episode without reconstructing the repository.
4. **Submit on demand.** `submit-kernel` verifies that one exact Agent commit already has a passing
   standard full Evaluate, then reviews its source/DSL and applicable Production policy. It does
   not launch another evaluation, mutate Agent Git, promote, roll back, or end the Session.
5. **Segment and continue.** An Episode is only an audit boundary created when the Agent chooses
   to stop a segment. The next segment resumes the same conversation, workspace and Git history.
6. **Archive.** Preserve Gateway Records, research claims, commit-review receipts and shared peer
   refs. Runtime does not choose or install an incumbent.

The default Journal configuration enables both Direction and Experiment annotations through
`light_journal.tool_modules`. They organize research but do not gate Session boundaries or select
a Kernel. Generated memory is optional and remains an audit aid rather than promotion state.

## Source and measurement identity

New tasks default to `source_tree`, with editable evaluator entry point `kernel.py` and optional
editable helpers below `impl/`.
Explicit alternatives are `single_file` and `cuda_manifest`. Layout implementations own
validation and packaging; Workflow uses their common artifact interface.

`kernel_commit_id` is the full Agent Git commit ID. Runtime accepts it only when the worktree is clean
and the commit contains the exact complete source snapshot, including headers and build settings.
A measurement identity additionally binds evaluator/input bytes, target,
options, transport, and repetition policy. Changing any required identity produces a new task.
Agent duplicates return the existing record reference; trusted acceptance can reuse compatible,
completed evidence. Unavailable or contradictory results cannot become cached verdicts.

See [candidate layouts](native-cuda.md), [measurement records](measurement-records.md),
and [evaluator outcomes](evaluator-outcomes/README.md).

## Measurement and commit review

Evaluate, Check, Profile, Dev, Command, Disassemble and ABBA produce measurements only when the
Agent requests them. Full-contract Evaluate remains the authoritative way to establish correctness;
targeted probes and Shape subsets remain diagnostic. Runtime projects the resulting Gateway Records
and derives `lineage_winner` plus `global_best` from accepted `submit-kernel` commits that have
passing full-contract Evaluate evidence. Pooled search broadcasts those observational winners with
their peer Git commits and Gateway Records, but never checks them out or uses them to control search.
Passing Evaluate results that were not accepted by `submit-kernel` remain available through
`records/`, but are excluded from the Research Board Pareto frontier and lineage/global winners.

`submit-kernel` is deliberately narrower: it archives and reviews one exact reachable Agent commit
for source layout, DSL and applicable Production dependency policy. It does not execute the Kernel,
certify correctness or performance, compare it with an incumbent, or modify Agent Git. The Legacy
workflow retains its separate acceptance and promotion semantics; see the [Legacy guide](legacy-workflow.md).

## Recovery and observability

The Supervisor persists job IDs, request identities, measurement batches, commit-review receipts,
private source archives and research audits. Agate SDK submissions use durable idempotency
keys. An uncertain non-idempotent submission without an ID returns a structured submission
timeout and is not blindly resubmitted.

Session throttling and output-limit recovery retain the Episode draft and observation history.
Agent active time excludes authenticated blocking Runtime requests. GPU, Wiki, and queue
deadlines remain independently enforced.

Capture is diagnostic; missing transcripts or usage counters remain explicit gaps.
Canonical state never depends on an Agent-writable projection.
See [Session observability](session-observability.md) and [Runtime Journal](runtime-journal.md).

## Verification

Package/composition, filesystem, HTTP, source sealing, and recovery checks are documented in
the [Light verification record](optional-light-modules.md#verification-scope).
[Native CUDA validation](native-cuda.md) records scoped live `<gpu_arch>` evidence.
[Evaluator-outcome checks](evaluator-outcomes/README.md) use local fault injection.
These are evidence for their documented paths, not a performance guarantee for every provider.

## Upgrade and rollback

Keep the installed packages, selected resources, and recorded configuration consistent.
Stop at an Episode boundary before changing deployment code. Restore the original matching
deployment to resume frozen continuations; use a new workspace for a different workflow,
source contract, or search topology. Do not rewrite recorded identities to force an upgrade.
Preserve private evidence and backups for operator repair.

See [installation and continuation](application-plugin-migration.md) and the
[v0.4.0 upgrade notes](releases/v0.4.0.md#upgrading-from-v030).
