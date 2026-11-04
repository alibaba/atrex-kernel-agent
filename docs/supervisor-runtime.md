# Supervisor GPU/Wiki Runtime

Light's private Tool Runtime serves agent GPU, Wiki, record, and Journal requests.
It is created by the selected Workflow behind `aka optimize`; no separate server command
is needed. The agent receives thin `tools/sandbox.py` and `tools/plugin.py` clients,
while credentials, private evaluator inputs, source snapshots, and records stay in the Supervisor.

## Lifecycle and authority

The Workflow supplies scoped tool handlers to the shared Agent. Agent owns the
Unix-socket ToolHost and enforces the Session binding; there is no separate `tool_runtime`
composition entry or public HTTP server to launch.

| Owner | Implementation |
| --- | --- |
| Session tool transport | `aka.agent.tools.host`, `aka.agent.tools.client` |
| Workflow handlers | `aka.workflow.light.host` |
| Measurement identity and recovery | `aka.experiment` |
| Native Eval/ABBA preparation and interpretation | `aka.evaluation` |
| Physical dispatch and job queries | `aka.execution` |
| Wiki resources and configuration | `aka.resources.wiki` |
| Journal/report transaction | `aka.workflow.light.host.journal` |

The Runtime stages the complete candidate selected by CandidateWorkspace, restores trusted
evaluator inputs, and delegates through these services. It does not trust a candidate-provided
evaluator or allow the agent to choose Supervisor storage paths. Known jobs and exact measurements
are durable across requests; Session tokens are not durable recovery credentials.

## GPU operations

These commands run **inside an active agent draft**:

```bash
python3 tools/sandbox.py --kind run --no-sync
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
python3 tools/sandbox.py --kind profile --profile-shape-id SHAPE_ID --no-sync
python3 tools/sandbox.py --kind dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py
python3 tools/sandbox.py --kind env
```

| Operation | Meaning |
| --- | --- |
| Evaluate (`run`) | Full or correctness-only evaluator request; native candidate checks default to six cases per shape |
| ABBA (`run --baseline-path ...`) | Native comparison against an exact baseline source snapshot |
| Profile | Profiler evidence for a selected shape; not correctness or promotion evidence |
| Dev | Bounded exploratory commands; a source-bound record does not prove the command measured that source |
| Check / Disassemble | Supported source diagnostics, subject to layout and worker capabilities |
| Env | Worker capabilities; not a Kernel measurement |

Profile uses `--profile-shape-id`, not Evaluate's `--shape-id`.
For hidden-shape production inputs, the Supervisor selects the requested opaque case without
publishing its dimensions. Native Eval does not silently fall back to Dev after failure.

Source-tree Evaluate/ABBA use native source archives; tree diagnostics use the supported
trusted diagnostic route. Native CUDA manifests support Evaluate, Profile, and ABBA, but
not Check/Disassemble or SOL/SSH. Light ABBA requires Agate Atrex-Bench.
See [source-layout compatibility](native-cuda.md) before selecting an operation.

The Episode prompt contains the exact acceptance ABBA command and the matching
`kernel-read` command. Use those values rather than assuming that every candidate is
a Python file or that any exploratory comparison can be reused for promotion.

## Public results and original GPU errors

GPU calls and `record-read` return a single JSON object containing `operation`,
`status`, `result`, `gateway_record_id`, and `kernel_id` when source-bound.
ABBA is an Evaluate result with `result.comparison` and `baseline_kernel_id`.
Dev output lives inside `result.stdout`, `result.stderr`, and its exit code.

A successful historical read exits zero even if the saved evaluator rejected the candidate.
Inspect the result's status and correctness. Initial failed operations retain nonzero CLI
outcomes. Legacy marker streams are private executor evidence, not the agent parsing contract.

Feedback retains compiler/runner errors, rank-local checks, output identity, numerical errors,
tolerances, and failing-case seeds. Private dimensions and credentials are redacted.
The projected diagnostic response is bounded to 8 MiB of ASCII-escaped JSON, with explicit
truncation metadata. Complete record serialization has a separate 32 MiB limit.
See [diagnostic and cache contracts](evaluator-outcomes/README.md#diagnostic-size-and-cache-contract).

Malformed agent requests return field-level repair guidance. A confirmed candidate-source
rejection returns `candidate_source_rejected` with known-not-submitted evidence.
Missing private records, failed credentials/configuration, and uncertain remote outcomes
require operator repair; they are not reasons to alter the kernel or bypass validation.

## Wiki

GPU Wiki is selected as a Resources plugin in the Light profile. The optional routed
Wiki service uses `workflow.wiki` and `workflow.wiki_options`; see
[Wiki configuration](optional-light-modules.md#independent-wiki-execution).

Agent tools use the mounted plugin client:

```bash
python3 tools/plugin.py list
python3 tools/plugin.py call gpu-wiki.query --input scratch/wiki-request.json
```

A provider with a compatible CLI root also supports `wiki-query`, `wiki-search`,
and `wiki-hardware` through the sandbox client. Wiki query caching, concurrency,
timeouts, and failures are separate from GPU jobs.
See [Wiki providers](optional-light-modules.md#wiki-providers).

## Deadlines, isolation and recovery

Evaluation controls measurement repetitions. Light Workflow configuration controls
queue/request timeouts and optional Wiki settings. Blocking authenticated Runtime requests pause the
requesting agent tree. Episode deadlines default to four hours and include blocked requests as wall-clock time.
The handler continues running and resumes the agent before sending its response.
Cancellation and capability revocation terminate owned work rather than granting more authority.

The default queue timeout is 60 seconds per queue stage. An omitted request timeout
uses the controller's derived queue/execution policy; an explicit positive value is the
whole-request limit. GPU allocation deadlines and Wiki execution budgets remain separate.
See [module options](optional-light-modules.md#other-module-options) and
[current deadline accounting](optional-light-modules.md#agent-active-time-limit-and-blocking-runtime-calls).

Accepted jobs are polled by ID. Direct HTTP replay requires a verified stable-key capability and preserves that key.
An uncertain CLI submission without an accepted ID cannot be blindly repeated.
The current [measurement recovery policy](measurement-records.md#failure-and-recovery-policy)
defines which failed jobs can be replaced and how completed ABBA batches are retained.

Native Light is cooperative same-UID execution. `--profile light-sandbox` enforces
the agent filesystem view on Linux; it does not change the GPU transport or add
network isolation. The controller and Core own service teardown.
See [workspace isolation](agent-workspace-isolation.md).

## Operator interfaces

The installed host-side Gateway CLI is `aka execution`.
Its requests are operator-owned and do not automatically acquire a campaign's Session
capability or measurement history. Agent clients outside a live Session fail explicitly.
Use `--help` for host-side arguments rather than calling removed checkout scripts.

For evidence and limitations, see [Light verification](optional-light-modules.md#verification-scope).
