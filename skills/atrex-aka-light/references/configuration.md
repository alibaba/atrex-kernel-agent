# AKA Light configuration

This page describes the 0.2.0 shared-module request contract. Inspect
`aka workflow --help` and the matching checkout before launch. All options use
one parser. The alias `aka optimize` accepts the same request fields.

## Launch and continuation

```bash
aka workflow --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox --problem /path/to/operator \
  --workspace-base /path/to/campaigns --platform L20D \
  --framework CuteDSL --agent-cli claude --optimization-mode production
```

`--repo-root` supplies matching checkout resources. `--profile` selects the
Workflow composition; omit it for native Light. `--patch` replaces selected
module configurations. There is no optimizer argument separator.

`--problem` selects a [supported operator](problem-inputs.md), not a candidate
directory. `--workspace-base` is the parent of new child campaigns;
`--workspace-suffix` adds a name suffix. Intentional changes to problem,
framework, mode, visibility, layout or composition require a new campaign.

Continuation uses the exact recorded `--workflow-request` value, with no new
request fields, Profile override or patch. Do not reconstruct a frozen request
from remembered flags.

## Public request fields

Native flags map directly to
[`WorkflowSettings` and `CampaignRuntimeSettings`](../../../src/aka/contracts/workflow.py).
Their dataclass defaults and the CLI help are authoritative. Alternatively pass
`--settings` and `--runtime` JSON objects; specifying the same field in JSON
and as a flag is rejected.

| Purpose | Current flags and behavior |
| --- | --- |
| Target | Required `--platform`; optional `--hardware` for a distinct remote executor alias. `--arch` is an intentional runtime architecture override. |
| Framework and policy | `--framework`, `--optimization-mode leaderboard\|production`, `--[no-]expose-evaluator-shapes`. Shapes default to exposed for new requests. |
| Agent | `--agent-cli claude\|qodercli\|codex\|pi\|dsh`; configure provider authentication separately. |
| Operator guidance | `--notes` injects constraints or known bottlenecks; keep credentials out. |
| Framework baseline | `--framework-baseline auto\|always\|never` controls V1 after measured V0. `--framework-baseline-timeout` bounds its session; default eight hours. |
| Independent baseline advice | Repeat `--framework-baseline-reviewers codex` or `qodercli`; these read-only reviewers do not edit or accept the candidate. |
| Repair/setup/review | `--handoff-resumes`, `--setup-timeout`, `--production-review-timeout` configure separate phases. Defaults are 2 repair turns, 7200 and 600 seconds. |
| Search | `--max-iters` is an inclusive canonical-version ceiling (20); `--token-budget` (0 disabled), `--target-util` (90), `--max-stall` (0 disabled), `--convert-after` (3) are independent admission/stop controls. |
| Promotion | `--verify-repeats` (2 per side, even 2–16 in Light), `--verify-run-timeout` (300 seconds), `--min-improvement-pct` (0, strict improvement). Complete ABBA and correctness evidence remain required. |
| Allocation | `--timeout` defaults to the derived enclosing Eval/ABBA allocation; an override must contain the complete schedule. It is separate from queue and stage limits. |

The old `--op-dir`, `--workspace`, `--sandbox-*`, `--v1-ask-*` and
`--verify-shape-batch-size` launch flags are retired. Transport options in
`tools/sandbox.py` belong to an Agent tool invocation, not this launch CLI.

Legacy-related fields still present in the shared DTO include `fast_episodes`,
`fast_trials`, `fast_episode_reviewers`, `full_episode_reviewers`,
`long_reviewer_session`, `review_trace_mode`, `gateway_candidate_recovery`,
`resume_framework_candidate` and `conversion_budget_action`. Parser
acceptance does not imply they configure Light's unified Episodes. Use the
selected Workflow implementation to confirm consumption.

## Measurement policy and wait limits

Evaluation's shared defaults are `bench_iters=20`, `warmup_iters=10`,
candidate timeout 600 seconds and performance timeout 300 seconds. The official
evaluator defines units: eager/Triton uses millisecond duration budgets;
CUDA-graph paths use replay counts. These values do not universally mean a
number of test/warmup executions. Explicit and historical frozen requests keep
their settings.

Light selects native ABBA, one Shape per batch, one measurement repetition
and bounded parallel batch workers (up to 16). The Runtime DTO's
`shape_batch_size` is not a substitute for the selected Evaluation policy.
[Module options](../../../docs/optional-light-modules.md#light-measurement-policy)
describe the defaults and their ownership.

`ATREX_SANDBOX_QUEUE_WAIT_GRACE` defaults to unset/`None`: no local client
wait deadline. An explicit non-negative integer limits client waiting to the
execution timeout plus the allowance and launcher cleanup slack; it never
extends the server Job execution budget. Workflow queue/request limits and
Agent active time are separate.

## Execution endpoints and SSH

The default Legacy Execution provider can select Agate or SSH. Select endpoints
through Execution configuration or its captured environment. Agate URL/profile
and SSH are mutually exclusive. The Gateway provider rejects SSH settings.

| Environment | Meaning |
| --- | --- |
| `ATREX_SANDBOX_URL`, `ATREX_SANDBOX_PROFILE` | Explicit Agate endpoint or profile; preserve authentication in the provider's configuration. |
| `ATREX_SANDBOX_SSH` | Non-interactive OpenSSH host or config alias. Requires one explicit framework. |
| `ATREX_SANDBOX_SSH_INIT` | Remote environment activation before jobs and health probes. |
| `ATREX_SANDBOX_SSH_GPU` | Physical GPU index assigned to this campaign; MIG selectors are unsupported. |
| `ATREX_SANDBOX_SSH_RUNTIME_BINDS` | JSON array of read-only remote directories, optionally `REMOTE=SANDBOX`. |
| `ATREX_SANDBOX_SSH_BWRAP` | Absolute remote Bubblewrap executable if absent from PATH. |
| `ATREX_SANDBOX_HEALTH_COMMAND` | Trusted readiness probe. |
| `ATREX_ENVIRONMENT_POLL_INTERVAL` | Recovery probe interval. |

The named Light profiles inherit these SSH configuration names from the Workflow
environment allowlist. See [SSH backend](../../../docs/ssh-backend.md) for GPU
isolation, explicit device maps and dependencies.

SSH supports typed requests, source archives, explicit input files and
same-allocation ABBA. It does not support closed `PreparedExecutionInputs`;
whole-tree diagnostics, private portable Profile and authorized Dev snapshot
conversion needing that transport reject before dispatch.

## Composition options

A patch supplies the whole configuration for the named module; omitted keys take
provider defaults. Retain `startup: true` for a Workflow launched by
`aka optimize`. Use current entries `workflow`, `agent`, `evaluation`,
`execution`, `resources`, `problem`, `source`, `candidate` and
`experiment`; the old `tool_runtime`, `light_journal` and `wiki` entries
are retired. The matching
[module guide](../../../docs/optional-light-modules.md) owns schemas and examples.

| Module | Relevant controls |
| --- | --- |
| `workflow` | Episode active allowance (four hours), optional numerical validation, Goal Episodes, generated memory, research records, inherited reusable state, layout/search topology, output-limit recovery (2), routed Wiki and queue/request limits. |
| `evaluation` | Shape batching, comparison method and measurement repetitions (1 or 3). |
| `agent` | `sandbox`, `bwrap_executable`, `proc_mode`, `timeout_seconds`, `session_name`, `state_directory`. Retired `capture`/`read_only_paths` keys are rejected. |
| `execution` | Backend profile and patches; physical dispatch and recovery configuration. |
| `resources` | Explicit plugin catalogs and contributed resources; retain selected bundles when replacing configuration. |

Pooled `workflow.search` controls epochs (10), trajectories (2),
episodes_per_trajectory (1) and max_parallel_episodes (2, at most trajectories).
Its schedule replaces the serial version budget. A shared token cap controls
admission; in-flight Episodes may overshoot it. Bootstrap/auxiliary usage is
excluded and missing provider usage cannot be fully enforced.

Completed Epoch winners publish verified Kernel bytes and performance facts.
Live drafts, tools, Skills, scratch, Conversation history and research state are
not merged between trajectories.

Routed Wiki is configured through `workflow.wiki`/`wiki_options`;
[WikiConfig](../../../src/aka/resources/wiki/config.py) owns model, effort,
settings_file, base_url, timeout, bridge_timeout, queue_timeout and concurrency.
Wiki timeouts do not change GPU or Agent budgets.

## Provider settings and Agent Runtime options

`ATREX_CODEX_SESSION_SETTINGS` and `ATREX_PI_SESSION_SETTINGS` are JSON
provider settings, falling back to `ATREX_SESSION_SETTINGS`. DSH uses its
configured headless profile, private Home and JSON events/checkpoints.
Native child-attempt accounting excludes seeded or resumed history.

The Agent's `tools/sandbox.py` discovers safe, operation-specific options
from the installed Agate client. Update the client and restart the Supervisor
to refresh that catalog. Credentials, endpoints, uploads, source selection,
timeouts and evaluator authority stay Supervisor-owned.

Unknown submission or an explicit replay prohibition requires recovery through
the original Job identity. Preserve its evidence and avoid blind resubmission.
