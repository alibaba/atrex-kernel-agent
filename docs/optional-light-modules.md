# Light modules and profiles

This guide describes the shared-module implementation in this checkout (package metadata
0.2.0). Install all matching packages using [Quick Start](quickstart.md). The earlier
private Light modules and their configuration tokens are no longer supported aliases.

## Select a composition

`aka optimize` and `aka workflow` default to `light`. Use `--profile workflow` for Legacy
or `--profile light-sandbox` for coordinator Bubblewrap isolation. All options share one
parser; no `--` separates launch options from request fields:

```bash
aka optimize --repo-root "$PWD" --profile light \
  --problem /path/to/atrex-bench/operator --platform '<gpu_arch>' \
  --framework Cuda --workspace-base /path/to/new-runs
```

Configure endpoint/authentication first as described in Quick Start. The example's
workspace path is a campaign parent, not a pre-existing candidate directory. Choose
Legacy for SOL/SSH campaigns requiring its verification path.

## Ownership and behavior

| Composition entry | Implementation | Responsibility |
| --- | --- | --- |
| `agent` | `aka.agent.plugin` | Coding sessions, isolation, scoped tool transport and observations |
| `resources` | `aka.resources.plugin` | Explicit plugin catalogs, Skills and contributed resources |
| `problem` | `aka.task.problem.plugin` | Trusted problem inputs and evaluator contract |
| `source` | `aka.task.source.plugin` | Initial candidate source |
| `candidate` | `aka.task.candidate_workspace.plugin` | Drafts, source transactions and sealed candidates |
| `evaluation` | `aka.evaluation.plugin` | Test/benchmark preparation, backend commands and result interpretation |
| `experiment` | `aka.experiment.plugin` | Measurement identity, execution coordination, records and safe recovery |
| `execution` | `aka.execution.plugin` | Physical dispatch, jobs and backend capabilities |
| `workflow` | `aka.workflow.light.plugin` | Episodes, reports, acceptance and promotion orchestration |

The Agent submits a candidate through a scoped tool. Experiment asks Evaluation to prepare
its execution request, dispatches through Execution and returns the interpreted evidence.
The native Atrex-Bench harness is selected/prepared by Evaluation. It is not an Agent-side
fallback or a second independent Light evaluator.

There are no current composition entries named `light_journal`, `light_evaluation`,
`light_experiments`, `light_acceptance`, `tool_runtime`, or `wiki`. Use the owning module's
configuration below. Changing a provider requires preserving its public contract and
frozen continuation identity; renaming an old private import is not an upgrade strategy.

## Options

### Other module options

A patch replaces an entry's whole configuration. Omitted keys take provider defaults;
retain explicit profile choices that you need. In particular, keep `startup: true` in a
Workflow patch used by `aka optimize`.

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {
      "startup": true,
      "numerical_validation": true,
      "generate_memory": true,
      "research_records": true,
      "file_system": "source_tree"
    }},
    {"id": "evaluation", "config": {
      "shape_batch_size": 1,
      "comparison_method": "native-abba",
      "measurement_repetitions": 1
    }}
  ]
}
```

Save as `/path/to/options.json` and add `--patch /path/to/options.json` to the launch.
Supplemental numerical validation, generated memory and research annotations default to
false. Standard correctness, automatic measurement records and report/promotion checks
remain required. `goal_episodes`, `inherit_state`, `search`, `wiki`, `wiki_options`,
`request_timeout`, `queue_timeout`, and `output_limit_recovery_retries` are also Light
Workflow options; see the [schema and defaults](../src/aka/workflow/light/plugin.py).

The shared Agent accepts `sandbox`, `bwrap_executable`, `proc_mode`, `timeout_seconds`,
`session_name`, and `state_directory`. It rejects the retired `capture` and
`read_only_paths` keys. See [isolation](agent-workspace-isolation.md) and
[current observations](session-observability.md).

### Independent Wiki execution

The Light Resources profile selects `plugins/gpu-wiki` from `--repo-root`. Agent-visible
catalog calls use the shared Resources implementation:

```bash
python -m aka.resources.cli call gpu-wiki.query --input wiki_request.json
```

Run this inside the prepared workspace with its resource lock. For a standalone catalog,
pass an explicit `--plugin-dir` and `--workspace`; see [local plugins](plugins.md).

Light's optional routed Wiki service is controlled by `workflow.wiki` and
`workflow.wiki_options`, not the removed `tool_runtime` or `wiki` composition entries.
Its options are `model`, `effort`, `settings_file`, `base_url`, `timeout`, `bridge_timeout`,
`queue_timeout`, and `concurrency`. Defaults and validation are maintained in
[WikiConfig](../src/aka/resources/wiki/config.py). For example:

```json
{"api_version":1,"patches":[{"id":"workflow","config":{
  "startup":true,"wiki":true,"wiki_options":{"timeout":120,"bridge_timeout":60}
}}]}
```

Wiki timeouts do not redefine GPU measurement or Agent invocation timeouts.

### Wiki providers

Select external tool catalogs through `resources.plugins` or `resources.plugin_dir`.
The selected plugin must implement the [local manifest contract](plugins.md#manifest-contract).
Resource selection and the optional Workflow Wiki route are different settings. A
Resources patch must retain desired bundles, such as `measurement-session` and
`research-records`, if it replaces the Light profile's Resources configuration.

### Light measurement policy

Ordinary evaluation defaults are shared across methods and entry points: `bench_iters=20`,
`warmup_iters=10`, candidate timeout 600 seconds, and performance timeout 300 seconds.
Explicit request values and frozen historical requests retain their values. These inner
limits are distinct from queue wait, the enclosing allocation and Workflow stage budgets.
The official evaluator determines the units: eager/Triton timing uses duration budgets
in milliseconds; CUDA-graph timing uses replay counts. They are not universally 20 tests
and 10 warmup executions.

Light selects `native-abba`, one Shape per batch and one measurement repetition; the
Legacy composition selects `legacy-abba` and four Shapes per batch. These express
comparison/batching policy, not separate bench/warmup defaults. See
[Evaluation configuration](../src/aka/contracts/evaluation.py).

### Agent active-time limit and blocking Runtime calls

The heading is retained for old links; the current limit is wall-clock time. Light defaults to a four-hour Episode deadline, configurable through `episode_timeout_seconds`.
Blocked tools count against that deadline; restart and repair retain its original expiry.
Agent invocations and framework baseline default to eight hours each. These are separate
scopes, not a Campaign-wide eight-hour limit. `agent.timeout_seconds`, GPU execution,
queue wait, Wiki and Workflow stage limits apply at their own scopes.

### Optional Epoch / Trajectory search

Set `workflow.search` to an object with `epochs`, `trajectories`,
`episodes_per_trajectory`, and `max_parallel_episodes`. An empty object uses serial
Episodes. Use a fresh workspace when changing search strategy. The accepted bounds
are in the Workflow schema; controller search does not create another Evaluation module.

### Single-file and contracted source-tree Kernels

Set `workflow.file_system` to `single_file`, `source_tree`, or `cuda_manifest`.
The default is `source_tree`. The Candidate plugin no longer accepts a `file_system`
patch, and old `candidate.source_tree`/`workflow.source_tree` import patches are not
accepted. See [native CUDA and source layouts](native-cuda.md).

### Reusable workspace

`workflow.inherit_state` defaults to true. Reusable Skills/tools are separate from candidate
promotion, while scratch is reset for a new Episode. Existing Episode continuation retains
its draft. Preserve source layout, resource identities and the original workflow on resume.

### Bubblewrap in restricted containers

Choose `agent.sandbox=bwrap` and `agent.proc_mode=projected` explicitly on supported Linux
hosts. Compiler/runtime validation is part of Agent configuration validation. It does not
prove namespace permissions or GPU availability. See [isolation](agent-workspace-isolation.md).

## Infrastructure recovery

Common Experiment retries allow two extra attempts after 5/10 seconds when execution
facts prove a transient failure happened before submission. Recorded steps retain their
120-second recovery admission window and counts. Known jobs are queried; uncertain jobs
are not blindly replaced. Remeasurement and backend changes remain explicit opt-in.
See [failure and recovery policy](measurement-records.md#failure-and-recovery-policy).

The retired `reliable` execution selection is rejected with migration guidance. Choose
an explicitly supported Execution provider; it does not silently select another transport.
Old retry-ledger schemas are not migrated automatically. See
[recovery compatibility](application-plugin-migration.md#recovery-compatibility).

## Verification scope

These are current configuration and command instructions. CLI/parser and schema checks
are not an end-to-end GPU or coding-agent qualification. Earlier Linux/GPU campaigns,
security limitations, old default values, private-module design details and rollout
procedures retain their original scope in the
[pre-migration guide](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md).
Its old flags, import paths, provider tokens and active-time accounting are historical;
do not copy them into a new 0.2.0 launch. Matching old installations are required to
reproduce those earlier runs.

## Historical topic index

The links below preserve the earlier documentation text and saved section links.
`1abdf3f3` identifies the documentation snapshot, not a compatible runtime for the
removed commands. Historical behavior predates the shared-module refactor; use the
implementation revision named in each validation record when reproducing evidence.

<a id="shared-production-policy"></a>
- [Shared production policy](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#shared-production-policy) (historical).

<a id="projected-proc-compiler-preflight"></a>
- [Projected-proc compiler preflight](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#projected-proc-compiler-preflight) (historical).

<a id="output-limit-recovery-and-response-accounting"></a>
- [Output-limit recovery and response accounting](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#output-limit-recovery-and-response-accounting) (historical).

<a id="optional-memory-summaries"></a>
- [Optional memory summaries](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#optional-memory-summaries) (historical).

<a id="historical-conversations"></a>
- [Historical Conversations](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#historical-conversations) (historical).

<a id="optional-research-recording"></a>
- [Optional research recording](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#optional-research-recording) (historical).

<a id="agent-measurement-results"></a>
- [Agent measurement results](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#agent-measurement-results) (historical).

<a id="agent-error-feedback"></a>
- [Agent error feedback](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#agent-error-feedback) (historical).

<a id="upgrade-of-source-level-integrations"></a>
- [Upgrade of source-level integrations](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#upgrade-of-source-level-integrations) (historical).

<a id="shared-implementations-and-retained-light-policy"></a>
- [Shared implementations and retained Light policy](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#shared-implementations-and-retained-light-policy) (historical).

<a id="infrastructure-recovery-and-agent-feedback"></a>
- [Infrastructure recovery and Agent feedback](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#infrastructure-recovery-and-agent-feedback) (historical).

<a id="security-migration-and-rollback"></a>
- [Security, migration and rollback](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/optional-light-modules.md#security-migration-and-rollback) (historical).
