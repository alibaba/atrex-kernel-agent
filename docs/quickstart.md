# Quick Start

The installed `aka optimize` command starts the default **Light** workflow: managed
source drafts, scoped Agent tools, unified Episodes, and Supervisor-owned promotion.
The primary path is native Atrex-Bench evaluation through Agate. For SOL-ExecBench,
OpenSSH, or the explicit Fast/Full workflow, use the [Legacy guide](legacy-workflow.md).

## Prerequisites

- Python 3.10+, `bash`, `git`, and `torch` on the coordinator.
- An authenticated coding CLI on `PATH`: `claude`, `qodercli`, `codex`, or `pi`.
- A configured Agate endpoint and GPU worker with the workload's framework and GPU stack.
- A native Atrex-Bench operator in its owning Bench checkout.
- For coordinator isolation: Linux, Bubblewrap, and permission to create the required
  namespaces. macOS supports native Light; it does not support `light-sandbox` directly.

NVIDIA profiling requires the appropriate NCU tools on the worker; AMD and PPU profiling
use their respective worker toolchains. Initialize only the reference-project submodules
you select. The Light resource profile includes GPU Wiki; its corpus is resolved on the first query,
not by an unconditional startup scan. See [Wiki configuration](optional-light-modules.md#independent-wiki-execution).

## Install from a matching checkout

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
python3 -m pip install build
python3 tools/build_distributions.py --output /tmp/aka-wheels \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light
python3 -m pip install --force-reinstall /tmp/aka-wheels/*.whl
aka optimize --help
```

Keep all three distributions and the resource checkout at the same revision. Their current
package metadata uses `0.2.0`, so installing a changed checkout requires reinstalling its
wheels even when the version string has not changed. Missing Light is an installation error.
See [installed startup](application-plugin-migration.md) for package and continuation details.

## Prepare the problem

A native Atrex-Bench operator contains `reference.py`, `input.py`, and `shapes.json`
inside a checkout with `scripts/run_eval.py` and `src/atrex_bench`.
The operator directory and AKA's `--repo-root` are separate inputs.

In production mode, a supplied `agent_problem.json` provides the generalized public contract.
Otherwise a trusted preprocessing session derives it from the evaluator-owned inputs.
Optimization sessions receive the public contract; detailed shapes and generators stay private
and are injected by the Supervisor during evaluation. Diagnostic feedback retains useful
case identifiers and errors with private dimensions redacted.

The default `--optimization-mode leaderboard` exposes the benchmark contract and allows
the permissive optimization policy. Select `production` explicitly for production review.
Use a new workspace when changing modes or migrating a campaign that exposed private shapes.

## Launch Light

From the AKA checkout:

```bash
aka optimize --repo-root "$PWD" \
  --problem /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --framework Cuda --agent-cli claude \
  --optimization-mode production \
  --workspace-base "$HOME/aka-opt" --max-iters 20
```

Replace the operator and platform with your workload and worker. `--hardware`
can specify a worker alias different from the logical platform. Configure one endpoint
on the coordinator before launch: `ATREX_SANDBOX_PROFILE=prod` (or `pre`), or
`ATREX_SANDBOX_URL=http://127.0.0.1:8000`. Clear the other endpoint selectors;
URL, profile and SSH selections are mutually exclusive.
Client waiting has no independent total deadline by default. The backend's queue and
execution limits still apply. Set `ATREX_SANDBOX_QUEUE_WAIT_GRACE` only when an explicit
client wait bound is required; expiry preserves the original job for reconciliation.

All launch and optimization options belong to the same parser; do not insert `--`.
Omitting `--profile` selects Light. Use `--profile light-sandbox` for Bubblewrap.
`--workspace-base` is the parent directory; AKA creates the named campaign beneath it.

To launch through a coding agent, ask it to translate the workload, worker, framework,
workspace, mode, and budgets into this same installed command.

### Agent backends and frameworks

Use `--agent-cli codex`, `qodercli`, or `pi` to change the coding backend.
Provider settings use `ATREX_CLAUDE_SESSION_SETTINGS`, `ATREX_QODER_SESSION_SETTINGS`,
`ATREX_CODEX_SESSION_SETTINGS`, or `ATREX_PI_SESSION_SETTINGS`, with
`ATREX_SESSION_SETTINGS` as the generic fallback. Keep credentials in the provider's
supported login/configuration mechanism.

Specify one `--framework` for one campaign. Omitting it dispatches separate campaigns
for the runtime-detected vendor: NVIDIA uses Triton/CuteDSL/Cuda/TileLang, AMD uses
Triton/FlyDSL/TileLang, PPU uses Triton/Cuda/TileLang, and unknown hardware uses
Triton/TileLang. Framework availability still depends on the worker.
Version and token budgets apply per dispatched campaign.

## What happens after launch

1. Resolve the selected providers, freeze the problem/runtime inputs, and create or resume
   a campaign below `--workspace-base`.
2. Evaluate V0 from the selected initial source. Production normally enables a framework
   baseline V1 through `--framework-baseline auto`; an imported source tree uses its
   pinned seed. `always` and `never` explicitly select baseline behavior.
3. Start an Episode in a managed Git-free draft. The agent researches, edits, profiles,
   evaluates, and submits an `episode-report` citing measurement record IDs.
4. Validate the exact source and required correctness/production evidence, seal it, and
   compare it to the incumbent through native same-allocation ABBA.
5. Promote only a geometric-mean latency reduction strictly above
   `--min-improvement-pct`. Preserve evidence for rejected or blocked attempts, and
   continue until campaign limits stop admission or the campaign finishes.

Light does not use the Legacy Fast/Full stage split. A report can be repaired and resubmitted;
an accepted report is immutable. Subsequent draft edits cannot replace the sealed source.

## Configure optional features

Save a composition patch, then pass `--patch /path/to/options.json`:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {
      "startup": true,
      "numerical_validation": true,
      "generate_memory": true,
      "research_records": true
    }}
  ]
}
```

This example enables supplemental numerical probes, generated memory, and research
annotations. All three are disabled by default. Standard evaluator correctness,
measurement records, report validation, and ABBA remain required with them disabled.

A patch replaces the provider's **whole configuration**; omitted values take provider
defaults. A Workflow patch used by `aka optimize` must retain `"startup": true` to
provide the launch entry point. An Agent patch for `light-sandbox` should retain
`"sandbox": "bwrap"` when isolation is desired. The shared Agent has no `capture` option.
See [all module options](optional-light-modules.md#other-module-options).

### Candidate layouts

New Light tasks use `source_tree`: a fixed `kernel.py` imports editable
`impl/model.py`, and the agent may add source files under `impl/`.
Native CUDA without a Python loader requires an explicit Workflow patch:

```json
{"api_version":1,"patches":[{"id":"workflow","config":{"startup":true,"file_system":"cuda_manifest"}}]}
```

Use it with `--framework Cuda`. CUDA sources, headers, and `solution.json` form the
candidate; Agate compiles and binds its launcher. The layout is fixed for the task.
See [source imports and native CUDA](native-cuda.md).

SOL/SSH requires `single_file` in Light, but Light ABBA has no SOL/SSH fallback.
Use [Legacy](legacy-workflow.md) for campaigns needing that verification path.

## Budgets and recovery

- `--max-iters` bounds canonical versions/Episodes; `--max-stall` can stop stalled campaigns.
- `--token-budget` controls Episode admission from available usage receipts. It excludes
  baseline/auxiliary usage, can be exceeded by an in-flight invocation, and cannot account
  for tokens a provider does not report.
- Light has no independent Episode wall-clock cap by default. An explicit
  `workflow.episode_timeout_seconds` sets a wall-clock cap; blocked tool time counts.
- `--framework-baseline-timeout` defaults to 10,800 seconds. GPU queue/execution
  budgets and shared Agent invocation limits are separate.
- Resume with the original workspace, composition, source layout, and frozen resources.
  Accepted job IDs and completed ABBA batches are reconciled before further GPU execution.

Use a new workspace when switching Legacy/Light or serial/pooled search. Do not edit
locks, private records, or job checkpoints to bypass an identity failure.
See [execution recovery](measurement-records.md#failure-and-recovery-policy) and
[promotion repair](supervisor-promotion.md#promotion-evidence-and-recovery).

## Inspect outputs

| Output | Meaning |
| --- | --- |
| Candidate source files | Retained implementation in the selected layout; inspect the whole tree for multi-file candidates |
| `.atrex_versions/vN.json` | Supervisor-owned version, correctness, measurement, and outcome facts when generated memory is disabled |
| `memory/vN.json` | Optional summaries, enabled by `workflow.generate_memory` |
| `.atrex_long_horizon/` | Episode state, attempts, usage observations, and Session archives |
| Private Supervisor scope | Measurement snapshots, Journals, drafts, promotion audits, and GPU checkpoints outside agent drafts |
| `profiles/` and requested artifacts | Available profiler outputs and diagnostic evidence |

Raw private stores and Session traces are not public submission artifacts. The optional
[dashboard](optimization-dashboard.md) reads persisted observations; its default file exporter
does not package complete multi-file source trees. Use the workflow's source-aware export
for those candidates.

Run `aka optimize --repo-root "$PWD" --help` for current CLI syntax; the public settings include Legacy stage
options, so use the [Light guide](optional-light-modules.md) for their workflow applicability.
