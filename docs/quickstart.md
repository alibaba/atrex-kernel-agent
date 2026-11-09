# Quick Start

The installed `aka optimize` command starts the default **Light** workflow: managed
source drafts, private HTTP tools, unified Episodes, and Supervisor-owned promotion.
Native Atrex-Bench evaluation runs through Agate or the [OpenSSH backend](ssh-backend.md).
For SOL-ExecBench or the explicit Fast/Full workflow, use the [Legacy guide](legacy-workflow.md).

## Prerequisites

- Python 3.10+, `bash`, `git`, and `torch` on the coordinator.
- An authenticated coding CLI on `PATH`: `claude`, `qodercli`, `codex`, `pi`, or `dsh`.
- A configured Agate endpoint or SSH host with the workload's framework and GPU stack.
- A native Atrex-Bench operator in its owning Bench checkout.
- For coordinator isolation: Linux, Bubblewrap, and permission to create the required
  namespaces. macOS supports native Light; it does not support `light-sandbox` directly.

NVIDIA profiling requires the appropriate NCU tools on the worker; AMD and PPU profiling
use their respective worker toolchains. Initialize only the reference-project submodules
you select. GPU Wiki is enabled by default, but its corpus is resolved on the first query,
not by an unconditional startup scan. See [Wiki configuration](optional-light-modules.md#independent-wiki-execution).

## Install from a matching checkout

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
python3 -m pip install build
python3 tools/build_distributions.py --output /tmp/aka-wheels \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light
python3 -m pip install --force-reinstall --no-index \
  --find-links /tmp/aka-wheels atrex-aka-light
aka optimize --help
```

Keep all three distributions and the resource checkout at the same revision. Their current
package metadata uses `0.1.0`, so installing a changed checkout requires reinstalling its
wheels even when the version string has not changed. Missing Light is an installation error.
See [installed startup](application-plugin-migration.md) for package and continuation details.

## Prepare the problem

A native Atrex-Bench operator contains `reference.py`, `input.py`, and `shapes.json`
inside a checkout with `scripts/run_eval.py` and `src/atrex_bench`.
The operator directory and AKA's `--repo-root` are separate inputs.

Production exposes the complete native Atrex-Bench contract by default. Optimization sessions can
read `shapes.json` and evaluator metadata; Runtime feedback remains compact but retains exact
dimensions. Production source review and promotion gates remain enabled. Pass
`--no-expose-evaluator-shapes` to use a supplied or automatically derived `agent_problem.json`, keep
detailed shapes private, inject them only during evaluation, and redact private dimensions from
diagnostic feedback.

The default `--optimization-mode leaderboard` exposes the benchmark contract and allows
the permissive optimization policy. Select `production` explicitly for production review.
Existing leaderboard workspaces retain their historical exposed snapshot key and
resume without migration. Existing hidden production workspaces must be resumed
with `--no-expose-evaluator-shapes`; the persisted workspace policy fails closed
and names that flag if a new default would otherwise change visibility. Use a new
workspace only when intentionally changing mode or Shape-visibility policy.

## Launch Light

From the AKA checkout:

```bash
aka optimize --repo-root "$PWD" -- \
  --op-dir /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --framework Cuda --agent-cli claude \
  --optimization-mode production \
  --workspace "$HOME/aka-opt" --max-iters 20
```

Replace the operator and platform with your workload and worker. `--sandbox-hardware`
can specify a worker alias different from the logical platform. Use `--sandbox-profile prod`
or `pre` to select an Agate endpoint profile, or `--sandbox-url URL` for an explicit endpoint;
otherwise the configured endpoint is retained.

Launch options (`--repo-root`, `--profile`, `--patch`) go **before** `--`.
Optimization arguments go after it. Omitting `--profile` selects Light.
Use `--profile light-sandbox` before the separator for Bubblewrap.

To launch through a coding agent, ask it to translate the workload, worker, framework,
workspace, mode, and budgets into this same installed command.

### Agent backends and frameworks

Use `--agent-cli codex`, `qodercli`, `pi`, or `dsh` to change the coding backend.
The [DSH guide](dsh-harness-adapter.md) walks through headless setup, provider
settings, and Session progress.
Provider settings use `ATREX_CLAUDE_SESSION_SETTINGS`, `ATREX_QODER_SESSION_SETTINGS`,
`ATREX_CODEX_SESSION_SETTINGS`, `ATREX_PI_SESSION_SETTINGS`, or
`ATREX_DSH_SESSION_SETTINGS`, with
`ATREX_SESSION_SETTINGS` as the generic fallback. Keep credentials in the provider's
supported login/configuration mechanism.

Specify one `--framework` for one campaign. Omitting it dispatches separate campaigns
for the runtime-detected vendor: NVIDIA uses Triton/CuteDSL/Cuda/TileLang, AMD uses
Triton/FlyDSL/TileLang, PPU uses Triton/Cuda/TileLang, and unknown hardware uses
Triton/TileLang. Framework availability still depends on the worker.
Version and token budgets apply per dispatched campaign.

## What happens after launch

1. Resolve the selected providers, freeze the problem/runtime inputs, and create or resume
   a campaign below `--workspace`.
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

Save a composition patch, then pass `--patch /path/to/options.json` before `--`:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {
      "startup": true,
      "numerical_validation": true,
      "generate_memory": true,
      "episode_timeout_seconds": 14400
    }},
    {"id": "light_journal", "config": {"research_records": true}}
  ]
}
```

This example enables supplemental numerical probes, generated memory, and research
annotations. All three are disabled by default. Standard evaluator correctness,
measurement records, report validation, and ABBA remain required with them disabled.

A patch replaces the provider's **whole configuration**; omitted values take provider
defaults. A Workflow patch used by `aka optimize` must retain `"startup": true` to
provide the launch entry point. An Agent patch for `light-sandbox` should retain
`"sandbox": "bwrap"` and `"capture": true` when those behaviors are desired.
See [all module options](optional-light-modules.md#other-module-options).

### Candidate layouts

New Light tasks use `source_tree`: `kernel.py` is the editable evaluator entry point
and may either contain the full implementation or import optional helpers under `impl/`.
Native CUDA without a Python loader requires an explicit candidate patch:

```json
{"api_version":1,"patches":[{"id":"candidate","config":{"file_system":"cuda_manifest"}}]}
```

Use it with `--framework Cuda`. CUDA sources, headers, and `solution.json` form the
candidate; Agate compiles and binds its launcher. The layout is fixed for the task.
See [source imports and native CUDA](native-cuda.md).

SSH supports source-tree and native CUDA Evaluate/ABBA. SOL requires `single_file`
and the [Legacy](legacy-workflow.md) verification path.

## Budgets and recovery

- `--max-iters` bounds canonical versions/Episodes; `--max-stall` can stop stalled campaigns.
- `--token-budget` controls Episode admission from available usage receipts. It excludes
  baseline/auxiliary usage, can be exceeded by an in-flight invocation, and cannot account
  for tokens a provider does not report.
- Light defaults to four hours of agent active time per optimization Episode. Blocking
  Tool Runtime requests pause the agent and do not consume this allowance. The active-time
  budget is shared across Session repairs within the running controller; it is not a
  persisted allowance across controller restarts.
- Framework baseline has its own eight-hour wall-clock default. GPU queue/execution and
  Wiki deadlines are independent of agent active time.
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

Run `aka optimize --repo-root "$PWD" -- --help` for CLI syntax; the parser retains Legacy
options, so use the [Light guide](optional-light-modules.md) for their workflow applicability.
