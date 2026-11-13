# Quick Start

The installed `aka optimize` command starts the default **Light** workflow: managed
source drafts, private HTTP tools, one persistent optimization Session, and auditable segments.
The primary path is native Atrex-Bench evaluation through Agate. For SOL-ExecBench,
OpenSSH, or the explicit Fast/Full workflow, use the [Legacy guide](legacy-workflow.md).

## Prerequisites

- Python 3.10+, `bash`, `git`, and `torch` on the coordinator.
- An authenticated coding CLI on `PATH`: `claude`, `qodercli`, `codex`, or `pi`.
- A configured Agate endpoint and GPU worker with the workload's framework and GPU stack.
- A native Atrex-Bench operator in its owning Bench checkout.
- For coordinator isolation, `light-sandbox` requires Linux, the `bwrap` executable, and
  permission to create the required namespaces. Native `light` does not require Bubblewrap.
  `light-sandbox` with `proc_mode: projected` also requires a working C compiler and libc
  development headers in the exact execution environment (`gcc glibc-devel` on the supported
  DNF image, or `build-essential libc6-dev` on Ubuntu). Run
  `python3 -m aka.light.agent.preflight` there before launch; startup fails closed if this
  compile check does not pass. macOS supports native Light but cannot run Bubblewrap directly.

NVIDIA profiling requires the appropriate NCU tools on the worker; AMD and PPU profiling
use their respective worker toolchains. Initialize only the reference-project submodules
you select. GPU Wiki is enabled by default, but its corpus is resolved on the first query,
not by an unconditional startup scan. See [Wiki configuration](optional-light-modules.md#independent-wiki-execution).

Multi-segment persistent Sessions require a backend with native resume support (currently Claude,
Codex, or DSH). A backend without resume support may run one segment, but Light fails explicitly
instead of silently starting a new conversation for the next Episode.

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

### macOS: run `light-sandbox` in Lima

To use Bubblewrap isolation from macOS, run the AKA coordinator inside a Linux VM. Lima's
default Ubuntu template is sufficient; GPU execution still happens remotely through Agate, so
the VM does not need GPU passthrough.

On the macOS host:

```bash
brew install lima
limactl start --name=aka-light template:default
limactl shell aka-light
```

Then, inside the Lima VM:

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap build-essential libc6-dev git python3 python3-venv
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
# Follow "Install from a matching checkout" above, then verify the guest:
python3 -m aka.light.agent.preflight
```

Install AKA and launch `--profile light-sandbox` from inside that VM. Keep the checkout,
Python environment, Agent CLI credentials and campaign workspace inside (or explicitly mounted
into) the VM; running only the Agent subprocess in Lima is not a supported split deployment.

### Aone Sandbox: use the projected `/proc` shim

The current Aone Sandbox container policy does not allow a nested Bubblewrap process to mount a
fresh procfs. Keep Bubblewrap isolation enabled, but select Light's bundled projected-proc shim.
Save this composition patch inside the Aone Sandbox:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "capture": true,
    "proc_mode": "projected"
  }}]
}
```

Install the local prerequisites in the same Aone execution image, then run the preflight:

```bash
sudo dnf install -y bubblewrap gcc glibc-devel
python3 -m aka.light.agent.preflight
```

Launch with the patch before the `--` separator:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox --patch /path/to/aone-projected-proc.json -- \
  --op-dir /path/to/atrex-bench/operator \
  --platform L20D --specify-dsl --framework Cuda --agent-cli claude \
  --optimization-mode ultra --workspace /workspace/aka-opt
```

Light compiles and loads the packaged compatibility shim automatically. Do not manually mount
the host `/proc`, expose the host process tree, or configure `LD_PRELOAD`. The projected view
contains only the bounded process metadata needed by the Agent CLI. Startup fails before the
Campaign begins if the shim cannot compile.

## Prepare the problem

A native Atrex-Bench operator contains `reference.py`, `input.py`, and `shapes.json`
inside a checkout with `scripts/run_eval.py` and `src/atrex_bench`.
The operator directory and AKA's `--repo-root` are separate inputs.

Exact evaluator Shapes are exposed by default through `shapes.json`. Pass
`--no-expose-evaluator-shapes` to use a supplied or Supervisor-derived generalized
`agent_problem.json` instead. In hidden mode, every Agent-visible Agate response is Shape-redacted;
in exposed mode it is not. Hidden mode is supported only for Performance/Ultra Native Atrex-Bench tasks;
an unsupported request fails before the Campaign starts instead of silently running exposed.
Evaluator generators and generated tensor values remain Supervisor-owned.

The default `--optimization-mode fast` exposes the benchmark contract, lets the Agent choose the
DSL and permits declared reproducible compute libraries. `performance` keeps Agent-selected DSL
but requires self-authored core compute. `ultra` additionally requires an explicit `--framework`.
Use a new workspace when changing optimization mode, DSL policy, or Shape visibility.
The selected workspace is a non-Git Campaign container. Light stores Campaign
state directly inside it and creates Git repositories only for
`lineages/tNNNN/agent-workspace`; do not initialize the Campaign root as Git or
reuse a legacy Controller repository in place.
The initial V0 full Evaluate is stored as an ordinary private Gateway Record in
`measurements/`. Light does not create a separate `baseline/` directory, and V0
does not participate in accepted-submission winner or Pareto selection.

## Launch Light

From the AKA checkout, use an explicit fixed-DSL Ultra launch:

```bash
aka optimize --repo-root "$PWD" -- \
  --op-dir /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --specify-dsl --framework Cuda --agent-cli claude \
  --optimization-mode ultra \
  --workspace "$HOME/aka-opt" --max-iters 20
```

For one Fast campaign in which the Agent chooses or mixes supported GPU DSLs and may use
declared reproducible compute libraries:

```bash
aka optimize --repo-root "$PWD" -- \
  --op-dir /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --agent-cli claude \
  --optimization-mode fast \
  --workspace "$HOME/aka-opt-fast" --max-iters 20
```

The startup line for the second command must contain
`optimization_mode=fast ... frameworks=open_dsl`. If it reports
`frameworks=Cuda`, the launch is still CUDA-pinned.

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

Use `--agent-cli codex`, `qodercli`, or `pi` to change the coding backend.
Provider settings use `ATREX_CLAUDE_SESSION_SETTINGS`, `ATREX_QODER_SESSION_SETTINGS`,
`ATREX_CODEX_SESSION_SETTINGS`, or `ATREX_PI_SESSION_SETTINGS`, with
`ATREX_SESSION_SETTINGS` as the generic fallback. Keep credentials in the provider's
supported login/configuration mechanism.

The three policies own DSL selection directly:

| Policy | Required arguments | Result |
| --- | --- | --- |
| Fast | `--optimization-mode fast` | Agent-selected DSL; declared reproducible compute libraries allowed |
| Performance | `--optimization-mode performance` | Agent-selected DSL guided by expertise; self-authored core compute required |
| Ultra | `--optimization-mode ultra --framework Cuda` | Configured DSL enforced; self-authored core compute required |

Fast and Performance normalize to one open-DSL campaign. Ultra rejects a missing framework,
`--no-specify-dsl`, and automatic framework dispatch before workspace creation. Historical
`leaderboard` and `production` names remain compatibility aliases for resuming existing runs;
new runs should use the three explicit policies above.

The first optimization segment starts directly from measured V0, and the same persistent Agent
Session may perform framework bring-up, correctness work, diagnostics and optimization before or
between commit reviews.

## What happens after launch

1. Resolve the selected providers, freeze the problem/runtime inputs, and create or resume
   a campaign below `--workspace`.
2. Evaluate V0 from the selected initial source, then start one persistent Agent Session directly
   from that source.
3. The Agent researches, edits, profiles and evaluates freely in its own persistent Git repository.
   It may call `submit-kernel` with a reachable commit after that exact commit has a passing full
   Evaluate. Submission verifies the saved evidence and applies the selected final-source policy
   review; it launches no new Evaluate or ABBA and performs no promotion, rollback or Session transition.
4. When the Agent voluntarily stops a segment, Light records an Episode audit boundary and resumes
   the same provider Session, workspace and Git history for the next segment. Runtime capabilities
   and shared evidence are refreshed in place.
5. Continue until campaign limits stop admission or the Agent/campaign finishes. Research files,
   Gateway Records and commit-review receipts remain auditable without selecting an incumbent.

Light does not use the Legacy Fast/Full stage split or an automatic promotion state machine.

## Configure optional features

Save a composition patch, then pass `--patch /path/to/options.json` before `--`:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {
      "startup": true,
      "numerical_validation": true,
      "episode_timeout_seconds": 14400
    }},
    {"id": "light_journal", "config": {"tool_modules": ["directions"]}}
  ]
}
```

This example explicitly enables supplemental numerical probes, adds a four-hour Episode limit,
and selects Direction-only research recording. Direction and Experiment
modules are both enabled by default; use `[]` for report-only mode. Standard evaluator correctness,
measurement records, report validation, and ABBA remain required regardless of these options.

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

SOL requires `single_file` in Light and has no native ABBA fallback. Explicit SSH supports
Native Atrex-Bench `source_tree`, CUDA manifest, Evaluate, ABBA and typed diagnostics; see
[Light OpenSSH backend](ssh-backend.md).

## Budgets and recovery

- `--max-iters` bounds canonical versions/Episodes; `--max-stall` can stop stalled campaigns.
- `--token-budget` controls Episode admission from main-Agent usage receipts. Independent Review
  calls are measured separately by kind and do not consume this budget. In-flight work may
  overshoot, and no counter can include tokens a provider does not report.
- Light defaults to four hours of agent active time per optimization Episode. Blocking
  Tool Runtime requests pause the agent and do not consume this allowance. The active-time
  budget is shared across Session repairs within the running controller; it is not a
  persisted allowance across controller restarts.
- GPU queue/execution, independent commit review and Wiki deadlines are independent of Agent active time.
- Resume with the original workspace, composition, source layout, and frozen resources.
  Accepted job IDs and saved Gateway Records are reconciled before further GPU execution.

Light uses three persistent search islands by default; set the Workflow plugin's `search`
configuration to `{}` for an explicit serial run. Use a new workspace when switching
Legacy/Light or serial/island search. Do not edit
locks, private records, or job checkpoints to bypass an identity failure.
See [execution recovery](measurement-records.md#failure-and-recovery-policy).

## Inspect outputs

| Output | Meaning |
| --- | --- |
| Agent Git repository | Complete implementation history, current HEAD, branches and imported peer refs |
| `.atrex_versions/vN.json` | Supervisor-owned segment and outcome facts when generated memory is disabled |
| `.atrex_long_horizon/` | Episode state, attempts, usage observations, and Session archives |
| Private Supervisor scope | Measurement snapshots, central Research/Submission stores, Runtime audit, source archives, and GPU checkpoints outside Agent Git |
| `profiles/` and requested artifacts | Available profiler outputs and diagnostic evidence |

Raw private stores and Session traces are not public submission artifacts. The optional
[dashboard](optimization-dashboard.md) reads persisted observations; its default file exporter
does not package complete multi-file source trees. Use the workflow's source-aware export
for those candidates.

Run `aka optimize --repo-root "$PWD" -- --help` for CLI syntax; the parser retains Legacy
options, so use the [Light guide](optional-light-modules.md) for their workflow applicability.
