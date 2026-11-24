# Quick Start

The installed `aka optimize` command starts the default **Light** workflow: managed
Agent-owned source trees, private Runtime tools, one persistent optimization Session, and auditable segments.
The primary path is native Atrex-Bench evaluation through Agate. For native Atrex-Bench
evaluation on a provisioned GPU host, see the [OpenSSH backend](ssh-backend.md).

## Prerequisites

- Python 3.10+, `bash`, `git`, and `torch` on the coordinator.
- An authenticated coding CLI on `PATH` (`claude`, `qodercli`, `codex`, `pi`, or `dsh`).
- GPU access through [Agate](#launch-light) or [OpenSSH](ssh-backend.md), with the workload's
  framework and GPU stack available on the worker.
- A native Atrex-Bench operator in its owning Bench checkout.
- For coordinator isolation, `light-sandbox` requires Linux, the `bwrap` executable, and
  permission to create the required namespaces. Native `light` does not require Bubblewrap.
  Ordinary Linux/Lima defaults to `proc_mode: private`; Aone Sandbox launches default to
  `proc_mode: host`. macOS supports native Light but cannot run Bubblewrap directly.

NVIDIA profiling requires the appropriate NCU tools on the worker; AMD and PPU profiling
use their respective worker toolchains. Initialize only the reference-project submodules
you select. GPU Wiki is enabled by default, but its corpus is resolved on the first query,
not by an unconditional startup scan. See [Wiki configuration](optional-light-modules.md#independent-wiki-execution).

Multi-segment persistent Sessions require a backend with native resume support
(Claude, Codex, Pi, or DSH). A backend without resume support can run a single segment.

## Install from a matching checkout

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
python3 -m pip install build
python3 tools/build_distributions.py --output /tmp/aka-wheels \
  atrex-aka-framework atrex-aka-light
python3 -m pip install --force-reinstall --no-index \
  --find-links /tmp/aka-wheels atrex-aka-light
aka optimize --help
```

Keep both distributions and the resource checkout at the same revision. The `v0.4.1`
repository release uses Python distribution version `0.3.0`. Rebuild and reinstall the
wheels when changing the checkout revision, even if the version string is the same.
See the [v0.4.1 upgrade guide](releases/v0.4.1.md#upgrading-from-v040) for changes to
launch flags, workspaces, and package installation.

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
sudo apt-get install -y bubblewrap git python3 python3-venv
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
# Follow "Install from a matching checkout" above.
```

Install AKA and launch `--profile light-sandbox` from inside that VM. Keep the checkout,
Python environment, Agent CLI credentials and campaign workspace inside (or explicitly mounted
into) the VM; running only the Agent subprocess in Lima is not a supported split deployment.

### Aone Sandbox default: `proc_mode: host`

The current Aone Sandbox container policy does not allow a nested Bubblewrap process to mount a
fresh procfs. Therefore every Aone Sandbox Light launch should default to `proc_mode: host` and share
the Aone Sandbox's existing PID namespace and complete `/proc`. Use a dedicated Sandbox because
this mode exposes every process in that Sandbox to the Agent. Save this composition patch inside
the Aone Sandbox. Do not omit this patch and fall back to the provider-wide `private` default:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "proc_mode": "host"
  }}]
}
```

Install Bubblewrap in the same Aone execution image:

```bash
sudo dnf install -y bubblewrap
```

Launch with the patch:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox --patch /path/to/aone-host-proc.json \
  --problem /path/to/atrex-bench/operator \
  --platform L20D --framework Cuda --agent-cli claude \
  --optimization-mode ultra --workspace-base /workspace/aka-opt
```

Light binds the Aone Sandbox's existing `/proc` directly into Bubblewrap. It does not compile or
load a compatibility shim and does not set `LD_PRELOAD`. `--cap-drop ALL` and the existing
filesystem allowlist remain active, but PID/process visibility is intentionally not isolated.

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
The Agent chooses whether to evaluate V0; any requested measurement is stored in
`measurements/`. Light does not create a separate `baseline/` directory, and V0
does not participate in accepted-submission winner or Pareto selection.

## Launch Light

From the AKA checkout, use an explicit fixed-DSL Ultra launch:

```bash
aka optimize --repo-root "$PWD" \
  --problem /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --framework Cuda --agent-cli claude \
  --optimization-mode ultra \
  --workspace-base "$HOME/aka-opt" --max-iters 20
```

For one Fast campaign in which the Agent chooses or mixes supported GPU DSLs and may use
declared reproducible compute libraries:

```bash
aka optimize --repo-root "$PWD" \
  --problem /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --agent-cli claude \
  --optimization-mode fast \
  --workspace-base "$HOME/aka-opt-fast" --max-iters 20
```

The startup line for the second command must contain
`optimization_mode=fast ... frameworks=open_dsl`. If it reports
`frameworks=Cuda`, the launch is still CUDA-pinned.

Replace the operator and platform with your workload and worker. `--hardware`
can specify a worker alias different from the logical platform. Configure the Agate endpoint
in the selected Execution provider or Supervisor environment (`AGATE_URL`); otherwise the
SDK uses its configured endpoint.

Launch and optimization options share one parser; no `--` separator is needed.
Omitting `--profile` selects Light. Use `--profile light-sandbox` for Bubblewrap.

To launch through a coding agent, ask it to translate the workload, worker, framework,
workspace, mode, and budgets into this same installed command.

### Agent backends and frameworks

Use `--agent-cli codex`, `qodercli`, `pi`, or `dsh` to change the coding backend.
Provider settings use `ATREX_CLAUDE_SESSION_SETTINGS`, `ATREX_QODER_SESSION_SETTINGS`,
`ATREX_CODEX_SESSION_SETTINGS`, `ATREX_PI_SESSION_SETTINGS`, or `ATREX_DSH_SESSION_SETTINGS`, with
`ATREX_SESSION_SETTINGS` as the generic fallback. Keep credentials in the provider's
supported login/configuration mechanism. See the [DSH setup guide](dsh-harness-adapter.md).

The three policies own DSL selection directly:

| Policy | Required arguments | Result |
| --- | --- | --- |
| Fast | `--optimization-mode fast` | Agent-selected DSL; declared reproducible compute libraries allowed |
| Performance | `--optimization-mode performance` | Agent-selected DSL guided by expertise; self-authored core compute required |
| Ultra | `--optimization-mode ultra --framework Cuda` | Configured DSL enforced; self-authored core compute required |

Fast and Performance use an open-DSL campaign. Ultra rejects a missing framework
before workspace creation. Only these three policy names are accepted.

The first optimization segment starts directly from the initial source, and the same persistent Agent
Session may perform framework bring-up, correctness work, diagnostics and optimization before or
between commit reviews.

## What happens after launch

1. Resolve the selected providers, freeze the problem/runtime inputs, and create or resume
   a campaign below `--workspace-base`.
2. Prepare the initial source and common Git root, then start the persistent Agent Session.
   The Agent decides whether to measure the initial implementation.
3. The Agent researches, edits, profiles and evaluates freely in its own persistent Git repository.
   It may call `submit-kernel` with a reachable commit after that exact commit has a passing full
   Evaluate. Submission verifies the saved evidence and applies the selected final-source policy
   review; it launches no new Evaluate or ABBA and performs no promotion, rollback or Session transition.
4. When the Agent voluntarily stops a segment, Light records an Episode audit boundary and resumes
   the same provider Session, workspace and Git history for the next segment. Runtime capabilities
   and shared evidence are refreshed in place.
5. Continue until campaign limits stop admission or the Agent/campaign finishes. Research files,
   Gateway Records and commit-review receipts remain auditable without selecting an incumbent.

Light uses only persistent Sessions and accepted submissions.

## Configure optional features

Save a composition patch, then pass `--patch /path/to/options.json` alongside the launch flags:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {
      "startup": true,
      "tool_modules": ["directions"]
    }}
  ]
}
```

This example enables Direction-only research. Both research modules
are disabled by default. Agent Sessions and Reviews have no active-time limit. All measurement
evidence is retained centrally; explicit ABBA is optional.

A patch replaces the provider's **whole configuration**; omitted values take provider
defaults. `startup` defaults to `true` so `aka optimize` installs its launch entry point. An Agent patch for `light-sandbox` should retain
`"sandbox": "bwrap"`  when those behaviors are desired.
See [all module options](optional-light-modules.md#research-and-parallel-search).

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

- `--max-iters` limits naturally completed Episode segments; it does not interrupt a running Session.
- `--token-budget` controls Episode admission from main-Agent usage receipts. Independent Review
  calls are measured separately by kind and do not consume this budget. In-flight work may
  overshoot, and no counter can include tokens a provider does not report.
- **Episode active time is intentionally unbounded.** There is no Episode timeout option.
  Natural Agent stops delimit Episodes; cancellation or Session revocation ends active execution.
  Reviews also have no active-time limit. An explicitly configured Agent provider
  `timeout_seconds` limits a provider invocation, independently of Episode segmentation.
- Runtime requests do not pause the Agent by default. Independent work can continue;
  `pause_agent_during_requests=true` opts into process suspension. With the default `false`,
  Runtime waits count toward an explicitly configured provider invocation time limit. Each
  request uses frozen source/input bytes; subsequent edits do not change that in-flight request.
- GPU queue, request and per-Shape execution budgets remain finite and separately configurable.
- Resume with the original workspace, composition, source layout, and frozen resources.
  Accepted job IDs and saved Gateway Records are reconciled before further GPU execution.

Light runs serially by default. Configure `search.trajectories` for pooled search; an empty
`search` object retains serial execution. Use a new workspace when switching
serial/island search. Do not edit
locks, private records, or job checkpoints to bypass an identity failure.
See [execution recovery](measurement-records.md#failure-and-recovery-policy).

## Inspect outputs

| Output | Meaning |
| --- | --- |
| Agent Git repository | Complete implementation history, current HEAD, branches and imported peer refs |
| Private Supervisor Campaign directory | Segment state, Session archives, Measurement snapshots, central Research/Submission stores, Runtime audit, source archives, and GPU checkpoints outside Agent Git |
| `profiles/` and requested artifacts | Available profiler outputs and diagnostic evidence |

Light does not write Campaign or Episode control state into the Agent workspace. Raw private
stores and Session traces are not public submission artifacts. The optional
[dashboard](optimization-dashboard.md) reads persisted observations; use `observability/export_light_kernel.py` to export the complete source tree and evidence.

Run `aka optimize --repo-root "$PWD" --help` for the native `WorkflowLaunch` CLI. All flags use one parser; no `--` separator is needed.
