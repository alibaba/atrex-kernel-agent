---
name: atrex-gpu-gateway
version: 0.15.0
description: Use the agate CLI to submit GPU operator evaluations, compilation checks, profiling, disassembly and controlled development commands to atrex-gpu-gateway, and query environments and asynchronous results. Covers CUDA solution.json configuration and AKA candidate file management and troubleshooting. Read operations references when managing gateway deployment or capacity.
---

# atrex-gpu-gateway

A stateless GPU gateway routes jobs over HTTP to GPU workers. Query and cancel jobs by `job_id`.
The local machine needs neither a GPU nor Ray or profiling tools. Consult `agate <command> --help` for exact options.

## Use within AKA

In an AKA-managed workspace, organize candidates according to the selected file system; the framework does not change the file format.
Read [AKA file organization and invocation](reference/aka-workspace.md) first, then
[manifest fields, CUDA ABI and repository dependencies](reference/eval-manifest.md) as needed.
Submit jobs through the workspace's `tools/sandbox.py`; the Supervisor owns environment selection, authentication, uploads and records.
The direct agate CLI installation and configuration instructions below are for standalone use, not for bypassing AKA Runtime.

## Choose a command

| Purpose | Command | Inputs and boundaries |
|---|---|---|
| Compilation checks and diagnostics only | `agate check kernel.py --gpu GPU` | Single file exposing `class Model`; submits a `compile` job without full correctness or timing |
| Check correctness and compare performance | `agate eval --gpu GPU --candidate kernel.py --reference-dir DIR` | Supply a reference implementation, input generator and shapes |
| Declare multiple candidate files and dependencies | `agate eval --gpu GPU --manifest solution.json --reference-dir DIR` | atrex manifest v1; prepares versioned Git repositories; the worker compiles CUDA and generates bindings without Python candidate code; see [declarative input](reference/eval-manifest.md) |
| Evaluate a SOL-ExecBench problem | `agate eval --backend sol --gpu GPU --solution solution.json` | Image-provided or caller-supplied problems |
| Find hotspots and performance bottlenecks | `agate profile --gpu GPU --candidate kernel.py --reference-dir DIR` | `survey` finds hotspots, `sol` analyzes compute/memory limits, and `deep` inspects kernels |
| Inspect GPU assembly | `agate disassemble --gpu GPU --candidate kernel.py --fmt auto` | Returns PTX/SASS/ISA; shares compilation check import/construction behavior |
| Query GPUs and toolchains | `agate env`, `agate env GPU --frameworks` | Read environment information without submitting development commands |
| Development actions without a dedicated API | `agate dev 'COMMAND' --gpu GPU` | Privileged raw shell, disabled by default; requires dedicated AK permissions |

`--backend atrex|sol` chooses the evaluation harness; gateway configuration chooses the execution backend.
`--profile prod|pre` chooses the gateway environment. The `sol` in `profile --level sol` means performance-limit analysis.
Use `eval` and `env` for new commands; legacy aliases `test`, `run`, `sol` and `list hw` remain compatible.

## Installation and authentication

```bash
pip install -U atrex-gateway-client \
  --index-url http://artlab.alibaba-inc.com/1/pypi/pypi-releases \
  --trusted-host artlab.alibaba-inc.com
export AGATE_AK='<ak>'
export AGATE_SK='<sk>'
agate config
agate env
```

The default is production at `https://atrex-gateway.alibaba-inc.com`; no extra URL configuration is required.
Alternatively, store `ak`/`sk` in `~/.atrex/config.json`. The CLI uses AK/SK signatures; corporate browser login cannot replace them.
Do not configure `AGATE_TOKEN`, which the server does not accept. `author` and a self-reported employee ID confer no permissions.
Switch to preproduction only when the user explicitly requests it. Examples here use production.

Use `agate update --check` to check for updates and `agate update` to install them. Older clients without this subcommand can use the pip command above.

## From compilation to evaluation

```bash
# Check compilation triggered by import and construction.
agate check my_kernel.py --gpu L20D
# Supply a JSON object when Model requires constructor arguments.
agate check my_kernel.py --gpu L20D --init-kwargs init_kwargs.json

# Check correctness and performance with real inputs.
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
# Check correctness only.
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --mode correctness_only

# Profile progressively when needed.
agate profile --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --level survey
agate profile --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --level deep --kernel-regex rms
```

`check` maps to `POST /v1/jobs/compile`; there is no `agate compile` CLI subcommand.
It imports the candidate and calls `Model(**init_kwargs)` without automatically invoking `forward`.
Implementations such as Triton that JIT-compile on first launch may produce only import/construction diagnostics and empty `ptxas` statistics.
`ok=true` does not prove those kernels compiled or ran correctly. Use `eval` or `profile` to exercise real inputs.
See [compilation checks](reference/compile.md) for result fields and the limits of `--arch` and sanitizers.

Python candidates and references must expose `class Model`; the worker generates that adapter for native CUDA candidates.
Read [reference directory format](reference/reference-dir.md) when preparing an operator package,
[solution.json format](reference/solution-json.md) for SOL candidates, and
[custom problems and minimal acceptance](reference/custom-sol-problem.md) for custom SOL problems.
All these references and examples ship with the Skill.

Shape parallelism is enabled by default for atrex eval; existing quotas and admission determine the worker budget.
Explicit `--shape-shards 1` forces serial execution. Older or unverified bench versions retain serial fallback.
See [advanced evaluation](reference/evaluation.md#shape-parallelism) for parallelism options, shard recovery and fallback boundaries.

### input.py size limit

The default limit for the `eval`/`profile` input generator `input.py` is **16 KiB = 16384 UTF-8 bytes**,
including comments and whitespace. It measures source bytes, not characters or generated tensor size.
Deployments may change `max_input_source_bytes`, effective after restart. Check the actual limit in
`GET /v1/env/{gpu}/capabilities` under `limits.max_input_source_bytes`.
Oversized input is rejected before queueing with `400 invalid_source`, including the actual and allowed byte counts for `reference.input_py`.
Candidates and `reference.py` have separate limits, each 128 KiB by default; see `limits.max_source_bytes`.
`check` does not accept a reference directory or `input.py`; the 16 KiB input-generator limit does not apply to its candidate.

## Submission, queries and failures

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --no-wait
agate get <job_id> --wait
agate jobs --status running
agate cancel <job_id>
```

`eval` waits for a terminal state by default, exiting 0 on success and nonzero on failure.
`check` also waits by default; use `--no-wait` to submit only.
Cancellation acceptance does not prove the remote job has stopped; continue querying the original `job_id`.
Phases distinguish gateway queueing, backend submission/queueing, execution and collection. A job keeps its identity while waiting for GPU scale-up.

- On quota exhaustion or temporary overload, honor `Retry-After` and reuse the original request and stable idempotency key. SDK retries also respect the call budget.
- If submission outcome is unknown, the connection breaks or storage is temporarily unavailable, retain the original `job_id`/idempotency key to confirm the result and avoid duplicate execution.
- `403 cluster_ak_denied` means the AK has no matching cluster permission; contact an administrator or choose an authorized environment.
- For `code/*`, inspect the candidate first; for `validation/*`, fix the request. Handle `infra/*` and `timeout/*` using error evidence and the available budget.

There are three timeout layers: `--job-timeout` limits execution, `--wait-timeout` limits CLI waiting,
and `--http-timeout` limits an individual HTTP request. Not every subcommand supports all options; check its help.
See [jobs and troubleshooting](reference/jobs-and-troubleshooting.md) for retries, error attribution, attachments and dependencies.

## Read as needed

| Scenario | Reference |
|---|---|
| Compilation results, JIT, constructor arguments, architecture or sanitizer | [Compilation checks](reference/compile.md) |
| Shape parallelism, ABBA, source archives, runner config, evaluator versions and clock locking | [Advanced evaluation](reference/evaluation.md) |
| Native CUDA, JSON build declarations and dependencies | [Manifest v1](reference/eval-manifest.md), [AKA workspace](reference/aka-workspace.md) |
| Built-in and custom SOL problems | [solution.json](reference/solution-json.md), [custom problems](reference/custom-sol-problem.md) |
| Multi-GPU collectives and NCU | [Collectives](reference/collectives.md) |
| Large OSS files, dependencies, asynchronous status, retries or diagnosis | [Jobs and troubleshooting](reference/jobs-and-troubleshooting.md) |
| Raw dev permissions, self-hosting, observations, migration and previews | [Operations boundaries](reference/operations.md) |
| Ray scaling, GPU floors, pause or environment handoff | [Capacity control](reference/capacity-control.md) |

Manual CLI, SDK and HTTP API documentation is served at `/sdk/cli/`, `/sdk/` and `/docs/api` respectively.
`/docs/api` provides static field documentation and JSON examples. Swagger/ReDoc and `/openapi.json` are not exposed.
