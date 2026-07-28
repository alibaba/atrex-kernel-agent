# Atrex Kernel Agent

AKA is an end-to-end Agent project for GPU kernel implementation, analysis, profiling, and iterative optimization. It helps an Agent turn PyTorch logic or an existing kernel into a high-performance GPU kernel through a structured, profile-driven workflow.

![Atrex architecture](assets/atrex-architecture.png)

![Atrex optimization loop](assets/atrex-optimization-loop.png)

## News

- [2026-07] We helped **Qwen3.8** rank **No. 1** on the **SOL-ExecBench FlashInfer operator optimization leaderboard**. [[Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)]
- [2026-07] We released **Atrex Kernel Agent v0.2.0** with a dual-route optimization system, an orchestrated clean-session loop, native SOL-ExecBench operator workflow, Triton-to-Gluon conversion support, and a fuller NVIDIA profiling toolchain. [[Release](https://github.com/alibaba/atrex-kernel-agent/releases/tag/v0.2.0)]
- [2026-07] We released **the Atrex paper**: [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541).
- [2026-06] We released **Atrex Kernel Agent v0.1.0** as the initial open-source version, with the interactive `gpu-kernel-optimizer` Skill route, GPU Wiki knowledge base, profile-driven optimization workflow, profiling tools, and reference templates. [[Release](https://github.com/alibaba/atrex-kernel-agent/releases/tag/v0.1.0)]

## What It Does

- Creates an isolated optimization workspace under `kernel_opt_<name>/`.
- Looks up target hardware specs from the local `gpu-wiki` knowledge base.
- Runs Roofline analysis and sets auditable performance targets.
- Implements a correct baseline kernel before entering optimization.
- Runs the profile-driven optimization loop: profile with `ncu` or `rocprofv3`, extract bottleneck evidence, query `gpu-wiki` / reference projects / web sources for relevant optimization knowledge, write an evidence-based plan, apply one optimization category, validate correctness and performance, record memory, commit, then repeat until Stop Conditions are met.
- Records plans, profile artifacts, structured memory, reports, and Git commits for every accepted iteration.

For the full architecture and workflow design, see [`docs/design.md`](docs/design.md).

## Quick Start

See the [Quick Start guide](docs/quickstart.md) for prerequisites, installation, and complete runnable paths for both the interactive Skill route and the orchestrated loop route.

## Optimization Routes

| Route | Driver | Termination | Best for |
|-------|--------|-------------|----------|
| [Route 1: Interactive Skill](#route-1-interactive-skill-skillmd) | `gpu-kernel-optimizer` Skill + hooks, invoked inside a coding session | In-session judgment, guarded by hooks | Hands-on, interactive optimization from a coding runtime |
| [Route 2: Orchestrated Loop](#route-2-orchestrated-loop-orchestratoroptimizepy) | `orchestrator/optimize.py`, spawning fresh clean sessions per iteration | Mechanical (max iterations / token budget / target utilization) | Unattended, budget-bounded, batch optimization |

Both routes share the same knowledge base (`gpu-wiki/`), reference projects, tools (`tools/`), and structured memory format (`memory/v<N>.json`).

## Route Details

### Route 1: Interactive Skill (`SKILL.md`)

This route installs the `gpu-kernel-optimizer` Skill and workflow hooks into your coding runtime. You then drive the optimization interactively from a coding session, and the hooks keep the workflow on track (memory reads, plan reads, correctness gates, stop-condition checks).

The optimization workspace `kernel_opt_<name>/` is created **in the current working directory** where you run the session, so all artifacts stay next to where you are working.

Internal users should configure git `insteadOf` URL redirect rules so that submodules and dependencies resolve against the internal network before running `git submodule update`. **External users can skip this step entirely.**

The install path is optional; defaults to `~/aka_kernel_opt`.

Common installer options:

```bash
bash install.sh --prefix ~/my_path    # Install to a custom directory
bash install.sh --hooks-only          # Install or update hooks only
bash install.sh --without-github      # Skip GitHub-hosted reference repos
bash install.sh --uninstall           # Remove hooks installed by this script
```

The installer detects supported runtime home directories and prepares local hooks when available. It ships **only** the Skill route; the orchestrator route (Route 2) runs from the source repo and is pruned from the installed skill directory.

### Route 2: Orchestrated Loop (`orchestrator/optimize.py`)

![route2 optimization loop](assets/optimize_workflow.png)

This route runs the optimization loop from the source repo without installing anything into your coding runtime. `orchestrator/optimize.py` owns the **outer loop** and spawns a fresh, clean Claude or Qoder CLI session for each iteration over the same git workspace. Select the backend with `--agent-cli claude|qodercli` (default: `claude`). State crosses the session boundary only through disk (`memory/v<N>.json`, `plans/`, `profiles/`, and git), and HEAD is always the best kernel.

Correctness/performance validation and profiling run on an atrex-gpu-gateway sandbox selected by
`--sandbox-hardware`. The gateway worker receives code and test/profile inputs only: optimizer `memory/`, plans,
edits, and Git state remain local. Structured test results and profile analysis artifacts are returned to the
local session. The same transport can be used directly:

```bash
python tools/sandbox.py --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
python tools/sandbox.py --hardware REMOTE_GPU --sync profiles/v1 -- \
  bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1 --source

# Same interface on the bundled localhost FIFO scheduler
# Start it first with: python tools/local_gateway.py serve
python tools/sandbox.py --hardware local --url http://127.0.0.1:8000 \
  --no-sync -- python test_kernel.py --no-memory
```

Local gateway mode preserves the request/packaging/result interface but is not a security sandbox:
submitted commands run directly as the server user. The bundled scheduler serializes jobs by default,
persists their status in SQLite, and speaks the same public `agate dev`/jobs API. See
[docs/local_gateway.md](docs/local_gateway.md) for startup, queue, cancellation, and compatibility details.

Termination is **mechanical**, not left to in-session judgment: the loop stops on a hard budget (max iterations or token budget) or a target-utilization short-circuit on a committed, correctness-passing iteration.

Everything op-specific (workspace name, the reference to optimize, the full workload/shape set, per-workload tolerances) is read from the SOL-ExecBench `--op-dir`; the ground-truth files (`definition.json`, `reference.py`, `workload.jsonl`) are used verbatim and never edited. `--platform` is required. In the default `leaderboard` mode, `--framework` may select one framework explicitly; when omitted, the orchestrator launches independent campaigns in parallel for Triton/CuteDSL/Cuda on NVIDIA, Triton/FlyDSL on AMD, or Triton on unknown hardware. A version that passes `test_kernel.py` in its workspace is directly submittable to SOL-ExecBench.

Key options:

```bash
--max-iters N        # Hard cap on optimization iterations
--token-budget N     # Hard token cap across all sessions (0 = no cap)
--agent-cli CLI      # Optimization session backend: claude (default) or qodercli
--optimization-mode MODE # leaderboard (default) or production
--framework DSL      # One explicit DSL; omit to parallel-dispatch all supported DSLs
--target-util PCT    # Peak-utilization %% short-circuit (default 90)
--sandbox-hardware GPU # agate selector/alias; independent of the logical --platform name
--sandbox-profile P  # Optional pre/prod endpoint; default uses agate config
--sandbox-url URL    # Explicit endpoint; use http://127.0.0.1:8000 with hardware=local
--sandbox-timeout S  # Remote command timeout, max 600 seconds
--workspace DIR      # Working directory for the campaign (default: current directory)
--max-stall N        # Stop after N consecutive no-commit iterations (0 = disabled)
--convert-after N    # Triton only: after N stalled iters, run one Triton->Gluon convert session
--arch ARCH          # Override auto-detected runtime arch, e.g. sm_103 or gfx942
```

Auto-dispatched campaigns use flat framework/hardware suffixes; for example,
`<workspace>/kernel_opt_<name>_triton_h20` and
`<workspace>/kernel_opt_<name>_cutedsl_h20`. Each campaign receives its own full iteration and
token budgets. Explicit `--framework` campaigns use the same naming convention.

`--optimization-mode leaderboard` preserves the existing permissive `CLAUDE.md` workflow: sessions may
use a different/mixed implementation or third-party kernel libraries when profiling evidence supports it.
`--optimization-mode production` also supports omitted `--framework`: the orchestrator auto-dispatches the
hardware-supported frameworks and binds every child campaign to its assigned framework. V0 may remain the
PyTorch correctness baseline, but every accepted optimized candidate must be implemented directly and
exclusively in that child's framework. Third-party kernel/operator imports, calls, and solution dependencies are forbidden. A mechanical
post-session gate rejects and reverts non-compliant kernel commits, records a `production_policy_rejection`,
and refuses to package a non-compliant final kernel. Triton-to-Gluon conversion is disabled in production
mode because the selected framework is exact.

```bash
python orchestrator/optimize.py \
  --op-dir /path/to/op --platform TARGET_GPU --sandbox-hardware REMOTE_GPU \
  --optimization-mode production --framework Triton
```

`--platform` is a logical optimization target while `--sandbox-hardware` is the gateway selector. The
orchestrator deliberately does not compare their names or reported GPU models because gateway inventory
may be aliased or desensitized. Runtime architecture probing remains authoritative when an omitted
`--framework` requires vendor-specific dispatch.

Both backends run non-interactively with a fresh session ID and the same workspace-local skills,
agents, prompts, sandbox constraints, and quality gates. Authenticate the selected CLI first with
`claude auth status` or `qodercli status`. Provider-specific settings can be supplied through
`ATREX_CLAUDE_SESSION_SETTINGS` or `ATREX_QODER_SESSION_SETTINGS`; `ATREX_SESSION_SETTINGS` remains
the generic fallback. Some Qoder models report zero token usage in stream JSON; in that case
`--token-budget` cannot be enforced and `--max-iters` remains the hard campaign bound.

## Main Files

```text
.
├── SKILL.md                         # Route 1: gpu-kernel-optimizer router manifest
├── install.sh                       # Route 1 installer / uninstaller
├── orchestrator/                    # Route 2: clean-session optimization orchestrator
│   ├── optimize.py                  # Outer optimization loop driver
│   └── prompts/                     # Per-session prompts (setup, iteration, convert)
├── agents/                          # Subagent definitions used by both routes
├── docs/                            # Detailed project design docs
├── reference/                       # Workspace, plan, memory, and profiling templates
├── skills/                          # Baseline, optimizer, restart, and output-contract modules
├── tools/                           # Profiling, utilization, memory, and measurement tools
└── gpu-wiki/                        # Local GPU knowledge base
```

## Acknowledgements

This project builds on and references many excellent open-source works. We gratefully acknowledge the authors and communities behind them.

Reference kernel projects (`reference-projects/`):

- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA Templates for Linear Algebra Subroutines
- [cutex](https://github.com/deciding/cutex) — CUDA Template Extensions
- [cuLA](https://github.com/inclusionAI/cuLA) — inclusionAI CUDA Linear Algebra
- [flash-attention](https://github.com/Dao-AILab/flash-attention) — Flash Attention
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Kernel library for LLM serving
- [FlyDSL](https://github.com/ROCm/FlyDSL) — ROCm FlyDSL
- [Triton](https://github.com/triton-lang/triton) — Triton language and compiler
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — DeepSeek DeepGEMM
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) — CUDA learning kernels
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — DeepSeek FlashMLA
- [Composable Kernel](https://github.com/ROCm/composable_kernel) — ROCm Composable Kernel
- [cute-gemm](https://github.com/reed-lau/cute-gemm) — CuTe GEMM examples
- [hpc-ops](https://github.com/Tencent/hpc-ops) — Tencent HPC Ops
- [aiter](https://github.com/ROCm/aiter) — ROCm AIter
- [quack](https://github.com/Dao-AILab/quack) — Dao-AILab Quack
- [tilelang](https://github.com/tile-ai/tilelang) — TileLang

Knowledge base and tooling (`gpu-wiki/3rdparty/`, `3rdparty/`):

- [KernelWiki](https://github.com/mit-han-lab/KernelWiki) — GPU kernel knowledge base
- [modern-gpu-programming-for-mlsys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys) — Modern GPU programming for MLSys
- [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) — Nsight Compute report parsing skill
- [humanize](https://github.com/PolyArch/humanize) — Plan generation plugin
- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — AKO4ALL
- [KDA](https://github.com/mit-han-lab/kernel-design-agents) — Kernel Design Agents

## Citation

Please cite our [paper](https://arxiv.org/abs/2607.14541) if it is helpful to your research.

```bibtex
@misc{atrex2026,
  title         = {Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent},
  author        = {Lingyun Yang and Yuxiao Wang and Shenghao Liang and Linfeng Yang and Daocheng Ying and Chunbo You and Rui Zhang and Luping Wang and Yinghao Yu and Guodong Yang and Liping Zhang},
  year          = {2026},
  eprint        = {2607.14541},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2607.14541}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
