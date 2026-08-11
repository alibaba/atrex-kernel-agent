# Atrex Kernel Agent

AKA is an end-to-end Agent system for GPU kernel implementation, profiling, and iterative
optimization. The current repository exposes one supported optimization entry point,
`orchestrator/optimize.py`; the native `long_horizon/` package is its internal episode engine,
not a second CLI.

![Atrex architecture](assets/atrex-architecture.png)

## News

- [2026-08] We slimmed down **Atrex Kernel Agent** by consolidating on a single orchestrated workflow and removing legacy paths and redundant context for a smaller context footprint and lower token usage.
- [2026-07] We helped **Qwen3.8** rank **No. 1** on the **SOL-ExecBench FlashInfer operator optimization leaderboard**. [[Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)]
- [2026-07] We released **Atrex Kernel Agent v0.2.0** with an orchestrated clean-session loop, native SOL-ExecBench operator workflow, Triton-to-Gluon conversion support, and a fuller NVIDIA profiling toolchain. [[Release](https://github.com/alibaba/atrex-kernel-agent/releases/tag/v0.2.0)]
- [2026-07] We released **the Atrex paper**: [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541).
- [2026-06] We released **Atrex Kernel Agent v0.1.0** as the initial open-source version, with the GPU Wiki knowledge base, profile-driven optimization workflow, profiling tools, and reference templates. [[Release](https://github.com/alibaba/atrex-kernel-agent/releases/tag/v0.1.0)]

## Current Design

- Accepts SOL-ExecBench operators and native Atrex-Bench shape operators.
- Creates one isolated Git workspace per framework and target, with separate campaign state for
  leaderboard and production optimization.
- Establishes a correctness-passing V0 and, by default in production mode, a self-contained
  framework-native V1 before optimization begins.
- Runs one Long Horizon campaign over the complete workload set in both modes.
- Lets each episode perform multiple profile/research/plan/edit/repair cycles, while the
  supervisor alone owns budgets, terminal validation, same-allocation ABBA verification, and
  squash promotion.
- Preserves Git history, canonical `memory/v<N>.json`, plans, profiler evidence, episode journals,
  verification artifacts, and aggregation provenance for recovery and audit.

For the full architecture and workflow design, see [`docs/design.md`](docs/design.md).

## Quick Start

See the [Quick Start guide](docs/quickstart.md) for prerequisites and complete runnable examples of the orchestrated optimization loop.

Or start a coding agent such as Claude Code, Codex, or Qoder in this repository and ask it to
launch an AKA optimization task. We recommend the following prompt:

```text
Use AKA's orchestrator/optimize.py to start one optimization task for atrex-bench/xx. Put the workspace under ~/aka-opt, set the platform to H20, use the local sandbox, use claude as the Agent CLI, set max-iters to 300, specify cuda as the framework, and run in production mode.
```

## Orchestrated Optimization

`orchestrator/optimize.py` is the repository's only supported optimization entry point. It owns
mechanical termination, state recovery, Agent session isolation, sandbox execution, workload
coordination, and final packaging.

![orchestrated optimization loop](assets/optimize_workflow.png)

```text
operator inputs
  -> V0 correctness baseline
  -> optional framework-native V1
  -> Long Horizon episode worktree
  -> live memory + journal + terminal handoff
  -> policy/protected-path checks + ABBA verification
  -> squash promotion
  -> finalization
```

Each canonical version is explored in an isolated Git branch and worktree. A fresh Claude, Qoder,
Codex, or Pi session owns one Long Horizon episode and may execute multiple engineering cycles before
publishing a structured terminal handoff. The supervisor validates the journal and candidate, runs
incumbent/candidate ABBA verification in one gateway allocation, and squash-promotes only a strict
correctness-passing improvement.

The uncommitted `memory/live.json` appears when an episode starts and refreshes after every journaled
experiment. It is an observability view, not promotion evidence; only `memory/v<N>.json` is canonical.

SOL and native Atrex-Bench campaigns optimize and validate the complete workload set together.

GPU validation and profiling execute through the configured gateway, while optimization memory,
plans, edits, episode state, and Git history remain local. Repository-scoped skills are prepared
inside each campaign workspace, and campaign termination remains mechanically controlled by explicit
budgets and promotion gates.

For prerequisites, runnable commands, backend configuration, operating modes, common options, local
gateway setup, and direct sandbox usage, see the [Quick Start guide](docs/quickstart.md). For the full
architecture and workflow design, see [docs/design.md](docs/design.md).

## Main Files

```text
.
├── orchestrator/                    # Public optimization entry and shared policy
│   ├── optimize.py                  # Long Horizon campaign driver
│   ├── agent_runtime/               # Claude/Qoder/Codex/Pi backend adapters
│   ├── telemetry/                   # Phase token aggregation
│   └── prompts/                     # Setup, inspection, baseline, and episode prompts
├── long_horizon/                    # Internal episode/worktree/ABBA engine
├── agents/                          # Workspace-local baseline Agent definition
├── docs/                            # Detailed project design docs
├── reference/                       # Workspace init, evaluator adapters, schemas, SOL packaging
├── reference-projects/              # Optional source-search repositories used by episodes
├── skills/                          # Workspace-local baseline skill used by Agent sessions
├── tools/                           # Sandbox, local gateway, profiling, memory, and measurement tools
├── gpu-wiki/                        # Architecture-scoped GPU knowledge base
└── 3rdparty/                        # Runtime planning and profiler-analysis dependencies
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
