# AI Agents for Automated GPU Kernel Generation and Optimization: A Comprehensive Overview

A synthesis of technical discussions and system analyses from the Zhihu community on AI agents for automatically generating and optimizing GPU kernels. Covers representative systems, technical approach comparisons, performance data, and engineering practices.

> **Source note**: This document consolidates core knowledge from approximately 10 Zhihu articles related to AI kernel agents, after deduplication, filtering, and structured organization.

---

## 1. Background and Motivation

### 1.1 Why AI Kernel Agents Are Needed

GPU kernel optimization is central to deep learning infrastructure but faces two fundamental contradictions:

**Contradiction 1: Demand growth far exceeds talent supply**
- Each new hardware generation (Hopper → Blackwell → …) requires re-adapting kernels
- Operator fusion, new precision formats (FP8/FP4), and new attention variants (GQA/MLA) continuously emerge
- Engineers with low-level CUDA optimization skills are extremely scarce

**Contradiction 2: The gap between LLM code generation capability and kernel optimization capability**
- General-purpose LLMs can generate correct CUDA code, but performance lags far behind `torch.compile`
- Kernel optimization fundamentally involves hardware-level reasoning about memory access patterns, SM utilization, tiling strategies, occupancy scheduling, etc.—skills that must be learned through actual execution feedback
- Single-round code generation achieves a "faster rate" of only about 14%, whereas a complete agent loop can reach 96.8%

### 1.2 Classification of Technical Approaches

Current AI kernel optimization methods fall into three categories:

| Approach | Representative System | Core Idea | Strengths | Limitations |
|----------|----------------------|-----------|-----------|-------------|
| Agent + Evolutionary Search | AVO (NVIDIA) | LLM-driven autonomous evolutionary mutation | Discovers microarchitecture optimizations unknown to humans | Long search time (7 days), hardware dependency |
| Agent + Reinforcement Learning | CUDA Agent (ByteDance Seed) | Large-scale RL-trained system optimizer | Internalizes optimization strategies, strong generalization | Large training resource requirements, complex data synthesis |
| Agent + Engineering Pipeline | KernelFalcon (Meta/PyTorch) | Deterministic control + parallel verification | 100% correctness rate, end-to-end | Dependent on strong models, speed optimization not yet addressed |
| Knowledge Engineering | Skills / AutoKernel | Structured knowledge-driven agent | Reusable, auditable | Requires manual knowledge base maintenance |

---

## 2. Detailed Analysis of Representative Systems

### 2.1 AVO — Evolutionary Search + Autonomous Agent (NVIDIA)

**Paper**: Agentic Variation Operators for Autonomous Evolutionary Search (2025.03)

**Core Innovation**: Upgrades the "mutation/crossover" operations in evolutionary search from template-based operations to a complete agent engineering workflow.

**Workflow**:
1. **Planning**: The agent analyzes population lineage information + knowledge base (CUDA/PTX documentation, GPU architecture manuals, existing kernel code) and formulates an optimization plan
2. **Implementation**: Multi-file, structural code modifications (not simple line replacements)
3. **Evaluation**: The agent independently compiles, checks correctness, and measures throughput
4. **Bug Fixing**: Analyzes error messages → fixes → re-tests (closed-loop repair)
5. **Supervisor Agent**: Monitors the evolution trajectory, detects stagnation, and triggers exploratory mutations

**Key Results** (NVIDIA B200 GPU, Attention forward kernel):

| Metric | vs cuDNN | vs FlashAttention-4 |
|--------|----------|---------------------|
| MHA Non-causal | +1.2% | +5.7% |
| MHA Causal | **+3.5%** | **+10.5%** |
| GQA (30-min adaptation) | +7.0% | +9.3% |

**Three Microarchitecture Optimizations Independently Discovered by the Agent**:
- **Branchless Accumulator Rescaling**: Converts conditional branches in online softmax to unconditional computation (`scale = exp(old_max - new_max)`, scale=1 when max is unchanged), eliminating warp divergence, yielding an 8.1% improvement in Non-causal
- **Correction/MMA Pipeline Overlap**: Executes rescaling correction (occupying ALU) in parallel with wgmma instructions (occupying Tensor Core), leveraging instruction-level parallelism
- **Cross-Warp-Group Register Rebalancing**: Reallocates register usage to avoid register spilling

**Evolution Trajectory Characteristics**: 7 days, 40 generations in the official version, exploring 500+ candidates. Performance exhibits stepwise jumps—sudden breakthroughs after long plateaus, consistent with real-world engineering optimization patterns. Ultimately surpasses cuDNN and FA4 starting from a baseline far below them.

### 2.2 CUDA Agent — Reinforcement Learning-Driven (ByteDance Seed)

**Paper**: CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation (2025.02)

**Core Innovation**: Uses large-scale agentic RL to transform LLMs from code generators into system optimizers.

**Three Major Components**:

**1. Data Synthesis Pipeline**
- Seed operator crawling: Extracts operators from torch and transformers libraries
- Compositional synthesis: LLM samples ≤5 operators stacked sequentially to construct fusion scenarios (key insight: the optimization space of fused multi-operator kernels ≠ the sum of single-operator optimizations)
- Execution filtering: Excludes random operators, constant outputs, and overly simple/difficult tasks
- Final dataset: CUDA-Agent-Ops-6K (6,000 training tasks)

**2. Workflow-Integrated Agent Loop**
- Follows the ReAct paradigm: iterative cycles of reasoning → action → observation
- Standard workflow: Performance analysis → CUDA implementation → compilation verification → iterative optimization
- Discrete reward design: r ∈ {-1, 1, 2, 3} (error / correct but not faster / faster than eager / also faster than compile)

**3. Stable RL Training**
- Core challenge: CUDA tokens account for <0.01% of pretraining data, leading to exploding importance ratios, rising entropy, and reward collapse
- Dual warm-up strategy: Single-round RL → RFT (Rejection Sampling Fine-Tuning, retaining high-reward trajectories) → Value Pretraining (pretraining the critic)
- Ablation validation: Removing RFT → training collapse; removing Value Pretraining → trajectory length explosion

**Performance Results** (KernelBench, model Seed1.6 with 230B/23B active parameters):

| Metric | CUDA Agent | Claude Opus 4.5 | Gemini 3 Pro |
|--------|------------|-----------------|--------------|
| Pass Rate | 98.8% | 95.2% | 91.2% |
| Faster Rate (vs compile) | **96.8%** | 66% | 69% |
| Geo Mean Speed-up | **2.11x** | — | — |Level 2 (Fusion tasks): 100% faster rate, 2.80x vs compile.

**Limitations**: Not compared with more complex compiler frameworks like TVM; training relies on a large-scale GPU resource pool (128 H20 GPUs as profiling sandbox).

### 2.3 KernelFalcon — Deep Agent Architecture (Meta/PyTorch)

**Source**: PyTorch team's open-source framework

**Core Philosophy**: "Structure the problem, not the prompt harder" — control the workflow through deterministic Python orchestration code, rather than letting an LLM decide the flow.

**Four-stage Pipeline**:
1. **FuserAgent (Operator Fusion)**: Operates directly on PyTorch source code, preserving control flow and Python semantics, supporting if/else, while, and dynamic shapes
2. **ExtractorAgent (Subgraph Extraction)**: Analyzes the fused code, extracts subgraph shape information, and constructs stable signatures for deduplication
3. **Dispatcher + KernelAgent (Parallel Triton Kernel Synthesis)**: Multiple workers explore in parallel; the first success stops the others (Early Stop)
4. **ComposerAgent (End-to-end Integration Verification)**: Combines multiple kernels and verifies that the overall pipeline is consistent with the original PyTorch model

**Key Design Decisions**:
- **Deterministic Control**: Orchestration logic is hardcoded in explicit Python code
- **Anti-Cheating**: Triton JIT functions must not contain PyTorch ops or random inputs; execution is isolated in a subprocess
- **Signature Deduplication**: Subgraphs with identical op sequences + shapes + weights generate a kernel only once

**Results**: All 250 KernelBench tasks (L1/L2/L3) achieved **100% correctness**.

**Current Limitations**: The code repository is in an early stage of construction; relies heavily on the model — a higher failure rate when using DeepSeek V3.1; speed optimization evaluation has not been addressed.

---

## 3. Engineering Practice: The Skills Mechanism

### 3.1 What Are Skills

Repository workflow modules, popularized by Claude Code, are becoming a standard configuration in kernel development repositories. They are not simply prompts; they are **development interfaces within the repository** that define the interaction protocol between the agent and the code repository:
- **Task Routing**: Determine which implementation path a given requirement should follow
- **Process Decomposition**: Break complex tasks into sequentially executable stages
- **Tool Exposure**: Turn internal repository abstractions, scripts, and commands into capabilities callable by the Agent
- **Boundary Control**: Clearly define which directories and patterns must not be touched

### 3.2 Representative Skills Designs

**Hugging Face Kernels — Knowledge Manual Style**
- A single unified entry point covering: write kernel → benchmark → integrate with diffusers/transformers → GPU architecture specialization
- Provides differentiated optimization advice for H100/A100/T4
- Reference material: scenario guides + templates + troubleshooting

**SGLang — Engineering SOP Style**
- 4 independent Skills: JIT kernel / AOT kernel / CI regression localization / test writing
- First make a technical choice (JIT vs AOT), then dive into implementation details
- Exposes internal repository APIs (`TensorMatcher`, `LaunchKernel`, `AlignedVector`)

**MNN (Alibaba) — Multi-stage State Machine Style**
- Each workflow module is broken into multiple steps as stepN-xxx.md, each with clear pass criteria
- Example for adding a new operator: Schema → Shape inference → Geometry computation/CPU implementation → Testing → Multi-backend extension
- Strong constraints: access to `schema/private/` and `source/internal/` is prohibited

**FlashInfer — Lifecycle Closure Style**
- 3 Skills covering the full development loop: write kernel → test benchmark → debug crashes
- Tightly integrated with the TVM-FFI mechanism
- Debugging workflow treats "recording inputs before a crash" as a core methodology

### 3.3 Key Principles for Skills Design

Common patterns distilled from multiple projects:
1. **A good workflow module must include a verification step**: going beyond "how to implement" to include "how to verify"
2. **Knowledge should be front-loaded, not back-filled**: structure the maintainers' tacit knowledge (selection strategies, pitfalls) into the workflow upfront
3. **Constraints are as important as tools**: defining what the Agent must not do is equally critical as defining what it can call
4. **Granularity varies by project**: different projects require different levels of workflow-module granularity

---

## 4. Benchmarks and Platforms

### 4.1 KernelBench

The mainstream benchmark for AI-generated kernel evaluation:
- Level 1: Single operator (100 tasks)
- Level 2: Operator sequences/fusion (100 tasks)
- Level 3: Complete submodules (50 tasks)
- Metrics: Pass Rate, Faster Rate (vs torch.compile), Geometric Mean Speed-up

Limitations: includes many toy cases and small shapes, not fully aligned with production workloads.

### 4.2 SOL-ExecBench + SOLAR (NVIDIA)

NVIDIA's next-generation evaluation standard:
- **235 benchmark problems**, extracted from 124 production/emerging AI models
- Evaluation criteria based on hardware **SOL bound** — incorporates Einsum normalization and the Orojenesis algorithm to compute the theoretical peak
- **SOL score**: measures how much of the gap between the baseline and the SOL bound the candidate kernel closes
- Supports BF16/FP8/NVFP4, covering both forward and backward propagation

### 4.3 Practice Platforms

- **LeetGPU**: An international online evaluation platform for GPU kernels
- **XPU OJ**: A domestic localized platform supporting H800/A800/H20/L20, CUDA/Triton/TileLang, with extensibility Nadu-specific heterogeneous computing power buoyant; designed with an agent auto-submission interface

---

## 5. Limitations and Outlook

### 5.1 Current Limitations

**Limited Search Space**: Current systems are primarily validated on Attention forward kernels and standard KernelBench tasks, and have not yet been extended to backward kernels, more complex production operators, or full-model optimization.

**Strong Hardware Dependency**: Whether optimization strategies discovered on specific GPUs (such as B200) can be directly transferred to other architectures (such as AMD MI300X, domestic accelerators) remains to be verified.

**High Computational Cost**: AVO's 7-day search + LLM API call costs, and CUDA Agent's 128-GPU profiling sandbox + large-scale RL training, all imply significant resource consumption.**Limited Interpretability**: The agent's complete decision-making process is difficult to explain—while it's possible to analyze what optimizations were discovered post-hoc, it's impossible to fully understand "why" the agent made a particular decision.

**Heavy Dependence on the Base Model**: When KernelFalcon uses non-top-tier models (such as DeepSeek V3.1), the failure rate increases significantly, indicating that the system's effectiveness still heavily depends on the capabilities of the underlying LLM.

### 5.2 Future Outlook

**From Single Kernel to End-to-End**: KernelFalcon is already exploring network-level optimization (operator fusion + subgraph extraction + kernel synthesis + end-to-end verification), which is a direction with greater practical value.

**Fusion of Knowledge-Driven and Search-Driven Approaches**: Skills provide structured domain knowledge, while RL provides search and exploration capabilities. Combining the two may be the ultimate form—an agent that starts from a knowledge base and continuously evolves through execution feedback.

**Shift in the Kernel Engineer's Role**: Engineers may transition from "hand-writing kernels" to "designing the agent's knowledge base and verification framework"—setting goals and constraints while letting the agent carry out specific optimizations. As the author of AVO said: "Blind coding is the future of software engineering, and human cognitive ability is the bottleneck."

**Standardized Evaluation Driving Progress**: SOL-ExecBench's hardware roofline-based evaluation approach makes comparisons between different systems fairer and is expected to drive rapid iteration across the entire field.

---

## Related Documents

- [Community GPU Optimization Survey and Learning Path](gpu-optimization-survey.md) — GPU optimization methodology and learning path
- [Community CUDA Performance Fundamentals](cuda-performance-fundamentals.md) — CUDA practical experience and common pitfalls
- [Community Operator Optimization Cookbook](operator-optimization-cookbook.md) — GEMM/Attention/Norm and other operator optimization techniques
- [GPU Execution Model and Thread Optimization](gpu-execution-model.md) — In-depth explanation of warp scheduling and thread parallelism
- [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) — A systematic guide to memory optimization
