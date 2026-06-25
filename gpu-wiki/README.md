GPU Wiki is a GPU kernel programming and optimization knowledge base for AI Agents (Claude Code). It provides structured domain knowledge for LLMs, enabling Agents to make accurate technical decisions in tasks such as GPU kernel development, optimization, and cross-platform migration.

**Core Usage**: The Agent locates relevant files through the index table in `README.md`, and uses the `Read` tool to directly read markdown content to acquire knowledge.

## Overall Architecture

**Design Rationale**: GPU kernel optimization is highly architecture-dependent — the optimization direction for the same kernel can be completely opposite across different architectures (for example, the same tile size may be compute-bound on H20 but memory-bound on MI300X). The three-level classification ensures the Agent can precisely access knowledge for the target architecture, avoiding the incorrect application of experience from other architectures.
## How to Use

1. Based on the directory structure and keyword index below, navigate to the relevant directory
2. Read the `README.md` of that directoryXR_XaE to get the file list and descriptions
3. Use the `Read` tool to read specific markdown files
4. If unsure of the exact location, use `Grep` to search for keywords under `docs/` or `reference-kernels/`
5. For API interfaces, source-level implementation details, or ISA manuals, clone the corresponding upstream project listed in the `reference-project` section into `/tmp/reference-projects/` and inspect it there
6. See the [Document Relationship Diagram](docs/RELATIONS.md) to understand conflicts, complements, and reading order between files
```
gpu-wiki/
├── CLAUDE.md                  # Agent entry: complete index table for 152 documents
├── README.md                  # Project description + knowledge source list
├── docs/                      # Knowledge documents (152 markdown files)
│   ├── hardware-specs/        #   Hardware compute specification tables (all GPUs)
│   ├── kernel-opt/            #   Optimization knowledge (121 files)
│   │   ├── generic/           #     Vendor-agnostic general theory
│   │   ├── amd/               #     AMD architecture-specific
│   │   └── nvidia/            #     NVIDIA architecture-specific
│   ├── ref-docs/              #   Reference documentation
│   ├── pitfalls/              #   Implementation/porting pitfall records
│   ├── converter/             #   Conversion knowledge (30 files)
│   │   ├── generic/           #     PyTorch → Triton
│   │   ├── amd/               #     Triton → Gluon (AMD)
│   │   └── nvidia/            #     Triton → Gluon (NVIDIA)
│   └── RELATIONS.md           #   Document relationship graph
├── reference-kernels/         # Reference kernel source code (253 Python files)
│   ├── nvidia/                #   Ampere / Hopper
│   ├── amd/                   #   CDNA / RDNA4
│   └── generic/               #   Architecture-agnostic
└── scripts/                   # Utility scripts
```
---
## Search Priority Guidelines

When an agent receives a GPU kernel-related task, it should follow the layered retrieval strategy below. **Do not load all documents at once**; expand layer by layer as needed.

### Five-Tier Retrieval Priority

| Priority | Tier | When to Retrieve | Relevant Directories |
|--------|------|----------|----------|
| **P0** | Hardware Specs | **First — after determining the target GPU** — get peak TFLOPS, bandwidth, ridge point, architecture parameters | `docs/hardware-specs/hardware_specs_{mi300x,mi308x,hopper}.md` |
| **P1** | Vendor-General | **After determining the target vendor** — general optimizations for AMD or NVIDIA | AMD: `docs/kernel-opt/amd/common/`; NVIDIA: `docs/kernel-opt/nvidia/common/` |
| **P2** | Architecture-Specific | **After determining the target chip** — optimization points for specific models | Gluon series: `docs/kernel-opt/{amd,nvidia}/gluon/{gfx942,sm90}/`; General architecture: `docs/kernel-opt/amd/common/gfx942/` |
| **P3** | DSL-Specific | **After determining the target DSL** — framework-specific knowledge | CuTeDSL: `docs/kernel-opt/nvidia/cutedsl/`; FlyDSL: `docs/kernel-opt/amd/flydsl/`; use `/tmp/reference-projects/` for source-level API details when needed |
| **P4** | Optimization Experience | **After encountering optimization issues** — accumulated experience | `docs/pitfalls/`, `docs/ref-docs/` |
### Quick Routing for Special Tasks

| Task | Direct Path |
|------|----------|
| **Look up API signatures / source code** | Clone and inspect the corresponding upstream project under `/tmp/reference-projects/` |
| **Look up ISA instructions** | Use vendor public documentation (see Architecture & Programming References section below) or the relevant upstream project under `/tmp/reference-projects/` |
| **Cross-architecture migration** | `docs/RELATIONS.md` (conflict/difference checklist) → `docs/converter/` (conversion rules) |
| **Troubleshooting / debugging pitfalls** | `docs/pitfalls/` — organized into five steps: trap → symptom → reality → why → lesson |
| **Writing inline ASM** | CuTeDSL: `docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md` → `reference-kernels/README.md` inline PTX table; FlyDSL: `docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md` → `reference-kernels/README.md` inline_asm table || **Profiling analysis** | NVIDIA: `docs/ref-docs/nvidia/common/ncu-profiling-guide.md`, `docs/ref-docs/nvidia/common/ncu-profile-driven-optimization-workflow.md`; AMD: `docs/ref-docs/amd/common/rocprofv3-profiling-guide.md`; Gluon per architecture: `*-profiling_guide.md` |
| **Code conversion** | PyTorch→Triton: `docs/converter/generic/`; Triton→Gluon: `docs/converter/{amd,nvidia}/` |---

## docs/hardware-specs/ — Hardware Compute Specification Tables

Centralized hardware specification tables for all target GPUs, providing peak TFLOPS, memory bandwidth, roofline ridge points, and architecture parameters needed for compute utilization and roofline analysis.

| File | GPU | Key Specs |
|------|-----|-----------|
| [`hardware_specs_mi300x.md`](docs/hardware-specs/hardware_specs_mi300x.md) | AMD MI300X (CDNA3, gfx942) | 304 CU, 1307T BF16, 5.3 TB/s, ridge ~247 |
| [`hardware_specs_mi308x.md`](docs/hardware-specs/hardware_specs_mi308x.md) | AMD MI308X (CDNA3, gfx942) | 80 CU, 206T BF16, 5.3 TB/s, ridge ~39 |
| [`hardware_specs_hopper.md`](docs/hardware-specs/hardware_specs_hopper.md) | NVIDIA H100/H20/H200 (sm_90) | 989T BF16 (H100), 228KB smem, ridge ~295/37 |
---

## docs/ref-docs/ — Complete Reference Articles (141 Articles)

Structured reference documentation, complete optimization reports, and summary content, including detailed sections, code examples, and tables. Organized by vendor/arch/dsl; `kernel-opt/` only retains short knowledge points, pattern cards, and quick references. **Read README.md first.Foreign the file list.** (249 markdown files under docs/)

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/`](docs/ref-docs/generic/) | GPU general optimization theory + community knowledge (11 articles) | Memory hierarchy, execution model, instruction optimization, Amdahl's Law, GEMM optimization, operator optimization, CUDA basics, LLM inference, AI Agent kernel |
| **AMD** | | |
| [`amd/common/`](docs/ref-docs/amd/common/) | AMD general optimization (14 articles) | MFMA, aiter, Composable Kernel, CK-Tile, TileDistribution, GEMM/FMHA pipeline, MX quantization, MoE, ML dispatcher, rocprofv3 |
| [`amd/flydsl/`](docs/ref-docs/amd/flydsl/) | FlyDSL framework and AMD FlyDSL optimization reports (22 articles) | Topic index: compilation pipeline, layout algebra, API reference, kernel authoring, benchmark, inline_asm, MI300X optimization reports, attention/no-mask ISA/mask/mask+LSE SHARE_KV_LDS/pk_fma/occupancy/backward/chunk-GDN, FP16 no-mask CK95 gap, etc.; see directory README and specific articles for details |
| [`amd/gluon/gfx942/`](docs/ref-docs/amd/gluon/gfx942/) | MI300X Gluon (9 articles) | General optimization, ISA patterns, profiling, CK GEMM, warp pipeline, optimization conclusions/effect summary |
| **NVIDIA** | | |
| [`nvidia/common/`](docs/ref-docs/nvidia/common/) | NVIDIA general optimization (19 articles) | PTX ISA/MMA/sync, NCU profiling, profile-driven optimization workflow, SMEM swizzling, software pipeline, FP8 accumulation, tile rasterization, warp specialization, register pressure, hierarchical reduction, GPU architecture |
| [`nvidia/cuda/`](docs/ref-docs/nvidia/cuda/) | CUDA C++ / inline PTX (1 article) | CUDA Graph / E2E validation |
| [`nvidia/cutedsl/`](docs/ref-docs/nvidia/cutedsl/) | CuTeDSL / CUTLASS / QuACK (25 articles) | Topic index: Layout algebra, GEMM tiling, programming model, pipeline, architectural features, Stream-K, FMHA/MLA, Conv, quantization, EVT, QuACK, inline PTX, as well as `sm90/` subdirectory; see directory README and specific articles for details |
| [`nvidia/gluon/sm90/`](docs/ref-docs/nvidia/gluon/sm90/) | Hopper Gluon (7 articles) | General optimization, ISA patterns, hardware specs, linear attention, matmul, pitfalls, profiling |

---

## docs/kernel-opt/ — Optimization Quick Reference + Hands-on Patterns

Concise optimization pattern cards (20-100 lines), distilled from code reading. The hands-on/ directory contains hands-on pattern quick references. **Read README.md first for the file list.** (89 markdown files under kernel-opt/)

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/hands-on/`](docs/kernel-opt/generic/hands-on/) | Triton kernel optimization patterns (8 items) | autotune, persistent kernel, online softmax, flash attention, fused kernel, grouped GEMM, cascade merge, Mamba SSM |
| **AMD** | | |
| [`amd/common/`](docs/kernel-opt/amd/common/) | AMD optimization quick reference (9 items) | hardware comparison, occupancy, LDS bank conflict, Triton tuning, GEMM tuning, PyTorch tuning, RCCL, profiling tools |
| [`amd/common/hands-on/`](docs/kernel-opt/amd/common/hands-on/) | AMD optimization hands-on (10 items) | MFMA selection, LDS swizzle, preshuffle, async DMA, MoE fusion, RDNA4 WMMA |
| [`amd/common/gfx942/`](docs/kernel-opt/amd/common/gfx942/) | MI300X optimization key points (4 items) | Flash Attention TileLang, Grouped GEMM, Composable Kernel, MI300X kernel practices |
| [`amd/gluon/gfx942/`](docs/kernel-opt/amd/gluon/gfx942/) | MI300X Gluon optimization knowledge points (SKILL + pattern cards) | config template, strategy, pattern overview, SE zigzag; see `docs/ref-docs/amd/gluon/gfx942/` for summary content |
| [`amd/flydsl/gfx942/`](docs/kernel-opt/amd/flydsl/gfx942/) | FlyDSL MI300X optimization key points (1 item) | Fused MoE |
| **NVIDIA** | | |
| [`nvidia/common/`](docs/kernel-opt/nvidia/common/) | NVIDIA optimization quick reference | Compute Capability, L2 persistence, async copy, TMA, occupancy, thread block cluster |
| [`nvidia/common/hands-on/`](docs/kernel-opt/nvidia/common/hands-on/) | Hopper Optimization Hands-On | TMA, WGMMA, mbarrier pipeline, warp specialization |
| [`nvidia/common/sm90/hands-on/`](docs/kernel-opt/nvidia/common/sm90/hands-on/) | Hopper (SM90) Optimization Hands-On (13 items) | TMA, WGMMA, mbarrier pipeline, warp specialization, FlashMLA seesaw |
| [`nvidia/cutedsl/`](docs/kernel-opt/nvidia/cutedsl/) | CuTeDSL Optimization Insights and Quick Reference | Full optimization reports: `docs/ref-docs/nvidia/cutedsl/` |
| [`nvidia/gluon/sm90/`](docs/kernel-opt/nvidia/gluon/sm90/) | Hopper Gluon Optimization Essentials (2+SKILL) | fused attention, softmax/reduce |

---

## docs/pitfalls/ — Implementation/Porting Pitfall Records

Each pitfall follows a five-step process: trap → symptom → reality → why → lesson, specifically documenting "counter-intuitive pitfalls that others are likely to get wrong" and "approaches that were tried but reverted." Complements the optimization reports in `ref-docs/` (reports cover "why we did this + benefits"; pitfalls cover "why we didn't do this + lessons learned").

| Directory | Description | Keywords |
|------|------|--------|
| [`amd/flydsl/`](docs/pitfalls/amd/flydsl/) | FlyDSL on AMD (6 articles, 90 traps) | Topic index: flash-attn, no-mask ISA scheduling, FP16 CK95 gap, bit-packed mask, mask+LSE SHARE_KV_LDS/pk_fma/occupancy, attention backward, mask integration, version iteration, chunk-GDN, etc.; specific traps, symptoms, causes, and fixes — see directory README and individual articles |
| [`nvidia/cuda/`](docs/pitfalls/nvidia/cuda/) | CUDA on NVIDIA (1 article, 7 traps) | Topic index: CUDA pitfalls; specific traps, symptoms, causes, and fixes — see directory README and individual articles |
| [`nvidia/cutedsl/`](docs/pitfalls/nvidia/cutedsl/) | CuTeDSL on NVIDIA (8 articles, 54 traps) | Topic index: GDN chunk fwd, TMA, GDN decode, NVFP4 GEMM, INT32 MoE data preparation, Fused FA-epilogue + NVFP4 quant, etc.; specific traps, symptoms, causes, and fixes — see directory README and individual articles |

---

## docs/converter/ — Converter Tool Knowledge

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/`](docs/converter/generic/) | PyTorch→Triton Conversion | API mapping, conversion rules, model configuration |
| [`amd/common/`](docs/converter/amd/common/) | Triton→Gluon General Rules | Porting rules, API mapping, learning guide, verification guide |
| [`amd/cdna3/`](docs/converter/amd/cdna3/) | CDNA3-Specific Triton→Gluon (7 items) | API mapping, pipeline, matrix_multiply, memory_access, layouts, pitfalls |
| [`nvidia/hopper/`](docs/converter/nvidia/hopper/) | Hopper-Specific Triton→Gluon (7 items) | wgmma, CP_ASYNC |

---

## reference-kernels/ — Reference Kernel Code

Python/CUDA GPU kernel implementations extracted from open-source repositories and local optimization projects, organized by **hardware architecture → framework/language → source project**. See the `README.md` in each directory and [`reference-kernels/README.md`](reference-kernels/README.md) for detailed file listings. (304 .py files in reference-kernels/)

| Directory | DSL | Description |
|------|-----|------|
| `nvidia/ampere/` | CuTeDSL, Gluon, Triton | SM80 (A100): CUTLASS GEMM, Flash Attention, DeepGEMM |
| `nvidia/hopper/` | CuTeDSL, Gluon | SM90 (H100/H20): CUTLASS, FlashInfer norm/GDN/Mamba, TMA/WGMMA |
| `amd/cdna/` | FlyDSL, Triton | CDNA3 general: FlyDSL GEMM/Attention/Norm/MoE + aiter Triton kernels (80+) |
| `amd/cdna3/` | FlyDSL | CDNA3 (gfx942) hardware-specific tuning: MI308X Flash Attention / no-mask / mask related reference kernels; see directory README for details |
| `amd/rdna4/` | FlyDSL, Gluon | RDNA4 (gfx1250): WMMA GEMM, Flash Attention |
| `generic/` | Triton, Gluon | Triton tutorials, triton-kernels library, Flash Attention, FlashInfer |### Search by Kernel Type

| Kernel Type | Where to Find |
|-------------|----------------|
| **GEMM / MatMul** | `nvidia/*/cutedsl/cutlass/`, `amd/cdna/flydsl/`, `amd/cdna/triton/aiter/gemm/`, `generic/triton/triton-tutorials/` |
| **Attention** | `nvidia/*/cutedsl/cutlass/fmha*`, `nvidia/*/cutedsl/cutlass/mla/`, `amd/cdna/triton/aiter/attention/`, `amd/cdna/flydsl/FlyDSL/flash_attn_func.py` (CDNA general), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py` (MI308X tuning + GQA), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_nomask_mi308x.py` (MI308X + BF16 no-mask), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_fp16_nomask_mi308x.py` (MI308X + FP16 no-mask CK95 gap), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py` (MI308X + free mask + LSE), `amd/cdna/flydsl/FlyDSL/sage_attn_flydsl.py`, `generic/triton/flash-attention/` |
| **Norm / Softmax** | `nvidia/hopper/cutedsl/flashinfer/`, `amd/cdna/triton/aiter/normalization/`, `amd/cdna/flydsl/FlyDSL/`, `generic/triton/triton-tutorials/` |
| **MoE** | `amd/cdna/triton/aiter/moe/`, `amd/cdna/flydsl/FlyDSL/moe_*.py` |
| **Quantization (FP8/FP4)** | `amd/cdna/triton/aiter/quant/` |
| **SSM / Mamba** | `nvidia/hopper/cutedsl/flashinfer/ssd_kernel.py` |

---

## reference-project — Upstream Source Projects

When the information in `docs/` or `reference-kernels/` is insufficient to determine implementation details, API behavior, performance patterns, or the latest code, the agent may `git clone` the upstream source repositories below for source-level study. Prefer shallow clones with `--depth 1` into a temporary directory, such as `/tmp/reference-projects/`. Treat cloned repositories as read-only references, and do not commit temporary source code into this repository.

**Usage Priority**:

1. First check the in-repository `docs/` and `reference-kernels/`
2. If ground truth, latest implementation details, or broader context is needed, clone the corresponding upstream project
3. After cloning, prefer `grep` / `rg` / direct file reading to locate relevant kernels, DSL APIs, scheduling strategies, and profiling clues
4. The temporary directory may be kept or cleaned up after the task, but do not update the gpu-wiki index to point to local temporary paths

| Repository | Git URL | Language / Framework | Description | Architectures |
|------|---------|-------------|------|----------|
| CUTLASS | `https://github.com/NVIDIA/cutlass.git` | Python (CuTeDSL) | Official NVIDIA CUTLASS CuTeDSL examples | Ampere, Hopper |
| cutex | `https://github.com/deciding/cutex.git` | Python (CuTeDSL) | Blackwell CuTeDSL GEMM/FA4 implementations | Blackwell |
| cuLA | `https://github.com/inclusionAI/cuLA.git` | Python (CuTeDSL) + Triton | Linear attention / KDA kernels | Blackwell |
| flash-attention | `https://github.com/Dao-AILab/flash-attention.git` | Python (CuTeDSL) + Triton | Flash Attention v4 implementations for multiple architectures | Ampere, Hopper, AMD |
| FlashInfer | `https://github.com/flashinfer-ai/flashinfer.git` | Python (CuTeDSL) + Triton | Inference acceleration library, including GDN decode, Norm, MLA, and MoE | Hopper |
| FlyDSL | `https://github.com/ROCm/FlyDSL.git` | Python (FlyDSL) | AMD FlyDSL framework kernel examples | CDNA3, RDNA4 |
| Triton | `https://github.com/triton-lang/triton.git` | Python (Triton/Gluon) | Official Triton/Gluon tutorials and the triton_kernels library | Generic |
| DeepGEMM | `https://github.com/deepseek-ai/DeepGEMM.git` | Triton + CUDA C++ | DeepSeek GEMM library, including Triton legacy kernels and sm90 C++ kernels | Ampere, Hopper |
| FlashMLA | `https://github.com/deepseek-ai/FlashMLA.git` | CUDA C++ | DeepSeek MLA decode, including Seesaw scheduling, DSM crossover, Split-KV, and TMA Gather | Hopper |
| composable_kernel | `https://github.com/ROCm/composable_kernel.git` | C++ (HIP) | AMD Composable Kernel template library, including TensorDescriptor transform trees | CDNA3 |
| cute-gemm | `https://github.com/reed-lau/cute-gemm.git` | CUDA C++ | CuTe C++ GEMM implementations | Hopper |
| hpc-ops | `https://github.com/Tencent/hpc-ops.git` | CUDA C++ | High-performance CUDA operator library | Hopper |
| aiter | `https://github.com/ROCm/aiter.git` | Python (Triton/Gluon) + C++ | Official AMD AI inference operator library, including Attention, GEMM, MoE, Norm, and Quant kernels | CDNA3 |
| QuACK | `https://github.com/Dao-AILab/quack.git` | Python (CuTeDSL) | Dao-AILab high-performance kernel library, including Reduction, GEMM, MLP, and TopK kernels, targeting around 90% SOL | Hopper |
| TileLang | `https://github.com/tile-ai/tilelang.git` | Python (TileLang DSL) | Unified GPU kernel DSL for CUDA/HIP backends, with CuTeDSL contrib inline PTX utilities | Hopper, CDNA3 |

Example command:

```bash
mkdir -p /tmp/reference-projects
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/reference-projects/cutlass
```

---

## Architecture & Programming References — Vendor Official Documentation

Official vendor documentation including ISA manuals, programming guides, and architecture references for low-level instruction lookup, inline assembly authoring, programming-model understanding, and performance modeling.

| Vendor | Scope | Document | Type | Link |
|--------|-------|----------|------|------|
| AMD | CDNA3 (MI300 Series) | AMD Instinct MI300 CDNA3 Instruction Set Architecture | ISA Manual | [PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) |
| AMD | HIP (General) | HIP Programming Guide — Programming Model | Programming Guide | [Link](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) |