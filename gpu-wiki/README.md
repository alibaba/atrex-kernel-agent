GPU Wiki is source knowledge and reference material for GPU kernel optimization, backend APIs, hardware behavior, implementation pitfalls, and local references. It provides structured domain knowledge for GPU kernel development, optimization, profiling, and cross-platform migration.

This source wiki does not define Atrex Server, CLI, task runtime input/output filenames, or task execution protocol. Those runtime contracts are injected by downstream projection and prompt layers; source content should use neutral terms such as input implementation, candidate implementation, optimized implementation, benchmark harness, and local reference material.

**Core Usage**: Start from this README, choose the smallest relevant directory, then read the linked Markdown or source files directly. Use any available file-reading and text-search tools; the wiki itself does not require a specific product or runtime.

## Overall Architecture

**Design Rationale**: GPU kernel optimization is highly architecture-dependent — the optimization direction for the same kernel can be completely opposite across different architectures (for example, the same tile size may be compute-bound on H20 but memory-bound on MI355X). The three-level classification makes it easier to access knowledge for the target architecture and avoid applying experience from the wrong hardware family.

## How to Use

1. Identify the target GPU, vendor, architecture, DSL, and kernel type.
2. Use the directory structure and routing tables below to choose the narrowest relevant directory.
3. Read that directory's `README.md` before opening individual files.
4. Search within `docs/`, `reference-kernels/`, and `reference-projects/` when the exact location is unclear.
5. Treat `reference-kernels/` and `reference-projects/` as local reference material for learning and adaptation.
6. For API signatures, type annotations, class behavior, decorator usage, or ISA details, prefer vendor official documentation or cloned upstream sources in `~/aka_kernel_opt/reference-projects/` over derived prose documentation.
7. For candidate implementation examples, check the usability status in `reference-kernels/README.md` and the nested README before assuming the code can run unchanged.
8. Examples are not guaranteed to be directly importable runtime dependencies; adapt them to the consuming benchmark harness and runtime contract.

## Search Priority Guidelines

For GPU-kernel work, follow the layered retrieval strategy below. **Do not load all documents at once**; expand layer by layer as needed.

### Five-Tier Retrieval Priority

| Priority | Tier | When to Retrieve | Relevant Directories |
|--------|------|----------|----------|
| **P0** | Hardware Specs | **First — after determining the target GPU** — get peak TFLOPS, bandwidth, ridge point, architecture parameters | `docs/hardware-specs/hardware_specs_{mi300x,mi308x,mi355x,hopper,blackwell}.md` |
| **P1** | Vendor-General | **After determining the target vendor** — general optimizations for AMD or NVIDIA | AMD: `docs/kernel-opt/amd/common/`; NVIDIA: `docs/kernel-opt/nvidia/common/` |
| **P2** | Architecture-Specific | **After determining the target chip** — optimization points for specific models | NVIDIA Blackwell: `docs/kernel-opt/nvidia/common/blackwell/`; Gluon series: `docs/kernel-opt/{amd,nvidia}/gluon/{gfx942,gfx950,sm90}/`; General architecture: `docs/kernel-opt/amd/common/{gfx942,gfx950}/` |
| **P3** | DSL-Specific | **After determining the target DSL** — framework-specific knowledge | CuTeDSL: `docs/kernel-opt/nvidia/cutedsl/`, `reference-kernels/`; FlyDSL: `docs/kernel-opt/amd/flydsl/`, `reference-kernels/` |
| **P4** | Optimization Experience | **After encountering optimization issues** — accumulated experience | `docs/pitfalls/`, `docs/ref-docs/` |
| **P5** | External References | When P0–P4 do not cover a topic: vendor official docs for ISA/API precision, or community wikis for alternative perspective | Vendor docs (see Architecture & Programming References below); `3rdparty/` — KernelWiki, modern-gpu-programming |

**Retrieval Flow**:

```
0. Identify target GPU → Load P0 (hardware specs: peak TFLOPS, ridge point, memory BW)
1. Identify target vendor → Load P1 (AMD common or NVIDIA common)
2. Identify target architecture → Load P2 (gfx942 / gfx950 / sm90, etc.)
3. Identify target DSL → Load P3 (CuTeDSL / FlyDSL / Gluon)
4. Optimization issues → Load P4 (trap → symptom → why → lesson)
5. Topic gaps, ISA/API precision, or alternative perspective → Load P5 (vendor official docs + 3rdparty community wikis)
```

### Quick Routing for Special Tasks

| Task | Direct Path |
|------|----------|
| **Look up API signatures / source code** | `~/aka_kernel_opt/reference-projects/` — clone upstream framework source (see reference-projects section below), or use vendor official API documentation |
| **Look up ISA instructions** | Use vendor public documentation (see Architecture & Programming References section below) or clone the relevant upstream project under `~/aka_kernel_opt/reference-projects/` |
| **Cross-architecture migration** | `docs/RELATIONS.md` (conflict/difference checklist) → `docs/converter/` (conversion rules) |
| **Troubleshooting / debugging pitfalls** | `docs/pitfalls/` — organized into five steps: trap → symptom → reality → why → lesson |
| **Writing inline ASM** | CuTeDSL: `docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md` → `reference-kernels/README.md` inline PTX table; FlyDSL: `docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md` → `reference-kernels/README.md` inline_asm table |
| **Profiling analysis** | NVIDIA: `docs/ref-docs/nvidia/common/ncu-profiling-guide.md`, `docs/ref-docs/nvidia/common/ncu-profile-driven-optimization-workflow.md`, `docs/ref-docs/nvidia/common/ncu-measurement-discipline.md` (trusting profile/timing numbers); AMD: `docs/ref-docs/amd/common/rocprofv3-profiling-guide.md`; Gluon per architecture: `*-profiling_guide.md` |
| **Code conversion** | PyTorch→Triton: `docs/converter/generic/`; Triton→Gluon: `docs/converter/{amd,nvidia}/` |

### Symptom-Driven Retrieval (NVIDIA vs AMD)

When a profiler reports a bottleneck symptom, translate it into search keywords
*here* before grepping — NVIDIA and AMD use different vocabularies and different
sub-trees, so the retrieval logic is vendor-split.

**NVIDIA** — start from the P1 index `docs/kernel-opt/nvidia/common/` (P2 arch
`sm90` / `blackwell/`, P3 DSL `cutedsl/`), then
`grep -ri "<keyword>" docs/kernel-opt/nvidia/`. The NVIDIA `ncu` /
`classify_ncu.py` `SYMPTOMS` line is controlled vocabulary that maps directly to
these keywords:

| Profiler symptom | gpu-wiki search keywords |
|---|---|
| `memory-bound` | coalesced access, vectorized load, async copy / TMA, cp.async |
| `low-sm-utilization` | occupancy, persistent kernel, split-k, tile rasterization |
| `register-pressure` | register spill, register pressure, occupancy tuning |
| `compute-bound` | WGMMA / tcgen05, warp specialization, MMA pipeline |
| `pipeline-stalls` | software pipeline, double buffering, mbarrier |

**AMD** — search `docs/kernel-opt/amd/` with:

- Memory issues: `coalesced access`, `vectorized load`, `async copy`, `double buffering`
- Compute issues: `MMA`, `MFMA`, `ILP`, `warp specialization`
- General issues: `bank conflict`, `register spill`, `occupancy`, `synchronization`

For a `FlyDSL` target, additionally study the `CuTeDSL` implementation of the
same kernel type under `reference-kernels/nvidia/`,
then map the optimization ideas onto AMD hardware.

---

## docs/hardware-specs/ — Hardware Compute Specification Tables

Centralized hardware specification tables for all target GPUs, providing peak TFLOPS, memory bandwidth, roofline ridge points, and architecture parameters needed for compute utilization and roofline analysis.

| File | GPU | Key Specs |
|------|-----|-----------|
| [`hardware_specs_mi300x.md`](docs/hardware-specs/hardware_specs_mi300x.md) | AMD MI300X (CDNA3, gfx942) | 304 CU, 1307T BF16, 5.3 TB/s, ridge ~247 |
| [`hardware_specs_mi308x.md`](docs/hardware-specs/hardware_specs_mi308x.md) | AMD MI308X (CDNA3, gfx942) | 80 CU, 206T BF16, 5.3 TB/s, ridge ~39 |
| [`hardware_specs_mi355x.md`](docs/hardware-specs/hardware_specs_mi355x.md) | AMD MI355X (CDNA4, gfx950) | 256 CU, 160KB LDS, ridge ~629 |
| [`hardware_specs_hopper.md`](docs/hardware-specs/hardware_specs_hopper.md) | NVIDIA H100/H20/H200 (sm_90) | 989T BF16 (H100), 228KB smem, ridge ~295/37 |
| [`hardware_specs_b200.md`](docs/hardware-specs/hardware_specs_b200.md) | NVIDIA B200 (GB200 / Blackwell / sm_100) | 160 SM, 2250T BF16, 8.0 TB/s, ridge ~281 |
| [`hardware_specs_b300.md`](docs/hardware-specs/hardware_specs_b300.md) | NVIDIA B300 (GB300 / Blackwell Ultra / sm_103) | 160 SM, 2250T BF16, 8.0 TB/s, 288GB HBM3e, ridge ~281 |
| [`hardware_specs_sm120.md`](docs/hardware-specs/hardware_specs_sm120.md) | NVIDIA RTX PRO 6000/5000 (sm_120) | Blackwell GeForce / RTX PRO |
| [`hardware-comparison-cdna3-cdna4.md`](docs/hardware-specs/hardware-comparison-cdna3-cdna4.md) | Cross-architecture | CDNA3 vs CDNA4 vs RDNA4 parameter comparison |

---

## docs/ref-docs/ — Complete Reference Articles (142 Articles)

Structured reference documentation, complete optimization reports, and summary content, including detailed sections, code examples, and tables. Organized by vendor/arch/dsl; `kernel-opt/` only retains short knowledge points, pattern cards, and quick references. **Read README.md first for the file list.**

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/`](docs/ref-docs/generic/) | GPU general optimization theory + community knowledge (11 articles) | Memory hierarchy, execution model, instruction optimization, Amdahl's Law, GEMM optimization, operator optimization, CUDA basics, LLM inference, AI Agent kernel |
| **AMD** | | |
| [`amd/common/`](docs/ref-docs/amd/common/) | AMD general optimization (14 articles) | MFMA, aiter, Composable Kernel, CK-Tile, TileDistribution, GEMM/FMHA pipeline, MX quantization, MoE, ML dispatcher, rocprofv3 |
| [`amd/flydsl/`](docs/ref-docs/amd/flydsl/) | FlyDSL framework and AMD FlyDSL optimization reports (22 articles) | Topic index: compilation pipeline, layout algebra, API reference, kernel authoring, benchmark, inline_asm, MI300X/MI355X optimization reports, attention/no-mask ISA/mask/mask+LSE SHARE_KV_LDS/pk_fma/occupancy/backward/chunk-GDN, FP16 no-mask CK95 gap, etc.; see directory README and specific articles for details |
| [`amd/gluon/gfx942/`](docs/ref-docs/amd/gluon/gfx942/) | MI300X Gluon (9 articles) | General optimization, ISA patterns, profiling, CK GEMM, warp pipeline, optimization conclusions/effect summary |
| [`amd/gluon/gfx950/`](docs/ref-docs/amd/gluon/gfx950/) | MI355X Gluon (7 articles) | matmul, MLA decode, pitfalls, profiling, softmax/reduce, chunk-GDN summary |
| **NVIDIA** | | |
| [`nvidia/common/`](docs/ref-docs/nvidia/common/) | NVIDIA general optimization (20 articles) | PTX ISA/MMA/sync, NCU profiling, profile-driven optimization workflow, measurement discipline (Duration≠latency, noise, graph-capture), SMEM swizzling, software pipeline, FP8 accumulation, tile rasterization, warp specialization, register pressure, hierarchical reduction, GPU architecture |
| [`nvidia/cuda/`](docs/ref-docs/nvidia/cuda/) | CUDA C++ / inline PTX (3 articles) | SM120 NVFP4 Split-K GEMV, consolidated decode/prefill GEMM production lessons, RMSNorm-MLP PDL handoff no-go; CUDA Graph / E2E validation |
| [`nvidia/cutedsl/`](docs/ref-docs/nvidia/cutedsl/) | CuTeDSL / CUTLASS / QuACK (25 articles) | Topic index: Layout algebra, GEMM tiling, programming model, pipeline, architectural features, Stream-K, FMHA/MLA, Conv, quantization, EVT, QuACK, inline PTX, as well as `sm90/`, `sm100/`, `sm120/` subdirectories; see directory README and specific articles for details |
| [`nvidia/gluon/sm90/`](docs/ref-docs/nvidia/gluon/sm90/) | Hopper Gluon (7 articles) | General optimization, ISA patterns, hardware specs, linear attention, matmul, pitfalls, profiling |

---

## docs/kernel-opt/ — Optimization Quick Reference + Hands-on Patterns

Concise optimization pattern cards (20-100 lines), distilled from code reading. The hands-on/ directory contains hands-on pattern quick references. **Read README.md first for the file list.**

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/hands-on/`](docs/kernel-opt/generic/hands-on/) | Triton kernel optimization patterns (8 items) | autotune, persistent kernel, online softmax, flash attention, fused kernel, grouped GEMM, cascade merge, Mamba SSM |
| **AMD** | | |
| [`amd/common/`](docs/kernel-opt/amd/common/) | AMD optimization quick reference (9 items) | hardware comparison, occupancy, LDS bank conflict, Triton tuning, GEMM tuning, PyTorch tuning, RCCL, profiling tools |
| [`amd/common/hands-on/`](docs/kernel-opt/amd/common/hands-on/) | AMD optimization hands-on (10 items) | MFMA selection, LDS swizzle, preshuffle, async DMA, MoE fusion, RDNA4 WMMA |
| [`amd/common/gfx942/`](docs/kernel-opt/amd/common/gfx942/) | MI300X optimization key points (4 items) | Flash Attention TileLang, Grouped GEMM, Composable Kernel, MI300X kernel practices |
| [`amd/gluon/gfx942/`](docs/kernel-opt/amd/gluon/gfx942/) | MI300X Gluon optimization guide and pattern cards | config template, strategy, pattern overview, SE zigzag; see `docs/ref-docs/amd/gluon/gfx942/` for summary content |
| [`amd/gluon/gfx950/`](docs/kernel-opt/amd/gluon/gfx950/) | MI355X Gluon optimization guide and key points | fused attention, hardware specs |
| [`amd/flydsl/gfx942/`](docs/kernel-opt/amd/flydsl/gfx942/) | FlyDSL MI300X optimization key points (2 items) | Fused MoE W4A16, FP8 PTPC checkpoint |
| [`amd/flydsl/gfx950/`](docs/kernel-opt/amd/flydsl/gfx950/) | FlyDSL MI355X optimization knowledge point index | Complete optimization reports have been migrated to `docs/ref-docs/amd/flydsl/gfx950/`; this directory only retains quick reference/index entries |
| **NVIDIA** | | |
| [`nvidia/common/`](docs/kernel-opt/nvidia/common/) | NVIDIA optimization quick reference (including 51 Blackwell knowledge cards) | Compute Capability, L2 persistence, async copy, TMA, occupancy, thread block cluster, Blackwell hardware mechanisms/kernel/migration/bottleneck patterns/optimization techniques |
| [`nvidia/common/blackwell/`](docs/kernel-opt/nvidia/common/blackwell/) | Blackwell / Hopper kernel optimization knowledge cards (51 cards) | tcgen05, TMEM, TMA, mbarrier, CLC, NVFP4, FlashMLA, DeepGEMM, MoE, WGMMA→tcgen05, register→TMEM, vectorized loads, warp specialization |
| [`nvidia/common/hands-on/`](docs/kernel-opt/nvidia/common/hands-on/) | Blackwell (SM100) Optimization Hands-On (11 items) | tcgen05/TMEM, three-role warp specialization, CLC, 2CTA, block-scaled MMA |
| [`nvidia/common/sm90/hands-on/`](docs/kernel-opt/nvidia/common/sm90/hands-on/) | Hopper (SM90) Optimization Hands-On (13 items) | TMA, WGMMA, mbarrier pipeline, warp specialization, FlashMLA seesaw |
| [`nvidia/cutedsl/`](docs/kernel-opt/nvidia/cutedsl/) | CuTeDSL Optimization Insights and Quick Reference | Subdirectory: `sm120/`; full optimization reports: `docs/ref-docs/nvidia/cutedsl/` |
| [`nvidia/gluon/sm90/`](docs/kernel-opt/nvidia/gluon/sm90/) | Hopper Gluon optimization guide and essentials | fused attention, softmax/reduce |

---

## docs/pitfalls/ — Implementation/Porting Pitfall Records

Each pitfall follows a five-step process: trap → symptom → reality → why → lesson, specifically documenting "counter-intuitive pitfalls that others are likely to get wrong" and "approaches that were tried but reverted." Complements the optimization reports in `ref-docs/` (reports cover "why we did this + benefits"; pitfalls cover "why we didn't do this + lessons learned").

| Directory | Description | Keywords |
|------|------|--------|
| [`amd/flydsl/`](docs/pitfalls/amd/flydsl/) | FlyDSL on AMD (6 articles, 90 traps) | Topic index: flash-attn, no-mask ISA scheduling, FP16 CK95 gap, bit-packed mask, mask+LSE SHARE_KV_LDS/pk_fma/occupancy, attention backward, mask integration, version iteration, chunk-GDN, etc.; specific traps, symptoms, causes, and fixes — see directory README and individual articles |
| [`nvidia/cuda/`](docs/pitfalls/nvidia/cuda/) | CUDA on NVIDIA (3 articles, 24 traps) | Topic index: SM120 NVFP4 Split-K GEMV, decode/prefill GEMM production pitfalls, RMSNorm-MLP PDL handoff pitfalls; see directory README and individual articles |
| [`nvidia/cutedsl/`](docs/pitfalls/nvidia/cutedsl/) | CuTeDSL on NVIDIA (8 articles, 54 traps) | Topic index: GDN chunk fwd, TMA, GDN decode, NVFP4 GEMM, INT32 MoE data preparation, Fused FA-epilogue + NVFP4 quant, etc.; specific traps, symptoms, causes, and fixes — see directory README and individual articles |
| [`nvidia/triton/`](docs/pitfalls/nvidia/triton/) | Triton on NVIDIA (2 articles, 17 traps) | Topic index: sm_120 fused RMSNormGated + SiLU gating; sm_100 DSA sparse-decode split-K (split-K structural win, online-softmax NaN guard, num_stages/static_range, cross-CTA atomic barrier, clusters ⊥ atomics, cache/eviction levers, `.item()` barrier, tl.sort/histogram); see directory README and individual articles |
| [`nvidia/gluon/`](docs/pitfalls/nvidia/gluon/) | Gluon on NVIDIA (1 article, 6 traps) | Topic index: sm_100 Blackwell primitives — `gl.dot_fma` vs `bw.tcgen05_mma`, dot_fma layout/dtype rules, tcgen05 small-H padding, `gl.barrier` not callable, explicit layouts, `translator_helpers`; see directory README and individual articles |

---

## docs/converter/ — Converter Tool Knowledge

| Directory | Description | Keywords |
|------|------|--------|
| [`generic/`](docs/converter/generic/) | PyTorch→Triton Conversion | API mapping, conversion rules, model configuration |
| [`amd/common/`](docs/converter/amd/common/) | Triton→Gluon General Rules | Porting rules, API mapping, learning guide, verification guide |
| [`amd/cdna3/`](docs/converter/amd/cdna3/) | CDNA3-Specific Triton→Gluon (7 items) | API mapping, pipeline, matrix_multiply, memory_access, layouts, pitfalls |
| [`amd/cdna4/`](docs/converter/amd/cdna4/) | CDNA4-Specific Triton→Gluon (7 items) | async copy DMA, mfma_scaled |
| [`nvidia/hopper/`](docs/converter/nvidia/hopper/) | Hopper-Specific Triton→Gluon (7 items) | wgmma, CP_ASYNC |

---

## reference-kernels/ — Local Reference Kernel Code

Python/CUDA GPU kernel implementations extracted from open-source repositories and local optimization projects, organized by **hardware architecture -> framework/language -> source project**. This directory is local reference material: study candidate implementation patterns, then adapt them to the consuming benchmark harness and runtime contract. See the `README.md` in each directory and [`reference-kernels/README.md`](reference-kernels/README.md) for detailed file listings.

| Directory | DSL | Description |
|------|-----|------|
| `nvidia/ampere/` | CuTeDSL, Gluon, Triton | SM80 (A100): CUTLASS GEMM, Flash Attention, DeepGEMM |
| `nvidia/hopper/` | CuTeDSL, Gluon | SM90 (H100/H20): CUTLASS, FlashInfer norm/GDN/Mamba, TMA/WGMMA |
| `nvidia/blackwell/` | CuTeDSL, Gluon, Triton | SM100 (B200): CUTLASS GEMM/MLA/MoE, tcgen05, CLC |
| `nvidia/blackwell-geforce/` | CuTeDSL, Triton, CUDA | SM120: CUTLASS, Flash Attention, FlashInfer, GDN chunk fwd, NVFP4 Split-K / CTA-3D TMA, prefill, RMSNorm-MLP PDL diagnostics; see directory README for details |
| `amd/cdna/` | FlyDSL, Triton | CDNA3+CDNA4 general: FlyDSL GEMM/Attention/Norm/MoE + aiter Triton kernels (80+) |
| `amd/cdna3/` | FlyDSL | CDNA3 (gfx942) hardware-specific tuning: MI308X Flash Attention / no-mask / mask related reference kernels; see directory README for details |
| `amd/cdna4/` | FlyDSL, Gluon | CDNA4 (gfx950): FlyDSL chunk-GDN, Gluon matmul, aiter GEMM/PA, and other reference kernels; see directory README for details |
| `amd/rdna4/` | FlyDSL, Gluon | RDNA4 (gfx1250): WMMA GEMM, Flash Attention |
| `generic/` | Triton, Gluon | Triton tutorials, triton-kernels library, Flash Attention, FlashInfer, LeetCUDA |

### Search by Kernel Type

| Kernel Type | Where to Find |
|-------------|----------------|
| **GEMM / MatMul** | `nvidia/*/cutedsl/cutlass/`, `nvidia/blackwell-geforce/cuda/nvfp4_*`, `nvidia/blackwell-geforce/cutedsl/flashinfer/dense_blockscaled_gemm_sm120_task39_diagnostic.py`, `amd/cdna/flydsl/`, `amd/cdna/triton/aiter/gemm/`, `amd/cdna4/gluon/`, `generic/triton/triton-tutorials/` |
| **Attention** | `nvidia/*/cutedsl/cutlass/fmha*`, `nvidia/*/cutedsl/cutlass/mla/`, `amd/cdna/triton/aiter/attention/`, `amd/cdna/flydsl/FlyDSL/flash_attn_func.py` (CDNA general), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py` (MI308X tuning + GQA), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_nomask_mi308x.py` (MI308X + BF16 no-mask), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_fp16_nomask_mi308x.py` (MI308X + FP16 no-mask CK95 gap), `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py` (MI308X + free mask + LSE), `amd/cdna/flydsl/FlyDSL/sage_attn_flydsl.py`, `generic/triton/flash-attention/` |
| **Norm / Softmax** | `nvidia/hopper/cutedsl/flashinfer/`, `nvidia/blackwell-geforce/cuda/rmsnorm_mlp_nvfp4_pdl/`, `amd/cdna/triton/aiter/normalization/`, `amd/cdna/flydsl/FlyDSL/`, `generic/triton/triton-tutorials/` |
| **MoE** | `nvidia/blackwell/cutedsl/flashinfer/`, `amd/cdna/triton/aiter/moe/`, `amd/cdna/flydsl/FlyDSL/moe_*.py`, `amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/` |
| **Quantization (FP8/FP4)** | `nvidia/blackwell/cutedsl/flashinfer/*quantize.py`, `nvidia/blackwell-geforce/cuda/rmsnorm_mlp_nvfp4_pdl/`, `amd/cdna/triton/aiter/quant/` |
| **SSM / Mamba** | `nvidia/blackwell/cutedsl/cutlass/mamba2_ssd/`, `nvidia/hopper/cutedsl/flashinfer/ssd_kernel.py` |

---

## Architecture & Programming References — Vendor Official Documentation

Official vendor documentation including ISA manuals, programming guides, and architecture references for low-level instruction lookup, inline assembly authoring, programming-model understanding, and performance modeling.

| Vendor | Scope | Document | Type | Link |
|--------|-------|----------|------|------|
| AMD | CDNA3 (MI300 Series) | AMD Instinct MI300 CDNA3 Instruction Set Architecture | ISA Manual | [PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) |
| AMD | HIP (General) | HIP Programming Guide — Programming Model | Programming Guide | [Link](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) |
| NVIDIA | PTX (General) | NVIDIA PTX ISA Reference | ISA Manual | [Link](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) |
| NVIDIA | Blackwell (SM100) | NVIDIA Blackwell Tuning Guide | Tuning Guide | [Link](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) |
| NVIDIA | Blackwell (SM100) | NVIDIA Blackwell Compatibility Guide | Compatibility Guide | [Link](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) |
| NVIDIA | Cross-Architecture | NVIDIA CUTLASS Documentation | Library Documentation | [Link](https://docs.nvidia.com/cutlass/latest/) |
| NVIDIA | Cross-Architecture | CUDA C++ Programming Guide | Programming Guide | [Link](https://docs.nvidia.com/cuda/cuda-programming-guide/) |

---

## 3rdparty/ — Community GPU Kernel Knowledge (P5)

External open-source GPU kernel wikis managed as git submodules.
Consult as **P5 priority** — after exhausting `docs/` (P0–P4) and `reference-kernels/`.

| Repository | Maintainer | Focus | Hardware | Best For |
|---|---|---|---|---|
| **KernelWiki** | MIT HAN Lab (Song Han) | Production-grade kernel optimization patterns from 2179 PRs | NVIDIA Blackwell (SM100), Hopper (SM90) | Performance diagnostics, optimization pattern matching, production kernel reference |
| **modern-gpu-programming-for-mlsys** | MLC-AI (Tianqi Chen) + CMU | Systematic GPU programming textbook: hardware → programming model → SOTA GEMM | NVIDIA Blackwell, Hopper, Ampere | Foundational learning, Roofline model, TMA pipeline, GEMM optimization theory |
