GPU Wiki is source knowledge and reference material for GPU kernel optimization, backend APIs, hardware behavior, implementation pitfalls, and local references. It provides structured domain knowledge for GPU kernel development, optimization, profiling, and cross-platform migration.

This source wiki does not define Atrex Server, CLI, task runtime input/output filenames, or task execution protocol. Those runtime contracts are injected by downstream projection and prompt layers; source content should use neutral terms such as input implementation, candidate implementation, optimized implementation, benchmark harness, and local reference material.

**Core Usage**: Start from this README, choose the smallest relevant directory, then read the linked Markdown or source files directly. Use any available file-reading and text-search tools; the wiki itself does not require a specific product or runtime.

**Other entry points**: flat catalog of every page → [`docs/index.md`](docs/index.md); maintenance schema and Ingest/Query/Lint operations → [`CLAUDE.md`](CLAUDE.md); change history → [`log.md`](log.md); architecture-scoped search → `python3 gpu-wiki/scripts/query.py "<keywords>" --arch <blackwell|hopper|cdna3|cdna4|...>` (restricts to one architecture and never leaks a competing one; `--vendor`/`--dsl` narrow further, `--list-arch` lists values).

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

For GPU-kernel work, expand layer by layer; do not load everything at once.

| Priority | Tier | Where |
|--------|------|-------|
| **P0** | Hardware specs | `docs/nvidia/common/hardware-specs/` (hopper/b200/b300/sm120), `docs/amd/hardware-specs/` (mi300x/mi308x/mi355x) |
| **P1** | Vendor-general | NVIDIA cross-arch: `docs/nvidia/common/` (ptx, profiling, cutedsl fundamentals, theory); AMD: `docs/amd/common/` |
| **P2** | Architecture | NVIDIA: `docs/nvidia/{hopper,blackwell,blackwell-geforce}/`; AMD: `docs/amd/{common,gluon,flydsl}/{gfx942,gfx950}/` |
| **P3** | DSL-specific | CuTeDSL/Gluon/Triton/CUDA under the matching arch dir; CuTeDSL fundamentals under `docs/nvidia/common/cutedsl/`; FlyDSL under `docs/amd/flydsl/` |
| **P4** | Pitfalls | each topic's `pitfalls/` subfolder, e.g. `docs/nvidia/blackwell-geforce/cutedsl/pitfalls/`, `docs/amd/flydsl/pitfalls/` |
| **P5** | External refs | vendor official docs (below) + `3rdparty/` community wikis |

## Repository Layout

`docs/` is vendor-first. Within **nvidia/** the second level is **architecture-first**; amd/ and generic/ stay topic-first.

```
docs/
├── generic/      theory + hands-on/ + converter/
├── amd/          common/(+gfx942,gfx950,hands-on)  flydsl/  gluon/  converter/  hardware-specs/
└── nvidia/
    ├── common/              cross-arch: ptx/  profiling/  cutedsl/ (CUTLASS/CuTe fundamentals)
    │                        gluon/  triton/  hardware-specs/  + general theory cards
    ├── hopper/        (sm90)  cutedsl/  gluon/  converter/  hands-on/
    ├── blackwell/     (sm100) hardware/ techniques/ kernels/ patterns/ languages/ migration/
    │                          hands-on/  articles/  cutedsl/  gluon/  triton/
    └── blackwell-geforce/ (sm120) cutedsl/  cuda/  triton/
```

### nvidia/ — architecture-first index

| Dir | Scope | Holds |
|-----|-------|-------|
| [`nvidia/common/`](docs/nvidia/common/) | Cross-arch (not tied to one GPU gen) | PTX ISA (`ptx/`), NCU/Nsight profiling (`profiling/`), CuTeDSL/CUTLASS fundamentals (`cutedsl/`), Gluon/Triton general notes, hardware-specs, general optimization theory cards |
| [`nvidia/hopper/`](docs/nvidia/hopper/) | Hopper sm90 | CuTeDSL, Gluon, Triton→Gluon converter, non-DSL hands-on (TMA/WGMMA/mbarrier/warp-spec) |
| [`nvidia/blackwell/`](docs/nvidia/blackwell/) | Blackwell datacenter sm100 (B200/B300) | structured knowledge tree (hardware/techniques/kernels/patterns/languages/migration), `hands-on/` (CuTeDSL examples), `articles/` (FA4/B300 deep-dives), CuTeDSL/Gluon, Triton pitfalls |
| [`nvidia/blackwell-geforce/`](docs/nvidia/blackwell-geforce/) | Blackwell GeForce / RTX PRO sm120 | CuTeDSL, CUDA, Triton (each with `pitfalls/`): NVFP4 GEMM/GEMV, GDN chunk-fwd/decode, MoE data-prep, RMSNorm-MLP PDL |

### amd/ and generic/

| Dir | Holds |
|-----|-------|
| [`amd/common/`](docs/amd/common/) | MFMA, CK, rocprofv3, roofline, LDS, occupancy + `hands-on/` + `gfx942/` `gfx950/` |
| [`amd/flydsl/`](docs/amd/flydsl/) | FlyDSL framework + MI300X/MI355X reports (`gfx942/` `gfx950/`) + `pitfalls/` |
| [`amd/gluon/`](docs/amd/gluon/) | MI300X/MI355X Gluon (`gfx942/` `gfx950/`) |
| [`amd/converter/`](docs/amd/converter/) | Triton→Gluon: `common/` + `cdna3/` + `cdna4/` |
| [`generic/`](docs/generic/) | vendor-agnostic theory + `hands-on/` (Triton patterns) + `converter/` (PyTorch→Triton) |

### Quick routing

| Task | Path |
|------|------|
| Writing inline ASM | CuTeDSL: `docs/nvidia/common/cutedsl/cutedsl-inline-ptx-patterns.md`; FlyDSL: `docs/amd/flydsl/flydsl-inline-asm-patterns.md` |
| Profiling | NVIDIA: `docs/nvidia/common/profiling/`; AMD: `docs/amd/common/rocprofv3-profiling-guide.md` |
| Cross-architecture migration | `docs/RELATIONS.md` → `docs/{nvidia/hopper,amd}/converter/` |
| Pitfalls | the `pitfalls/` subfolder inside the relevant arch/topic |

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

