# GPU Wiki Knowledge Base Design Notes

## 1. Project Positioning

GPU Wiki is a GPU kernel programming and optimization knowledge base designed for AI Agents. It is not a conventional documentation site for human readers; it is a structured technical memory that helps Agents make accurate decisions during GPU kernel development, optimization, profiling, and cross-platform migration.

Core usage:

1. The Agent enters through `README.md`, `CLAUDE.md`, and directory-level `README.md` files.
2. The Agent selects the target knowledge area by architecture, DSL/framework, and task type.
3. The Agent reads concise pattern cards in `docs/kernel-opt/` first, then follows links to full reports in `docs/ref-docs/`, upstream source clones in reference-projects/ (in /tmp/), or runnable examples in `reference-kernels/` when deeper evidence is needed.

The repository is intentionally optimized for Agent retrieval: files are split by topic, directory names encode architecture/framework context, and relationship documents call out conflicts between architectures.

## 2. Current Repository Layout

```text
gpu-wiki/
├── AGENTS.md                  # Agent instruction: read README.md first
├── CLAUDE.md                  # Agent-oriented global index and navigation entry
├── README.md                  # Project overview, usage policy, source list, and module index
├── gpu-wiki-design.md         # This design note
├── docs/                      # Curated markdown knowledge base
│   ├── README.md              # docs/ module index
│   ├── RELATIONS.md           # Reading paths, cross-architecture relations and conflicts
│   ├── hardware-specs/        # Hardware compute specification tables
│   ├── kernel-opt/            # Concise optimization quick references and hands-on cards
│   ├── ref-docs/              # Full reference articles and optimization journeys
│   ├── pitfalls/              # Counter-intuitive implementation and porting pitfalls
│   └── converter/             # PyTorch→Triton and Triton→Gluon conversion knowledge
├── reference-kernels/         # Runnable/reference kernel implementations
│   ├── amd/
│   ├── generic/
│   └── nvidia/
└── .skill/                    # Repository maintenance skills and automation docs
```

Current scale, based on the working tree at the time this file was rewritten:

| Category | Count |
|----------|------:|
| Total markdown files | 983 |
| `docs/` markdown files | 254 |
| `docs/ref-docs/` markdown files | 109 |
| `docs/kernel-opt/` markdown files | 89 |
| `docs/converter/` markdown files | 30 |
| `docs/pitfalls/` markdown files | 21 |
| `docs/hardware-specs/` markdown files | 3 |
| `reference-kernels/` Python files | 304 |

## 3. Organization Principles

### 3.1 Architecture and Framework Are First-Class Dimensions

GPU kernel optimization is highly architecture-dependent. The same technique can be beneficial on one GPU and harmful on another because of differences in memory bandwidth, tensor-core/MFMA shape, LDS/shared-memory behavior, pipeline primitives, occupancy constraints, and profiling bottlenecks.

Therefore, knowledge is organized by a combined hierarchy:

```text
Domain → Vendor → Framework/DSL → Architecture → Topic
```

Representative examples:

```text
docs/kernel-opt/amd/gluon/gfx942/
docs/ref-docs/amd/flydsl/
docs/kernel-opt/nvidia/common/sm90/hands-on/
docs/ref-docs/nvidia/cutedsl/
reference-kernels/amd/cdna3/flydsl/
reference-kernels/nvidia/hopper/cutedsl/
```

This structure prevents Agents from accidentally applying NVIDIA Hopper experience to AMD CDNA3/CDNA4 kernels, or applying FlyDSL-specific implementation details to Gluon or CuTeDSL tasks.

### 3.2 Short Cards First, Full Reports Second

The repository separates quick decision support from long-form evidence:

- `docs/kernel-opt/`: concise optimization cards, pattern summaries, and hands-on notes. These files are meant to be read early during problem solving.
- `docs/ref-docs/`: complete reference articles, deep optimization reports, profiling workflows, framework notes, and detailed implementation journeys. These files are read when the Agent needs justification, exact constraints, or richer examples.
- `docs/pitfalls/`: negative knowledge. These files explain what failed, why it failed, and when not to use a technique.
- `reference-kernels/`: concrete runnable or near-runnable kernel implementations.

This layered design keeps common lookups fast while preserving enough source-level evidence for difficult optimization and migration tasks.

### 3.3 One File, One Focus

Most topic files are intentionally narrow. A single file should cover one optimization technique, one pitfall group, one conversion rule set, one hardware table, or one framework topic.

This reduces the amount of irrelevant text an Agent must read and makes retrieval safer. Broad overview pages are placed in directory-level `README.md` files, while implementation details live in focused child documents.

## 4. Main Knowledge Domains

### 4.1 `docs/hardware-specs/` — Hardware Compute Specification Tables

`docs/hardware-specs/` centralizes hardware facts needed for roofline analysis and compute-utilization reasoning.

Current files:

| File | Scope |
|------|-------|
| `hardware_specs_mi300x.md` | AMD MI300X / CDNA3 / gfx942 hardware specification table |
| `hardware_specs_mi308x.md` | AMD MI308X / CDNA3 / gfx942 hardware specification table |
| `hardware_specs_hopper.md` | NVIDIA Hopper / SM90 hardware specification table |

These files provide peak FLOPS, memory bandwidth, ridge points, and architecture parameters used by profiling and optimization documents.

### 4.2 `docs/kernel-opt/` — Optimization Quick References and Hands-On Cards

`docs/kernel-opt/` contains concise optimization knowledge. It is the first stop when an Agent needs a practical direction for kernel optimization.

Key subtrees:

| Directory | Role |
|-----------|------|
| `generic/hands-on/` | Vendor-agnostic Triton and GPU optimization patterns |
| `amd/common/` | AMD common optimization references: occupancy, LDS, RCCL, profiling, GEMM tuning |
| `amd/common/hands-on/` | AMD hands-on optimization cards such as MFMA selection, LDS swizzle, async DMA, MoE fusion |
| `amd/common/gfx942/` | MI300X-oriented optimization notes and practice cases |
| `amd/gluon/gfx942/` | MI300X Gluon optimization skill and pattern cards |
| `amd/flydsl/gfx942/` | FlyDSL MI300X optimization notes |
| `nvidia/common/` | NVIDIA common optimization references: compute capability, L2 persistence, async copy, TMA, occupancy |
| `nvidia/common/hands-on/` | NVIDIA hands-on cards for TMA, WGMMA, mbarrier pipeline, warp specialization |
| `nvidia/common/sm90/hands-on/` | Hopper-specific hands-on cards |
| `nvidia/cutedsl/` | CuTeDSL optimization insights and quick references |
| `nvidia/gluon/sm90/` | Hopper Gluon optimization essentials |

The design intent is that an Agent reads these compact files before consulting long reports or source code.

### 4.3 `docs/ref-docs/` — Full Reference Articles and Optimization Reports

`docs/ref-docs/` stores detailed articles, complete optimization reports, framework explanations, profiling guides, and ISA/API summaries. It is the deep-reading layer of the repository.

Key subtrees:

| Directory | Role |
|-----------|------|
| `generic/` | GPU general theory and community optimization knowledge |
| `amd/common/` | AMD general optimization, MFMA, Composable Kernel, CK-Tile, rocprofv3, MoE, quantization |
| `amd/flydsl/` | FlyDSL framework references and AMD FlyDSL optimization reports |
| `amd/gluon/gfx942/` | MI300X Gluon optimization reports, profiling guides, ISA patterns, CK GEMM references |
| `nvidia/common/` | PTX, NCU, software pipeline, FP8 accumulation, tile rasterization, warp specialization |
| `nvidia/cuda/` | CUDA C++ and inline PTX related references |
| `nvidia/cutedsl/` | CuTeDSL/CUTLASS/QuACK references, layout algebra, GEMM/FMHA/MLA, pipeline, quantization |
| `nvidia/gluon/sm90/` | Hopper Gluon reports and profiling notes |
| `nvidia/triton/` | NVIDIA Triton-related references |

This area is intended to preserve the reasoning behind optimization decisions, not just the final recommendation.

### 4.4 `docs/pitfalls/` — Negative Knowledge and Porting Traps

`docs/pitfalls/` records failures, traps, and reverted approaches. It complements `docs/ref-docs/`: reference docs explain why a successful technique works, while pitfall docs explain why a tempting approach is wrong or fragile.

Pitfall files generally follow this logic:

```text
trap → symptom → real cause → why it happens → lesson / fix
```

Key subtrees:

| Directory | Role |
|-----------|------|
| `amd/flydsl/` | FlyDSL-on-AMD pitfalls, especially attention, mask, LSE, CK95 gap, occupancy, backward and chunk-GDN issues |
| `nvidia/cuda/` | CUDA implementation pitfalls |
| `nvidia/cutedsl/` | CuTeDSL pitfalls around TMA, GDN, NVFP4 GEMM, MoE preparation, fused epilogue and quantization |
| `nvidia/triton/` | NVIDIA Triton pitfalls |

Agents should read pitfall documents whenever a technique looks counter-intuitive, cross-architecture behavior conflicts, or an implementation has suspicious performance regressions.

### 4.5 `docs/converter/` — Code Conversion Knowledge

`docs/converter/` captures conversion rules for moving code across DSLs and architectures.

Main conversion stages:

- PyTorch → Triton
- Triton → Gluon

Key subtrees:

| Directory | Role |
|-----------|------|
| `generic/` | PyTorch→Triton conversion rules, API mapping, model configuration |
| `amd/common/` | General Triton→Gluon porting rules for AMD |
| `amd/cdna3/` | CDNA3-specific Triton→Gluon conversion: API mapping, pipeline, matrix multiply, memory access, layouts, pitfalls |
| `nvidia/hopper/` | Hopper-specific Triton→Gluon conversion: WGMMA, CP_ASYNC and related rules |

The converter documents are structured as operational guides rather than general tutorials.

### 4.6 `docs/RELATIONS.md` — Relationship and Conflict Graph

`docs/RELATIONS.md` is the repository-level relationship graph. It exists because GPU optimization knowledge is not monotonic: an optimization that is correct for one architecture or workload may conflict with another.

It records:

- hierarchical reading paths
- cross-architecture comparison tables
- conflict and difference inventories
- complementary document groups that should be read together

Agents should consult it when a task spans multiple architectures, multiple DSLs, or migration between NVIDIA and AMD.

## 5. Reference Code and Ground Truth

### 5.1 `reference-kernels/` — Reference Kernel Implementations

`reference-kernels/` stores Python/CUDA kernel implementations extracted from upstream repositories and local optimization projects. It is organized by hardware architecture, then DSL/framework, then source project.

Current high-level structure:

| Directory | Main DSLs / Frameworks | Role |
|-----------|------------------------|------|
| `nvidia/ampere/` | CuTeDSL, Gluon, Triton | SM80/A100 examples such as CUTLASS, Flash Attention, DeepGEMM |
| `nvidia/hopper/` | CuTeDSL, Gluon | SM90/H100/H20 examples such as CUTLASS, FlashInfer, QuACK, TileLang, TMA/WGMMA kernels |
| `amd/cdna/` | FlyDSL, Triton | CDNA general examples, including FlyDSL and aiter Triton kernels |
| `amd/cdna3/` | FlyDSL | CDNA3/gfx942-specific MI308X attention and mask/no-mask tuning kernels |
| `amd/rdna4/` | FlyDSL, Gluon | RDNA4/gfx1250 WMMA and attention examples |
| `generic/` | Triton, Gluon | Generic Triton tutorials, triton-kernels, FlashAttention, FlashInfer, LeetCUDA |

Typical lookup by kernel type:

| Kernel Type | Useful Areas |
|-------------|--------------|
| GEMM / MatMul | `nvidia/*/cutedsl/cutlass/`, `amd/cdna/flydsl/`, `amd/cdna/triton/aiter/gemm/`, `generic/triton/triton-tutorials/` |
| Attention / MLA | `nvidia/hopper/cutedsl/`, `amd/cdna*/flydsl/`, `amd/cdna/triton/aiter/attention/`, `generic/triton/flash-attention/` |
| Norm / Softmax / Reduction | `nvidia/hopper/cutedsl/flashinfer/`, `amd/cdna/triton/aiter/normalization/`, `generic/triton/triton-tutorials/` |
| MoE | `amd/cdna/triton/aiter/moe/`, `amd/cdna/flydsl/` |
| Quantization | `amd/cdna/triton/aiter/quant/`, NVIDIA CuTeDSL quantization reports and examples |
| SSM / Mamba | `nvidia/hopper/cutedsl/flashinfer/` |

### 5.2 `reference-projects/` — Upstream Project References

`reference-projects/` is reserved for optional local snapshots or source summaries of upstream projects. For API/ISA ground truth, clone the relevant upstream project to `/tmp/reference-projects/` or consult vendor official documentation.

## 6. Source Provenance

The knowledge base combines three source categories:

1. **Official documentation**: CUDA Programming Guide, PTX ISA, CUTLASS/CuTeDSL docs, Nsight Compute, ROCm documentation, AMD ISA references, profiling guides.
2. **Open-source repositories**: CUTLASS, cutex, cuLA, flash-attention, FlashInfer, FlyDSL, Triton, DeepGEMM, LeetCUDA, FlashMLA, composable_kernel, cute-gemm, hpc-ops, aiter, QuACK, TileLang.
3. **Local optimization experience**: architecture-specific Gluon/FlyDSL/CuTeDSL optimization reports, pitfalls, profiling conclusions, and distilled pattern cards.

The repository keeps distilled knowledge in `docs/`, executable or illustrative code in `reference-kernels/`, and references to upstream projects in reference-projects/.

## 7. Agent Reading Workflow

For a new task, the recommended workflow is:

1. **Identify task type**: optimization, conversion, profiling, API clarification, implementation, or pitfall investigation.
2. **Identify target architecture**: generic, AMD CDNA/CDNA3/RDNA4, NVIDIA Ampere/Hopper, or another explicitly named target.
3. **Read the closest directory README**: start from `README.md`, then `docs/README.md`, then the relevant module README.
4. **Read quick knowledge first**: use `docs/kernel-opt/` for concise optimization direction.
5. **Read full evidence when needed**: use `docs/ref-docs/` and `docs/pitfalls/` to understand constraints, failures, and tradeoffs.
6. **Check ground truth for implementation**: clone upstream projects to /tmp/reference-projects/ for API/ISA behavior, use reference-kernels/ for concrete kernel patterns, and prefer vendor official documentation for ISA/architecture references.
7. **Consult `docs/RELATIONS.md` for cross-architecture work**: especially before transferring an optimization from AMD to NVIDIA or from one GPU generation to another.

## 8. Coverage Summary

Current coverage focuses on high-performance AI kernels and DSL-level GPU programming.

### Architectures

| Vendor | Architecture / Generation | Coverage |
|--------|---------------------------|----------|
| NVIDIA | Hopper / SM90 | Deep coverage: Gluon, CuTeDSL, profiling, TMA/WGMMA, attention, MLA, GDN, Mamba, quantization |
| NVIDIA | Ampere / SM80 | Reference-kernel level coverage |
| NVIDIA | Blackwell-related knowledge | Present mainly through source/project references and selected CuTeDSL/optimization materials where available |
| AMD | CDNA / CDNA3 / gfx942 | Deep coverage: FlyDSL, Gluon, aiter Triton, MI300X/MI308X specs, attention, GEMM, MoE, profiling, pitfalls |
| AMD | RDNA4 / gfx1250 | Reference-kernel level coverage with FlyDSL/Gluon examples |
| Generic | Architecture-agnostic GPU/Triton | General optimization theory and Triton examples |

### Kernel Types

Covered kernel families include:

- GEMM / MatMul
- Flash Attention / MLA Decode
- Softmax / LayerNorm / Reduction
- MoE
- SSM / Mamba
- Quantization, including FP8/FP4-related materials
- Linear Attention / GDN-style kernels

### DSLs and Frameworks

Covered programming models include:

- Triton
- Gluon
- CuTeDSL / CUTLASS Python DSL
- FlyDSL
- CUDA C++ / inline PTX
- AMD ISA / MFMA / inline assembly concepts

## 9. Maintenance Rules

To keep the knowledge base useful for Agents:

1. **Preserve narrow topic boundaries**: do not merge unrelated techniques into a monolithic file.
2. **Keep directory README files current**: Agents depend on README files for navigation.
3. **Record conflicts explicitly**: if a technique behaves differently across architectures, update `docs/RELATIONS.md` or a relevant pitfall file.
4. **Separate facts from examples**: API and ISA truth belongs in vendor documentation or upstream source repositories; runnable usage patterns belong in `reference-kernels/`; distilled guidance belongs in `docs/`.
5. **Prefer evidence-backed optimization notes**: include profiling data, hardware constraints, or source references when possible.
6. **Update counts and structure after large imports**: this design document should reflect the current repository structure, not historical organization.
