# Document Relationship Diagram

This document describes the relationships between files in gpu-wiki: reading order, complementary relationships, conflicts and differences.

---

## 1. Reading Path Diagram

### General Fundamentals (required reading, prerequisite for all architectures)

```Tier 0 — Core Concepts (Read First)
  gpu-memory-hierarchy.md ─── Registers, shared memory, coalescing, bank conflict
  gpu-execution-model.md ─── Thread hierarchy, warp, CTA, SM, grid

Tier 1 — General Optimization (Read Next)
  gpu-instruction-optimization.md ── fast math, roofline, vectorization
  gpu-application-optimization.md ── Amdahl's Law, host-device transfer, operator fusion```

### Architecture-Specific (read after selecting target architecture)

```Tier 2 — Architecture Specs
  ┌─ NVIDIA:  nvidia-compute-capabilities.md (includes CC spec tables for each generation)
  └─ AMD:     amd-gpu-kernel-tuning.md (includes hardware spec tables)

Tier 3 — Deep Dive
  ├─ NVIDIA CuTeDSL Chain (nvidia/cutedsl/): cutlass-cute-fundamentals.md → cutlass-gemm-optimization.md
  │                     → kernel-opt/nvidia/cutedsl/ (production optimization)
  ├─ NVIDIA PTX Chain (nvidia/common/):  ptx-programming-model.md → ptx-instruction-set.md
  │                     → kernel-opt/nvidia/ptx/ (production optimization: NVFP4 Split-K GEMV)
  ├─ AMD Framework (amd/common/): amd-kernel-optimization-frameworks.md + amd-mfma-matrix-cores.md
  │                     → kernel-opt/amd/flydsl/ (production optimization: Flash Attention, Chunk-GDN)
  └─ AMD FlyDSL Chain (amd/flydsl/): flydsl-programming-guide.md → flydsl-layout-algebra.md
                    → kernel-opt/amd/flydsl/gfx942/ (CDNA3 production optimization: Chunk-GDN megakernel)

Tier 4 — Gluon DSL Architecture-Specific Optimization (Reading chains within each architecture)
  AMD FlyDSL (gfx942): SKILL.md → layouts.md → api_mapping.md
  NVIDIA CuTeDSL (sm90): cutedsl-programming-model.md → cutedsl-pipeline-patterns.md
          → [GEMM Chain] pattern_overview → optimization_strategy → warp_pipeline_stage
          → [Attention Chain] optimization_results + se_level_zigzag
```

### Converter Tool Reading Chain

```Stage 1: PyTorch → Triton (Vendor-Agnostic)
  → porting_rules.md (Conversion principles)
  → api_mapping.md (API mapping table)

Stage 2: Triton → Gluon (Select Target Architecture)
  General rules (read first): amd/common/porting_rules.md + learning_guide.md + verification_guide.md
  Architecture-specific (read later): SKILL.md → layouts.md → api_mapping.md```

---

## 2. Cross-Architecture Comparison Table

### Kernel Optimization Document Comparison

| Topic | CDNA3 (MI308X) | Hopper (H100/H20) |
|------|---------------|-------------------|
| Hardware Specs | `hardware-specs/hardware_specs_mi300x.md` | `hardware-specs/hardware_specs_hopper.md` |
| General Optimization Checklist | `cdna3-..--common_optimizations.md` | `hopper-..--common_optimizations.md` |
| Profiling | `cdna3-..--profiling_guide.md` (rocprofv3) | `hopper-..--profiling_guide.md` (ncu) |
| ISA Instruction Reference | `cdna3-..--isa_patterns.md` | `hopper-..--isa_patterns.md` |
| GEMM Optimization | `pattern_overview` + `optimization_strategy` + `warp_pipeline_stage` + `final_config_template` + `key_conclusions` (5 files) | `hopper-..--matmul.md` (1 file) |
| Attention Optimization | `optimization_results` + `se_level_zigzag` (2 files) + `cdna3-flash-attention-bf16-nomask-isa-scheduling.md` (FlyDSL BF16 no-mask ATT scheduling) | `hopper-..--fused_attention.md` (skeleton) |
| Linear Attention / GDN Megakernel | `cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md` | FlashQLA/Hopper warp-specialization as migration source |
| Softmax/Reduction | *embedded in common_optimizations* | `hopper-..--softmax_reduce.md` |
| Pitfalls & Lessons Learned | *scattered across multiple files* | `hopper-..--pitfalls.md` (11 items) |
| MLA Decode | *none* | *none* |
| Linear Attention | *none* | `hopper-..--linear_attention.md` |
| CuTeDSL Reference | *none (AMD)* | `hopper-cutedsl-sm90.md` |
| CK GEMM Reference | `skill_reference_ck_gemm_optimization.md` | *none (NVIDIA)* |
| Gluon API Reference | `gluon_AMD_gfx942_optimization .md` | *no standalone file* |
| FP8 GEMM Hands-on | *none* | *none* |

### Converter Documentation Cross-Reference

| Topic | CDNA3 | Hopper |
|------|-------|--------|
| SKILL Entry | `cdna3-..--SKILL.md` | `hopper-..--SKILL.md` |
| API Mapping Table | `cdna3-..--api_mapping.md` | `hopper-..--api_mapping.md` |
| Pipeline Mode | `cdna3-..--pipeline.md` (software-only) | `hopper-..--pipeline.md` (CP_ASYNC) |
| Matrix Multiply Mode | `cdna3-..--matrix_multiply.md` (mfma) | `hopper-..--matrix_multiply.md` (wgmma) |
| Memory Access Mode | `cdna3-..--memory_access.md` | `hopper-..--memory_access.md` |
| Layout Mapping | `cdna3-..--layouts.md` | `hopper-..--layouts.md` |
| Common Pitfalls | `cdna3-..--common_pitfalls.md` | `hopper-..--common_pitfalls.md` |

### Converter → Kernel-Opt Relationships

| Converter Document | Produced/Consumed Kernel-Opt Documents |
|---------------|---------------------------|
| `*--pipeline.md` | → `cdna3-..--warp_pipeline_stage.md` (optimized pipeline output code) |
| `*--matrix_multiply.md` | → `amd-mfma-matrix-cores.md` (MFMA instruction details) |
| `*--matrix_multiply.md` | → `nvidia-ptx-mma-instructions.md` (MMA instruction evolution) |
| `*--layouts.md` | → `hardware-specs/hardware_specs_*.md` (hardware constraints determine legal layouts) |
| `*--common_pitfalls.md` (conversion errors) | ↔ `*--pitfalls.md` (runtime performance issues) |
| `*--api_mapping.md` | → `*--isa_patterns.md` (underlying instruction reference for API mapping) |

---

## 3. Conflicts and Differences

### 🔴 Direct Conflicts (contradictory advice on the same issue)

#### Conflict 1: Value of Warp Pipeline Stage

| Document | Conclusion |
|------|------|
| `cdna3-..--warp_pipeline_stage.md` | WPS is a key optimization for large-tile GEMM, **+27% performance** |
| Hopper Docs | Does not use the WPS concept; uses `fence` + `commit_group` + `wait_group` instead |

**Interpretation**: WPS is applicable to pure GEMM (no cross-iteration dependencies), but not to fused attention. The +27% conclusion in CDNA3 docs applies only to GEMM scenarios.

#### Conflict 2: Benefit of Manual ISA Optimization

| Document | Conclusion |
|------|------|
| `cdna3-..--common_optimizations.md` | Manual ISA optimization is effective: removing `other=0.0` +1%, `tl.assume` +2%, loop-invariant hoisting +4% |
| `hopper-..--pitfalls.md` (#1) | Manual code restructuring (hoisting loop invariants, adjusting prefetch) is **almost always counter-productive**, since the compiler's global optimization for CSE and scheduling is coupled |

**Interpretation**: AMD compiler (Gluon for gfx942) responds well to manual ISA tuning. NVIDIA compiler (sm_90) has more aggressive global optimization where manual intervention tends to disrupt compiler strategies.#### Conflict 3: NVIDIA Bias in Generic Documentation

| Document | Issue |
|------|------|
| `gpu-memory-hierarchy.md` | Overview table uses NVIDIA-specific values: 255 regs/thread, 64-228 KB/SM, without noting AMD equivalents |
| `gpu-execution-model.md` | Correctly mentions NVIDIA 32-thread warp vs AMD 64-thread wavefront |

**Interpretation**: `gpu-memory-hierarchy.md` conclusion of 32 banks does not apply to all AMD architectures. When reading AMD-related documentation, be aware of generic documents' NVIDIA defaults.

### 🟡 Significant Architectural Differences (Not Conflicts, But Must Be Noted When Migrating Across Architectures)

#### Difference 1: Massive Ridge Point Differences

| Architecture | BF16 Ridge Point | State at Same Tile AI=237 |
|------|-----------------|----------------------|
| CDNA3 (MI308X) | ~247 | Near ridge point (boundary) |
| H100 | ~295 | memory-bound |
| H20 | ~37 | **compute-bound** (far above ridge) |

**Impact**: The same kernel may require completely opposite optimization directions on different architectures.

#### Difference 2: FP8 Format Incompatibility

| Architecture | FP8 Format | Notes |
|------|---------|------|
| CDNA3 | E4M3**FNUZ** (bias=8), E5M2**FNUZ** (bias=16) | AMD non-standard format |
| NVIDIA | `.e4m3`, `.e5m2` (OCP) | OCP standard |

**Impact**: CDNA3 FP8 data is not binary-compatible with NVIDIA; cross-platform migration requires format conversion.

#### Difference 3: Completely Different Pipeline Mechanisms

| Architecture | Method | Core API |
|------|------|---------|
| CDNA3 | Pure software: `buffer_load` → registers → `smem.store` | Manual buffer management |
| Hopper | CP_ASYNC DMA: `async_copy_global_to_shared`, bypasses registers | `async_copy` series |

**Impact**: CDNA3 pipeline code cannot be ported to Hopper, and vice versa.

#### Difference 4: Matrix Multiply Instructions

| Architecture | Instruction | Operand Source | Warp/Wave Size |
|------|------|-----------|---------------|
| CDNA3 | `v_mfma` | Registers | 64 threads |
| Hopper | `wgmma` (warpgroup MMA) | **Shared memory** direct read | 128 threads (4 warps) |

#### Difference 5: LDS/Shared Memory Capacity

| Architecture | LDS/SMEM per CU/SM | Impact |
|------|-------------------|------|
| CDNA3 | 64 KB | Tile size is limited |
| Hopper | **228 KB** | Maximum capacity |

#### Difference 6: BF16 vs FP16 Performance

| Architecture | Behavior |
|------|------|
| AMD (MI308X) | BF16 matrix core execution is **significantly faster** than FP16 |
| NVIDIA | BF16 and FP16 Tensor Core throughput is identical |

#### Difference 7: Block Size Recommendations

| Source | Recommendation |
|------|------|
| `gpu-execution-model.md` (generic) | 128 or 256 threads/block |
| Hopper (CC 9.0) | Must be a multiple of 128 (wgmma warp group requirement) |
| AMD | Recommend 1024/2048/4096 (64-thread wavefront, needs more threads to fill a CU) |

#### Difference 8: V RHS Staging in GDN Chunk-Forward

| Architecture / Implementation | Conclusion |
|-------------|------|
| AMD FlyDSL chunk-GDN fwd_h | Focus is on O=3 patch, col-major k-LDS, barrier removal, and SmemPtr double buffering |

**Interpretation**: FlashInfer/FlashQLA-style "fusion" is not simply about reducing shared staging. The effective fusion point involves pushing decay terms to the v_new side, removing scratch buffers, and using transposed LDSM atoms to absorb transposed views.### 🟢 Content Duplication (Not a Conflict, but Redundant)

| Duplicated Content | Files Involved | Suggestion |
|---------|---------|------|
| Host-device Transfer | `gpu-memory-hierarchy.md` + `gpu-application-optimization.md` | `gpu-application-optimization.md` is more comprehensive |
| Warp Divergence Concept | `gpu-execution-model.md` + `gpu-instruction-optimization.md` | The former defines the concept, the latter provides optimization tips — acceptable |
| Profiling Principles | `gpu-instruction-optimization.md` + `ncu-profiling-guide.md` + `amd-gpu-kernel-tuning.md` | Layering is reasonable: generic → vendor-specific |

---

## 4. Complementary Relationship Map

### AMD vs NVIDIA Comparison for the Same Concepts

| Concept | AMD Documentation | NVIDIA Documentation |
|------|---------|------------|
| Matrix Core Programming | `amd-mfma-matrix-cores.md` | `nvidia-ptx-mma-instructions.md` |
| Optimization Frameworks | `amd-kernel-optimization-frameworks.md` (FlyDSL/CK/TileLang) | `cutlass-cute-fundamentals.md` + `cutlass-gemm-optimization.md` (CUTLASS/CuTe) |
| DSL Programming | `amd/flydsl/flydsl-programming-guide.md` (FlyDSL) | `nvidia/cutedsl/cutedsl-programming-model.md` + `cutedsl-pipeline-patterns.md` (CuTeDSL) |
| Profiling Tools | `amd/common/rocprofv3-profiling-guide.md` (Generic) + `amd/gluon/gfx942-..--profiling_guide.md` (ATT Instruction-Level) | `nvidia/common/ncu-profiling-guide.md` + `nvidia/gluon/sm90/hopper-..--profiling_guide.md` (Nsight Compute) |
| Hardware Specifications | `hardware-specs/hardware_specs_mi300x.md` | `nvidia/common/nvidia-compute-capabilities.md` + `hardware-specs/hardware_specs_hopper.md` |

### Document Groups That Should Be Read Together

**Group 1: Memory Optimization Panorama**
- `gpu-memory-hierarchy.md` (General Principles)
- `nvidia-arch-specific-optimization.md` (NVIDIA-specific: L2 persistence, TMA)
- `amd-gpu-kernel-tuning.md` (AMD-specific: LDS bank conflict, XOR swizzle)

**Group 2: GEMM Optimization Panorama**
- `cutlass-gemm-optimization.md` (NVIDIA tiling strategy)
- `cdna3-..--warp_pipeline_stage.md` (AMD WPS technology)
- Architecture-specific `matmul.md` (Roofline + architecture-specific configurations)

**Group 3: Profiling Toolchain**
- `gpu-instruction-optimization.md` (Roofline principles)
- `ncu-profiling-guide.md` (NVIDIA tools)
- `cdna3-..--profiling_guide.md` (AMD tools)
- Architecture-specific `hardware-specs/hardware_specs_*.md` (peak TFLOPS for compute utilization calculation)

**Group 4: Conversion + Optimization Closed Loop**
- `converter/*--pipeline.md` (Conversion produces pipeline code)
- `kernel-opt/*--common_optimizations.md` (Optimizing the ISA of that code)
- `kernel-opt/*--profiling_guide.md` (Verifying optimization results)

**Group 5: FlyDSL Chunk-GDN Optimization Trilogy**
- `pitfalls/amd/flydsl/chunk-gdn-pitfalls.md` (11 pitfalls: SmemPtr large memref, scf.IfOp syntax, O=3 patch, pre-load+barrier coordination, iter_args VGPR regression, ds_read_tr regression, IR explosion, nested scf.for_, hardcoded grid, grid dimension order, TP compile-time constants)
- Prerequisites: `ref-docs/amd/flydsl/flydsl-programming-guide.md` + `pitfalls/amd/flydsl/flash-attn-pitfalls.md`

**Group 6: CUDA NVFP4 Split-K GEMV Trilogy**
- Complementary background: `ref-docs/nvidia/cutedsl/cutlass-tile-scheduling.md` (Stream-K / split-K scheduler concepts)

---

## 5. Known Issues

### Content displacement in converter/amd/common/triton-to-gluon-skill.md

This file is located under `amd/common/` but actually contains **Hopper (NVIDIA sm_90)** content: the YAML frontmatter says `name: hopper-triton-to-gluon-converter`, and the content references `cuda:90`, `wgmma`, `NVMMASharedLayout`, and other NVIDIA APIs. It is suspected to have been copied from Hopper SKILL.md without modification.

### Inconsistent CDNA3 documentation structure

CDNA3's GEMM optimization is spread across 5 files (pattern_overview, optimization_strategy, warp_pipeline_stage, final_config_template, key_conclusions), whereas Hopper uses only 1 `matmul.md`. This is because the CDNA3 docs were accumulated incrementally, while Hopper was consolidated later in a unified manner.
