# Document Relationship Diagram

This document describes the relationships between files in gpu-wiki: reading order, complementary relationships, conflicts and differences.

---

## 1. Reading Path Diagram

### General Fundamentals (required reading, prerequisite for all architectures)

```
Tier 0 — Core Concepts (Read First)
  gpu-memory-hierarchy.md ─── Registers、shared memory、coalescing、bank conflict
  gpu-execution-model.md ─── Thread hierarchy、warp、CTA、SM、grid

Tier 1 — General Optimization (Read Next)
  gpu-instruction-optimization.md ── fast math、roofline、vectorization
  gpu-application-optimization.md ── Amdahl's Law、host-device transfer、operator fusion
```

### Architecture-Specific (read after selecting target architecture)

```
Tier 2 — Architecture Specs
  ┌─ NVIDIA:  nvidia-compute-capabilities.md (includes CC spec tables for each generation)
  └─ AMD:     amd-gpu-kernel-tuning.md (includes hardware spec tables)

Tier 3 — Deep Dive
  ├─ NVIDIA CuTeDSL Chain (nvidia/common/cutedsl/): cutlass-cute-fundamentals.md → cutlass-gemm-optimization.md
  │                     → nvidia/blackwell-geforce/cutedsl/ (production optimization: sm120 GDN chunk-fwd V113)
  ├─ NVIDIA PTX Chain (nvidia/common/):  ptx-programming-model.md → ptx-instruction-set.md
  │                     → nvidia/common/ptx/ (production optimization: NVFP4 Split-K GEMV)
  ├─ AMD Framework (amd/common/): amd-kernel-optimization-frameworks.md + amd-mfma-matrix-cores.md
  │                     → kernel-opt/amd/flydsl/ (production optimization: Flash Attention, Chunk-GDN)
  └─ AMD FlyDSL Chain (amd/flydsl/): flydsl-programming-guide.md → flydsl-layout-algebra.md
                    → kernel-opt/amd/flydsl/gfx942/ (CDNA3 production optimization: Chunk-GDN megakernel)
                    → kernel-opt/amd/flydsl/gfx950/ (CDNA4 production optimization: chunk-GDN fwd_h)

Tier 4 — Gluon DSL Architecture-Specific Optimization (Reading chains within each architecture)
  AMD FlyDSL (gfx942/gfx950): conversion-guide.md → layouts.md → api_mapping.md
  NVIDIA CuTeDSL (sm90): cutedsl-programming-model.md → cutedsl-pipeline-patterns.md
          → [GEMM Chain] pattern_overview → optimization_strategy → warp_pipeline_stage
          → [Attention Chain] optimization_results + se_level_zigzag
```

### Converter Tool Reading Chain

```
Stage 1: PyTorch → Triton (Vendor-Agnostic)
  → porting_rules.md (Conversion principles)
  → api_mapping.md (API mapping table)

Stage 2: Triton → Gluon (Select Target Architecture)
  General rules (read first): amd/common/porting_rules.md + learning_guide.md + verification_guide.md
  Architecture-specific (read later): conversion-guide.md → layouts.md → api_mapping.md
```

---

## 2. Cross-Architecture Comparison Table

### Kernel Optimization Document Comparison

| Topic | CDNA3 (MI308X) | CDNA4 (MI355X) | Hopper (H100/H20) |
|------|---------------|----------------|-------------------|
| Hardware Specs | `hardware-specs/hardware_specs_mi300x.md` | `hardware-specs/hardware_specs_mi355x.md` | `hardware-specs/hardware_specs_hopper.md` |
| General Optimization Checklist | `cdna3-..--common_optimizations.md` | `cdna4-..--common_optimizations.md` | `hopper-..--common_optimizations.md` |
| Profiling | `cdna3-..--profiling_guide.md` (rocprofv3) | `cdna4-..--profiling_guide.md` (rocprofv3) | `hopper-..--profiling_guide.md` (ncu) |
| ISA Instruction Reference | `cdna3-..--isa_patterns.md` | *No standalone file* | `hopper-..--isa_patterns.md` |
| GEMM Optimization | `pattern_overview` + `optimization_strategy` + `warp_pipeline_stage` + `final_config_template` + `key_conclusions` (5 files) | `cdna4-..--matmul.md` (1 file) | `hopper-..--matmul.md` (1 file) |
| Attention Optimization | `optimization_results` + `se_level_zigzag` (2 files) + `cdna3-flash-attention-bf16-nomask-isa-scheduling.md` (FlyDSL BF16 no-mask ATT scheduling) | `cdna4-..--fused_attention.md` | `hopper-..--fused_attention.md` (skeleton) |
| Linear Attention / GDN Megakernel | `cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md` | `cdna4-chunk-gdn.md` | FlashQLA/Hopper warp-specialization as migration source |
| Softmax/Reduction | *embedded in common_optimizations* | `cdna4-..--softmax_reduce.md` | `hopper-..--softmax_reduce.md` |
| Pitfalls & Lessons Learned | *scattered across multiple files* | `cdna4-..--pitfalls.md` (10 items) | `hopper-..--pitfalls.md` (11 items) |
| MLA Decode | *none* | `cdna4-..--mla_decode.md` | *none* |
| Linear Attention | *none* | *none* | `hopper-..--linear_attention.md` |
| CuTeDSL Reference | *none (AMD)* | *none (AMD)* | `hopper-cutedsl-sm90.md` |
| CK GEMM Reference | `ck_gemm_optimization_reference.md` | *none* | *none (NVIDIA)* |
| Gluon API Reference | `gluon-amd-gfx942-optimization.md` | *no standalone file* | *no standalone file* |
| FP8 GEMM Hands-on | *none* | `cdna4-fp8-gemm-optimization.md` | *none* |

SM120 CuTeDSL GDN chunk-forward is the NVIDIA Blackwell GeForce counterpart for this row: optimization report at
`nvidia/blackwell-geforce/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md`,
final V113 is no-cache directional final-state `0.531-0.533ms = 1.51× same-process FLA`,
complementary optimization axes with AMD FlyDSL chunk-GDN.

### Converter Documentation Cross-Reference

| Topic | CDNA3 | CDNA4 | Hopper |
|------|-------|-------|--------|
| Conversion Guide | [CDNA3 conversion guide](amd/converter/cdna3/conversion-guide.md) | [CDNA4 conversion guide](amd/converter/cdna4/conversion-guide.md) | [Hopper conversion guide](nvidia/hopper/converter/hopper/conversion-guide.md) |
| API Mapping Table | [CDNA3 API mapping](amd/converter/cdna3/api_mapping.md) | [CDNA4 API mapping](amd/converter/cdna4/api_mapping.md) | [Hopper API mapping](nvidia/hopper/converter/hopper/api_mapping.md) |
| Pipeline Mode | [CDNA3 pipeline notes](amd/converter/cdna3/pipeline.md) (software-only) | [CDNA4 pipeline notes](amd/converter/cdna4/pipeline.md) (HW DMA reference) | [Hopper pipeline notes](nvidia/hopper/converter/hopper/pipeline.md) (CP_ASYNC) |
| Matrix Multiply Mode | [CDNA3 matrix multiply notes](amd/converter/cdna3/matrix_multiply.md) (mfma) | [CDNA4 matrix multiply notes](amd/converter/cdna4/matrix_multiply.md) (mfma) | [Hopper matrix multiply notes](nvidia/hopper/converter/hopper/matrix_multiply.md) (wgmma) |
| Memory Access Mode | [CDNA3 memory access notes](amd/converter/cdna3/memory_access.md) | [CDNA4 memory access notes](amd/converter/cdna4/memory_access.md) | [Hopper memory access notes](nvidia/hopper/converter/hopper/memory_access.md) |
| Layout Mapping | [CDNA3 layout mapping](amd/converter/cdna3/layouts.md) | [CDNA4 layout mapping](amd/converter/cdna4/layouts.md) | [Hopper layout mapping](nvidia/hopper/converter/hopper/layouts.md) |
| Common Pitfalls | [CDNA3 common pitfalls](amd/converter/cdna3/common_pitfalls.md) | [CDNA4 common pitfalls](amd/converter/cdna4/common_pitfalls.md) | [Hopper common pitfalls](nvidia/hopper/converter/hopper/common_pitfalls.md) |

### Converter → Kernel-Opt Relationships

| Converter Document | Produced/Consumed Kernel-Opt Documents |
|---------------|---------------------------|
| Pipeline notes | → warp pipeline stage notes (optimized pipeline output code) |
| Matrix multiply notes | → AMD MFMA matrix-core reference (MFMA instruction details) |
| Matrix multiply notes | → NVIDIA PTX MMA instruction reference (MMA instruction evolution) |
| Layout notes | → hardware specification pages (hardware constraints determine legal layouts) |
| Conversion pitfalls | ↔ runtime performance pitfall notes |
| API mapping notes | → ISA pattern references (underlying instruction reference for API mapping) |

---

## 3. Conflicts and Differences

### 🔴 Direct Conflicts (contradictory advice on the same issue)

#### Conflict 1: Value of Warp Pipeline Stage

| Document | Conclusion |
|------|------|
| `cdna3-..--warp_pipeline_stage.md` | WPS is a key optimization for large-tile GEMM, **+27% performance** |
| `cdna4-..--pitfalls.md` (#7) | WPS is **counter-productive** for attention (cross-iteration dependency in online softmax) |
| Hopper Docs | Does not use the WPS concept; uses `fence` + `commit_group` + `wait_group` instead |

**Interpretation**: WPS is applicable to pure GEMM (no cross-iteration dependencies), but not to fused attention. The +27% conclusion in CDNA3 docs applies only to GEMM scenarios.

#### Conflict 2: Benefit of Manual ISA Optimization

| Document | Conclusion |
|------|------|
| `cdna3-..--common_optimizations.md` | Manual ISA optimization is effective: removing `other=0.0` +1%, `tl.assume` +2%, loop-invariant hoisting +4% |
| `hopper-..--pitfalls.md` (#1) | Manual code restructuring (hoisting loop invariants, adjusting prefetch) is **almost always counter-productive**, since the compiler's global optimization for CSE and scheduling is coupled |

**Interpretation**: AMD compiler (Gluon for gfx942) responds well to manual ISA tuning. NVIDIA compiler (sm_90) has more aggressive global optimization where manual intervention tends to disrupt compiler strategies. CDNA4 falls somewhere in between.

#### Conflict 3: NVIDIA Bias in Generic Documentation

| Document | Issue |
|------|------|
| `gpu-memory-hierarchy.md` | Claims shared memory has **32 banks** (generic), but CDNA4 actually has **64 banks** |
| `gpu-memory-hierarchy.md` | Overview table uses NVIDIA-specific values: 255 regs/thread, 64-228 KB/SM, without noting AMD equivalents |
| `gpu-execution-model.md` | Correctly mentions NVIDIA 32-thread warp vs AMD 64-thread wavefront |

**Interpretation**: `gpu-memory-hierarchy.md` conclusion of 32 banks does not apply to CDNA4. When reading AMD-related documentation, be aware of generic documents' NVIDIA defaults.

### 🟡 Significant Architectural Differences (Not Conflicts, But Must Be Noted When Migrating Across Architectures)

#### Difference 1: Massive Ridge Point Differences

| Architecture | BF16 Ridge Point | State at Same Tile AI=237 |
|------|-----------------|----------------------|
| CDNA3 (MI308X) | ~247 | Near ridge point (boundary) |
| CDNA4 (MI355X) | ~629 | **memory-bound** (far below ridge) |
| H100 | ~295 | memory-bound |
| H20 | ~37 | **compute-bound** (far above ridge) |

**Impact**: The same kernel may require completely opposite optimization directions on different architectures.

#### Difference 2: FP8 Format Incompatibility

| Architecture | FP8 Format | Notes |
|------|---------|------|
| CDNA3 | E4M3**FNUZ** (bias=8), E5M2**FNUZ** (bias=16) | AMD non-standard format |
| CDNA4 | E4M3**FN** (OCP, bias=7), E5M2 (OCP, bias=15) | OCP standard |
| NVIDIA | `.e4m3`, `.e5m2` (OCP) | OCP standard |

**Impact**: CDNA3 FP8 data is not binary-compatible with CDNA4/NVIDIA; cross-platform migration requires format conversion.

#### Difference 3: Completely Different Pipeline Mechanisms

| Architecture | Method | Core API |
|------|------|---------|
| CDNA3 | Pure software: `buffer_load` → registers → `smem.store` | Manual buffer management |
| CDNA4 | Hardware DMA: `async_copy.buffer_load_to_shared`, bypasses registers | `async_copy` series |
| Hopper | CP_ASYNC DMA: `async_copy_global_to_shared`, bypasses registers | `async_copy` series |

**Impact**: CDNA3 pipeline code cannot be ported to CDNA4 or Hopper, and vice versa.

#### Difference 4: Matrix Multiply Instructions

| Architecture | Instruction | Operand Source | Warp/Wave Size |
|------|------|-----------|---------------|
| CDNA3 | `v_mfma` | Registers | 64 threads |
| CDNA4 | `v_mfma` / `v_mfma_scale` | Registers | 64 threads |
| Hopper | `wgmma` (warpgroup MMA) | **Shared memory** direct read | 128 threads (4 warps) |

#### Difference 5: LDS/Shared Memory Capacity

| Architecture | LDS/SMEM per CU/SM | Impact |
|------|-------------------|------|
| CDNA3 | 64 KB | Tile size is limited |
| CDNA4 | **160 KB** | Can use larger tiles and deeper pipelines |
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
| SM120 CuTeDSL GDN chunk fwd V113 | Retain cp.async staging, use LDSM/R2S + scaled-vnew + reuse-B LDSM; `0.531-0.533ms = 1.51× same-process FLA` |
| SM120 CuTeDSL GDN chunk fwd V31 | Retain cp.async staging, use LDSM/R2S + scaled-vnew; `0.615ms = 1.42× FLA varlen` |
| SM120 CuTeDSL V30 direct global V RHS | Correct but slower, repeat-100 is `0.7200ms` vs V29 `0.6861ms`; scalar `LDG.E.U16` falls on RHS critical path |
| AMD FlyDSL chunk-GDN fwd_h | Focus is on O=3 patch, col-major k-LDS, barrier removal, and SmemPtr double buffering; do not reverse-apply SM120's direct-global-V conclusion to AMD |

**Interpretation**: FlashInfer/FlashQLA-style "fusion" is not simply about reducing shared staging. In SM120 CuTeDSL's GDN chunk-forward, reading V RHS directly via global scalar load moves the bottleneck to the RHS load critical path; V31's effective fusion point is pushing `exp_decay[t]` to the `v_new[t,v]` side, removing the K-decay scratch, and using a transposed LDSM atom to absorb the transposed `sK` view. V113's newly effective point is local B-fragment reuse, not preprocess cache or full TMA fusion.

### 🟢 Content Duplication (Not a Conflict, but Redundant)

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
| DSL Programming | `amd/flydsl/flydsl-programming-guide.md` (FlyDSL) | `nvidia/common/cutedsl/cutedsl-programming-model.md` + `cutedsl-pipeline-patterns.md` (CuTeDSL) |
| Profiling Tools | `amd/common/rocprofv3-profiling-guide.md` (Generic) + `amd/gluon/gfx942|gfx950-..--profiling_guide.md` (ATT Instruction-Level) | `nvidia/common/profiling/ncu-profiling-guide.md` + `nvidia/hopper/gluon/hopper-..--profiling_guide.md` (Nsight Compute) |
| Hardware Specifications | `hardware-specs/hardware_specs_mi300x.md` + `hardware-specs/hardware_specs_mi355x.md` | `nvidia/common/nvidia-compute-capabilities.md` + `hardware-specs/hardware_specs_hopper.md` |

### Document Groups That Should Be Read Together

**Group 1: Memory Optimization Panorama**
- `gpu-memory-hierarchy.md` (General Principles)
- `nvidia-arch-specific-optimization.md` (NVIDIA-specific: L2 persistence, TMA)
- `amd-gpu-kernel-tuning.md` (AMD-specific: LDS bank conflict, XOR swizzle)

**Group 2: GEMM Optimization Panorama**
- `cutlass-gemm-optimization.md` (NVIDIA tiling strategy)
- `nvidia/blackwell-geforce/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md` (CUDA Split-K: fixing K-dimension parallelism for small `M*N` tiles)
- `cdna3-..--warp_pipeline_stage.md` (AMD WPS technology)
- Architecture-specific `matmul.md` (Roofline + architecture-specific configurations)

**Group 3: Profiling Toolchain**
- `gpu-instruction-optimization.md` (Roofline principles)
- `ncu-profiling-guide.md` (NVIDIA tools)
- `ncu-measurement-discipline.md` (NVIDIA: trusting the numbers — Duration≠latency, noise floor, graph-capture pitfalls)
- `ncu-rule-est-speedup-meta-rules.md` (NVIDIA: `Est. Speedup %` is a ceiling, not a wall-time delta)
- `cdna3/cdna4-..--profiling_guide.md` (AMD tools)
- Architecture-specific `hardware-specs/hardware_specs_*.md` (peak TFLOPS for compute utilization calculation)

**Group 4: Conversion + Optimization Closed Loop**
- `converter/*--pipeline.md` (Conversion produces pipeline code)
- `kernel-opt/*--common_optimizations.md` (Optimizing the ISA of that code)
- `kernel-opt/*--profiling_guide.md` (Verifying optimization results)

**Group 5: FlyDSL Chunk-GDN Optimization Trilogy**
- `ref-docs/amd/flydsl/gfx950/cdna4-chunk-gdn.md` (Full pipeline optimization: V0→V8, 2.64x→0.78x Triton, including detailed fwd_h optimization journey)
- `pitfalls/amd/flydsl/chunk-gdn-pitfalls.md` (11 pitfalls: SmemPtr large memref, scf.IfOp syntax, O=3 patch, pre-load+barrier coordination, iter_args VGPR regression, ds_read_tr regression, IR explosion, nested scf.for_, hardcoded grid, grid dimension order, TP compile-time constants)
- `reference-kernels/amd/cdna4/flydsl/FlyDSL/chunk_gdn_*.py` (5 kernels + pipeline final implementation code)
- Prerequisites: `ref-docs/amd/flydsl/flydsl-programming-guide.md` + `pitfalls/amd/flydsl/flash-attn-pitfalls.md`

**Group 6: FlyDSL FP8 PTPC Fused MoE Archive**
- `ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md` (proj007 task66 isolated stage checkpoint, bandwidth `target_us` method, and continuation boundaries)
- `ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md` (atrex-open integrated v2 full-pipeline archive with same-machine parity against atrex-open)
- `reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/` (task66 checkpoint source and bandwidth harness)
- `reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/` (atrex-open v2 standalone full-pipeline package)
- `pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md` (shared task66 and atrex-open v2 traps; do not mix isolated target_us and full-pipeline parity gates)

**Group 6: CUDA NVFP4 Split-K GEMV Trilogy**
- `nvidia/blackwell-geforce/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md` (C1 CUTLASS + C2 Split-K journey, E2E evidence, dispatch recipe)
- `nvidia/blackwell-geforce/cuda/pitfalls/nvfp4-split-k-gemv-pitfalls.md` (7 pitfalls: full shape substitution, cold standalone, K/S alignment, SF layout, workspace, custom-op graph boundaries, Stream-K confusion)
- `reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/` (CUDA kernel + vLLM CUTLASS dispatch example)
- Complementary background: `nvidia/common/cutedsl/cutlass-tile-scheduling.md` (Stream-K / split-K scheduler concepts)**Group 7: SM120 CuTeDSL GDN chunk-forward three-piece set**
- `nvidia/blackwell-geforce/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md` (V0→V113 journey, ncu/nsys evidence, V30 direct V rejection, V31 scaled-vnew, V113 reuse-B LDSM, V122 NCU bandwidth)
- `nvidia/blackwell-geforce/cutedsl/pitfalls/gdn-chunk-fwd-pitfalls.md` (26 pitfalls: cp.async layout, TMA deadlock/regression, direct V regression, K-decay scratch algebra removal, transposed LDSM atom, no-cache acceptance, NCU duration interpretation)
- `reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/` (V113 production kernel + V31 0.615ms backup + pre-V31 1.18ms backup)
- Complementary comparison: `ref-docs/amd/flydsl/gfx950/cdna4-chunk-gdn.md` (AMD FlyDSL chunk-GDN fwd_h different optimization axes)

---

## 5. Known Issues

### Content displacement in converter/amd/common/hopper-conversion-guide.md

This file is located under `amd/common/` but actually contains **Hopper (NVIDIA sm_90)** content. Its original package metadata identified a Hopper converter, and the content references `cuda:90`, `wgmma`, `NVMMASharedLayout`, and other NVIDIA APIs. It is suspected to have been copied from Hopper conversion guidance without modification.

### Inconsistent CDNA3 documentation structure

CDNA3's GEMM optimization is spread across 5 files (pattern_overview, optimization_strategy, warp_pipeline_stage, final_config_template, key_conclusions), whereas CDNA4 and Hopper each use only 1 `matmul.md`. This is because the CDNA3 docs were accumulated incrementally, while CDNA4/Hopper were consolidated later in a unified manner.
