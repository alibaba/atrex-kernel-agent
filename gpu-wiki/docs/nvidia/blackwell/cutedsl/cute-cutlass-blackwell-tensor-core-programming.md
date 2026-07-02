# CuTe and CUTLASS Programming for Blackwell Tensor Cores

A comprehensive guide to programming Blackwell's fifth-generation Tensor Cores using the CuTe abstraction library and CUTLASS 4.0 framework, based on the GTC25 presentation covering architecture features, CuTe MMA programming, TMA, 2-SM operations, TMEM epilogues, and the CUTLASS CollectiveBuilder.


**Last updated**: 2026-06-30

---

## 1. Introduction

CUTLASS is a C++ template-based CUDA kernel library for implementing high-performance matrix operations, specifically optimized for Tensor Cores. It provides highly optimized GEMM (General Matrix Multiply) and its variants (convolutions, grouped GEMM, sparse GEMM, etc.) decomposed into reusable, composable modular components.

CUTLASS serves as a foundation for NVIDIA's official libraries (cuBLAS, cuDNN) and is a core dependency for PyTorch, TensorRT-LLM, FlashAttention, and other top-tier frameworks. Since its 2018 debut, CUTLASS has evolved alongside Volta, Ampere, Hopper, and Blackwell architectures.

## 2. Blackwell Architecture Features

Three core hardware innovations form the foundation of Blackwell's performance:

### 2.1 Fifth-Generation Tensor Cores (tcgen05)

Key changes from Hopper to Blackwell:

| Property | Hopper | Blackwell |
|----------|--------|-----------|
| Operand A source | Registers or SMEM | TMEM or SMEM |
| Accumulator location | Registers | TMEM |
| Async scope | Compute only | Compute + Epilogue |
| Instruction issue | Warpgroup (4 warps) | Single thread |
| Cross-SM execution | No | Yes (2-SM MMA) |

The 2-SM MMA mechanism allows a CTA pair in a 2x1 physical cluster to cooperatively execute a single MMA instruction. One thread in the leader CTA issues the instruction, but computation is distributed across both SMs, effectively doubling the MMA M-dimension (e.g., from 128 to 256) and achieving 2x throughput for all Hopper-supported data types.

### 2.2 New Block-Scale Data Types

Blackwell natively supports MXFP8, MXFP6, MXFP4, and NVIDIA's NVFP4 with hardware block-scaling. The `tcgen05.mma` instruction loads scale factor matrices from TMEM alongside the data. Performance improvements: 2x throughput for MXFP8/FP6/FP4 vs. Hopper FP8; 4x throughput for NVFP4 and non-mixed MXFP4.

### 2.3 Tensor Memory (TMEM)

TMEM is a new 256 KB on-chip memory per SM (same capacity as the register file), organized as 128 lanes x 512 columns (4B each). Key characteristics:

- Each warp in a warpgroup can only access its corresponding 32 lanes
- Explicitly managed via `tcgen05.alloc` and `tcgen05.dealloc`
- Data movement via `tcgen05.load/store` (TMEM↔RMEM) or `tcgen05.cp` (SMEM↔TMEM)
- Fixed access patterns based on pre-defined block layouts
- Not addressable by general SIMT instructions

### 2.4 New Scheduling Capabilities

**Preferred Thread Block Clusters:** Blackwell allows a grid to use two cluster shapes simultaneously — a preferred larger size and a fallback smaller size — to minimize SM fragmentation. CUTLASS supports runtime cluster shape specification.

**Dynamic Tile Scheduling:** The new `clusterlaunchcontrol` (CLC) PTX instruction enables SMs to dynamically request work at runtime. CUTLASS implements a Dynamic Persistent Scheduler as the default on Blackwell, eliminating tail effects from static scheduling.

## 3. CuTe Programming for Blackwell

### 3.1 MMA Atom and TiledMMA

CuTe abstracts MMA instructions through two components:

- **MMA Atom:** Inline PTX assembly wrapper (MMA Op) + metadata template (MMA Traits) including data partition definitions and layout patterns
- **TiledMMA:** Higher-level abstraction that tiles an MMA Atom across a larger tile, automatically decomposing macro-level GEMM tasks into hardware-supported MMA instruction sequences

Key distinction: Hopper MMA partitions work among threads within a CTA; Blackwell MMA partitions work among one or two CTAs. This is reflected in `ThrID = Layout<_1>` (single CTA) or `Layout<_2>` (CTA pair).

### 3.2 Blackwell MMA GEMM Example

A complete GEMM kernel construction follows these steps:

1. **Define global memory tensors** using `make_tensor` with pointer and layout
2. **Create MMA tiler view** using `mma_tiler` and `local_tile` (scheduling unit is now MMA-tile, not CTA-tile)
3. **Partition within MMA-tile** using `tiled_mma.get_slice(mma_v)` and `mma.partition_A/B/C`
4. **Allocate SMEM** using `UMMA::tile_to_mma_shape` for optimized layouts
5. **Create MMA fragments** — descriptors for A/B (SMEM addresses) and C (TMEM tensor)
6. **Copy GMEM→SMEM** using `cooperative_copy` or TMA
7. **Execute GEMM** via `gemm(tiled_mma, tCrA, tCrB, tCtC)`

### 3.3 TMA (Tensor Memory Accelerator)

TMA Atoms encapsulate `cp.async.bulk.tensor` instructions. The `make_tma_atom` factory function binds a GMEM tensor, SMEM layout, and tiler together. TMA enables async data loading overlapped with MMA computation (pipelining).

Key TMA features on Blackwell:
- Up to 5D tensor descriptors
- Cluster-scope shared memory targeting (multicast)
- mbarrier-based synchronization

### 3.4 MMA.2SM and TMA.2SM

For 2-SM MMA, CuTe provides:

- `MMA_Traits` with `ThrID = Layout<_2>` and `FrgTypeC = UMMA::tmem_frg_2sm<c_type>`
- MMA Op corresponding to `tcgen05.mma.cta_group::2.kind::f16`
- Copy Atoms using `SM100_TMA_2SM_LOAD` for cross-SM data loading

The 2SM tiling scheme combines two adjacent tiles into one larger MMA tile (doubled M dimension). Only the leader CTA (Peer ID = 0) issues the `gemm()` call, but the instruction automatically triggers cooperative execution on both SMs.

### 3.5 TMEM Epilogue

After the GEMM main loop, accumulator results reside in TMEM. The epilogue transfers them to global memory:

1. **Create TiledCopy** using `make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tCtC)`
2. **Partition** source (TMEM), destination (GMEM), and intermediate (RMEM) tensors
3. **Execute TMEM→RMEM copy:** `copy(tiled_cpy, tDtC, tDrC)`
4. **Execute RMEM→GMEM write:** `axpby(alpha, tDrC, beta, tDgC)` (fused scaling + store)

The two-stage design (TMEM→RMEM→GMEM) enables arbitrary element-wise operations between stages (activation functions, quantization, etc.).

## 4. CUTLASS 4.0 for Blackwell

### 4.1 CollectiveBuilder Programming Model

CUTLASS solves the combinatorial explosion of kernel variants through its Collective Layer. The `CollectiveBuilder` acts as an intelligent factory — specify high-level requirements and it generates optimized implementations.

**CollectiveMainloop Builder parameters:**
- Architecture tag (`cutlass::arch::Sm100`)
- Input data types and layouts
- MMA Shape (e.g., `Shape<_256,_128,_64>`)
- Cluster Shape (e.g., `Shape<_2,_2,_1>`)
- Kernel Schedule (`KernelScheduleAuto`)

**CollectiveEpilogue Builder parameters:**
- Architecture, MMA Shape, Cluster Shape (matching mainloop)
- Output C/D types and layouts
- Epilogue Schedule (`EpilogueScheduleAuto`)
- FusionOperation (Epilogue Visitor Tree hook)

### 4.2 Complete Kernel Assembly

```cpp
// Step 1: Build CollectiveMainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, /* ... */>::CollectiveMma;

// Step 2: Build CollectiveEpilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, /* ... */>::CollectiveOp;

// Step 3: Compose into GemmUniversal
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;
```

### 4.3 Pre-tuned Mainloop Collectives

| Scenario | Collective Name |
|----------|----------------|
| Dense GEMM (non-block-scale) | `MainloopSm100TmaUmmaWarpSpecialized` |
| Dense GEMM (block-scale) | `MainloopSm100TmaUmmaWarpSpecializedBlockScaled` |
| Grouped GEMM | `MainloopSm100ArrayTmaUmmaWarpSpecialized` |
| Grouped GEMM (block-scale) | `MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled` |
| Implicit GEMM Convolutions | TMA-based im2col variant |
| Fast emulated FP32 | Low-precision hardware simulation |
| Software block-scaling | Custom scale factor handling |

### 4.4 Pre-tuned Epilogue Collectives

- **TMA Store:** `Sm100TmaWarpSpecialized` / `Sm100ArrayTmaWarpSpecialized` — async writeback via TMA, best for large output tiles
- **Direct Store:** `Sm100NoSmemWarpSpecialized` / `Sm100PtrArrayNoSmemWarpSpecialized` — SIMT cores write directly from registers, saves SMEM

### 4.5 Runtime Configuration

CUTLASS 4.0 introduces runtime flexibility to reduce kernel compilation explosion:

- **Runtime Thread Block Clusters:** Cluster shape passed as launch-time parameter
- **Runtime Data Types:** `type_erased_dynamic_float8_t` handles all same-bitwidth FP8 variants with one compiled kernel

### 4.6 Scheduling Strategies

| Strategy | Description |
|----------|-------------|
| Dynamic Persistent (CLC) | Default on Blackwell; dynamic load balancing |
| Stream-K | K-dimension parallelism for thin/flat matrices |
| Static Persistent | Minimal overhead, fixed mapping |

### 4.7 Warp-Specialized Kernel Structure (Hopper vs. Blackwell)

**Hopper:** Accumulator in registers; MMA and Epilogue executed by same warps; ping-pong between two MMA warp groups to overlap.

**Blackwell:** Accumulator in TMEM (visible to all warps); MMA and Epilogue fully decoupled into separate warp groups; natural producer-consumer pipeline without ping-pong. Roles: Data Loading Warps → MMA Warp → Epilogue Warps (+ optional Scheduling Warp for CLC).

### 4.8 Migration from Hopper to Blackwell

Migrating a CUTLASS Hopper kernel requires only two changes:
1. Change architecture tag: `Sm90` → `Sm100`
2. Reinterpret tile shape semantics: CTA Shape → MMA Shape

The scheduler automatically switches from Static Persistent to Dynamic Persistent.

## 5. Summary

CUTLASS 4.0 provides Day-0 high-performance support for Blackwell through:

- Full adaptation of tcgen05 MMA instructions and TMEM operations
- New software pipelines designed for Blackwell's async model
- Out-of-the-box kernels with 2x performance for dense GEMM, convolutions, and grouped GEMM
- Complete support for MXFP8/FP6/FP4 block-scaled data types (up to 4x Hopper FP8)
- Dynamic persistent scheduling, runtime cluster shapes, and runtime data types

The layered architecture (CuTe Atoms → TiledCopy/TiledMMA → Collective Layer → Kernel Layer) provides different levels of control: hardware experts can innovate at the atom level, operator developers compose at the collective level, and application developers use device-level universal kernels.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [Tensor Core from Volta to Blackwell](../../common/tensor-core-volta-to-blackwell.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
