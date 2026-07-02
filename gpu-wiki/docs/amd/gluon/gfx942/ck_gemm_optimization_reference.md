# AMD MI308X (gfx942) GEMM Optimization Techniques Reference

> **Purpose**: This document is an optimization methodology reference for understanding and adapting high-performance GEMM patterns on the AMD MI308X (gfx942, CDNA3) platform. It covers core optimization concepts, design patterns, and decision frameworks applicable to this platform.

**Last updated**: 2026-06-30

>
> **Target Hardware**: AMD Instinct MI308X — gfx942 Architecture (CDNA3)
>
> **Key Hardware Characteristics**:
> - **Compute Units**: 80 CUs, each with 4 SIMDs, each SIMD executing 16 lanes (Wavefront = 64 threads)
> - **Matrix Multiply Instructions**: MFMA (Matrix Fused Multiply-Add) and SMFMAC (Structured Sparse), **does not support scaled MFMA instructions**
> - **Memory Hierarchy**: HBM3 (5.3 TB/s bandwidth) → L3 Cache (256 MB, shared across dies) → L2 Cache (4 MB/XCD) → LDS (64 KB per CU) → VGPR/SGPR
> - **Multi-die Architecture**: 4 XCDs (Accelerated Compute Die), each XCD containing 20 CUs, interconnected via Infinity Fabric
> - **LDS Capacity**: 64 KB per CU, maximum 64 KB usable per Workgroup
> - **VGPR**: 512 KB per CU (128 KB per SIMD = 512 32-bit VGPRs)
> - **Cache Hierarchy**: L2 Cache 4 MB/XCD (shared within XCD), L3 Cache 256 MB (globally shared)

---

## 1. Overall Design Philosophy

### 1.1 Layered Abstraction Architecture

High-performance GEMM implementations adopt a layered architecture, with each layer focusing on optimization at different granularities:

| Layer | Focus | Optimization Goal |
|------|--------|---------|
| **Wave Level** | Encapsulation and scheduling of MFMA/SMFMAC instructions | Maximize instruction throughput |
| **Workgroup Level** | LDS buffering, pipeline orchestration | Hide HBM memory access latency |
| **Grid/Kernel Level** | Mapping from Workgroup to output tiles, load balancing across 4 XCDs | Maximize GPU utilization |
| **Device/Host Level** | Kernel launch, parameter configuration, instance selection | Ease of use and flexibility |

### 1.2 Core Design Principles

- **Tile-based Programming Model**: Decompose matrices into multi-level tiles, each mapped to different hardware levels of gfx942 (Grid → Workgroup → Wave → Thread)
- **Compute and Memory Access Overlap**: Use pipelining and multi-buffering techniques to keep MFMA units and memory units busy simultaneously
- **Compile-time Specialization**: Eliminate runtime branches at compile time through templates, reducing instruction overhead
- **Multi-XCD Awareness**: Workgroup mapping must consider the 4-XCD multi-die topology of MI308X to maximize L2 Cache locality

---

## 2. Pipeline Optimization (Core Technique)

Pipelining is central to GEMM performance, controlling the overlap of HBM loading, LDS buffering, and MFMA computation. On MI308X, HBM3 latency is approximately ~400 cycles, while a single MFMA instruction latency is 8-16 cycles. Deep pipelining is therefore essential HD to hide memory latency.

### 2.1 Pipeline Evolution Path

From simple to complex, pipeline design follows this evolution path:

| Stage | Core Idea | Prefetch Depth | Buffering Strategy | Resource Consumption | Applicable Scenarios |
|------|---------|---------|---------|---------|---------|
| **Basic** | Serial load-compute | No prefetch | Single buffer | Lowest | Small-scale GEMM, resource-constrained |
| **Simple Prefetch** | Preload next round of data | 1 level | Single buffer | Low | Medium-scale |
| **Instruction-Level Scheduling** | Precisely control MFMA/VMEM/DS instruction issue timing | 2 levels | Single buffer | Medium | Compute-intensive |
| **Double Buffering** | Ping-Pong LDS alternating read/write | 3 levels | Double buffer | High | High-performance scenarios |
| **Wave-Level Interleaving** | Different Waves take distinct roles (data movement vs. computation) | 3 levels | Double buffer | Medium-High | Memory-latency-sensitive |
| **Deep Prefetch** | Configurable multi-stage prefetch | 1-7 levels | Single buffer | Variable | Memory-bandwidth-bound |

### 2.2 Global Prefetch

**Core Idea**: While the current K-block is undergoing MFMA computation, preload data for the next (or multiple subsequent) K-blocks from HBM into LDS, overlapping MFMA computation with HBM access.

**Design Considerations**:
- **Prefetch Depth**: The more tiles prefetched, the stronger the ability to hide HBM latency, but the more LDS/VGPR space consumed
- **Tail Handling**: When the K dimension is not divisible by the tile size, special tail handling logic (such as padding or conditional loading) is required
- **Prefetch Depth Selection on MI308X**: HBM3 latency is ~400 cyclesql; the MFMA computation time for a single tile depends on KPerBlock and the number of MFMA instructions. Typically, 2-3 levels of prefetch are sufficient to effectively hide latency

### 2.3 LDS Multi-Buffering (Double/Triple Buffering)

**Core Idea**: Use multiple copies of LDS space for alternating reads and writes — one for current MFMA computation455, another for loading the next round of data from HBM — enabling full overlap of computation and LDS writes.

**Ping-Pong Double Buffering Flow**:
```
Ping phase:// Use current buffer for computation
Pong phase:
  2. Read data from LDS buffer0 to VGPR
  3. Write HBM data to LDS buffer1
  4. Prefetch next round of HBM data
  5. Execute MFMA using buffer0 VGPR data
```

**Trade-offs on MI308X**: Double buffering doubles LDS usage (64 KB LDS per CU), which may reduce occupancy (the number of simultaneously active Workgroups per CU). For example, when double buffering uses 48 KB of LDS, only 1 Workgroup can run per CU.

### 2.4 Instruction-Level Scheduling (Intrawave Scheduling)

**Core Idea**: Leverage gfx942's hardware scheduling barrier `s_sched_group_barrier` to precisely control the issue timing of four categories of instructions, achieving fine-grained overlap of computation and memory access.**Four Instruction Types on gfx942**:
- **MFMA Instructions**: Matrix multiply-accumulate, latency 8–16 cycles
- **VMEM Read**: Load data from HBM to VGPR, latency ~400 cycles
- **DS Read**: Read data from LDS to VGPR, latency ~20 cycles
- **DS Write**: Write VGPR data to LDS, latency ~20 cycles

**Scheduling Objective**: Interleave these four instruction types so that the MFMA unit, VMEM unit, and LDS unit work concurrently. The typical scheduling pattern is: issue several DS reads → issue several MFMAs → issue DS writes → issue VMEM reads, and repeat.

### 2.5 Wave-Level Interleaved Execution

**Core Idea**: Divide the Waves within a Workgroup into two groups—one dedicated to data movement (Memory Ops), the other dedicated to MFMA computation (Compute Ops)—achieving maximum overlap through role specialization.

**Design Points**:
- Use LDS as the data exchange medium between the two groups of Waves
- The two groups execute alternately, switching roles via rotating operation_id
- Finally merge the computation results in VGPRs
- Particularly effective on MI308X because each CU has 4 SIMDs and can schedule multiple Waves simultaneously

### 2.6 Scheduler Selection Strategy

| Scheduling Strategy | Description | MI308X Applicable Scenarios |
|---------|------|----------------|
| **Default Scheduling** | Rely on hardware default scheduling | General scenarios |
| **Intrawave** | Precisely control MFMA/VMEM/DS instruction issue order within a single Wave | Compute-bound (large matrices), primary scheduling method |
| **Interwave** | Coordinate across Waves, reducing block_sync overhead | Memory bandwidth-bound (large K dimension), paired with deep prefetch |

---

## III. MFMA Instructions and Wave-Level Optimization

### 3.1 MI308X (gfx942) MFMA Instruction Set

MI308X (gfx942) supports a rich set of MFMA instructions, which form the foundation of GEMM performance. **Note: MI308X does not support MFMA instructions with scale (such as `v_mfma_scale_f32_*_f8f6f4`), but does support standard FP8/BF8 MFMA instructions (without scale).**

**Single-Block (1B) Instructions** (most commonly used in GEMM kernels):

| Data Type | Matrix Dimensions | Accumulator Type | Latency (cycles) | Description |
|---------|----------|---------|--------------|------|
| FP16 | 32×32×8 | FP32 | 32 | Basic FP16 |
| FP16 | 16×16×16 | FP32 | 16 | Small tile FP16 |
| BF16 | 32×32×8 | FP32 | 32 | Basic BF16 |
| BF16 | 16×16×16 | FP32 | 16 | Small tile BF16 |
| FP8/BF8 | 32×32×16 | FP32 | 32 | FP8/BF8 (without scale) |
| FP8/BF8 | 16×16×32 | FP32 | 16 | FP8/BF8 (without scale) |
| INT8 | 32×32×16 | INT32 | 32 | Integer quantization |
| INT8 | 16×16×32 | INT32 | 16 | Integer quantization |
| XF32 | 32×32×4 | FP32 | 32 | Reduced-precision F32 (10-bit mantissa) |
| XF32 | 16×16×8 | FP32 | 16 | Reduced-precision F32 (10-bit mantissa) |
| FP64 | 16×16×4 | FP64 | 32 | Double precision |
| FP32 | 32×32×2 | FP32 | 64 | Full-precision F32 |
| FP32 | 16×16×4 | FP32 | 32 | Full-precision F32 |

**Multi-Block (nB) Instructions** (for small tile scenarios):

| Data Type | Matrix Dimensions | Block Count | Latency (cycles) |
|---------|----------|------|--------------|
| FP16 | 32×32×4_2B | 2 | 64 |
| FP16 | 16×16×4_4B | 4 | 32 |
| FP16 | 4×4×4_16B | 16 | 8 |
| INT8 | 32×32×4_2B | 2 | 64 |
| INT8 | 16×16×4_4B | 4 | 32 |
| INT8 | 4×4×4_16B | 16 | 8 |
| FP64 | 4×4×4_4B | 4 | 16 |

**Key Notes**:
- MI308X's FP8/BF8 MFMA are standard instructions (`V_MFMA_F32_*_FP8/BF8`), without a scale factor, with inputs directly in FP8/BF8 format
- Does not support `v_mfma_scale_f32_*_f8f6f4` (scaled FP8/FP6/FP4 instructions), and therefore does not support hardware acceleration for FP4/FP6/MX Format
- Latency refers to the number of cycles from instruction issue to result availability, and is related to the pipeline depth of the MFMA unit

### 3.2 Wave-Level GEMM Design

**Core Idea**: Encapsulate MFMA instructions as Wave-level GEMM primitives, where a single Wave (64 threads) collaboratively performs matrix multiplication.

**Design Points**:
- **Instruction Selection**: Automatically choose the optimal MFMA instruction based on data type and tile dimensions (e.g., FP16 prefers 32×32×16)
- **K-dimension Iteration**: When KPerBlock exceeds the K dimension of a single MFMA, accumulate results through iterative MFMA instructions
- **Data Distribution**: MFMA instructions require input data to be distributed in a specific pattern across 64 lanes, requiring management of A/B/C matrix lane distribution
- **Swizzle Optimization**: Reduce LDS bank conflicts by rearranging the lane distribution of input data
- **C Matrix Transpose Distribution**: Support transposing the distribution of the C matrix output by MFMA to accommodate different write-back patterns

### 3.3 SMFMAC Structured Sparsity Acceleration

**Core Idea**: gfx942 supports SMFMAC instructions, leveraging 2:4 structured sparsity (at most 2 non-zero values out of every 4 consecutive elements), compressing the A matrix to half its size while using specialized sparse matrix multiplication instructions Hedge acceleration.

**Key Features**:
- 2:1 compression ratio, effectively doubling HBM bandwidth
- Requires additional storage of 2-bit index information for non-zero elements
- Only supports sparsity in the A matrix (B matrix remains dense)
- Requires preprocessing to convert dense matrices to 2:4 sparse format + indices

**Supported SMFMAC Instructions**:
- FP16: 16×16×32, 32×32×16
- BF16: 16×16×32, 32×32×16

## IV. Data Layout and Memory Optimization

### 4.1 Multi-Level Tile Blocking

**Core Idea**: Decompose the M×N×K dimensions of GEMM into multi-level tiles, with each level mapped to a different hardware layer of MI308X.

| Level | Mapping Target | MI308X Resource Constraint | Optimization Goal |
|------|---------|----------------|---------|
| **Block Tile** (M_block × N_block × K_block) | Workgroup | LDS 64 KB/CU | Maximize LDS data reuse |
| **Wave Tile** (M_wave × N_wave) | Wave (64 threads) | MFMA instruction size | Match MFMA instruction size |
| **Thread Tile** (M_repeat × N_repeat) | Single thread | VGPR 512/SIMD | Maximize VGPR reuse |

**Tile Size Selection Principles on MI308X**:
- Block Tile is constrained by LDS 64 KB: A_tile + B_tile ≤ 64 KB (single buffering) or ≤ 32 KB (double buffering)
- Wave Tile must be an integer multiple of the MFMA instruction size (e.g., 32×32 or 16×16)
- Larger Thread Tile yields better VGPR reuse, but increased VGPR usage reduces occupancy (fewer Workgroups can run concurrently per CU)

### 4.2 Output Matrix Shuffle Write-Back (CShuffle)

**Core Idea**: The output of MFMA instructions is distributed across the VGPRs of 64 lanes within a Wave. The distribution pattern is determined by the MFMA instruction and is usually not suitable for direct coalesced write-back to HBM. By routing through LDS, the scattered C matrix data is rearranged into a contiguous layout before being written back.

**Flow**:
```
MFMA output (VGPR, distributed across 64 lanes)
  → Write to LDS (rearrange layout)
  → Read from LDS (contiguous layout)
  → Apply post-processing operations (such as activation function, Bias addition)
  → Write back to HBM (coalesced access, 128-bit store)
```

**Optimization Significance**: Converts non-coalesced HBM writes into coalesced writes, significantly improving write-back bandwidth utilization. Post-processing operations can also be fused during the LDS stage, avoiding additional kernel launches.

### 4.3 LDS Direct Load

**Core Idea**: Load data directly from HBM to LDS, bypassing the VGPR intermediate step to reduce VGPR pressure.

**Applicable Conditions**:
- Each thread loads 4 bytes (1 DWORD)
- 64 threads within a Wave write to contiguous LDS DWORDs
- Source and destination data types must be the same (no type conversion supported)

**Optimization Significance on MI308X**: Reduces VGPR usage impression, improving occupancy. Particularly effective in configurations with high VGPR pressure (e.g., large Thread Tile + double buffering).

### 4.4 Coordinate Transformation Abstraction

**Core Idea**: Through a coordinate transformation layer, complex memory access patterns (such as convolution's im2col, transpose, tiling) are unified as tensor coordinate transformation operations, freeing upper-level algorithms from needing to concern themselves with the details of underlying memory layouts.

**Application Scenarios**:
- Convolution rearrangement: Image to Column / Column to Image
- Matrix transpose: Row-major ↔ Column-major
- Multi-dimensional tiling: Mapping high-dimensional tensors to low-dimensional memory space
- Strided access: Handling non-contiguous memory layouts

### 4.5 Vectorized Data Transfer

**Core Idea**: Use gfx942's vectorized load/store instructions (e.g., `buffer_load_dwordx4` = 128-bit) to transfer multiple data elements at once, maximizing HBM bandwidth utilization.

**MI308X Design Points**:
- **Optimal Vector Size**: 128-bit (e.g., FP16 × 8 = 128 bit, INT8 × 16 = 128 bit)
- **Alignment Requirement**: Starting address must be aligned to the vector size
- **Vectorization Dimension**: Choose the memory-contiguous dimension (typically the innermost dimension of K) for vectorization
- **Buffer Instructions**: gfx942 supports `buffer_load/store` instructions, providing out-of-bounds checking and automatic padding

### 4.6 K-Dimension Parallelism (SplitK)

**Core Idea**: Split the K dimension across multiple Workgroups for parallel computation, where each Workgroup computes a partial sum, and results are finally merged through reduction.

**MI308X Applicable Scenarios**: Matrices with small M and N but large K, where the traditional M×N parallelism cannot fully utilize MI308X's 80 CUs.

**Reduction Strategies**:
- **Atomic Reduction**: Use HBM atomic operations for direct accumulation, simple but may have contention
- **Explicit Reduction**: Use workspace buffer to store partial sums, then launch a separate reduction kernel to merge results

---

## V. FP8 Quantization and Low-Precision Optimization

MI308X (gfx942) supports the standard FP8/BF8 MFMA instruction set, enabling direct matrix multiplication with FP8/BF8 inputs and FP32 accumulation. The theoretical peak throughput is double that of FP16.

### 5.1 Weight Preshuffle

**Core Idea**: Pre-rearrange quantized weight matrices offline so that they match the data consumption pattern of MFMA instructions when loaded, eliminating runtime data rearrangement within the kernel.

**MI308X Applicable Scenarios**: Inference phase (weights are fixed). gfx942's MFMA instructions impose strict requirements on the lane distribution of input data, and preshuffling can completely eliminate runtime data rearrangement.

**Optimization Significance**: Transfers the runtime data rearrangement overhead to an offline preprocessing stage, achieving zero runtime overhead.

### 5.2 Online Dequantization

**Core Idea**: Integrate dequantization operations into the GEMM computation pipeline. Quantized weights are dequantized to a high-precision format immediately after being loaded from LDS into VGPRs, participating in MFMA computation without the overhead of launching a separate dequantization kernel.

**Design Points**:
- Dequantization is performed within VGPRs (VGPR-to-VGPR), adding no HBM/LDS access
- Requires coordination with Intrawave scheduling to insert dequantization instructions into the gaps between MFMA and DS read operations
- Can be fused with GEMM + Scale + Scale to implement Weight-Only Quantization

### 5.3 FP8/BF8 MFMA Native Support

**Core Idea**: MI308X supports standard FP8/BF8 MFMA instructions (`V_MFMA_F32_32x32x16_FP8/BF8` and `V_MFMA_F32_16x16x32_FP8/BF8`), allowing direct matrix multiplication with FP8/BF8 format inputs and accumulation in FP32.**Design Points**:
- Supports FP8×FP8, FP8×BF8, BF8×BF8, and other combinations
- No hardware scale factor; per-block scale must be managed at the software level (apply scale correction in VGPR after MFMA computation)
- FP8 → FP32 type conversion instructions (`V_CVT_F32_FP8`, `V_CVT_PK_F32_FP8`) can be used in scenarios requiring explicit dequantization
- HBM bandwidth requirement is only half that of FP16, providing significant benefits for bandwidth-bound scenarios

**Difference from scaled MFMA**: MI308X does not support `v_mfma_scale_f32_*_f8f6f4` (supported by higher-end models such as MI300X/MI350), so scale factors must be manually applied in VGPR via VALU instructions after MFMA computation is complete.

### 5.4 Per-Block Scaling (AB Scale)

**Core Idea**: Apply independent per-block scaling operations to the A and B matrices, used in scenarios such as FP8 training that require dynamic range adjustment.

**Design Points**:
- Each tile block has an independent scale factor
- Scale factor granularity affects the trade-off between precision and performance
- Scale correction must be applied in VGPR after MFMA computation

---

## VI. System-Level Optimization

### 6.1 Workgroup-to-Tile Mapping Strategy

**Core Idea**: Optimize the mapping relationship between workgroups and output matrix tiles to improve L2 cache hit rates and cross-XCD load balancing.

| Strategy | Description | MI308X Applicability |
|------|------|----------------|
| **2D Mapping** | blockIdx.x → M, blockIdx.y → N | General-purpose, simple and straightforward |
| **1D Mapping** | Single blockIdx linear mapping | When flexible grid size control is needed |
| **Spatial Locality Grouping** | Group adjacent workgroups onto the same XCD | **Critical MI308X optimization**, maximizes intra-XCD L2 cache reuse |
| **Swizzle Mapping** | Apply swizzle transformation to workgroup indices | Reduce cross-XCD L2 cache conflicts |
| **Dynamic Mapping (StreamK)** | Workgroups dynamically claim tasks | Irregular matrix dimensions |

**MI308X Multi-XCD Spatial Locality Optimization**: MI308X has 4 XCDs, each with an independent 4 MB L2 cache. Grouping spatially adjacent workgroups onto the same XCD for execution maximizes L2 cache data reuse and avoids cross-die Infinity Fabric traffic. The grouping parameter `GroupNum` controls the number of large groups (typically set to the XCD count of 4), and `M01` controls the local grouping size within the M dimension.

### 6.2 Persistent Kernel

**Core Idea**: A workgroup does not exit after completing one output tile but continues processing the next tile, reducing kernel launch overhead and workgroup scheduling overhead.

**MI308X Design Points**:
- Dynamically determine grid size based on occupancy (= 80 CUs × max workgroups per CU)
- Workgroups claim tasks via a global counter or pre-allocated table
- Effectively reduces kernel launch and workgroup scheduling overhead on MI308X

### 6.3 Dynamic Load Balancing (StreamK)

**Core Idea**: Break away from the traditional data-parallel decomposition (where each workgroup processes a fixed output tile for the entire K iteration), allowing workgroups to process a variable number of K blocks for more uniform load distribution.
**MI308X Applicable Scenarios**:
- Small K dimensions where traditional decomposition cannot fully utilize all 80 CUs
- Irregular matrix dimensions leading to uneven load across 4 XCDs

**Reduction Strategies**:
- **Atomic Reduction**: Multiple workgroups accumulate to the same output tile via HBM atomic operations
- **Explicit Reduction**: Store partial results in a workspace buffer, then merge with a reduction kernel

### 6.4 Grouped GEMM / Batched GEMM

**Core Idea**: Efficiently handle multiple GEMM problems of varying sizes, avoiding separate kernel launches for each GEMM. Particularly important on MI308X, as its 80 CUs require sufficient parallelism to be fully utilized.

**Two Implementation Strategies**:
- **Parameter Packing**: Pack all GEMM parameters into a single buffer, launching one kernel to process all problems
- **Tile Loop**: Workgroups iterate over all GEMM problems in a loop without needing to know problem sizes in advance

**Combining with Persistent Kernel**: After processing a tile for one GEMM, a workgroup can continue by processing a tile for another GEMM, fully utilizing all CUs on MI308X.

### 6.5 Compile-Time Padding Specialization

**Core Idea**: Based on whether matrix dimensions are aligned to tile size, select different code paths at compile time — eliminate boundary checks when aligned, add padding handling when unaligned.

| Specialization Type | Description |
|---------|------|
| **No Padding** | All dimensions aligned, no boundary checks, optimal performance |
| **Single-Dimension Padding** | Padding needed for one of the M/N/K dimensions |
| **Multi-Dimension Padding** | Padding needed for multiple dimensions |
| **Full Padding** | Padding needed for all dimensions, most general but highest overhead |

**Optimization Significance**: Eliminate runtime branches through compile-time specialization. On MI308X, the gfx942 `buffer_load` instruction has built-in out-of-bounds checking (returns 0), which can partially replace software padding.

### 6.6 Epilogue Fusion (Post-Processing Fusion)

**Core Idea**: Fuse elementwise operations (such as Bias addition, activation functions, Scale) after GEMM output into the GEMM kernel, avoiding additional kernel launches and HBM read/write operations.

**Fusion Modes**:
```
E = ElementwiseOp(GEMM(A, B), D0, D1, ...)
```

**Common Fusion Operations**:
- **Bias Addition**: E = C + Bias
- **Activation Functions**: E = ReLU(C) / GeLU(C) / SiLU(C)
- **Scale**: E = C × Scale
- **Combined Operations**: E = ReLU(C + Bias)
**MI308X Optimization Significance**: Reduce the number of kernel launches and HBM accesses. Although MI308X has high HBM3 bandwidth (5.3 TB/s), kernel launch overhead and HBM access remain bottlenecks for small matrices.

### 6.7 MOE (Mixture of Experts) GEMM

**Core Idea**: Efficiently handle sparse GEMM computations in MOE models, where each token is routed to only a few experts.

**Key Design**:
- Sort tokens by expert, so that tokens for the same expert are stored contiguously
- Support two-stage GEMM: input-side GEMM (expansion) and output-side GEMM (contraction)
- Output-side GEMM uses atomic operations to accumulate results from different experts
- Support routing weight multiplication and per-token quantization
- Can be combined with weight pre-reordering + blockscale

### 6.8 Occupancy Optimization

**Core Idea**: Maximize the number of concurrently active Workgroups per CU by controlling each Workgroup's resource consumption (VGPR count, LDS size).

**MI308X Resource Constraints and Occupancy Relationship**:

| Resource | Per CU Total | 1 Workgroup | 2 Workgroups | 4 Workgroups |
|------|-----------|-------------|-------------|-------------|
| **VGPR** | 512/SIMD | ≤512 | ≤256 | ≤128 |
| **LDS** | 64 KB | ≤64 KB | ≤32 KB | ≤16 KB |

**Key Trade-offs**:
- **VGPR vs Occupancy**: More VGPRs → Larger Thread Tile → Better compute efficiency, but lower occupancy
- **LDS vs Occupancy**: Double buffering doubles LDS usage reinsurance, potentially reducing occupancy from 2 to 1
- **Wave Group Partitioning**: Group Waves to execute different tasks (data movement vs computation), increasing parallelism without increasing resource usage

---
## VII. MI308X (gfx942) Data Type Support

| Data Type | MFMA Native Support | Optimal Instruction (Latency) | Accumulation Type | Typical Scenario | Optimization Tips |
|---------|-------------|----------------|---------|---------|---------|
| FP64 | ✅ | 16×16×4 (32c) | FP64 | HPC | Standard pipeline |
| FP32 | ✅ | 32×32×2 (64c) | FP32 | Training | Standard pipeline |
| XF32 | ✅ | 32×32×4 (32c) | FP32 | Training (reduced precision) | Double throughput, 10-bit mantissa |
| FP16 | ✅ | 32×32×8 (32c) | FP32 | Training/Inference (primary) | Most commonly used |
| BF16 | ✅ | 32×32×8 (32c) | FP32 | Training/Inference | Most commonly used |
| FP8 (E4M3) | ✅ without scale | 32×32×16 (32c) | FP32 | Inference | Requires software per-block scale |
| BF8 (E5M2) | ✅ without scale | 32×32×16 (32c) | FP32 | Inference | Requires software per-block scale |
| INT8 | ✅ | 32×32×16 (32c) | INT32 | Quantized inference | Requires online dequantization or scale correction |

**Note**: MI308X **does not support** scaled MFMA instructions (`v_mfma_scale_f32_*_f8f6f4`), nor does it support hardware acceleration for FP4/FP6/MX Format. FP8/BF8 uses standard MFMA instructions, and the scale factors must be applied manually via VALU instructions after the MFMA computation.

**Mixed Precision Strategy**: Use low precision for inputs (to save HBM bandwidth), accumulate in FP32 (to guarantee precision), and convert outputs as needed. The gfx942 MFMA instructions natively support FP16/BF16/FP8/BF8/INT8 low-precision input + FP32/INT32 accumulation.
---

## VIII. MI308X Performance Tuning Decision Framework

### 8.1 Scenario Analysis Decision Tree

```
Matrix dimension analysis (MI308X: 80 CU, 4 XCD, 5.3 TB/s HBM3, 64 KB LDS/CU, 4 MB L2/XCD, 256 MB L3):
├── M, N large, K large → Compute-intensive
│   ├── Choose 3-stage prefetch + Intrawave scheduling
│   ├── Use LDS double buffering
│   └── Use spatial locality Workgroup mapping (GroupNum=4, matches 4 XCDs)
├── M, N large, K small → Potential cross-XCD load imbalance
│   ├── Consider StreamK dynamic load balancing
│   └── Use Persistent Kernel
├── M, N small, K large → HBM bandwidth-limited
│   ├── Use SplitK to parallelize K dimension (fully utilize 80 CUs)
│   ├── Use deep prefetch pipeline (multi-stage)
│   └── Use Interwave scheduling
└── M, N small, K small → Kernel launch overhead is significant
    ├── Use Persistent Kernel
    ├── Consider Grouped GEMM batching
 └── decrease pipeline , low VGPR/LDS high occupancy
```

### 8.2 MI308X Performance Tuning Checklist

- [ ] **Pipeline Selection**: Use Intrawave + double buffering for compute-bound scenarios, deep prefetch + Interwave for HBM bandwidth-bound scenarios
- [ ] **LDS Buffering Strategy**: Double buffering vs single buffering (trade off between occupancy and latency hiding under the 64 KB LDS limit)
- [ ] **Prefetch Depth**: HBM3 latency ~400 cycles, typically 2-3 levels of prefetch are sufficient
- [ ] **Vectorization Width**: Use 128-bit vectorization (FP16×8 / FP8×16 / INT8×16)
- [ ] **Tile Size**: Ensure A_tile + B_tile ≤ LDS capacity (single buffer 64 KB / double buffer 32 KB)
- [ ] **Workgroup Mapping**: Use spatial locality grouping (GroupNum=4 to match 4 XCDs)
- [ ] **Padding Specialization**: Use non-padded version when dimensions are aligned; leverage buffer_load's out-of-bounds-returns-0 behavior
- [ ] **Occupancy**: Check whether VGPR (≤512/SIMD) and LDS (≤64 KB/CU) affect occupancy
- [ ] **Weight Pre-reordering**: Offline pre-reordering must be considered for inference scenarios (eliminates MFMA lane distribution reordering overhead)
- [ ] **K-Dimension Parallelism**: Consider SplitK or StreamK when K is large (fully utilize 80 CUs)
- [ ] **FP8 Optimization**: MI308X supports standard FP8 MFMA (without scale), scale factors must be applied manually via VALU after MFMA
- [ ] **MFMA Instruction Selection**: Prefer 32×32×8 (32c) for FP16/BF16, prefer 32×32×16 (32c) for FP8/INT8

## 9. MI308X Technical Combination Recommendation Matrix

| Scenario | Pipeline Strategy | MFMA Instruction | Special Technique | System Optimization |
|------|----------|----------|---------|---------|
| **FP16 Large Matrix Training** | Double Buffering + Intrawave | 32×32×8 FP16 (32c) | CShuffle Write-back | Spatial Locality Mapping (4 XCD) |
| **FP16 Small Matrix Inference** | Intrawave | 32×32×8 FP16 (32c) | Weight Pre-reorder | Persistent Kernel |
| **FP8 Inference** | Intrawave + Pre-reorder | 32×32×16 FP8 (32c, without scale) | Preshuffle + AB Scale | Persistent Kernel |
| **INT8 Quantized Inference** | Intrawave | 32×32×16 INT8 (32c) | Online Dequantization | Default Mapping |
| **2:4 Sparse Inference** | Intrawave | SMFMAC | 2:4 Structured Sparsity | Default Mapping |
| **MOE Inference** | Intrawave + Pre-reorder | 32×32×8 FP16 (32c) | MOE + Dequantization + blockscale | Grouped GEMM |
| **Irregular Matrices** | Intrawave | 32×32×8 FP16 (32c) | StreamK | Dynamic Load Balancing |
| **Grouped GEMM** | Intrawave | 32×32×8 FP16 (32c) | Tile Loop | Persistent Kernel |
| **Large K Dimension** | Deep Prefetch + Interwave | 32×32×8 FP16 (32c) | SplitK | Interwave Scheduling |
| **GEMM + Activation Fusion** | Intrawave | 32×32×8 FP16 (32c) | Epilogue Fusion | Default Mapping |

---

#// Prefetch next iteration to another buffer

### 10.3 Manual MFMA Loop Unrolling

**Purpose**: Manually unroll the MFMA loop in the K dimension to increase instruction-level parallelism.

```python
# BLOCK_K=64, MFMA 16, requires 4
a_slice = a_smem.slice(0, 16, dim=1)
accumulator = gl.amd.cdna3.mfma(a_slice.load(...), b_slice.load(...), accumulator)

a_slice = a_smem.slice(16, 16, dim=1)
accumulator = gl.amd.cdna3.mfma(a_slice.load(...), b_slice.load(...), accumulator)

a_slice = a_smem.slice(32, 16, dim=1)
accumulator = gl.amd.cdna3.mfma(a_slice.load(...), b_slice.load(...), accumulator)

a_slice = a_smem.slice(48, 16, dim=1)
accumulator = gl.amd.cdna3.mfma(a_slice.load(...), b_slice.load(...), accumulator)
```

**Paired with slice operations**:
```python
# A matrixby K dimension (dim=1)
a_sub = a_tile.slice(offset, 16, dim=1)
a_load = a_sub.load(layout=dot_layout_a)

# B matrixby K dimension (dim=0)
b_sub = b_tile.slice(offset, 16, dim=0)
b_load = b_sub.load(layout=dot_layout_b)
```

**Effect**: Better instruction scheduling, reducing stalls.

---

### 10.4 Scheduling Barrier

**Purpose**: Use hardware scheduling barriers to separate load/store and compute regions.**Choice with warp_pipeline_stage**:
- `warp_pipeline_stage`: warp-level overlap, more flexible
- `sched_barrier + s_set_prio`: more precise hardware control

---

### 10.5 gl.assume() Optimization Hints

**Purpose**: Tells the compiler the value range of variables, helping optimize address calculations.

```python
gl.assume(pid_m >= 0)
gl.assume(pid_n >= 0)
gl.assume(stride_am > 0)
gl.assume(stride_ak > 0)
gl.assume(stride_bn > 0)
gl.assume(stride_bk > 0)
gl.assume(stride_cm > 0)
gl.assume(stride_cn > 0)
gl.assume(num_k_iter > 3)
```

**Effect**: The compiler can eliminate some bounds checks Tresand generate more efficient code.

---

### 10.6 Simplifying PID Remapping

**Purpose**: Simplifies PID calculation at the XCD level.

```python
# (test_gluon_v3)
pid = gl.program_id(0)
xcd = pid % NUM_XCDS
local_pid = pid // NUM_XCDS
pid = xcd * PIDS_PER_XCD + local_pid

# (requiresfunction)
pid = gl.program_id(axis=0)
pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)
```

**Selection Principle**: The simple version suffices for most scenarios.

---

### 10.7 Shared Memory Pre-allocation Strategy

**Pre-allocation + index reuse**:
```python
# ( batch , pingpong)
a_smem = ttgl.allocate_shared_memory(gl.bfloat16, [1, BLOCK_M, BLOCK_K], shared_layout_a)

# loopuse index access
a_tile = a_smem.index(0)
a_tile.store(val)
a_tile.load(...)
```

**vs In-loop allocation**:
```python
# ❌ Wrong: iterationmemory
a_smem = gl.allocate_shared_memory(..., value=a_val) # loop

# ✅ Correct:
a_smem = ttgl.allocate_shared_memory(gl.bfloat16, [1, BLOCK_M, BLOCK_K], shared_layout_a)
a_tile = a_smem.index(buf_idx)
a_tile.store(val) #
```

---

### 10.8 Performance Comparison Summary

| Optimization Technique | Effect |
|------------------------|--------|
| warp_pipeline_stage | +20-30% |
| pingpong buffering | Eliminates LDS stall |
| Manual MFMA unrolling | +10-15% |
| gl.assume() | Bounds check optimization |
| Scheduling barriers | Separates instruction types |

**Practical Cases**:
- matmul_gluon_new (basic optimization): 132 TFLOPs (66% of Triton)
- test_gluon_v3 (full optimization): 163 TFLOPs (81% of Triton)


## Related

- [Changelog for Preview 0.1.4](CHANGELOG.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [CDNA3 (gfx942) ISA Instruction Patterns and Optimization Reference](isa_patterns.md)
- [CUTLASS GEMM Optimization Strategy](../../../nvidia/common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
