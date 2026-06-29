# GPGPU Architecture: Blackwell Instruction Analysis

A comprehensive analysis of Blackwell (B200, SM100) architecture features, instruction set, and GEMM design — covering hardware specifications, tcgen05 ISA, Tensor Memory, SMEM layout, and CUTLASS tiling strategies.

---

## 1. Hardware Architecture

### 1.1 GB202 SM Specifications

The GB200 (B200 dual-die) key parameters:

- **CUDA Cores**: 18,944 FP32/INT32 cores
- **SMs**: 148 (128 CUDA Cores/SM, 4 SMSPs, 32 FP32/INT32 cores per SMSP)
- **Unified L1/Shared Memory per SM**: 256 KB total, SMEM configurable up to 228 KB/SM (same as Hopper)
- **Global Memory**: 178 GB HBM3E (marketed as 180 GB), 7680-bit bus width, 4000 MHz clock. Single-die bandwidth ~3.84 TB/s, dual-die reaches 7.68 TB/s
- **L2 Cache**: 126 MB
- **Constant Memory**: 64 KB

Blackwell returns to the Kepler/Maxwell/Pascal-era Unified INT32/FP32 Cores design. Starting from Volta, INT32 and FP32 cores were separated at a 1:1 ratio. From Turing onward, INT32 cores had independent datapaths — INT32 address calculations no longer interrupted the FP32 instruction stream (important for SGEMM performance).

Ada adjusted the INT32:FP32 ratio to 1:2, meaning peak INT32 throughput was only half of FP32. Blackwell readjusts to 1:1, likely with each core operating as either FP32 or INT32 per clock cycle (similar to pre-Volta behavior). This saves die area, though actual application impact requires testing.

### 1.2 Compute Estimates

**CUDA Core throughput**:
- FP32/INT32: 18,944 × 1.97 GHz × 2 / 1e12 = 74.6 TFLOPS

**Tensor Core throughput**:
- FP16/BF16: 148 SMs × 4 SMSPs × 1.97 GHz × 2048 MACs/cycle/SMSP = 4.8 PFLOPS
- FP8: 148 × 4 × 1.97 GHz × 4096 MACs/cycle/SMSP = 9.55 PFLOPS
- FP4: 148 × 4 × 1.97 GHz × 8192 MACs/cycle/SMSP = 19.1 PFLOPS

### 1.3 Memory Latency and Bandwidth

NVIDIA has consistently increased L2 cache size across generations: V100 6 MB → A100 40 MB → H100 50 MB → Blackwell 126 MB.

Blackwell has two ~63 MB L2 partitions (63 MB × 2 = 126 MB total). Overall:
- Near L2 partition latency: ~300 cycles
- Far L2 partition latency (on cache miss with duplication): ~800 cycles
- For comparison, H200 far partition: ~700 cycles; A100 near partition: ~200 cycles, far partition: ~550 cycles

---

## 2. Preferred Thread Block Clusters

Hopper introduced Thread Block Clusters with TMA multicast load and higher-bandwidth L2 cache mediation to reduce global memory bandwidth pressure.

Hopper requires all CTAs within a cluster to reside within a single GPC. Each H100/H200 GPC has 9 TPCs (2 SMs each). SM90 supports max cluster size 8 (range [1,8]); SM90A supports max cluster size 16 (range [1,16]).

This can leave SMs idle (e.g., 100 CTAs with cluster size 4 wastes 2 SMs per GPC). Blackwell introduces **Preferred Cluster Launch**:
- Hardware first launches with a larger cluster size to fully utilize GPC resources
- When idle SMs remain, falls back to a smaller cluster size to leverage fragmented resources

```cpp
/// Launches a kernel using the CUDA Extensible Launch API and Threadblock Clusters.
/// This API is for preferred cluster launch; a preferred and a fallback cluster shapes are
/// considered for launch respectively.
virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const fallback_cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;
```

---

## 3. Dynamic Tile Scheduling

CUTLASS designed persistent kernels for Hopper GEMM (persistent WASP pingpong/cooperative kernels). These execute CTAs in waves for better wave quantization handling and latency hiding (especially pingpong mode for prologue/epilogue).

Hopper uses **static scheduling**: tiles are mapped to SMs at kernel launch and never migrated. If other kernels preempt SMs, the GPU does not reschedule displaced work tiles, causing severe wave quantization effects (low tail-stage SM utilization).

Blackwell introduces **dynamic scheduling** (via `clusterlaunchcontrol` PTX instruction): the scheduler can cancel mappings and reschedule preempted work tiles to SMs that finish early, minimizing wave quantization effects.

---

## 4. tcgen05 ISA Overview

Blackwell's 5th-generation Tensor Cores represent NVIDIA's most aggressive GPGPU DSA step:

- **2x throughput** at same precision (FP16/BF16/TF32/INT8/FP8) vs. Hopper
- Micro-scaling format support: MXFP8/FP6 at 2x over Hopper FP8; MXFP4 at 4x over Hopper FP8
- Expanded from single-SM warpgroup-level WGMMA to **2-SM-level cooperative execution**
- Fully asynchronous execution — single-thread (leader thread) dispatch for better overlap
- New **Tensor Memory (TMEM)** — no register file needed for accumulators. Each SM: 128 lanes × 512 banks = 256 KB (same size as register file)

Key tcgen05 instructions:
- **TMEM alloc/dealloc**: `tcgen05.alloc/dealloc`, `tcgen05.relinquish_alloc_permit`
- **TMEM read/write**: `tcgen05.ld/st/cp/shift`
- **MMA**: `tcgen05.mma`
- **Synchronization**: `tcgen05.wait::*`, `tcgen05.commit`, `tcgen05.fence::*`

TMEM read/write and MMA instructions are asynchronous; TMEM alloc/dealloc and sync operations are synchronous (blocking).

---

## 5. Tensor Memory

Tensor Memory is a dedicated 2D on-SM memory ([rows/lanes, columns]) accessible only by tcgen05 instructions — not by thread-level SIMT instructions.

TMEM serves as both input (operand A) and output (operand D) for `tcgen05.mma`, completely eliminating register storage for accumulators. For FP16 with FP32 accumulator at maximum size (m×n×k = 128×256×32B), the accumulator occupies 128×256×4B = 128 KB — 32 KB per warp, consuming half the SMSP register file.

For micro-scaling computation (MXFP8/6/4), all scaling factors must be stored in TMEM.

Each SM's TMEM: 128 lanes (rows) × 512 columns = 64K cells × 32-bit/cell. Address encoding uses 32 bits: low 16 bits [0:15] = column index, high 16 bits [16:31] = lane index. Each warp can only access 32 lanes; accessing all 128 lanes requires the full warpgroup.

### 5.1 TMEM ↔ RF/SMEM Data Flow

- `tcgen05.ld`: Load TMEM data → register file
- `tcgen05.st`: Store register file data → TMEM
- `tcgen05.cp`: Copy SMEM data → TMEM

All operations use fixed shapes (not arbitrary granularity).

### 5.2 tcgen05.alloc/dealloc

TMEM is dynamically allocated by a single warp within a CTA and must be freed by that same warp. Allocation granularity: 32 columns (powers of 2: 32/64/128/256/512). When all columns are allocated, all 128 lanes are committed.

Key properties:
- Both alloc and dealloc are **synchronous** — blocking until TMEM is available
- Supports CTA-pair operations (`.cta_group::2`): both CTAs in a pair must jointly execute alloc/dealloc

```
// Single-CTA version:
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sMemAddr1], 32;
ld.shared.b32 taddr, [sMemAddr1];
// use taddr ...
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 32;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;

// CTA-pair version:
tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [sMemAddr2], 32;
ld.shared.b32 taddr, [sMemAddr2];
// use taddr ...
tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, 32;
tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
```

### 5.3 tcgen05.ld/st

`tcgen05.ld` asynchronously loads TMEM data to registers in a specified layout, executed per-warp. The `.shape` specifies the atomic load shape; `.num` specifies repetition count.

Shape and repetition combinations:
- `.16x32bx2` / `.16x64b` / `.32x32b`: support x1/2/4/8/16/32/64/128
- `.16x128b`: support x1/2/4/8/16/32/64
- `.16x256b`: support x1/2/4/8/16/32

Maximum single instruction: 256 columns. Each CTA's TMEM is divided into 4 chunks (32 lanes × 512 columns each); each warp accesses one chunk. Warp ID mapping within warpgroup:
- warp_rank % 4 == 0: lanes 0–31
- warp_rank % 4 == 1: lanes 32–63
- warp_rank % 4 == 2: lanes 64–95
- warp_rank % 4 == 3: lanes 96–127

Optional packing/unpacking modes:
- **Pack mode load**: Merges two 16-bit TMEM chunks into a single 32-bit register
- **Unpack mode store**: Splits a 32-bit register into two 16-bit TMEM chunks

### 5.4 tcgen05.cp

Asynchronously copies SMEM sub-byte data (e.g., FP4/6) to TMEM with decompression to bytes. Issued by a single thread within a warp.

Supports CTA-group multicast:
- `.cta_group::1`: Copy to current CTA's TMEM only
- `.cta_group::2`: Copy to both current and peer CTA's TMEM

Shape support varies by warp count:
- Single warp: `.32x128b`, `.4x256b`
- Dual warps: `.64x128b`
- Warpgroup: `.128x128b`, `.128x256b`

Multicast within warpgroup:
- `.warpx2::02_13` / `.warpx2::01_23`: Copies 64×128b SMEM to current warp and peer-warp's TMEM
- `.warpx4`: Copies 32×128b SMEM to all warps' corresponding lanes

Supports decompression (4/6-bit → 8-bit) during copy.

---

## 6. tcgen05.mma Instruction

### 6.1 Shapes

The MMA shape depends on precision, `.ws` mode, CTA group count, and sparsity:

For FP16/BF16, single-SM `tcgen05.mma`:
- m=64: n starts at 8, step 8, max 256 (identical to corresponding WGMMA)
- m=128: n starts at 16, step 16, max 256
- K is always 32B (16 elements for FP16)

For 2-SM `tcgen05.mma`:
- m extends to 128/256, K=32B, n starts at 32, step 32, max 256

### 6.2 SMEM Layout and Swizzle

Like Hopper's WGMMA, `tcgen05.mma` operands A/B can be read directly from SMEM. Hardware swizzle strategies prevent bank conflicts (TMEM does not have this issue since it doesn't support arbitrary thread-level reads).

Swizzle must match TMA hardware swizzle. Supported modes:
- **128B swizzle with 16B atomicity**: Equivalent to standard 128B swizzle (M/N-major and K-major)
- **128B swizzle with 32B atomicity**: Combines 2×128-bit cells into 2×16B=32B cells before swizzling
- **64B swizzle**: For both MN-major and K-major
- **32B swizzle**: K-major

### 6.3 Weight Stationary Mode

`tcgen05.mma` supports convolution via weight stationary mode with a **Collector Buffer** for operand reuse. The Tensor Core caches the reused operand (A or B) in the collector buffer, avoiding repeated loads from SMEM/TMEM.

Collector qualifiers:
- `::fill`: Load new matrix into buffer
- `::use`: Reuse cached matrix
- `::lastuse`: Final use, allow discard
- `::discard`: Explicit discard

### 6.4 Block Scaling

Supports micro-scaling format with block-wise dequantization: `D += (A * scale_A) * (B * scale_B) + D`

Scale factors stored in TMEM. Supported types:
- `.kind::mxf8f6f4`: Operands A/B can be any f8/f6/f4 combination; scale = UE8M0 (range 0–255)
- `.kind::mxf4`: FP4 with UE8M0 scale
- `.kind::mxf4nvf4`: NVFP4 with UE4M3 scale

Scale granularity options: `.scale_vec::1X`, `.scale_vec::2X`, `.scale_vec::4X`

---

## 7. Instruction Dispatch and Synchronization

### 7.1 Dispatch Model

Key differences from Hopper WGMMA:
- Both are asynchronous requiring explicit synchronization
- WGMMA: all threads in a warpgroup participate in dispatch and sync (single SM)
- UMMA: **single thread** dispatches — more lightweight, enables better overlap (like TMA)

For single-CTA UMMA: executes on one SM (similar to WGMMA). For CTA-pair UMMA: two adjacent SMs cooperate. The leader CTA (even-numbered CTA in the pair) dispatches via a single leader thread.

```cpp
int cta_rank = static_cast<int>(cute::block_rank_in_cluster());
auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank);
auto elect_one_cta = cute::get<0>(cta_in_cluster_coord_vmnk) == cute::Int<0>{};
if (elect_one_cta) {
    // Issue pair-UMMA from single thread
}
```

After dispatch, explicit arrive (`tcgen05.commit`) marks instruction submission — must come from the dispatching CTA's thread.

### 7.2 Synchronization for Pair-UMMA

MMA barrier arrive count equals the number of MMAs (not CTAs). For a cluster with M-mode size 2 and N-mode size 4: despite 10 CTAs involved, arrive count is 5 (only leader CTAs arrive).

Special `umma_arrive_multicast_2x1SM` is used because `cta_group::2` commits use a separate pipeline from `cta_group::1`.

### 7.3 TMA Multicast Synchronization for 2SM

TMA launches with bitmask restricting multicast to same-parity CTAs. However, the wait barrier (called from even CTAs only) must wait for the entire MMA tile. Odd CTAs must somehow arrive at even CTAs' mbarrier.

CUTLASS solves this via:
1. SM100 `cta_group::2` qualifier for TMA copy: allows arriving at either the executing CTA's or its peer CTA's mbarrier
2. Modified mbarrier address: `cast_smem_ptr_to_uint(mbar_ptr) & 0xFEFFFFFF` — clearing bit 24 finds the leader CTA's mbarrier (bit 24 corresponds to CTA ID bit 0 in the unified shared address space)

---

## 8. GEMM on Blackwell

### 8.1 Design Overview

Unlike Hopper's pingpong design (WG1 Epilogue overlaps with WG2 WGMMA), Blackwell's mainloop Tensor Core execution is single-thread dispatched and fully async. After completing the current tile's mainloop, the warpgroup handles epilogue while the next tile's mainloop dispatches immediately — no resource contention between mainloop and epilogue.

A simple Blackwell GEMM design:
1. TMA: global → shared memory data transfer
2. UMMA: MMA computation
3. `tcgen05.ld`: MMA results from TMEM → registers
4. Epilogue computation on registers, write results to GMEM

### 8.2 Tiling Strategy

For 1-SM UMMA: standard [BM, BN, BK] partitioning per CTA. Thread Block Clusters can use TMA multicast for efficient data movement.

For 2-SM UMMA: two CTAs form an atom cluster, computing a larger work tile. CUTLASS uses **MMA-centric tiling** (vs. CTA-centric tiling):

```cpp
TiledMMA tiled_mma = make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<
        TypeA, TypeB, TypeC,
        MmaM, MmaN,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>{}
);
```

This abstraction hides whether 1-SM or 2-SM UMMA is used at the MMA tile level. The `cluster_layout_vmnk` variable encodes CTA-pair relationships:

For `AtomThrID{} = 2` with cluster shape (4,4,1): `tiled_divide` produces `((2,2,4,1):(1,2,4,16))`, where mode-0 represents CTA-pairs.

### 8.3 TMA Multicast with 2-SM UMMA

For `TMA_MULTICAST_LOAD_A`: each MMA tile [MMA_M, MMA_K] multicasts to CTAs 0/1, 4/5, 8/9, 12/13.
For `TMA_MULTICAST_LOAD_B`: each MMA tile [MMA_N, MMA_K] multicasts to CTAs 0/1/2/3.

```cpp
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord);
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord);
uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(...) |
                             create_tma_multicast_mask<0,2>(...);
```

Resulting masks:
- `tma_mcast_mask_a`: 0x1111
- `tma_mcast_mask_b`: 0x0005
- `mma_mcast_mask_c`: 0x333f

### 8.4 Execution Loop

```cpp
// Init MMA barrier
if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
}
int mma_barrier_phase_bit = 0;
__syncthreads();

tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // ... copy data in ...
    if (elect_one_warp) {
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
}
// ... copy data out ...
```

---

## 9. References

- NVIDIA GTC 2025 Keynote
- CUTLASS Blackwell Cluster Launch Control documentation
- CUTLASS Blackwell Functionality documentation
- Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis
- NVIDIA PTX ISA documentation
