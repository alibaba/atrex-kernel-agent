# Blackwell Matrix Multiplication Part 3: Achieving 85% SOTA

Multicast TMA, 2×SM MMA, warp-specialized pipelines, and output double-buffering techniques that push Blackwell GEMM performance to 85% of cuBLAS state-of-the-art.


**Last updated**: 2026-06-30

---

## 1. Kernel 5: Multicast and 2×SM MMA

Since Hopper, CTAs within the same SM group can access each other's shared memory (DSMEM). Blackwell introduces two advanced optimizations:

- **TMA Multicast** (since Hopper): Multiple SMs cooperatively load a tile into shared memory
- **2×SM MMA** (Blackwell-new): Two SMs' Tensor Cores work together, using shared memory inputs from both SMs to execute a large-scale MMA

Cluster shape is set at compile time via the `@__llvm_metadata` decorator, enabling CTAs within a cluster to access each other's shared memory.

### 1.1 CTA Memory Multicast

Without multicast: 4 CTAs each load complete A and B tiles — redundant loads at tile granularity.

With a 2×2 cluster: Two CTAs in the same row each load half of the A tile and broadcast to the neighbor; B tiles are similarly handled by column. Each CTA performs only half the loads while still obtaining the full tile.

API: `async_copy_multicast(..., a_multicast_mask, ...)`. The `a_multicast_mask` is a 16-bit value corresponding to indices of up to 16 CTAs in the cluster.

### 1.2 2×SM MMA

Multicast reduces GMEM→SMEM traffic, but tiles are still duplicated in DSMEM. `tcgen05.mma.cta_group::2` solves this: CTA 0 and CTA 1 work as a pair, each loading only half of the B tile. The 2×SM MMA instruction sees both halves in shared memory and coordinates both SMs' Tensor Cores to complete 2× the work of a single-SM MMA.

- Shared memory → TMEM traffic for B tiles is also halved
- `elect_one_cta` selects the leader CTA (even-ID in the pair) to issue the instruction
- `mma_arrive_multicast(cta_group=2)` signals the barrier on the leader CTA
- MMA_M = 128 or 256; when MMA_M=256, each half-TMEM layout matches single-SM MMA

**Performance: 360.2 TFLOPS (20% of SOTA).** Bottleneck remains global memory throughput.

---

## 2. Kernel 6: 2-SM Pipeline (Warp Specialization)

Refactored into `load_AB()` / `consume_AB()` / `store_C()`. Problem: current CTA data dependencies prevent TC and TMA from working simultaneously.

### 2.1 MMA and TMA Pipelining

Blackwell SM has up to 227 KB SMEM. Under 256×256×16 2×SM MMA (BF16):
- A tile: 16 KB / B tile: 16 KB / C tile: 64 KB

Less than half the available SMEM. Introduce a **5-stage circular buffer** to overlap TMA and MMA (one side computes while the other prefetches).

### 2.2 Warp Specialization

- One warp issues TMA instructions
- One warp issues MMA instructions
- Four warps handle output (TMEM → registers → shared memory)
- Multiple memory barriers communicate across warps: TMA→MMA, MMA→output (compute_barrier)

**Performance: 1429 TFLOPS (81% of SOTA).**

---

## 3. Kernel 7: Output Double-Buffering

The C store pipeline (TMEM → registers → shared memory → GMEM) applies the same pipelining principle.

- Declare 2 output tile buffers (stageN=32 initially) for ping-pong switching
- First iteration does not wait (uses the other buffer); iterations 2 through num_stage-1 use `wait_group[1]`; final iteration waits for all TMA stores to complete
- `commit_group` / `wait_group[N]` controls TMA async store grouping

**Additional free benefit**: Previously the C tile occupied 64 KB (~40% of SMEM). Now using `2 × BM × StageN × 2B = 16 KB`, the **saved 48 KB can increase load-side pipeline depth**.

**Performance: 85% of SOTA** (+64 TFLOPS over Kernel 6).

---

## 4. Next Steps

The remaining 15% comes from: GMEM write-out overhead and CTA scheduling launch overhead. The next article introduces persistent kernels + Cluster Launch Control.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Software Pipeline Depth Optimization](../../common/software-pipeline-depth-optimization.md)
