# PTX Instruction Evolution: A100, H100, B200

A comparative analysis of PTX instruction set evolution across three generations of NVIDIA AI-accelerating GPUs, covering memory access, synchronization, and Tensor Core MMA instructions.

---

## 1. Background

NVIDIA's AI-accelerating GPUs have achieved a claimed 40-million-fold performance increase from Pascal to Rubin over 10 years — seemingly impossible under Moore's Law's doubling every two years. In reality, this results from multiple compounding factors:

- Biennial generation updates, each with process improvement
- 2–3x same-datatype throughput gain per generation
- TF32 on A100 delivered 10x over Volta
- Each generation introduces lower precision types, each yielding 2x
- Increasing dies per package (A100/H100: single die; B200: dual die)
- Increasing GPUs per node (8-GPU → NVL72 supernode)
- New hardware units per generation (A100: `cp.async`; H100: TMA; B200: TMEM)

---

## 2. A100 / H100 / B200 Specifications

| Metric | A100 SXM4 80GB | H100 SXM5 80GB | B200 (HGX) |
|--------|----------------|----------------|-------------|
| Architecture | Ampere (GA100) | Hopper (GH100) | Blackwell (GB100, dual-die) |
| Process | TSMC 7nm | TSMC 4nm | TSMC 4NP |
| Transistors | 54.2B | 80B | 208B |
| Die Area | 826 mm² | 814 mm² | ~750 mm² × 2 |
| SM Count | 108 | 132 | 148 |
| GPC Count | 7 | 8 | 8 |
| FP32 CUDA Cores/SM | 64 | 128 | 128 |
| Tensor Cores (Gen) | 432 (3rd) | 528 (4th) | 576 (5th) |
| Memory | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| Memory BW | 2,039 GB/s | 3,350 GB/s | 8,000 GB/s |
| L2 Cache | 40 MB | 50 MB | ~192 MB |
| SMEM/SM | 192 KB (max 164 shared) | 256 KB (max 228) | 256 KB (max 228) |
| NVLink | 3rd, 600 GB/s | 4th, 900 GB/s | 5th, 1,800 GB/s |
| TDP | 400W | 700W | 1,000W |
| FP64 | 9.7 TFLOPS | 34 | 37 |
| FP64 TC | 19.5 | 67 | 37 |
| TF32 TC (dense/sparse) | 156/312 | 495/990 | 1,100/2,200 |
| BF16/FP16 TC | 312/624 | 990/1,979 | 2,250/4,500 |
| FP8 TC | — | 1,979/3,958 | 4,500/9,000 |
| FP4 TC | — | — | 9,000/18,000 |
| INT8 TC | 624/1,248 TOPS | 1,979/3,958 | 4,500/9,000 |

---

## 3. PTX Instruction Comparison

### 3.1 Memory Access Instructions

- **A100**: introduced `cp.async`
- **H100**: introduced `cp.async.bulk.tensor` (TMA)
- **B200**: introduced `tcgen05` family (`tcgen05.cp/ld/st/alloc`, etc.)

| Feature | A100 | H100 | B200 |
|---------|------|------|------|
| GMEM → SMEM async copy | `cp.async` (LDGSTS), 16B/thread | `cp.async` + `cp.async.bulk.tensor` (TMA, multi-dim) | `cp.async` + enhanced TMA |
| SMEM → RF | `ldmatrix` (LDSM) warp-level | `ldmatrix`; `wgmma` directly consumes B from SMEM | `ldmatrix`; `tcgen05.mma` directly consumes from SMEM/TMEM |
| SMEM ↔ TMEM | — | — | `tcgen05.cp`; `tcgen05.ld/st` |
| TMEM management | — | — | `tcgen05.alloc` / `dealloc` / `relinquish_alloc_permit` |
| DSMEM | — | Cluster-internal cross-CTA read/write | Same as Hopper |

### 3.2 Synchronization Instructions

- **A100**: `commit_group` / `wait_group` (async copy)
- **H100**: `mbarrier.arrive` / `try_wait`, wgmma and cluster barriers
- **B200**: similar but integrated into tcgen05

| Feature | A100 | H100 | B200 |
|---------|------|------|------|
| Async copy sync | `cp.async.commit/wait_group` | mbarrier + TMA auto-decrement tx-count | `cp.async.bulk.wait_group` |
| mbarrier | Paired with `cp.async` | init/arrive/expect_tx/try_wait | + `tcgen05.commit.mbarrier::arrive` |
| MMA sync | `mma.sync` implicit warp sync | `wgmma.commit/wait/fence`; warp-group async | `tcgen05.commit` + mbarrier; fully async |
| Cluster sync | — | `barrier.cluster.arrive/wait` | Same as Hopper |

### 3.3 Compute (Tensor Core MMA)

| Feature | A100 | H100 | B200 |
|---------|------|------|------|
| Primary MMA instr | `mma.sync` (HMMA) | `wgmma.mma_async` (HGMMA) | `tcgen05.mma` (UMMA) |
| Issue granularity | warp (32 threads) | warp-group (128 threads = 4 warps) | Single thread on behalf of entire CTA |
| Execution mode | Synchronous | Async commit/wait/fence | Fully async (via mbarrier) |
| Operand source | A, B from RF | A from RF or SMEM descriptor; B from SMEM | A, B from SMEM or TMEM; D stored in TMEM |
| Accumulator | RF | RF | TMEM; requires `tcgen05.ld` to read back to RF |
| Typical tile | m16n8k16 | m64n{8..256}k16 | m{64..256}n{64..256}k16 |
| Single-instr latency | ~tens of cycles | m64n256k16 → ~128 cycles (linear) | m256n256k16 → **~11 cycles** (near-constant) |
| Precision | FP16/BF16/TF32/FP64/INT8/INT4/INT1 | + FP8 | + FP4/FP6 |

---

## 4. Typical GEMM Kernel Data Flow

```
A100: GMEM --[cp.async]--> SMEM --[ldmatrix]--> RF --[mma.sync]--> RF (accumulator)
        Sync: cp.async.commit_group / wait_group

H100: GMEM --[TMA]--> SMEM --[wgmma descriptor]--> RF (accumulator)
        Sync: mbarrier (TMA) + wgmma.commit/wait/fence (MMA)

B200: GMEM --[TMA]--> SMEM --[tcgen05.cp]--> TMEM --[tcgen05.mma]--> TMEM (accumulator)
       TMEM --[tcgen05.ld]--> RF --[store]--> GMEM
        Sync: mbarrier (TMA + MMA via tcgen05.commit)
```

---

## 5. Key Evolution Summary

- **Volta → Ampere**: Introduced `cp.async` for asynchronous GMEM→SMEM copy (bypassing RF), reducing register pressure; introduced `ldmatrix` for optimized SMEM→RF loading; 3rd-gen Tensor Core added TF32/BF16/FP64
- **Ampere → Hopper**: Introduced TMA hardware engine for single-thread multi-dimensional tensor copies; MMA upgraded from warp-level to warp-group-level async (`wgmma`); introduced Thread Block Clusters and DSMEM; introduced FP8 + Transformer Engine
- **Hopper → Blackwell**: Introduced TMEM so accumulators no longer occupy RF; MMA reduced from warp-group to single-thread dispatch (`tcgen05.mma`), fully async; **MMA latency is near-constant (~11 cycles) regardless of tile size**; added FP4/FP6 precision support
