# TCA 51% but MFU Under 8%: Hidden GPU Performance Losses

An investigation into the surprising gap between Tensor Core Activity (TCA) and Model FLOPs Utilization (MFU), uncovering clock frequency scaling, tile padding waste, and instruction-level inefficiency as root causes.

---

## 1. Background

TCA (Tensor Core Activity) is the fraction of clock cycles during which the Tensor Core is active. MFU (Model FLOPs Utilization) is the ratio of actual model FLOPs to the theoretical peak FLOPs over wall-clock time. In theory, these two metrics should be closely correlated, typically within a 10-20% gap. However, under certain conditions the gap becomes unstable — and in the case of Flash Attention 2, TCA reaches 51% while MFU stays below 8%, a consistent 4x discrepancy after clock correction.

This investigation reveals three root causes:
1. **Power-wall clock throttling** — GPU frequency drops under heavy load
2. **Tile padding waste** — cuBLAS pads matrices to tile boundaries
3. **Legacy instruction inefficiency** — FA2 uses HMMA instructions that only drive 1/4 of Blackwell Tensor Core capacity

---

## 2. GPU Architecture Overview

### 2.1 SM Structure

A GPU comprises multiple Streaming Multiprocessors (SMs). B200 contains 148 SMs. Each SM is the true counterpart to a CPU core, possessing a complete set of registers/caches, compute units, and schedulers.

SM-level shared resources:
- **L1 Instruction Cache** at the top for instruction fetch
- **TMA (Tensor Memory Accelerator)** for asynchronous data movement
- **256 KB L1 Data Cache / Shared Memory** — programmer-managed on-chip storage

Each SM contains 4 identical Sub-Partitions (SMSP / Sub-Cores). Each SMSP includes:
- **Warp Scheduler + Dispatch Unit**: selects one instruction per cycle for one warp
- **Register File** (16,384 × 32-bit = 64KB): thread-private high-speed storage
- **Compute Pipelines**: INT32, FP32 (CUDA Core), FP64, LD/ST, SFU
- **Tensor Core**: one per SMSP, the primary driver of AI compute

### 2.2 Tensor Core

A CUDA Core performs one scalar FMA per cycle. A Tensor Core performs one matrix operation per cycle: D = A × B + C, computing an entire small matrix tile in a single instruction.

A single small matrix's data exceeds any single thread's register capacity, so Tensor Core operands are provided cooperatively by an entire warp (32 threads). Each thread holds a fragment of the matrix; 32 threads' fragments assemble the complete A, B, C, D operands.

The cooperation granularity has grown across generations:
- **A100**: 1 warp (32 threads)
- **H100**: 1 warp-group (4 warps = 128 threads)
- **B200**: Returns to single-thread issue semantics, but operands come from Shared Memory and TMEM rather than registers

### 2.3 GEMM Tiling

AI training/inference compute is dominated by GEMM (General Matrix Multiply). A large GEMM C = A × B (A: M×K, B: K×N, C: M×N) requires 2MNK FLOPs.

In practice, the output matrix C is tiled: each tile is assigned to a Thread Block, which cooperatively loads the corresponding A/B tiles into Shared Memory and uses Tensor Core for matrix multiply-accumulate.

---

## 3. MFU vs TCA: The Expected Relationship

$$\text{MFU} = \frac{\text{counted\_FLOPs}}{\text{wall\_time} \times \text{peak\_TFLOPS}}$$

- **wall_time**: program execution time
- **peak_TFLOPS**: hardware peak (B200 FP16 = 2,250 TFLOPS)
- **counted_FLOPs**: actual model FLOPs (verified via manual computation and `torch.utils.flop_counter.FlopCounterMode`)

TCA is a hardware-level counter requiring no user computation. Since GPU peak TFLOPS comes from Tensor Core, and TCA records TC active time (excluding data-wait time), MFU and TCA should be closely matched.

Note: Flash Attention backward performs one additional recompute, so total FLOPs = 4× forward FLOPs (vs. the standard training 3× forward).

---

## 4. Power Wall and Clock Throttling

### 4.1 Discovery

Running large matmul benchmarks on B200 revealed MFU consistently below TCA, with MFU capped at ~83.22% regardless of matrix size. Monitoring showed:
- GPU clock at 1643 MHz (not the 1965 MHz peak)
- Power at 1 kW (the air-cooled baseline), well below the 1.2 kW peak

TCA is computed against the actual (throttled) clock, while MFU uses the physical peak. Under light load (no power wall), they match perfectly:

1024³ matmul: 409W, 1965 MHz, MFU = 9.66% = TCA = 9.6%

### 4.2 Clock-Corrected MFU

$$\text{MFU}_{\text{corrected}} = \text{MFU} \times \frac{\text{actual\_clock}}{\text{max\_clock}}$$

| Config | MFU | TCA | Avg Clock | Power | Corrected MFU | Gap |
|--------|-----|-----|-----------|-------|---------------|-----|
| 1024³ | 9.66% | 9.6% | 1965 MHz | 409W | 9.66% | -0.06% |
| 2048³ | 46.63% | 50.0% | 1939 MHz | 964W | 47.26% | 2.74% |
| 4096³ | 58.50% | 74.25% | 1586 MHz | 984W | 72.48% | 1.77% |

### 4.3 Clock Measurement Accuracy (H100)

nvidia-smi samples at only 5Hz — insufficient for measuring a ~1.5 GHz clock under varying load. NVML at 200Hz still has significant error. The solution: use CUDA `clock64()` to read the SM hardware cycle counter directly.

| Config | NVML Clock | clock64 Actual | Overclock % | TCA | Corrected MFU | Gap |
|--------|-----------|----------------|-------------|-----|---------------|-----|
| 4096³ | 1515 MHz | 1376 MHz | 10.1% | 90.26% | 89.68% | 0.58% |
| 16384×4K×16K | 1471 MHz | 1340 MHz | 9.8% | 94.70% | 93.07% | 1.63% |

---

## 5. Tile Padding Waste

### 5.1 Mechanism

When matrix dimensions are not integer multiples of the tile size, cuBLAS pads to tile boundaries. The padded FLOPs are computed by Tensor Core (counted by TCA) but contribute nothing to the model (not counted by MFU).

### 5.2 NCU Verification

| Config | Kernel | Tile | tensor_active | Theoretical (FLOPs/FPC) | Ratio | Corrected MFU | Gap |
|--------|--------|------|---------------|--------------------------|-------|---------------|-----|
| 4096³ | nvjet_tst_256x128 | 256×128 | 134,217,728 | 134,217,728 | 1.000000 | 89.68% | 0.58% |
| 16384×4K×16K | nvjet_tst_320x128 | 320×128 | 2,181,038,080 | 2,147,483,648 | 1.01563 | 94.52% | 0.18% |

Key insight: cuBLAS kernel selection prioritizes arithmetic intensity over padding-free alignment. Larger tiles have higher compute-to-memory ratio (volume vs. surface area scaling), keeping Tensor Core fed rather than waiting for data.

---

## 6. Flash Attention 2 Makes TCA Artificially High

### 6.1 Transformer-Level Observations

| Config | Attn % | MFU | TCA | MFU/(TCA×0.588) | Clock |
|--------|--------|-----|-----|------------------|-------|
| d=1024, seq=512 | 7.7% | 19.91% | 24.45% | 0.814 | 1965 |
| d=256, seq=1024 | 40% | 7.13% | 14.36% | 0.496 | 1965 |
| d=256, seq=2048 | 57% | 8.20% | 20.68% | 0.396 | 1965 |
| d=128, seq=2048 | 73% | 6.49% | 18.19% | 0.357 | 1965 |

Pattern: higher attention FLOPs fraction → larger TCA/MFU gap.

### 6.2 Isolating Attention Components

| Component | MFU | TCA | Ratio (clock-corrected) |
|-----------|-----|-----|------------------------|
| bmm Q@K^T (N=512) | 9.28% | 15.7% | 1.005 |
| bmm P@V (N=64) | 11.67% | 39% | 0.509 |
| Flash Attention 2 | 7.65% | 51% | 0.255 |

After clock correction, Flash Attention 2 shows a consistent **4x** gap between TCA and MFU.

### 6.3 The 4x Factor is Universal

| Config | D | B×H | MFU | TCA | Ratio |
|--------|---|-----|-----|-----|-------|
| S512_D64_H16 | 64 | 512 | 7.63% | 51% | 0.254 |
| S512_D128_H8 | 128 | 256 | 8.81% | 59% | 0.254 |
| S512_D32_H32 | 32 | 1024 | 5.81% | 39% | 0.253 |
| S1024_D64_H16 | 64 | 1024 | 6.68% | 45% | 0.252 |
| S2048_D64_H16 | 64 | 256 | 8.57% | 57% | 0.256 |
| S4096_D64_H4 | 64 | 16 | 7.92% | 53% | 0.254 |
| flash_attn_v2 (Dao) | 64 | 512 | 8.49% | 57% | 0.253 |
| pytorch_sdpa_flash | 64 | 512 | 7.63% | 51% | 0.254 |

The ratio is exactly 0.25 — completely independent of D, S, H, or B.

---

## 7. NCU Root Cause Analysis

### 7.1 FA2 Occupancy

NCU data shows FA2 can fit 2 blocks/SM:

| Limiting Factor | Per Block | SM Total | Blocks Possible |
|----------------|-----------|----------|-----------------|
| Registers | 255 × 128 = 32,640 | 65,536 | 2 |
| Shared Memory | 66,560 B | 233,472 B | 3 |
| → Occupancy Limit | | | 2 blocks |

FA2 grid = 256 blocks, 160 SMs × 2 slots = 320. Some SMs run 2 blocks (2 warps/SMSP), others only 1 block (1 warp/SMSP).

### 7.2 SMSP-Level Metrics (1-block vs 2-block SMs)

| SMSP Metric | min (1-block SM) | max (2-block SM) | Ratio |
|-------------|-----------------|-----------------|-------|
| TCA | 37.75% | 75.50% | 2.0x |
| Warps Active | 0.66 | 2.04 | 3.1x |
| Tensor Instructions | 8,192 | 16,384 | 2.0x |
| HMMA FLOPs | 134,217,728 | 268,435,456 | 2.0x |

More warps → higher TCA, but MFU/TCA = 0.25 remains constant in both cases.

### 7.3 The Critical Discovery: Instruction Difference

| Counter | FA2 | cuDNN |
|---------|-----|-------|
| ops_path_tensor_op_hmma (per-warp) | 34,359,738,368 | 0 |
| ops_path_tensor_op_utchmma (per-warp-group) | 0 | 34,359,738,368 |

Same 34.4B FLOPs. FA2 uses exclusively HMMA; cuDNN uses exclusively UTCHMMA. Zero crossover.

| Metric | FA2 (HMMA) | cuDNN (UTCHMMA) | Ratio |
|--------|-----------|-----------------|-------|
| Tensor Instructions | 8,388,608 | 65,536 | 128x |
| FLOPs/Instruction | 4,096 | 524,288 | 1/128 |
| TC Active Cycles | 67,108,864 | 16,777,216 | 4x |
| FLOPs/Active-Cycle | 512 | 2,048 | 1/4 |

HMMA m16n8k16 = 4,096 FLOPs per instruction. UTCHMMA tile = 524,288 FLOPs per instruction. Per TC-active cycle, UTCHMMA throughput is exactly **4x** that of HMMA.

### 7.4 Confirming the Root Cause

| Metric | FA2 | cuDNN |
|--------|-----|-------|
| Block Size | 128 (4 warps) | 512 (16 warps) |
| Registers/Thread | 255 | 128 |
| SharedMem/Block | 66,560 B | 233,472 B |
| SMSP Warps Active (avg) | 1.64 | 3.99 |

FA2's Tensor Inst Rate = TCA (65.25% = 65.25%). This proves that during HMMA execution, the TC is not intermittently idle (temporal model) but **continuously active with only 1/4 of channels computing**. If TC were idle 3/4 of the time, TCA would be far below Inst Rate.

---

## 8. The 1/4 Root Cause: Wrong Instruction Generation

The 1/4 factor comes from the HMMA instruction itself, not from warp count.

FA2 uses HMMA — a per-warp instruction originating from Ampere (A100). Each HMMA drives only 1 of 4 compute lanes inside the Tensor Core. Regardless of whether the SMSP has 1 or 2 warps, each HMMA produces only 1/4 of peak FLOPs. More warps simply keep the tensor pipe busier (higher TCA) without making each instruction more efficient.

cuDNN uses UTCHMMA — a per-warp-group instruction native to Blackwell (B200). Four warps cooperatively drive all 4 compute lanes, producing full-throughput FLOPs per instruction.

### 8.1 Verification by Back-Calculation

B200 FP16 Dense Peak = 2,250 TFLOPS. With 148 SMs × 4 SMSP = 592 Tensor Cores, and 2,048 FLOPs/active-cycle:

$$\text{Clock} = \frac{2{,}250 \times 10^{12}}{592 \times 2{,}048} = 1{,}856 \text{ MHz}$$

This closely matches the 1965 MHz boost clock specification.

### 8.2 Attention Backend Comparison

| Backend | ms/step | Corrected MFU | TCA | Corrected Ratio |
|---------|---------|---------------|-----|-----------------|
| FA2 (PyTorch SDPA) | 0.200 | 7.63% | 51% | 0.254 |
| FA2 (Dao flash_attn) | 0.180 | 8.49% | 57% | 0.253 |
| cuDNN Attention | 0.100 | 15.25% | 26% | 0.997 |
| Efficient (xFormers) | 0.425 | 3.59% | 48% | 0.127 |

cuDNN Attention achieves 2x the speed and 2x the MFU of FA2, with a corrected ratio of ~1.0 (no instruction-level waste).

---

## 9. GPU Architecture and Instruction Evolution

### 9.1 A100 (Ampere, SM80): Thread-Centric

MMA instruction: **HMMA** (per-warp). One warp (32 threads) issues `mma.sync m16n8k16` → 4,096 FLOPs. Each thread holds matrix fragments in private registers. This instruction saturates A100's Tensor Core — the TC was designed for this granularity.

Programming model: task → thread, data → register, warp = minimum scheduling unit. HMMA latency ~30 cycles, requiring ~30 warps (Little's Law) to saturate throughput. High occupancy is essential.

### 9.2 H100 (Hopper, SM90): Warp-Group Cooperation

MMA instruction: **WGMMA** (per-warp-group). Four warps (128 threads) cooperatively issue one async instruction. Tile jumps from m16n8k16 to m64n256k16 → 524,288 FLOPs (128x larger than HMMA).

Key changes:
- **Asynchronous issue**: warp-group does not block after issuing WGMMA; can immediately prefetch next tile via TMA
- **TMA (Tensor Memory Accelerator)**: dedicated hardware moves entire tiles from global to shared memory without using CUDA Cores or registers
- **Reduced occupancy dependence**: a single warp-group's internal pipeline suffices to hide latency

### 9.3 B200 (Blackwell, SM100): Tensor Memory Liberation

MMA instruction: **tcgen05.mma / UTCHMMA** (single-thread issue semantics). Even larger tiles (M up to 256), lower latency, hardware auto-reads operands from shared memory.

Critical innovation — **Tensor Memory (TMEM)**:
- 256 KB dedicated storage adjacent to Tensor Core
- Accumulator D resides in TMEM, not register file
- Tensor Core performs in-place accumulate directly on TMEM
- Register file (unchanged at 64KB/SM since Kepler 2012) is freed for other use
- Enables larger tiles without register pressure or occupancy penalty

Measured: B200 achieves 80.7% peak on FP64 large GEMM vs. H200's 55.6% — a 25 percentage point architectural efficiency difference.

### 9.4 Backward Compatibility ≠ Efficiency

Each generation's old instructions run on new hardware (backward compatible), but at reduced efficiency:
- HMMA on A100: saturates TC (100%)
- HMMA on H100: ~63% of TC capacity
- HMMA on B200: ~25% of TC capacity

FA2 → FA4 upgrade is essentially HMMA → UTCHMMA migration, requiring full kernel rewrite:
- Operand B moves from registers to shared memory
- Thread organization: per-warp → 4-warp group cooperative
- Synchronization: warp-level → warp-group level
- Data path: register file → tensor memory path

---

## 10. Attention Backend Benchmark

Systematic comparison of 7 attention implementations on B200 across 4 configurations.

### 10.1 MFU Comparison

| Backend | S=512 D=64 | S=1024 D=128 | S=2048 D=128 | S=4096 D=128 |
|---------|-----------|-------------|-------------|-------------|
| cuDNN | 23.98% | 40.98% | 43.80% | 47.64% |
| FA4 (CuTe-DSL) | 16.84% | 16.82% | 16.90% | 33.53% |
| FA2 (Dao) | 14.03% | 15.06% | 15.46% | 16.01% |
| SDPA Flash | 12.73% | 13.80% | 14.06% | 14.48% |
| Flex (Triton) | 12.50% | 12.73% | 13.55% | 14.52% |
| Efficient | 5.97% | 7.04% | 6.16% | 6.21% |
| Math (unfused) | 0.78% | 1.20% | 1.24% | 1.25% |

### 10.2 Latency (ms/step)

| Backend | S=512 D=64 | S=1024 D=128 | S=2048 D=128 | S=4096 D=128 |
|---------|-----------|-------------|-------------|-------------|
| cuDNN | 0.064 | 0.037 | 0.035 | 0.064 |
| FA4 (CuTe-DSL) | 0.091 | 0.091 | 0.090 | 0.091 |
| FA2 (Dao) | 0.109 | 0.101 | 0.099 | 0.191 |
| SDPA Flash | 0.120 | 0.111 | 0.109 | 0.211 |
| Flex (Triton) | 0.122 | 0.120 | 0.113 | 0.210 |
| Efficient | 0.256 | 0.217 | 0.248 | 0.492 |
| Math (unfused) | 1.955 | 1.276 | 1.232 | 2.450 |

### 10.3 Peak Memory (MB)

All fused implementations: 224-225 MB (S=512 D=64), 12-64 MB (larger configs). Math (unfused): 160-1272 MB depending on config.

### 10.4 Why No FA3?

FA3 (Flash Attention 3) is designed exclusively for Hopper (SM90). Its core innovation is warp specialization: producer warps handle TMA async data movement while consumer warps focus on Tensor Core matmul, overlapping via pipeline. This leverages Hopper-specific TMA hardware.

---

## 11. Conclusions

All observed MFU < TCA gaps have three sources:

1. **Power-wall clock throttling**: actual_clock / max_clock (significant for large matrices; B200 drops from 1965 → ~1460 MHz under load)

2. **Legacy HMMA instructions**: FA2 and some older components still use HMMA, causing Tensor Core to operate at only 1/4 capacity on Blackwell

3. **Tile padding waste**: when dimensions are not tile-aligned, cuBLAS pads to tile boundaries — TC computes padding FLOPs (counted by TCA) but they contribute nothing to MFU

**Practical takeaway**: If using the correct (latest-generation) instructions and tile-aligned dimensions, TCA can be treated as a proxy for MFU. The critical lesson when upgrading hardware: verify that kernels use the current-generation MMA instruction (mma.sync → WGMMA → UTCHMMA). Paper specifications are meaningless if 75% of the silicon is running legacy instructions at 1/4 efficiency.
