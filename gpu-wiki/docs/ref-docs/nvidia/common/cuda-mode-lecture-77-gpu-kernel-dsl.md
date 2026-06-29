# CUDA-MODE Lecture 77: DSLs for GPU Kernels

Notes from CUDA-MODE Lecture 77 by Tri Dao, covering the GPU kernel DSL ecosystem from PyTorch through Triton to CuTe-DSL, demonstrating performance and development efficiency tradeoffs across abstraction levels via Softmax, GEMM, and Attention examples. Code: `Dao-AILab/quack`.

---

## 1. Why DSLs?

Tri Dao's computational efficiency formula:

> **Intelligence/Dollar = (Intelligence/FLOPS) x (FLOPS/Dollar)**

This decomposes AI cost-effectiveness into two parts: **algorithmic efficiency** (Intelligence/FLOPS) and **hardware efficiency** (FLOPS/Dollar).

A Venn diagram illustrates: algorithm research pursues better models, hardware optimization pursues better scaling, and **DSLs sit at their intersection** — maintaining research productivity while fully utilizing hardware. Bonus: **DSLs + good abstractions make it easier for LLMs to generate efficient GPU kernels**.

---

## 2. DSL Spectrum

**PyTorch is the first DSL** — writing programs in PyTorch turns code into GPU kernels. PyTorch 2.x Dynamo captures programs and executes them through Triton compilation.

**Triton (second DSL) vs CUDA:**

| Optimization | CUDA | Triton |
|---|---|---|
| Memory coalescing | Manual | Automatic |
| Shared memory management | Manual | Automatic |
| Intra-SM scheduling | Manual | Automatic |
| Cross-SM scheduling | Manual | Manual |

Triton's **tile-based** programming model lets developers focus on algorithm logic.

**CuTe-DSL (third DSL):** Embeds CUTLASS C++ into Python. Elevator analogy:

- **Triton ≈ Express elevator** (high-level abstraction)
- **CUTLASS ≈ Escalator** (fine-grained control)
- **PTX ≈ Spiral staircase** (low-level detail)

CuTe-DSL **fully exposes all 4 GPU thread/memory hierarchy levels:**

1. Thread registers and local memory
2. Block shared memory
3. Cluster distributed shared memory
4. Global memory

Triton only exposes thread block and grid — two levels — limiting hardware control.

---

## 3. Other Notable DSLs

- **ThunderKittens** (Stanford): Simple and fast AI kernel framework
- **TileLang**: Tile-based GPU programming abstraction
- **Mojo**: Combines Python usability with systems-level performance
- **Mosaic GPU**: Alternative GPU programming abstraction

---

## 4. Triton Extensions

- **Gluon**: Lower-level language built on Triton compiler technology, exposing fine-grained control over layout, scheduling, and memory
- **TLX**: Low-level, warp-aware Triton extension providing hardware-specific intrinsics like `wgmma`, `async_copy`, `barrier`

---

## 5. Softmax Example

- **Liger Kernel Triton implementation**: `@triton.jit` — get row ID and offsets → load data with boundary handling → compute max (numerical stability) → compute exp and normalize → output.
- **Triton Softmax multi-block**: Two main loops — first computes global max and accumulated exponential sum (online algorithm), second computes final softmax using global statistics. Handles data exceeding single-block capacity.
- **CuTe-DSL async copy optimization**: Allocate shared memory → create async copy atom (`CopyG2SOp`) → async data transfer → manage commit and synchronization. Achieves compute/memory transfer overlap.

### 5.1 CuTe-DSL 4-Level Reduction

**Thread reduction**: `X.reduce(cute.ReductionOp.MAX, init_val=float('-inf'), reduction_profile=0)`.

**Warp reduction**: `@cute.jit` + `cute.arch.shuffle_sync_bfly` butterfly pattern; 32 threads progressively reduce to a single result through multiple shuffle rounds.

**Block reduction**: (1) Each warp lane 0 writes to SMEM → (2) sync → (3) subset of threads reads SMEM → (4) `warp_reduce` completes final block-level reduction.

**Cluster reduction**: Each warp writes reduction results to its own block and other blocks' reduction buffers within the cluster (H100 distributed shared memory), then reads from its own buffer for final reduction. **Breaks the traditional limitation that thread blocks cannot directly communicate.**

**Complete flow**: Load data → thread reduction → warp reduction → conditional (multiple warps per row) → select block or cluster reduction based on cluster configuration. Adaptive design.

---

## 6. Softmax Performance (H100 BF16 M=32k)

| Implementation | Result |
|---|---|
| Torch compile | Baseline |
| Liger Triton | Moderate |
| Quack CuTe-DSL | Stable high performance at long sequences |

Performance is similar at small sequence lengths (1k-4k); differences emerge as sequence length grows. Two key regions: warp reduction without block reduction, and cluster reduction.

---

## 7. Hopper GEMM A@B Performance (BF16, M=N=8k)

CuTe-DSL outperforms cuBLAS 13.0 at all test points. **At K=2k-3k (Ping-Pong overlap epilogue): CuTe-DSL achieves 800 TFLOPS vs cuBLAS at only 760 TFLOPS.**

Ping-Pong Schedule: A technique where two CTAs alternate epilogue and mainloop phases to hide epilogue latency.

**On Blackwell, cuBLAS currently outperforms CuTe-DSL-based implementations by ~3%, expected to be resolved.**

### 7.1 GEMM + SwiGLU Fusion (Hopper, BF16, M=8k, N=5.3d, K=d)

CuTe-DSL achieves stable ~790 TFLOPS; cuBLAS + Triton ranges from 530 up to 740 but remains consistently lower. **Epilogue fusion yields 7-15% performance improvement.**

### 7.2 FA4 vs cuDNN

CuTe-DSL-based FA4 outperforms cuDNN implementation.

---

## 8. DSL Selection Matrix

| DSL | Memory-Bound Scenarios | Compute-Bound Scenarios | Ramp-Up Time |
|---|---|---|---|
| Torch compile | ~90% | ~70-80% | Hours to days |
| Triton | ~90% | ~80-90% | Days to weeks |
| CuTe-DSL | 100% | 100% | Weeks to months |

Overall positioning: Torch (high productivity / relatively lower performance) → Triton (balanced) → CuTe-DSL / CUDA / PTX (high performance / more development effort).

---

## 9. Supplementary: H200 Softmax Bandwidth

Quack (CuTe-DSL) achieves highest bandwidth compared to Torch Naive / Torch Compile / FlashInfer across virtually all configurations.

Selected data points (token_num x hidden):

| token_num | hidden | HF | Torch Compile | FlashInfer | Quack |
|---|---|---|---|---|---|
| 4096 | 16384 | 1357.6 | 1069.5 | 608.5 | **1965.9** |
| 4096 | 32768 | 1425.4 | 1112.0 | 580.1 | **2024.5** |
| 8192 | 32768 | 1462.5 | 609.3 | 246.7 | **2061.6** |
| 16384 | 32768 | 543.9 | 489.6 | 247.8 | **2086.1** |
| 32768 | 16384 | 535.7 | 485.8 | 255.1 | **2079.2** |
