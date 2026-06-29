# FlashAttention-4 Release: Algorithm Pipeline Redesign

Overview of FlashAttention-4's official release, covering the fundamental pipeline redesign that achieves near-matmul-speed attention on Blackwell GPUs through co-design with asymmetrically scaled hardware.

---

## 1. Overview

FlashAttention-4 is officially released after a year of development. On Blackwell GPUs, attention execution speed now approaches that of matrix multiplication — even though the bottlenecks are fundamentally different from previous generations.

Current Tensor Cores are so fast that the forward pass bottleneck is no longer matmul but exponential operations on the SFU; the backward pass bottleneck is shared memory bandwidth. FA4's redesigned algorithm contains a series of mechanisms to overcome these bottlenecks: polynomial-emulated exponential, novel online softmax avoiding 90% of rescaling, and 2-CTA MMA enabling two thread blocks to share operands and reduce SMEM traffic.

## 2. Hardware Trend: Asymmetric Scaling

FA3 targeted Hopper (H100); the industry has rapidly shifted to Blackwell (B200/GB200). Blackwell continues the pattern of **asymmetric hardware scaling**: Tensor Core throughput growth far outpaces other resources (SMEM bandwidth, SFU, ALU).

From H100 to B200: BF16 Tensor Core throughput scales from 1 to 2.25 PFLOPS (**2.25×**), but SFU and SMEM bandwidth remain essentially unchanged.

Attention contains two GEMMs with intermediate softmax, accompanied by extensive data movement, synchronization, layout conversion, and mask handling. The conventional view is that performance is determined by GEMM. But speeds-and-feeds analysis on B200 reveals:

- **Forward:** Bottleneck is SFU for softmax exponential operations
- **Backward:** Bottleneck is SMEM bandwidth

## 3. FA4 Co-Design

Core objective: maximize overlap between matrix multiplication and other bottleneck resources, achieving 1605 TFLOPS/s (71% utilization), 1.3× faster than cuDNN 9.13, 2.7× faster than Triton.

- **Novel pipeline:** Separate software pipelines designed for forward and backward, leveraging Blackwell's fully-async MMA and larger tile sizes to maximize overlap between Tensor Core, softmax, and memory operations
- **Forward (FWD):** Software-emulated exponential via polynomial approximation on FMA units; **conditional softmax rescaling** skips unnecessary rescales to relieve SFU bottleneck
- **Backward (BWD):** TMEM stores intermediate results to relieve SMEM traffic; **2-CTA MMA** further reduces SMEM access and halves atomic reduction count; supports deterministic execution mode
- **Scheduling:** New tile scheduler addresses load imbalance from causal masks and variable-length sequences

## 4. Blackwell Hardware Features

**Tensor Memory (TMEM):** B200's 148 SMs each have 256 KB TMEM, directly connected to Tensor Core, used for warp-synchronous intermediate result storage.

**Fully-Async Fifth-Generation Tensor Core:** `tcgen05.mma` supports asynchronous execution with accumulators stored in TMEM. Under BF16/FP16, single-CTA maximum UMMA tile is 128×256×16 (approximately 2× Hopper's largest WGMMA atom). UMMA is initiated by a single thread, reducing register pressure and enabling larger tiles and deeper pipelines; it also makes warp specialization (data-moving warps / MMA-issuing warps) more viable. `tcgen05.mma` can also read operand A directly from TMEM.

**2-CTA MMA:** A pair of CTAs in the same cluster jointly execute one UMMA, spanning TMEM across both CTAs. A single thread in the leader CTA initiates the MMA, but both CTAs must remain active during execution. By splitting M and N dimensions across the pair, MMA tile can expand to **256×256×16**, reducing redundant data transfer and lowering per-CTA resource requirements. CTA group size (1 or 2) must be consistent between TMEM operations and Tensor Core computations within a kernel.

## 5. Programming Language: CuTe-DSL

FA4 is implemented entirely in CuTe-DSL (CUTLASS's Python kernel DSL): write kernels in Python → DSL lowers to PTX → CUDA toolchain compiles to GPU machine code. Abstraction level is consistent with CuTe/CUTLASS, while providing PTX-level escape hatches. **Compilation time reduced approximately 20–30× compared to C++ templates** — installation/compilation takes seconds rather than minutes/hours.

## 6. Performance Benchmarks

- **Forward:** 1.1–1.3× faster than cuDNN 9.13, 2.1–2.7× faster than Triton
- **Backward:** Consistently outperforms other baselines on long sequences

PyTorch has announced that **FlexAttention now supports the FA4 backend**: PyTorch can automatically generate CuTe-DSL score/mask modification code and instantiate FA4 for custom attention variants via JIT compilation; achieves 1.2–3.2× over Triton in compute-bound scenarios.

## 7. Significance

FlashAttention-4 represents a milestone. On Blackwell, attention can now approach matmul speed, meaning the computational bottleneck will shift entirely to memory and communication. The 1600 TFLOPS attention performance represents a 2–3× improvement over FA3.
