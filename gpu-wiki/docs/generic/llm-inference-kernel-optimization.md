# Community LLM Inference Kernel Optimization

> This article synthesizes multiple technical articles from the Zhihu community, distilling core knowledge of GPU kernel optimization for LLM inference scenarios.

**Last updated**: 2026-06-30

> It covers FlashAttention evolution, KV Cache optimization, MoE GEMM, quantization kernels, linear attention,
> and DeepSeek model inference optimization practices.
>
> See the "Information Sources" section at the end of the article for the list of source articles.

---

## FlashAttention Principles and Evolution (v1 to v4)

### Core Idea: Tiling + Online Softmax

The core optimization motivation of FlashAttention is to avoid writing the complete attention matrix to HBM. Traditional implementations require O(N^2) intermediate storage, whereas FlashAttention loads Q/K/V in blocks into SRAM (shared memory) via tiling, performs softmax computation on-chip, and directly outputs the result, reducing HBM access from O(N^2) to O(N).

Online softmax is the key algorithm enabling tiling: while iterating through K/V blocks, it maintains a running max and running sum, correcting the existing accumulator with the scaling factor `exp(m_old - m_new)` whenever a new block is processed. For specific implementation details, see [Online Softmax and Flash Attention](hands-on/online-softmax-flash-attention.md).

### FlashAttention v2/v3 Improvements

- **v2**: Improved work distribution among warps, reduced the proportion of non-matrix-multiplication FLOPs (such as rescale operations), increasing GPU utilization from ~30% in v1 to ~70%
- **v3** (Hopper architecture): Leveraged TMA (Tensor Memory Accelerator) for asynchronous data loading, combined with warp specialization to overlap computation and memory access, fully utilizing the WGMMA instruction throughput of SM90

### FlashAttention-4: Breakthrough on Blackwell Architecture

FA4 is the latest implementation based on the NVIDIA Blackwell (SM100) architecture, written using CuTe-DSL (replacing hand-written CUDA), reducing compilation time from approximately 55 seconds to approximately 2.5 seconds. Core optimizations include:

**1. Polynomial exp approximation replacing hardware exp2 instructions**

FA4 uses a degree-2 polynomial approximation of `exp(x)` with a precision error of about 1%, but freeing significant SFU (Special Function Unit) resources. On Blackwell, SFU throughput is 1/16 of FMA, while the polynomial approximation is implemented using FMA instructions, eliminating the SFU bottleneck.

**2. Tensor Memory (TMEM) utilization**

Blackwell introduces TMEM—dedicated storage directly coupled with Tensor Cores. FA4 stores softmax intermediate results in TMEM, avoiding shared memory read/write latency. TMEM allocation is per-warp exclusive, requiring no synchronization.

**3. Conditional Rescale Skipping**

When the max value of a new block does not exceed the current running max, the rescale operation is skipped. In practical sequences, approximately 90% of blocks can be skipped.

**4. Asynchronous Pipelining**

A 3-stage pipeline (load → compute → store) is implemented using TMA asynchronous loading, fully overlapping computation and memory access.

**Performance Data**: FA4 achieves 1613 TFLOPS on the Blackwell B200, approximately 71% hardware utilization, nearly 2x the performance of FA3 on Hopper.

### Triton Compilation Flow and Kernel Debugging

Triton kernel compilation goes through 5 stages:

```
Python → TTIR → TTGIR → LLIR → PTX → SASS
```

| Stage | Primary Work |
|------|---------|
| TTIR | Type inference, basic operator fusion |
| TTGIR | GPU-specific optimizations: shared memory allocation, layout inference, warp synchronization |
| LLIR | LLVM IR general optimizations: register allocation, loop unrolling |
| PTX | Virtual ISA: thread-level parallelism, memory hierarchy mapping |
| SASS | Actual GPU machine instructions |

The TTGIR stage is where Triton's core value lies, automatically handling shared memory placement, TMA optimization, warp-level reduction, etc. The performance gap between FlashAttention's Triton implementation and hand-written CUDA mainly stems from optimization strategy differences at the TTGIR stage.

**IR Debugging Tools**: TeraLang provides an IR Viewer (`deciding.github.io/txl`), supporting bidirectional line-level mapping between Python source code and TTIR/TTGIR/PTX. Use `@txl.jit(diff_mode='ttgir', diff_select=0)` to inspect optimization differences pass by pass.

---

## Paged KV Cache Mechanism and Optimization

### Basic Mechanism

The Paged KV Cache mechanism draws analogy to virtual memory paging in operating systems, partitioning the KV Cache into pages of fixed size (typically `page_block_size=64`), and mapping logical positions to physical positions via a block table.

Execution flow:
1. Look up the physical page block list for the sequence using `block_table[batch_idx]`
2. Compute the page block containing the target token: `block_idx = token_pos // page_block_size`
3. Compute the intra-page offset: `offset = token_pos % page_block_size`
4. Load the K/V vectors from the corresponding physical address

### Split-KV Parallelism

When KV sequences are long, a Split-KV strategy is employed to partition the sequence by `kv_seq_sub` (typical value 512), with multiple thread blocks processing different KV sub-sequences in parallel, and results merged via reduction.

For the decode stage (`seqlen_q=1`), each query has only one token in a memory-bound computation pattern, so the degree of Split-KV parallelism directly determines GPU utilization.

### GQA/MHA Support

The Paged Attention kernel supports GQA (Grouped Query Attention) through the ratio relationship between `num_kv_heads` and `num_heads`: multiple query heads share the same group of KV heads, with the kernel internally indexing KV via `head_idx // num_groups`.

## KV Cache Compression: TurboQuant

TurboQuant adopts different compression strategies for K and V:

**K Compression**:
- Random Rotation: Eliminates cross-channel correlation cells for more uniform quantization
- 4-bit non-uniform quantization using offline-trained Lloyd-Max codebooks
- QJL residual compensation for critical information loss

**V Compression (PolarQuant-KV)**:
- Polar decomposition: Decomposes V vectors into magnitude (scalar) and direction (unit vector)
- Direction vectors quantized at low bit-width; magnitude preserved at higher precision
- Leverages the weighted summation property of attention weights—directional quantization errors partially cancel during aggregation

**Performance Data**: 4-bit quantization saves approximately 99% of KV Cache memory. At sequence length 512, attention computation is accelerated by 2.4×. On LLaMA-3-8B, TurboQuant 4-bit achieves perplexity degradation of less than 0.1.

---

## MoE Group GEMM Optimization

### Expert Parallelism vs Tensor Parallelism

GEMM optimization for MoE (Mixture of Experts) layers requires consideration of parallelism strategies:

- **Tensor Parallelism (TP)**: Each GPU holds a portion of every expert's weights. AllReduce communication volume = 2 × hidden_size × batch_size
- **Expert Parallelism (EP)**: Each GPU holds the complete weights of a subset of experts. AllToAll communication volume depends on token routing distribution

DeepSeek-V3/R1 uses 256 experts (top-8 routing). Under EP, communication volume is significantly lower than TP.

### Top-K Fusion Optimization

In TensorRT-LLM, NVIDIA fuses MoE Top-K routing with GEMM to avoid intermediate results spilling to HBM. The approach is to perform top-k selection and token rearrangement from the gate network output directly inside the GEMM kernel, eliminating one global memory read/write round trip.

**Performance Data**: Top-K fusion yields approximately 7.4% speedup on DeepSeek-R1's MoE layers.

### FP4 AllGather Optimization

Under EP mode, AllGather communication between experts is the bottleneck. Transmitting weights in FP4 format (instead of FP8/FP16) halves communication volume, while GEMM performs online FP4→FP8 dequantization on the receiving end.

**Performance Data**: FP4 AllGather delivers approximately 4% end-to-end speedup on MoE layers.

---

## FP8/FP4 Quantization Kernels

### FP8 Quantization Strategy

In MLPerf Inference v5.0, both AMD and NVIDIA adopted the OCP FP8-e4m3 format:

**Per-tensor symmetric static quantization**:
```
scale = absmax(X) / 448
X_quantized = clamp(round(X / scale), -448, 448)
```

where 448 is the maximum representable value for FP8-e4m3. The quantization process uses a calibration dataset to precompute scales.

**Quark Quantization Tool** (AMD):
- Supports AutoSmoothQuant algorithm130 to reduce quantization error
- QKV weights share a common scale (taking the maximum of their individual scales)
- KV Cache is also quantized to FP8
- Exported format can be directly loaded and deployed by vLLM

### FP4 Quantization (Blackwell)

Blackwell architecture natively supports FP4 Tensor Core instructions. After applying FP4 quantization to DeepSeek-R1 on B200:
- Per-GPU TPS increased from approximately 2000 (FP8) to approximately 4600 (TensorRT-LLM)
- 8-card B200 system throughput reached 21,088 tokens/s, approximately 25× that of an 8-card H100 system

### GEMM Tuning

GEMM accounts for over 70% of end-to-end LLM inference computation. Key tuning techniques:

- **Offline GEMM tuning**: Use hipBLASLt (AMD) or cuBLAS (NVIDIA) auto-tuning to select optimal tile sizes for specific GEMM shapes
- **Performance Data**: FP8 GEMM on AMD achieves 1.5K TFLOP for prefill (batch=65K) and 1.5K TFLOP for decode (batch=2048)

---

## Linear Attention & GDN Kernel Optimization

### Linear Attention Overview

Linear attention reduces standard attention's O(N²) complexity to O(N) by decomposing softmax(QK^T)V into Q(K^T V), avoiding explicit computation of the N×N attention matrix.

Representative methods:
- **GLA** (Gated Linear Attention): Introduces gating mechanisms into linear attention
- **KDA** (Kernel-based Dual Attention): Combines linear attention with local attention
- **GDN** (Gated Delta Network): Fuses gating, linear attention, and delta rules

### GDN Prefill Kernel Optimization (Blackwell Gluon)

Qwen3.5's GDN layer prefill adopts a chunk-wise algorithm:

1. Partition the sequence into chunks
2. Within each chunk: use standard attention computation (quadratic portion)
3. Across chunks: use recurrence to propagate state (linear portion)

**Gluon Kernel optimization highlights**:
- **Bitmask Causal Mask**: Use bitmask instead of float mask to save shared memory and bandwidth
- **TMA async_scatter**: Use TMA scatter mode Facts to asynchronously write discontinuous outputs in varlen scenarios, improving performance by approximately 12% over manual scatter
- **Cumsum vectorization**: Vectorize prefix sum operations in GDN to reduce serial dependencies
- Leverage Blackwell's TMEM and tcgen05_mma instructions

## DeepSeek Model Inference Optimization Practices

### DeepSeek-R1 Optimization on Blackwell (NVIDIA Official)

NVIDIA uses TensorRT-LLM to optimize DeepSeek-R1 (671B parameters) on B200. The core optimizations are divided into three layers:

**MLA Layer Optimization**:
- 2CTA MMA: Two CTAs collaborate to execute matrix multiplication, improving Tensor Core utilization
- FP8 KV Cache: Quantizing KV Cache saves memory, resulting in approximately 6% speedup
- Weight absorption: Absorbing MLA projection matrices into QK computation to reduce intermediate results

**MoE Layer Optimization**:
- Top-K fusion: Approximately 7.4% speedup
- FP4 AllGather: Approximately 4% speedup
- Group GEMM tuning: Optimizing GEMM kernels for MoE's uneven expert load

**Runtime Optimization**:
- CUDA Graph: Pre-recording kernel launch sequences to eliminate per-launch CPU overhead, resulting in approximately 22% end-to-end speedup
- ADP (Adaptive Parallelism): Dynamically switching TP/EP strategies based on batch size, approximately 400% speedup
- EP (Expert Parallelism): Distributing 256 experts across multiple GPUs, approximately 142% speedup

**Overall Performance**: Single GPU TPS improved from approximately 2000 (baseline) to approximately 4600, and 8-card B200 achieved 21,088 tokens/s.

### MLPerf Inference v5.0: MI325X Inference Optimization

AMD used MI325X (256GB HBM3E, 6TB/s bandwidth) to run Llama 2 70B in MLPerf Inference v5.0:

**vLLM Optimization**:
- Multi-step scheduling: Scheduling multiple decode steps at once, allowing the GPU to execute forward computation continuously without waiting for CPU instructions
- Hyperparameter tuning: `max_num_batched_tokens` (prefill max batch), `max_num_seqs` (decode max batch), QPS balancing
- Load balancing: Dynamically distributing incoming queries based on each GPU's query count and size
**Quantization and GEMM**:
- Using AMD Quark for FP8 quantization, with AutoSmoothQuant to reduce error
- hipBLASLt offline GEMM tuning to select optimal tile size
- Prefill batch=65K achieving 1.5K TFLOP, decode batch=2048 achieving 1.4K TFLOP

**System-level Tuning**:
```bash
# Performance Mode
sudo cpupower frequency-set -g performance
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
# GPU frequency locking to avoid power throttling
sudo rocm-smi --setperfdeterminism 1700
```

**Performance Results**: MI325X competes directly with H200. The MangoBoost 4-node MI300X cluster achieved 103K tokens/s, setting the MLPerf Llama 2 70B offline performance record.

### AI-Assisted Kernel Generation

NVIDIA experimented with using DeepSeek-R1 to automatically generate GPU attention kernels:
- Closed-loop workflow: R1 generates kernel → verifier analyzes → generates new prompt → R1 iterates
- 15 minutes of inference time produces optimized kernels
- 100% solve rate on KernelBench Level-1 problems, 96% solve rate on Level-2 problems
- Some results outperform hand-optimized kernels written by skilled engineers

---

## GPU Profiling Tools

### Nsight Systems

NVIDIA's official profiling tool, consisting of CLI (`nsys`) and GUI components:
- `nsys profile` sampling generates `.nsys-rep` files
- GUI displays timeline, kernel calls, memory transfers, etc.
- Supports NVTX markers for annotating code regions

### PyTorch Profiler

Embed a profiler context manager in code to output a list of functions ranked by highest GPU time share. The generated JSON data can be used for:
- `chrome://tracing`: Import JSON to view timeline
- TensorBoard: `pip install tensorboard`, launch and view at `localhost:6006/#pytorch_profiler`

### AI Flame Graphs

The AI flame graph proposed by Brendan Gregg displays CPU and GPU call stacks in a mixed view:
- Green: GPU instructions
- Red/Yellow/Orange: CPU code paths (C/C++/kernel)
- Pink: PyTorch functions

This visualization method helps identify CPU-GPU interaction bottlenecks.

---

## Information Sources

This content is compiled from the following Zhihu articles:

- 《GPU Compute Doubled FlashAttention-4》(zhuanlan.zhihu.com/p/2016244587212059939)
- 《Paged KV Cache Execution Flow》(zhuanlan.zhihu.com/p/2022751653233739039)
- 《TurboQuant K+V Dual Compression》(zhuanlan.zhihu.com/p/2021821004310127750)
- 《Optimizing DeepSeek R1 Throughput on NVIDIA Blackwell GPUs》(zhuanlan.zhihu.com/p/1937300334126040902)
- 《cuLA CUDA Linear Attention》(zhuanlan.zhihu.com/p/2023109718873220097)
- 《A Beginner's Perspective on cuLA Development Notes》(zhuanlan.zhihu.com/p/2023828507680056143)
- 《Qwen3.5 GDN Prefill Kernel Optimization》(zhuanlan.zhihu.com/p/2007935329550766500)
- 《[Triton Low-Level Modification Series] From Vector Add to FlashAttention-Level Optimization in Practice》(zhuanlan.zhihu.com/p/2015703290181083527)
- 《CUDA Launch Kernel from an Intelligent Computing Low-Level Software Perspective》(zhuanlan.zhihu.com/p/2017706822077786108)
- 《Large Model Inference Benchmark List Update: GPU Performance Tuning Essentials》(zhuanlan.zhihu.com/p/1891510263217356976)
- 《DeepSeek R1 Generates GPU Kernels Without Programming》(zhuanlan.zhihu.com/p/23456689226)- 《NVIDIA Optimizes DeepSeek-R1 on B200》(zhuanlan.zhihu.com/p/30981405406)
- 《Performance Analysis in the AI Era: A Preliminary Exploration of GPU Profiling》(zhuanlan.zhihu.com/p/31148258173)
- 《Basic Methods for CUDA Kernel Optimization》(zhuanlan.zhihu.com/p/693645814)

## Related

- **Flash Attention Implementation**: [Online Softmax and Flash Attention](hands-on/online-softmax-flash-attention.md) -- Triton-based online softmax and causal mask optimization code
- **Split-KV Merge**: [Cascade / State Merge](hands-on/cascade-state-merge.md) -- Split-KV attention result merging patterns
- **Fused Kernel**: [Fused Kernel Patterns](hands-on/fused-kernel-patterns.md) -- Common patterns such as softmax fusion and reduction fusion
- **GPU Memory Hierarchy**: [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) -- HBM/L2/SRAM bandwidth bottleneck analysis
- **GPU Execution Model**: [GPU Execution Model and Thread Optimization](gpu-execution-model.md) -- SIMT, warp, and occupancy fundamentals
- **Hopper Hands-On**: [Hopper SM90 Optimization Hands-On](../nvidia/hopper/hands-on/README.md) -- TMA, WGMMA, warp specialization
- **Blackwell Hands-On**: [Blackwell SM100 Optimization Hands-On](../nvidia/blackwell/hands-on/README.md) -- tcgen05, TMEM, CLC
- **AMD Inference Optimization**: [AMD General Optimization](../amd/common/README.md) -- MFMA, LDS, GEMM tuning
- **CuTeDSL**: [CuTeDSL Programming Model](../nvidia/common/cutedsl/README.md) -- The DSL framework used by FA4
