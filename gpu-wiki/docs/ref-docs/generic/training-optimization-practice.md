# Community Large Model Training Optimization Practices

> This document synthesizes technical articles from the Zhihu community, distilling knowledge on GPU optimization practices in large model training scenarios.
> It covers GPU resource optimization techniques, training performance analysis and diagnosis, mixed-precision training, Triton compiler optimization,
> single-GPU training of very large models, and engineering practices for sparse model training.
>
> See the "Information Sources" section at the end of the article for a list of source articles.

---

## GPU Resource Optimization: 12 Practical Tips

The following tips are based on community practice summaries, covering common optimization points from single GPU to clusters:

### 1. Mixed-Precision Training (AMP)

Use PyTorch's `torch.cuda.amp` automatic mixed precision to switch some computations to FP16/BF16:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Key Points**:
- BF16 is more stable than FP16 (dynamic range equivalent to FP32); Ampere and newer architectures recommend using BF16
- GradScaler prevents gradient underflow under FP16
- Keep key layers (e.g., LayerNorm, Softmax) in FP32 computation

### 2. Data Loading Optimization

```python
DataLoader(
    dataset,
    num_workers=4,         # Number of CPU prefetch threads, typically set to half of CPU core count
    pin_memory=True,       # Pinned memory, accelerates CPU→GPU transfer
    prefetch_factor=2,     # Number of batches prefetched per worker
    persistent_workers=True # Avoid recreating workers each epoch
)
```

**Key Points**:
- Excessive `num_workers` can cause CPU memory pressure and process switching overhead
- `pin_memory=True` provides significant acceleration for large batches
- Use memory mapping (mmap) to load large datasets and avoid repeated I/O

### 3. Tensor Core Utilization

Ensure matrix dimensions are aligned to multiples of 8 (FP16) or 16 (INT8/FP8):
- Align hidden_size, embedding_size, and batch_size
- When misaligned, the GPU falls back to CUDA Cores, reducing throughput by 10-50×

### 4. Batch Size Tuning

- Too small: Low GPU utilization, kernel launch overhead dominates
- Too large: Insufficient memory or degraded convergence
- Use gradient accumulation to increase the effective batch size without increasing memory usage

### 5. GPU Profiling

Use Nsight Systems or PyTorch Profiler to identify bottlenecks:
- If GPU utilization is low (<70%), check data loading, CPU preprocessing, and synchronization points
- If kernel time is long, check whether Tensor Cores are being used and review memory access patterns

### 6. Model Architecture Optimization

- Use `nn.functional` instead of custom loops
- Prefer fused operators whenever possible (e.g., `F.scaled_dot_product_attention`)
- Reduce small kernel invocations and use kernel fusion

### 7. Memory Management

- `torch.cuda.empty_cache()` frees cache (note: does not release tensors themselves)
- Gradient checkpointing: Trade computation for memory, reducing activation memory by 2-3×
- Use `torch.no_grad()` for operations that don't require gradients

### 8. Minimizing CPU-GPU Transfers

- Avoid frequent `.cpu()` / `.cuda()` conversions
- Move preprocessing logic to the GPU
- Use `non_blocking=True` for asynchronous transfers

### 9. XLA Compilation Optimization

For TPU or specific GPU scenarios, use the XLA compiler:
- Automatic operator fusion
- Memory layout optimization
- Supports the `torch.compile` backend

### 10. Distributed Training

- DDP (DistributedDataParallel): Gradient AllReduce
- FSDP (Fully Sharded Data Parallel): Sharding of parameters, gradients, and optimizer states
- Pipeline Parallelism: Split layers across different GPUs
- Overlapping communication and computation is key

### 11. Checkpoint Strategy

- Asynchronous saving: Avoid blocking the training loop
- Incremental saving: Save only changed parameters
- Use distributed checkpoint formats (e.g., `torch.distributed.checkpoint`)

### 12. GPU Cluster Management

- Monitor GPU temperature and power consumption to avoid throttling
- Enable persistence mode: `nvidia-smi -pm 1`
- NUMA-aware process binding

---

## Training Performance Bottleneck Analysis and Diagnosis

### Nsight Systems in Practice

Nsight Systems is NVIDIA's official system-level profiling tool, suitable for analyzing end-to-end training performance:

**CLI Sampling**:
```bash
nsys profile -t cuda,nvtx,osrt -o report python train.py
```

**Key Analysis Dimensions**:
- **Timeline View**: Examine the timeline of GPU kernels, CUDA APIs, and CPU threads
- **Kernel Statistics**: Sort by time/invocation count to identify hotspot kernels
- **Stream Concurrency**: Check whether multiple streams overlap effectively
- **CPU-GPU Interaction**: Identify synchronization points and idle gaps

**NVTX Markers**: Insert markers in code to partition profiling intervals:
```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("forward")
output = model(input)
nvtx.range_pop()
```

**Common Bottleneck Patterns**:
1. **Data Loading Bottleneck**: GPU utilization appears sawtooth-shaped, with long idle periods at the start of each iteration
2. **Communication Bottleneck**: AllReduce/AllToAll time accounts for >30%
3. **Small Kernel Overhead**: Numerous short kernels cause launch overhead to exceed computation time
4. **Memory Bottleneck**: Frequent allocation/deallocation triggers CUDA malloc

**Megatron-LM Profiling Experience**: Launch with the `--profile` flag; Nsight Systems can clearly display the bubble ratio of each stage in Pipeline Parallelism, helping to adjust the number of micro-batches and the pipeline schedule.

### PyTorch Profiler + TensorBoard

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
) as prof:
    for step, data in enumerate(dataloader):
        # training step
        prof.step()
```

TensorBoard provides multi-dimensional analysis:
- GPU kernel time distribution
- Memory usage timeline
- Operator-level CPU/GPU time comparison

---

## Mixed Precision Training Optimization

### FP16 vs BF16

| Feature | FP16 | BF16 |
|------|------|------|
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Value range | ±65504 | ±3.4e38 (same as FP32) |
| Precision | Higher | Lower |
| Stability | Requires loss scaling | Usually not needed |
| Hardware support | Volta+ | Ampere+ |

**Practical recommendations**:
- Prefer BF16 for Ampere and later architectures
- If training encounters NaN/Inf, check for FP16 overflow
- Maintain FP32 precision during gradient accumulation
- Keep BatchNorm/LayerNorm statistics in FP32

### Tensor Core Alignment

Tensor Core requires input matrix dimension alignment:
- FP16/BF16: multiples of 8
- FP8: multiples of 16
- INT8: multiples of 16

Misalignment can cause 10-50x performance degradation. Common parameters needing alignment:
- `hidden_size`, `intermediate_size`
- `num_attention_heads` (ensure head_dim is aligned)
- `vocab_size` (pad to multiples of 8)

---

## Triton/Gluon Compiler Optimization

### AMD Triton Compilation Process

The AMD Triton compiler adds AMD-specific optimizations on top of the standard pipeline (frontend → optimizer → backend):

**Optimizer layer (AMD-specific)**:
- **LDS Bypass**: When data is used only once, skip LDS (Local Data Share) and load directly from global memory to registers
- **amd_mfma Layout**: Automatically derive data layout based on MFMA (Matrix Fused Multiply-Add) instruction operand requirements
- **Blocked/Shared Layout Selection**: Automatically choose the optimal data layout based on memory access patterns

**Backend layer**:
- Generates hsaco (Heterogeneous System Architecture Code Object) binaries
- Differs from NVIDIA's PTX → SASS path

### GEAK-Triton v2: AI-Driven Kernel Optimization

GEAK-Triton v2 uses an AI agent to automatically optimize Triton kernels on AMD GPUs:

**OptimAgentv2 Architecture**:
- Multi-offspring Evolution: Generate multiple variants for parallel evaluation each iteration
- LLM Evaluator: Uses large language models to analyze profiling data and generate optimization suggestions
- Profiler-Analyzer: Hardware-aware performance analysis feedback

**OpenEvolve Framework**:
- Quality-Diversity MAP-Elites algorithm: Maintains a pool of kernel variants balancing diversity and quality
- Automatically explores different combinations of tile size, unroll factor, and pipeline stage

**Performance data**: Achieves an average speedup of 3.32-7.02x on AMD GPUs.

---

## Single-GPU Ultra-Large Model Training: MegaTrain

The MegaTrain approach enables training 100B+ parameter models on a single GPU, with CPU-centric design as its core concept:

### Streaming Forward/Backward

The model is divided into multiple chunks by layers, with only one chunk resident on the GPU at any time:

1. Load layer i parameters from CPU memory to GPU
2. Execute forward computation
3. Transfer activations back to CPU or discard them (paired with gradient checkpointing)
4. Offload parameters back to CPU
5. Load layer i+1 parameters, repeat

### Double Buffering

Uses two GPU buffers that alternate:

- Buffer A: Computing layer i
- Buffer B: Asynchronously loading layer i+1 parameters

Hides transfer latency by overlapping CPU-GPU transfers with GPU computation.

**Performance data**:
- Double buffering yields a 31.3% throughput improvement
- Can train a 1200B parameter model on a single H200
- 14B model training reaches 264 TFLOPS

### Stateless Templates

Design stateless layer templates to avoid maintaining additional state information during parameter offload/load. Optimizer states are also stored in CPU memory and loaded on demand.

---

## Sparse Model Training: JD Advertising Practices

Training architecture for ultra-large sparse models (TB-level embedding parameters) in JD's advertising scenario:

### CPU-DRAM Parameter Server

Sparse embedding parameters cannot all fit into GPU HBM, using tiered storage:
- **CPU DRAM**: Stores the complete embedding table (TB-level)
- **GPU HBM Cache**: Caches frequently accessed embedding vectors
- LRU-based cache replacement strategy

### Distributed Pipeline

```
Data sharding → Embedding Lookup (CPU/GPU collaboration) → Dense network (GPU) → Gradient update
```

- Embedding lookup handled by parameter servers, supporting asynchronous updates
- Dense components use standard data parallelism
- Pipeline parallelism overlaps communication with computation

### Inference Optimization

- **TensorBatch**: Merge multiple small batches into a large batch for batched inference
- **Multi-stream GPU**: Use multiple CUDA streams(Node to execute different inference requests in parallel
- **XLA Compiler Adaptation**: Optimize compilation strategy for sparse access patterns

---

## PTX and Low-Level Optimization

### PTX Virtual ISA

PTX (Parallel Thread Execution) is NVIDIA's virtual instruction set, positioned between LLVM IR and SASS machine code:**Register Allocation**:
- PTX uses infinite virtual registers, which are mapped to physical registers by the compiler
- Physical registers are limited (SM90: 65,536 32-bit registers per SM)
- Excessive register pressure causes spills to local memory (which physically resides in HBM), leading to dramatic performance degradation

**Warp-Level Primitives**:
- `shfl.sync`: Threads within a warp directly exchange data without going through shared memory
- Used for patterns such as reduction, broadcast, and butterfly exchange
- Latency is approximately 1 cycle, far lower than shared memory's ~20 cycles

**SFU Optimization**:
- Special Function Units execute transcendental functions such as sin/cos/exp/log/rsqrt
- Throughput is far lower than FMA (1/16 of FMA on Blackwell)
- For performance-sensitive kernels, use polynomial approximations instead of SFU instructions

**Architectural Portability**: PTX programs can run on different GPU architectures (via JIT compilation to SASS), but performance is not guaranteed to be optimal.

---

## Information Sources

This article synthesizes content from the following Zhihu articles:

- "12 Practical Tips for GPU Resource Optimization" (zhuanlan.zhihu.com/p/1902442351688922566)
- "AMD GPU Triton Kernel Optimization" (zhuanlan.zhihu.com/p/1904621841001190537)
- "Advanced CUDA Kernel Optimization Techniques: Handwritten PTX" (zhuanlan.zhihu.com/p/1926327814820435340)
- "Nsight Systems Tool Principles" (zhuanlan.zhihu.com/p/1955933603209917837)
- "GEAK-Triton v2" (zhuanlan.zhihu.com/p/1996972082488157726)
- "MegaTrain" (zhuanlan.zhihu.com/p/2025389159984973476)
- "JD.com Advertising Sparse Large Model Training and Inference" (zhuanlan.zhihu.com/p/713692019)

---

## Related Documents

- **GPU Execution Model**: [GPU Execution Model and Thread Optimization](gpu-execution-model.md) -- SIMT, warp, and occupancy fundamentals
- **GPU Memory Hierarchy**: [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) -- Register, shared memory, and HBM bandwidth analysis
- **GPU Instruction Optimization**: [GPU Instruction-Level Optimization](gpu-instruction-optimization.md) -- Fast math, precision vs. throughput trade-offs
- **Application-Level Optimization**: [GPU Application-Level Optimization Strategies](gpu-application-optimization.md) -- Amdahl's Law, parallelization assessment
- **LLM Inference Optimization**: [Community LLM Inference Kernel Optimization](llm-inference-kernel-optimization.md) -- Kernel optimization practices for inference scenarios
- **NVIDIA Profiling**: [NCU/Nsight Optimization](../nvidia/common/README.md) -- Detailed documentation on NVIDIA profiling tools
- **AMD Optimization**: [AMD General Optimization](../amd/common/README.md) -- AMD GPU optimization techniques
- **Triton Hands-On**: [Triton Kernel Optimization Patterns Hands-On](../../kernel-opt/generic/hands-on/README.md) -- 8 Triton optimization patterns
