# System-Level GPU Optimization

> Synthesized from multiple technical articles in the Zhihu community, covering GPU utilization diagnostics, CUDA Graph, Multi-Stream parallelism, Kernel Launch overhead optimization, inference serving architecture, and other system-level optimization topics.

---

## 1. GPU Utilization: Understanding Real Performance

### 1.1 GPU Utilization ≠ GPU Performance

The GPU utilization (GPU Util) displayed by nvidia-smi is defined as: **the percentage of time during the past sampling period that one or more kernels were executing on the GPU**.

This means:
- As long as 1 SM is executing any operation (even just reading/writing memory), it counts as 100%
- Does not reflect the actual usage of compute units (CUDA Core / Tensor Core)
- Does not reflect SM spatial utilization (only a small number of SMs may be active)

**Real-world case**: An LLM training job showed 100% GPU utilization, but MFU (Model FLOPS Utilization) was only 20%. After optimizations such as kernel fusion (FlashAttention, FusedMLP, etc.), MFU increased to 38%, and training speed improved by 4×.

### 1.2 Multi-Level Utilization Metrics

To accurately assess GPU performance, a multi-level metric system is needed:

```
┌────────────────────────────────────────────┐
│ Level 1: GPU Utilization (nvidia-smi)      │
│ "GPU ？" - meansnone kernel execute │
├────────────────────────────────────────────┤
│ Level 2: SM Active (SM utilization) │
│ " SM ？" - SM SM │
├────────────────────────────────────────────┤
│ Level 3: SM Occupancy (SM Warp utilization) │
│ "SM warp ？" - │
├────────────────────────────────────────────┤
│ Level 4: Tensor Core Active                │
│ "Tensor Core compute？" - matrixcompute │
├────────────────────────────────────────────┤
│ Level 5: MFU (Model FLOPS Utilization)     │
│ "actualcompute vs peak" - standard │
└────────────────────────────────────────────┘
```

**SM Active (SM Spatial Utilization)**:

```
SM_Active = Σ( SM ) / ( SM × )
```

Example: 10 time slices, 10 SMs, where 8 SMs are active during T0-T1, 1 SM during T4, and 3 SMs during T6-T8:
```
SM_Active = (2×8 + 1×1 + 3×3) / (10×10) = 26%
```

**SM Occupancy (SM Warp Utilization)**:

```
SM_Occupancy = warp / SM warp
```

Taking A100 as an example: with 256 threads per block estimating to 255 registers per thread → one block occupies all 65536 registers → only 1 block per SM → only 8 warps (256/32) → Occupancy = 8/64 = 12.5%.

**Note**: High Occupancy does not always mean high performance, but low Occupancy always reduces the ability to hide memory latency.

### 1.3 MFU: The Ultimate Performance Metric

MFU (Model FLOPS Utilization) = Actual Throughput / Theoretical Peak Throughput

```
MFU = (actual tokens/s × token FLOP) / GPU peak FLOPS
```

Current MFU reference values for LLM training:
- Baseline (unoptimized): 15-25%
- Good: 35-45%
- Excellent: >50%

Common causes of low MFU:
- Memory transfer time >> compute time (memory-bound kernel)
- Kernels not fused, launch overhead accounts for a large proportion
- Communication (AllReduce/AllGather) not overlapped with computation
- Data loading pipeline bottleneck

---

## 2. Common Causes of Low GPU Utilization and Optimization

### 2.1 Data Loading Bottleneck

**Symptoms**: GPU periodically idles, showing a sawtooth utilization pattern. nsys timeline shows large gaps between kernels.

**Cause**: Data preprocessing is executed on the CPU Look at the sentence think tooks on 2c5ness matching GPU consumption speed.

**Optimization solutions**:

| Solution | Description | Effect |
|------|------|------|
| Increase `num_workers` | Number of parallel loading processes in DataLoader | Typically set to half the number of CPU cores |
| Enable `pin_memory=True` | Use pinned memory to accelerate HtoD copy | HtoD bandwidth doubled |
| Increase `prefetch_factor` | Prefetch more batches to CPU memory | Smoother GPU waiting |
| DALI / GPU Decoding | Offload image decoding/augmentation to GPU | Eliminate CPU bottleneck |
| Increase batch size | Reduce kernel launch frequency | Increase GPU time proportion |

### 2.2 CPU Preprocessing Bottleneck

**Symptoms**: Low GPU utilization, high CPU utilization.

**Common preprocessing operations**:
- Image decoding (JPEG/PNG)
- Data augmentation (crop/rotation/color transform)
- Tokenization (NLP scenarios)
- Feature engineering

**Optimization solutions**:
- Move preprocessing to GPU (NVIDIA DALI, TorchVision GPU transforms)
- Precompute datasets (offline preprocessing stored in binary format)
- Use faster CPU decoding libraries (turbojpeg, pillow-simd)

### 2.3 Kernel Launch Overhead

**Symptoms**: Many short kernels in nsys, each separated by ~5-10μs gaps.

**Essence**: Each kernel launch requires the CPU to submit a command to the GPU, passing through driver API → command queue → GPU parsing, with an inherent latency of approximately 5-15μs.

**Impact**:
- When kernel execution time is very short (<50μs), launch overhead accounts for a significant proportion
- On small-scale problems, launch latency rather than memory bandwidth is the primary bottleneck



---

## 3. CUDA Graph

### 3.1 What is CUDA Graph

CUDA Graph records a sequence of kernel launches and dependency relationships into a graph structure, which can then be submitted to the GPU for execution in a single operation radiotherapy.

**Principle**:

```
Traditional mode:                  CUDA Graph mode:
CPU: launch K1 → wait              CPU: launch Graph → done
     launch K2 → wait                   (GPU automatically executes K1→K2→K3)
     launch K3 → wait
GPU: ──K1──gap──K2──gap──K3        GPU: ──K1─K2─K3── (no gaps)
```

**Core advantages**:
- **Eliminates kernel launch overhead**: A single graph launch replaces N individual kernel launches, reducing N×5μs launch time
- **Reduces CPU-GPU synchronization**: Graph is completely executed on the GPU side
- **Enables global optimization**: GPU driver performs graph-level optimization (e.g., instruction prefetching)

### 3.2 Applicable Scenarios and Limitations

**Applicable scenarios**:
- Repetitive workloads with fixed computational graphs (e.g., each training iteration)
- Large number of small kernels (e.g., element-wise operations)
- Inference serving scenarios

**Limitations**:
- The computational graph must be static (dynamic shapes are not inherently supported; instance pools are needed)
- Debugging is relatively difficult
- CPU-side logic in the middle of graph execution is not supported
- Conditional branches and loops have limited support (partial support in recent versions)

### 3.3 Basic Usage

```python
# CUDA Graph in PyTorch
graph = torch.cuda.CUDAGraph()

# Warm up to stabilize GPU memory layout
with torch.cuda.stream(stream):
    for _ in range(3):
        output = model(static_input)

# Capture
with torch.cuda.graph(graph):
    output = model(static_input)

# Replay: each inference only needs graph replay
static_input.copy_(real_input)
graph.replay()
```

```cpp
// Stream Capture in CUDA C++
cudaGraph_t graph;
cudaGraphExec_t instance;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int iteration = 0; iteration < 1000; ++iteration) {
    kernel<<<grid, block, 0, stream>>>(args);
}
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
cudaGraphLaunch(instance, stream);
```

**Performance impact**: When there are many small kernels, latency can be reduced by 20-50%.

### 3.4 CUDA Graph in PyTorch

PyTorch provides high-level encapsulation of CUDA Graph:

```python
static_input = torch.empty_like(real_input)
graph = torch.cuda.CUDAGraph()

# Capture once, replay many times
with torch.cuda.graph(graph):
    static_output = model(static_input)

for real_input in requests:
    static_input.copy_(real_input)
    graph.replay()
    output = static_output.clone()
```

---

## 4. GPU Multi-Stream Concurrency

### 4.1 Stream Model

CUDA Stream represents a GPU operation queue. Operations within the same stream execute in order; operations between different streams can execute concurrently.

**Default behavior**: If no stream is specified, all operations use the default stream (`stream 0`), which is a synchronous stream (implicit synchronization between operations).

**Inference**: Even if default_stream is not explicitly used, operations by default are in stream 0apat, and implicit synchronization occurs between host/device operations.

### 4.2 Key Concurrency Mechanisms

Some operations can overlap:
- Data copy (HtoD/DtoH) and kernel execution
- Different types of kernels (compute-bound and memory-bound)
- CUDA operations and CPU operations tenant

**Typical async overlapping pipeline**:

```
Stream 0: HtoD(batch N+1) ───────────────
Stream 1: ─────── Kernel(batch N) ───────
Stream 2: ───────────────── DtoH(batch N-1)
```

### 4.3 Stream Overlap in PyTorch

**Note**: PyTorch's CUDA caching allocator is stream-ordered. Using streams allows overlap, but attention must be paid to tensor lifecycle management.

```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    next_batch = next_batch.to('cuda', non_blocking=True)
    next_output = model(next_batch)

# Wait only when the result is needed
torch.cuda.current_stream().wait_stream(stream)
```

### 4.4 Concurrency Limitations

Not all GPUs support the same degree of concurrency:
- **Copy Engine count**: Most GPUs have 1-2 copy engines. Simultaneously running 3+ async copying streams provides no performance benefit.
- **Kernel concurrency**: It is generally difficult fault for multiple compute-intensive (compute-bound) kernels to truly execute in parallel
- **Hardware scheduling**: SM allocation between streams is managed by hardware

---

## 5. Inference Serving Optimization

### 5.1 Typical Inference Service Architecture

```
┌──────────┐    ┌───────────┐    ┌───────────┐
│  Proxy   │───→│CPU Process│───→│GPU Process│
│ (router) │    │(pre/post) │    │(inference)│
└──────────┘    └───────────┘    └───────────┘
                      ↑                ↑
              scalable processes   load multiple models
              shared-memory IPC    into GPU memory
```

### 5.2 Key Optimization Techniques

| Technique | Description | Expected Benefit |
|------|------|------|
| **Batching** | Aggregate multiple requests for batch inference | Improve throughput by 2-10× |
| **Continuous Batching** | Dynamically adjust batch when requests complete | Reduces latency by 30-50% compared to static batching |
| **KV Cache** | Cache Key/Value tensors to avoid recomputation | Reduces computation by 50% per decode step |
| **PagedAttention** | Manage KV Cache in block/paged units | Reduce memory fragmentation, improve utilization |
| **Quantization** | INT8/FP8/INT4 quantization of weights and activations | Reduce memory by 2-4×, improve throughput |
| **Speculative Decoding** | Use a draft model to generate candidate tokens | Improve latency by erness 2-3× |
| **Multi-GPU TP/PP** | Tensor parallelism / Pipeline parallelism | Support larger models, linear scaling |

### 5.3 PagedAttention Principle

Core mechanism: similar to virtual memory paging in OS, dividing KV Cache into fixed-size blocks.

```
Request A: [block1][block2][block3]
Request B: [block4][block5]
Request C: [block1][block2][block6]  # shared prefix

KV cache is managed through a page table:
logical block id → physical GPU memory block
```

**Advantages**:
- No need to pre-allocate contiguous memory
- Dynamic block allocation, reducing fragmentation
- Memory sharing during beam search and parallel sampling
- Typical memory waste reduced from >50% to <4%

### 5.4 Continuous Batching

Traditional static batching must wait for all requests in a batch to complete before proceeding. Continuous batching immediately returns completed requests and inserts new requests.

```
Time t0: [Req A token 1][Req B token 1][Req C token 1]
Time t1: [Req A token 2][Req B done   ][Req C token 2][Req D token 1]
Time t2: [Req A token 3][Req C done   ][Req D token 2][Req E token 1]
```

### 5.5 Speculative Decoding

**Principle**: Use a smaller draft model to quickly generate k candidate tokens, then the target model validates all k tokens in a single forward pass.

```
Draft model:  token1, token2, token3, token4  (fast generation)
Target model: verify these 4 tokens in one forward pass
Accepted:     token1, token2, token3
Rejected:     token4 → regenerate from token3
```

**Key metrics**: Acceptance rate — the probability that draft tokens are accepted. Higher acceptance rate → better speedup.

---

## 6. GPU Memory Optimization

### 6.1 Memory Hierarchy and Characteristics

| Level | Size (H100) | Bandwidth | Latency | Typical Content |
|------|------|------|------|------|
| Register File | 256KB/SM | ~8TB/s | ~0 cycles | Temporary variables |
| L1 Cache/Shared Memory | 256KB/SM | ~4TB/s | ~30 cycles | Frequently accessed data |
| L2 Cache | 50MB | ~4TB/s | ~200 cycles | Cross-SM shared data |
| HBM | 80GB | 3.35TB/s | ~300 cycles | Activations, weights |
| NVLink | - | 900GB/s | - | Cross-GPU communication |

### 6.2 Common Memory Issues

**Memory Fragmentation**:
- Allocating and releasing tensors of varying sizes leads to fragmentation
- Solution: Use a memory pool, pre-allocate large blocks

**Reserved but Unused**:
- PyTorch caches freed memory but does not return it to the OS
- Symptoms: nvidia-smi shows high memory usage, but PyTorch reports low allocated memory
- Solution: `torch.cuda.empty_cache()` or use `PYTORCH_NO_CUDA_MEMORY_CACHING=1`

**Activation Memory Overhead**:
- Backpropagation requires saving forward intermediate results
- Solutions: Gradient Checkpointing (trading computation for memory), Activation Offloading

---

## 7. End-to-End Optimization Practice

### 7.1 Systematic Diagnostic Process

```
1. Check nvidia-smi: Is GPU Util high?
   ├─ Low → data loading / CPU preprocessing / launch overhead
   └─ High → continue

2. Check SM Active: Are most SMs active?
   ├─ Low → kernel grid too small / poor parallelism
   └─ High → continue

3. Check SM Occupancy: Are enough warps resident?
   ├─ Low → register pressure / shared-memory usage / block size
   └─ High → continue

4. Check Tensor Active / memory bandwidth:
   ├─ Low Tensor Active + high bandwidth → memory-bound
   ├─ High Tensor Active → compute-bound
   └─ Both low → scheduling / dependency / launch overhead
```

### 7.2 Common Bottleneck Identification

| Tool | Usage |
|------|------|
| nvidia-smi | Quick check for overall GPU usage |
| nsys / ncu | Fine-grained timeline and kernel analysis |
| PyTorch Profiler | PyTorch-level operator analysis |
| torch.cuda.Event | Manual insertion of timing points |
| DCGM | Cluster-level monitoring |

### 7.3 Optimization Priority

In order of return on investment:

1. **Eliminate data loading/CPU bottlenecks** (largest ROI): Most common issue, simple fixes have dramatic effects
2. **Kernel fusion and algorithm optimization**: FlashAttention, fused optimizers, fused MLP
3. **Communication overlap**: Overlap AllReduce and computationLow
4. **CUDA Graph**: Eliminate launch overhead
5. **Memory optimization**: Reduce peak memory to support larger batch sizes
6. **Precision optimization**: BF16/FP8 training, INT4/INT8 inference

### 7.4 Verify That Optimization Is Effective

Each optimization must quantitatively verify its effect:

- **Throughput improvement**: samples/sec, tokens/sec
- **Latency reduction**: P50/P95/P99 latency
- **MFU improvement**: Model FLOPS Utilization
- **Memory savings**: Peak memory reduction

---

## 8. Summary

GPU system-level optimization covers multiple dimensions from hardware utilization to software architecture:

- **Utilization metrics**: Do not be deceived by nvidia-smi's simple GPU Util; assess performance using MFU + SM Active + SM Occupancy
- **System-level optimization**: Address data loading bottlenecks, launch overhead, and CPU preprocessing bottlenecks
- **CUDA Graph**: Eliminates launch overhead, suitable for static computation needs graphCre scenarios
- **Multi-Stream**: Overlap computation and data transferCopy
- **Inference optimization**: Continuous batching, PagedAttention, speculative decoding, quantizationbre
- **Memory optimization**: Gradient checkpointing, memory pool, activation offloading

Optimization should follow a "measurement → identification → optimization → verification" iterative process, with each change quantitatively verifying its effect.

## 3. CUDA Graph: Eliminating Kernel Launch Overhead

### 3.1 Principle

CUDA Graph captures a series of CUDA operations (kernel launches, memory copies, etc.) as a "graph" and submits them to the GPU in a single batch. Subsequent executions only need to replay the graph, eliminating the per-kernel CPU submission overhead.

```
mode: CUDA Graph mode:
CPU: launch K1 → wait         CPU: launch Graph → done
 launch K2 -> wait (GPU automaticexecute K1->K2->K3)
     launch K3 → wait
GPU: ──K1──gap──K2──gap──K3 GPU: ──K1─K2─K3──(none)
```

### 3.2 Usage

**Method 1: Stream Capture**

```python
# PyTorch CUDA Graph
g = torch.cuda.CUDAGraph()

# ( GPU memorylayout)
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)

# English note
with torch.cuda.graph(g):
    output = model(static_input)

# ( replay)
static_input.copy_(real_input)
g.replay()
```

```cpp
// CUDA C++ Stream Capture
cudaGraph_t graph;
cudaGraphExec_t instance;

// English comment
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < 1000; i++) {
    kernel<<<grid, block, 0, stream>>>(args);
}
cudaStreamEndCapture(stream, &graph);

// createexecute
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// execute
cudaGraphLaunch(instance, stream);
```

### 3.3 CUDA Graph Limitations

| Limitation | Description | Workaround |
|------|------|---------|
| Fixed shapes | Input/output tensor shapes and addresses must remain unchanged | Use padding to fixed shapes |
| No dynamic control flow | Conditional branches are not allowed in the graph | Precompile different graphs for different branches |
| No dynamic memory allocation | cudaMalloc is not allowed during graph execution | Pre-allocate fixed-size buffers |
| Capture overhead | Initial capture incurs some overhead | Capture only during initialization, replay thereafter |

### 3.4 Performance Benefits

Measured data from a recommendation system scenario (Alibaba RTP):

| Scenario | Without Graph | With Graph | Improvement |
|------|---------|---------|------|
| Online inference QPS | 1000 | 2000+ | ~2× |
| Single inference latency | 15ms | 8ms | ~47% |

On the B200 for small-scale problems (10KB data), CUDA Graph alone can deliver approximately 50% performance improvement.

---

## 4. Multi-Stream Parallelism

### 4.1 CUDA Stream Basics

A CUDA Stream is an ordered queue of commands. Operations within the same stream execute sequentially, while operations across different streams can run in parallel.

```
Stream 0: ──K1────K2────K3────
Stream 1: ──────K4────K5──────
Stream 2: ────K6──────K7──────
```

Scenarios for multi-stream parallelism:
- Multiple independent kernels can execute concurrently
- Overlapping memory copies with computation (requires different streams)
- Multi-model inference (different models on different streams)

### 4.2 Multi-Stream + MPS

Naive multi-stream usage within a single CUDA context is still subject to serialization constraints. NVIDIA MPS (Multi-Process Service) enables true parallelism across contexts:

```
                 ┌──── Context 1 (Stream 0, 1)
MPS Server ──────┤
                 └──── Context 2 (Stream 0, 1)
```

Benefits of MPS:
- Reduces context switching overhead when multiple processes share a GPU
- Enables true concurrent kernel execution
- Suitable for multi-model deployment scenarios such as recommendation systems

### 4.3 CUDA Graph vs Multi-Stream Comparison

| Dimension | CUDA Graph | Multi-Stream |
|------|-----------|-------------|
| Optimization target | Eliminate launch overhead | Increase parallelism |
| Applicable scenarios | Repeated execution of fixed patterns | Concurrent independent tasks |
| Shape constraints | Must be fixed | No constraints |
| Implementation complexity | Moderate | Lower |
| Compatibility | No dynamic control flow support | General-purpose |
| Typical gains | 2× throughput (small-kernel-intensive scenarios) | 1.5–2× throughput (multi-model scenarios) |

**Recommended combination**: For scenarios such as recommendation systems, inference for multiple models can each be captured with CUDA Graph and then executed in parallel on different streams, achieving the benefits of both approaches.

---

## 5. Inference Serving Architecture Optimization

### 5.1 CPU-GPU Process Separation

Bottleneck in traditional Python inference services (Flask/KServe): CPU preprocessing and GPU inference run in the same thread, resulting in low GPU utilization (QPS < 4).

**Root cause analysis**:
- Single-threaded mode: CPU and GPU wait serially, insufficient GPU kernel scheduling
- Multi-threaded mode: The Python GIL causes GPU kernel launch threads to be frequently interrupted

**Solution — CPU-GPU Process Separation**:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Proxy   │───→│CPU Process│───→│GPU Process│
English description
└──────────┘    └──────────┘    └──────────┘
                      ↑               ↑
 load
 passedsharedmemory
```**Effect**: QPS improved by more than 7x.

### 5.2 TensorRT Model Acceleration

TensorRT optimization techniques:

| Optimization Phase | Technique | Description |
|----------|------|------|
| Network Construction | Operator Fusion | Horizontal/vertical fusion to reduce kernel count |
| | Node Elimination | Remove unnecessary nodes (e.g., Identity) |
| | Multi-Precision Support | Automatic selection of FP32/FP16/INT8 |
| | Hardware-Specific Optimization | Generate optimal kernels for specific GPU architecture |
| Runtime | Serialized Loading | Directly load optimized engine file |
| | Memory Management | Automatic VRAM lifecycle management |

**FP16 Precision Loss Localization**:
1. Mark all operators as outputs, compare FP32 and FP16 output differences
2. Find the earliest operator that fails to meet precision requirements
3. Mark that operator (or its parent operator) as FP32, keep the rest as FP16
4. Iterate until overall precision meets requirements

### 5.3 Kernel Fusion

Kernel fusion is one of the most effective means of improving inference performance:

```
English description
K1: LayerNorm -> HBM K_fused: LayerNorm + Attention + Residual
K2: Attention -> HBM dataregister/sharedmemory
K3: Residual -> HBM ,
```

**Commonly used fused kernels**:

| Fusion Pattern | Library/Implementation | Benefit Source |
|----------|---------|---------|
| FlashAttention | flash-attn | Avoid writing QK^T matrix back to HBM |
| Fused MLP | apex / Triton | Gate+Up+Down fusion |
| Fused LayerNorm + Residual | apex | Reduce 2 HBM reads/writes |
| Fused Optimizer | apex / DeepSpeed | Parameter updates without write-back and re-read |

---

## 6. System-Level Optimization Checklist

### 6.1 Training Scenarios

- [ ] **Data Loading**: `num_workers` ≥ 4, `pin_memory=True`, `prefetch_factor` ≥ 2
- [ ] **Mixed Precision**: Enable AMP (bf16/fp16), use Tensor Core
- [ ] **Kernel Fusion**: Use FlashAttention, Fused Optimizer
- [ ] **Communication Overlap**: Overlap AllReduce with backward computation (FSDP/DeepSpeed ZeRO)
- [ ] **Compilation Optimization**: Evaluate `torch.compile` effectiveness
- [ ] **Profile**: Use nsys to check for GPU idle gaps

### 6.2 Inference Scenarios

- [ ] **Model Optimization**: TensorRT / ONNX Runtime acceleration
- [ ] **Batching**: Dynamic batching to improve GPU utilization
- [ ] **CUDA Graph**: Enable Graph for fixed-shape scenarios
- [ ] **CPU-GPU Separation**: Use independent processeslib for pre/post processing
- [ ] **Multi-Model Deployment**: Use MPS or multi-Stream concurrency
- [ ] **Quantization**: FP16 / INT8 / FP8 to reduce VRAM and computation

### 6.3 Performance Monitoring

- [ ] **GPU Util**: nvidia-smi monitors basic utilization (>90% as baseline requirement)
- [ ] **SM Active**: DCGM monitors SM spatial utilization
- [ ] **SM Occupancy**: nsys monitors warp utilization
- [ ] **Tensor Active**: nsys monitors Tensor Core activity
- [ ] **MFU**: Calculate actual compute utilization (the ultimate metric for training scenarios)

---

## Related Documents

- [NCU Performance Analysis Guide](ncu-profiling-guide.md) — Nsight Compute metrics analysis
- [Architecture Profiling Tools](../kernel-opt/profiling-tools-by-arch.md) — nsys/ncu support per architecture
- [Occupancy Tuning](../kernel-opt/occupancy-tuning-by-arch.md) — Occupancy calculation and tuning strategies
- [NVIDIA Compute Capabilities](../kernel-opt/nvidia-compute-capabilities.md) — Hardware resource limits per architecture
- [NVIDIA Architecture-Specific Optimization](../kernel-opt/nvidia-arch-specific-optimization.md) — Optimization features per architecture generation
- [L2 Cache Persistence](../kernel-opt/l2-cache-persistence.md) — L2 Persistence configuration
