# PyTorch Performance Profiling, Tuning, and Scaling

A comprehensive methodology for profiling, debugging, and system-level tuning of PyTorch workloads on modern NVIDIA GPUs, covering the full stack from Python interpreter overhead to multi-node distributed training.

---

## 1. Overview

AI training and inference pipelines can encounter performance bottlenecks at every layer — from Python interpreter overhead and CPU-side data loading stalls, to CUDA kernel underutilization and GPU memory contention. Effective optimization requires multi-tool profiling across multiple stack levels.

Core technical areas:
- NVTX markers and profiling toolchain for cross-tool correlation
- CPU-GPU joint profiling with Linux perf
- PyTorch compiler optimization via torch.compile
- CUDA stream concurrency and CUDA graphs
- Memory profiling and tuning (allocator, checkpointing, offloading)
- Data pipeline optimization
- Multi-GPU/multi-node scaling (DDP, FSDP)
- CI performance regression testing

---

## 2. NVTX Markers and Profiling Toolchain

### 2.1 NVTX Marker Mechanism

NVIDIA Tools Extension (NVTX) is a lightweight code annotation mechanism that inserts named time ranges into profiler timeline views. Its core value is **cross-tool performance correlation** — the same NVTX range (e.g., "forward") appears simultaneously in PyTorch Profiler traces and Nsight Systems timelines.

Injection methods:
- **PyTorch Python API**: `torch.profiler.record_function("name")` for high-level marks; `torch.cuda.nvtx.range_push/pop()` for low-level ranges
- **CUDA C++ API**: `nvtxRangePush()/nvtxRangePop()`
- **Third-party libraries**: Triton, PyCUDA, CuPy, CUTLASS all support NVTX injection

Performance overhead is near-zero when no profiler is attached. When a profiler (e.g., Nsight Systems) attaches, CUPTI captures NVTX push/pop events and projects asynchronous GPU kernel execution times onto these CPU-defined intervals.

### 2.2 Profiling Tool Landscape

| Tool | Scope | Core Function | Typical Use |
|------|-------|---------------|-------------|
| PyTorch Profiler (Kineto) | Operator-level (CPU/GPU) | NVTX, shape recording, memory stats, trace export | Identify slow operators, kernel launch overhead |
| Nsight Systems (nsys) | System-level timeline | Unified CPU thread + GPU stream timeline, multi-process | End-to-end pipeline; detect data loader stalls |
| Nsight Compute (ncu) | Per-kernel GPU | Hardware metrics, source correlation, roofline | Deep kernel analysis; compute vs memory bound |
| PyTorch Memory Profiler | Per-operation GPU memory | Memory snapshot timeline, peak memory per op | Diagnose fragmentation, memory anomalies |
| Linux perf | CPU profiling | CPU cycles/instructions/cache sampling, flame graphs | Python overhead, GIL contention, data loading |
| HTA (Holistic Trace Analysis) | Distributed training | Multi-worker trace aggregation, Perfetto backend | Multi-GPU/node balance, communication overlap |

### 2.3 Tool Selection Strategy

Follow a top-down, layer-by-layer approach:
1. **PyTorch Profiler** → operator-level overview, identify slowest operations
2. **Nsight Systems** → system timeline, check CPU-GPU overlap, data loading stalls
3. **Nsight Compute** → roofline analysis of hot kernels, compute/memory characterization
4. **Linux perf** → CPU-side hotspots, Python interpreter overhead
5. **HTA** → multi-GPU/node load balancing and communication efficiency

---

## 3. PyTorch Profiling in Practice

### 3.1 Operator-Level Profiling

```python
from torch import profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    with profiler.record_function("train_step"):
        torch.cuda.nvtx.range_push("forward")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        optimizer.zero_grad()
        torch.cuda.nvtx.range_pop()
```

Key parameters:
- `record_shapes=True`: records tensor shapes for understanding memory/compute characteristics
- `profile_memory=True`: tracks per-operation GPU memory usage
- `with_flops=True`: captures FLOPs counters for measuring compute intensity

Always warm up (5-10 iterations) before profiling to avoid capturing one-time initialization overhead.

### 3.2 System-Level Profiling with Nsight Systems

```bash
nsys profile --output=profile --stats=true -t cuda,nvtx python train.py

# Generate NVTX GPU projection summary
nsys stats --report=nvtx_gpu_proj_sum profile.nsys-rep
```

Example NVTX GPU projection results:

| NVTX Range | GPU Time (ms) | Self GPU Time (ms) | Instances |
|------------|--------------|-------------------|-----------|
| train_step | 138.0 | 0.0 | 1 |
| forward | 60.5 | 60.5 | 130 |
| backward | 58.3 | 58.3 | 130 |
| optimizer_step | 19.2 | 19.2 | 1 |

### 3.3 Kernel Roofline Analysis

```bash
ncu \
  --target-processes all \
  --kernel-name-regex "matmul" \
  --metrics \
    gpu__time_duration.avg, \
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed, \
    lts__throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum, \
    sm__warps_active.avg.pct_of_peak_sustained_active \
  --csv full \
  -o matmul_roofline_report \
  python train.py
```

| Kernel State | Peak FLOPS % | Peak Memory BW % | SM Occupancy | Characteristic |
|-------------|-------------|-----------------|-------------|----------------|
| Baseline | 50% | 70% | 60% | Memory-bound |
| Optimized | 85% | 40% | 80% | Compute-bound (near roofline) |

Note: There is no universal occupancy target. Many high-performance kernels achieve full latency hiding at 25-50% occupancy.

---

## 4. CPU-GPU Joint Profiling

### 4.1 CPU-Side Analysis with Linux perf

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses python train.py

perf record -F 2000 -g --call-graph dwarf -o perf.data python train.py
perf report --stdio -n -g -i perf.data
```

Typical CPU hotspot breakdown:

| CPU Hotspot | Share | Root Cause | Optimization |
|-------------|-------|-----------|-------------|
| Python interpreter forward | 45.0% | Python dispatch overhead | torch.compile |
| matmul | 20.5% | Heavy GEMM computation | Compiler optimization or custom CUDA kernels |
| dataloader_iter_next | 10.2% | DataLoader iterator bottleneck | Increase num_workers, persistent_workers=True |
| ncclAllReduce | 8.7% | NCCL communication | Increase bucket_cap_mb, gradient compression |
| read (I/O) | 5.3% | Host I/O system calls | pin_memory=True, batched file reads, optimized formats |

### 4.2 NVIDIA PMU Monitoring

```bash
perf list | grep -i nvidia
perf stat -a \
  -e nvidia_nvlink_c2c0_pmu_0/cycles/ \
  -e cycles,cache-misses \
  python train.py
```

Linux perf NVIDIA PMU support is limited to device-level link/fabric events (e.g., NVLink-C2C). SM pipeline, warp stall, and memory throughput counters still require CUPTI and Nsight tools.

---

## 5. PyTorch Compiler (torch.compile)

### 5.1 Architecture

Three core components:
- **TorchDynamo**: Captures Python bytecode at runtime, traces forward graph, detects graph breaks
- **AOT Autograd**: Generates backward graph at compile time, enables cross-forward/backward fusion
- **TorchInductor**: Backend code generator using Triton for GPU, with autotuning

```python
compiled_model = torch.compile(model)                                    # default
compiled_model = torch.compile(model, mode="reduce-overhead")            # minimize overhead
compiled_model = torch.compile(model, mode="max-autotune")               # maximum performance
compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs") # no CUDA graphs
```

### 5.2 Performance Impact

On MoE model: Eager ~248 ms/iter → Compiled (max-autotune) ~173 ms/iter (**~30% speedup**).

Acceleration sources:
- **Operator fusion**: Multiple small ops merged into fewer larger kernels
- **Python overhead elimination**: No interpreter dispatch after compilation
- **Reduced kernel launches**: Fewer fused kernels per iteration
- **On-chip data retention**: Intermediate data stays in registers/SMEM

Benefit varies by model type:
- Sparse (MoE): hundreds of medium matmuls → significant benefit (~30%+)
- Dense: dominated by large GEMM → limited benefit (<10%)

### 5.3 Compilation Modes

| Mode | Description | Compile Time | Notable Features |
|------|-------------|-------------|-----------------|
| default | Balanced (good speed, moderate compile time) | Low-Med | General fusion, basic autotune |
| reduce-overhead | Minimize per-iteration overhead (small batch ideal) | Med | Uses CUDA graphs |
| max-autotune | Maximize runtime (long-running tasks) | High | Aggressive Triton autotune + CUDA graphs |
| max-autotune-no-cudagraphs | Same but no graph capture | High | Maintains flexibility for dynamic shapes |

### 5.4 Optimized Attention Mechanisms

| Technique | Description | Use Case |
|-----------|-------------|----------|
| SDPA | `torch.nn.functional.scaled_dot_product_attention`, auto-selects fastest backend | Standard attention |
| FlexAttention | Compiler-based, generates optimized kernels for custom sparse patterns | Block-sparse, sliding window |
| FlexDecoding | Decode-phase counterpart to FlexAttention | LLM inference generation |
| Context Parallel | Shards attention along sequence length across devices | Ultra-long context |

### 5.5 torchao: Quantization, Sparsity, and Pruning

PyTorch Architecture Optimization (torchao) unifies:
- **Quantization**: PTQ, QAT covering INT8, FP8 formats
- **Pruning**: Model pruning techniques
- **Sparsity**: 2:4 sparse and block-sparse patterns

All integrate with torch.compile() for hardware-aware kernel generation.

---

## 6. CUDA Stream Concurrency

### 6.1 Fundamentals

By default, PyTorch schedules all operations on the default stream sequentially. Multiple streams enable concurrency when:
- Operations have no data dependencies
- GPU has sufficient resources (SMs, memory bandwidth, DMA engines)
- Operations access different memory regions

### 6.2 Compute-Communication Overlap

```python
transfer_stream = torch.cuda.Stream(device=device)
compute_stream = torch.cuda.default_stream(device=device)

# Pre-load first batch
with torch.cuda.stream(transfer_stream):
    next_inputs = first_batch[0].to(device, non_blocking=True)

for _ in range(len(dataloader)):
    compute_stream.wait_stream(transfer_stream)
    inputs = next_inputs

    # Start next transfer
    batch = next(dataloader_iter, None)
    if batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs = batch[0].to(device, non_blocking=True)

    # Compute on current batch
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
```

Key techniques:
- `wait_stream()`: Lightweight synchronization (vs. full device barrier)
- `.to(device, non_blocking=True)`: Async DMA copy without blocking CPU
- Always have one batch in transfer while another is computing

### 6.3 CUDA Events for Fine-Grained Synchronization

```python
event = torch.cuda.Event(enable_timing=False)
# In stream1:
event.record()
# In stream2:
stream2.wait_event(event)  # Wait only for this specific point
```

Events provide more precise dependency control than `wait_stream()` (which waits for all prior operations).

---

## 7. CUDA Graphs

### 7.1 Principle

CUDA graphs capture a sequence of GPU operations as a whole, eliminating per-kernel launch overhead on replay. Workflow:
1. Warm up (initialize data, allocations)
2. Capture operations on a dedicated stream
3. Replay the captured graph with new input data

```python
g = torch.cuda.CUDAGraph()
capture_stream = torch.cuda.Stream()
static_input = torch.randn(batch_shape, device='cuda')
static_output = torch.empty(output_shape, device='cuda')

# Warmup
with torch.cuda.stream(capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
capture_stream.synchronize()

# Capture
with torch.cuda.graph(g, stream=capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
capture_stream.synchronize()

# Replay
static_input.copy_(new_batch)
g.replay()
result = static_output.clone()
```

### 7.2 Best Practices

| Practice | Details |
|----------|---------|
| Pre-allocate all memory | All tensors needed during capture must be allocated beforehand |
| Keep graph structure fixed | Cannot change operations, shapes, or memory sizes after capture |
| Capture as much as possible | Ideally capture entire iteration (forward + backward + optimizer + all-reduce) |
| Plan memory reuse | Cannot free/reallocate graph tensors post-capture |
| Memory pool sharing | Use `torch.cuda.graph(pool=...)` to share pools across graph instances |

---

## 8. Memory Profiling and Tuning

### 8.1 CUDA Memory Allocator Tuning

PyTorch's caching allocator manages CUDA memory. Variable-size allocations (common in MoE) cause fragmentation.

```bash
export PYTORCH_ALLOC_CONF=\
  max_split_size_mb:256,\
  roundup_power2_divisions:[256:1,512:2,1024:4,>:8],\
  backend:cudaMallocAsync
```

| Parameter | Effect |
|-----------|--------|
| max_split_size_mb:256 | Keeps large free blocks intact (up to 256MB) |
| roundup_power2_divisions | Groups allocation sizes into fixed buckets for reuse |
| backend:cudaMallocAsync | Async allocator avoids synchronization on free events |

### 8.2 Activation Checkpointing

Trade compute for memory: don't store intermediate activations during forward; recompute them during backward.

```python
import torch.utils.checkpoint as checkpoint
# Wrap Transformer blocks — ~33% additional forward compute, significant memory savings
```

### 8.3 Parameter Offloading

For infrequently accessed components (e.g., less-used MoE experts), offload to CPU memory:
- **DeepSpeed ZeRO-Infinity**: Automated prefetch, layer-by-layer streaming from CPU/NVMe
- **Async transfer**: Pinned memory + non-blocking DMA overlapped with compute
- **Unified Memory**: For Grace Blackwell GB200/GB300 with NVLink-C2C high-speed interconnect
- **GPUDirect Storage**: GPU reads directly from NVMe without CPU involvement

### 8.4 FSDP with Checkpointing and Offloading

```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,       # ZeRO Stage-3
    cpu_offload=CPUOffload(offload_params=True, pin_memory=True),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.bfloat16),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    activation_checkpointing_policy={
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    }
)
```

FSDP sharding strategies:

| Strategy | ZeRO Stage | Shards | Memory Efficiency | Communication |
|----------|-----------|--------|-------------------|---------------|
| FULL_SHARD | 3 | Params + Gradients + Optimizer | Highest | Highest |
| HYBRID_SHARD | 3 (hybrid) | Intra-node shard, inter-node replicate | Medium | Medium |
| SHARD_GRAD_OP | 2 | Gradients + Optimizer only | Lower | Lower |

---

## 9. Data Pipeline Optimization

### 9.1 DataLoader Tuning

| Parameter | Recommendation |
|-----------|---------------|
| num_workers | Start at 4/GPU, sweep (4,8,16,32) |
| pin_memory=True | Standard systems: always enable |
| persistent_workers=True | Avoid per-epoch process restart overhead |
| prefetch_factor | Adjust with num_workers; avoid over-fetching |
| non_blocking=True | Non-blocking GPU copies with pin_memory |

### 9.2 Data Format Optimization

- Pre-compute tokenized datasets (cache once, reuse)
- Use optimized formats: Arrow, WebDataset (tar shards), TFRecord, Parquet
- Pack small files into optimized dataset formats
- Apply mixed precision and compression to reduce I/O bandwidth

### 9.3 NVIDIA DALI

Data Loading Library for parallel CPU/GPU preprocessing, particularly for image/video workloads (decode + augment). Feeds data directly to training code via CUDA pipeline.

---

## 10. Multi-GPU Distributed Scaling

### 10.1 DDP with torch.compile

DistributedDataParallel synchronizes gradients via all-reduce. When combined with torch.compile:
- PyTorch automatically creates graph breaks at synchronization points
- DDP buckets gradients, overlapping communication with computation
- Each bucket's backward is compiled as a separate graph

Bucket tuning: Default 25MB. Increasing to 50MB reduces per-message overhead (if network bandwidth allows). A single huge bucket loses communication-computation overlap.

### 10.2 FSDP with torch.compile

Best practice — block-level wrapping:
- Each block's parameters materialize only when needed (peak memory ∝ block size / total params)
- Next block's weights prefetch asynchronously during current block computation
- TorchDynamo inserts graph breaks at shard boundaries, preserving overlap

### 10.3 Tensor Parallel and Pipeline Parallel

Orthogonal to torch.compile:
- TorchInductor optimizes intra-segment computation (cublasLt/cuDNN/Triton kernels)
- NCCL collectives handled by the distributed strategy
- Compiler does not fuse or reschedule communication operations

### 10.4 Advanced Distributed Systems

| Project | Description |
|---------|-------------|
| TorchTitan | Reference implementation for large-scale training (FSDP+TP+AsyncTP) |
| AsyncTP | Asynchronous tensor parallel: dual-stream + SM-wave-aware scheduling |
| AutoParallel | Automatic parallel strategy planning (FSDP+TP+PP combination) |
| SimpleFSDP | torch.compile-friendly FSDP reimplementation; -28% memory, +69% throughput |

---

## 11. Multi-GPU Profiling with HTA

Meta's Holistic Trace Analysis (HTA):
- Merges multi-worker traces into unified timeline
- Aligns NVTX markers across ranks to reveal execution differences
- Reports GPU idle time, identifies stragglers and sync issues
- Provides optimization recommendations

Typical analysis: discover one rank enters backward late (waiting for all-reduce), or identify load imbalance across ranks.

---

## 12. CI Performance Regression Testing

### 12.1 Performance Regression CI

```yaml
- name: Run MoE benchmark
  run: torchbench run --model moe --iters 10 --batch-size 4 --json results.json
- name: Compare throughput
  run: python scripts/compare_perf.py baseline.json results.json
```

Best practices:
- Consistent hardware for CI runners
- Regression threshold ≥5% sustained over 3 runs
- Record memory usage and data loading time alongside throughput
- Use `torch.allclose` with strict tolerance for correctness tests
- Capture `torch.cuda.max_memory_allocated()` for memory monitoring

### 12.2 Training Iteration Time Breakdown

| Component | Time (ms) | Percentage |
|-----------|----------|-----------|
| Forward | 10.5 | 43.8% |
| Backward | 9.0 | 37.5% |
| All-reduce (gradient sync) | 4.0 | 16.7% |
| Other overhead | 0.5 | 2.1% |
| **Total step time** | **24.0** | **100%** |

Compute (forward + backward) = 81.3%; communication = 16.7%. The ~1/6 spent on gradient sync is the target for further optimization (async all-reduce, pipeline parallel, activation compression).

---

## 13. Key Principles Summary

### 13.1 Profiling Principles

- Maintain a **profiling-first** approach: bottlenecks hide at any layer
- Use **multi-tool holistic analysis** to capture data at every level
- Prefer **compiled mode** over eager mode (one line: `torch.compile`)
- Use the highest optimization mode the workload permits
- Save compilation artifacts for reuse across runs

### 13.2 Synchronization and Precision

- Avoid sync traps: `tensor.item()` synchronizes; use `non_blocking=True`
- Never use `time.time()` for GPU timing (implicit sync); use `torch.cuda.Event(enable_timing=True)`
- Use `torch.autocast` for Tensor Core utilization; prefer BF16 over FP16

### 13.3 Memory and Compute

- Fuse small operations (<1ms) using torch.compile or custom kernels
- Pre-allocate large tensors; use fixed shapes; tune CUDA allocator
- Apply activation checkpointing for models >10B parameters
- Offload to CPU/NVMe with async transfers overlapping compute

### 13.4 Data and Communication

- Sufficient DataLoader workers + prefetch_factor
- pin_memory=True + non_blocking transfers
- Tune DDP bucket_cap_mb; consider gradient compression
- Place frequently communicating processes on same node/switch

### 13.5 Optimization Priority

| Priority | Technique | Expected Benefit | Difficulty |
|----------|-----------|-----------------|-----------|
| 1 | torch.compile (default) | 10-30% | Very Low |
| 2 | Mixed precision (BF16/TF32) | 2-4x | Low |
| 3 | DataLoader tuning | Significant (if data-bound) | Low |
| 4 | Compile mode upgrade (max-autotune) | Additional 5-15% | Low |
| 5 | CUDA stream concurrency | 10-30% | Medium |
| 6 | Memory allocator tuning | Reduce fragmentation/OOM | Medium |
| 7 | Activation checkpointing | 30-60% memory savings | Medium |
| 8 | CUDA graphs | Eliminate launch overhead | Medium-High |
| 9 | Custom fused kernels | 5-15% | High |
| 10 | FSDP + CPU offload | Train larger models | Medium |
