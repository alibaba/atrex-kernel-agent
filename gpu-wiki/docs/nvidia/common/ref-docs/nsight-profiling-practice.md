# Nsight Profiling in Practice

> Compiled from multiple technical articles in the Zhihu community, covering practical topics including Nsight Systems system-level analysis, Nsight Compute kernel-level analysis, Warp Stall diagnosis, and Roofline analysis.

---

## 1. Tool Overview: Nsight Systems vs Nsight Compute

NVIDIA provides two layers of performance analysis tools, targeting performance bottlenecks at different granularities:

| Dimension | Nsight Systems (nsys) | Nsight Compute (ncu) |
|------|----------------------|---------------------|
| Analysis Granularity | System-level timeline | Single kernel-level |
| Primary Use Case | CPU-GPU interaction, kernel launch overhead, stream parallelism | Kernel internal bottlenecks (occupancy/memory/compute) |
| Overhead | Low (~2%) | High (replay mode, kernel replayed multiple times) |
| Workflow | Use nsys first to identify hotspot kernels | Then use ncu-Type for in-depth analysis of hotspot kernels |
| Command Line | `nsys profile` | `ncu --set detailed` |

**Recommended Workflow**: nsys global scan first → identify hotspots → then ncu for precise analysis.

---

## 2. Nsight Systems in Practice

### 2.1 Basic Commands and Parameters

```bash
# profiling
nsys profile -o output_report ./my_app

# recommendedparameter
nsys profile \
 --trace=cuda,nvtx,osrt \ # CUDA API, NVTX , OS row
 --duration=30 \ # 30 seconds
 --delay=5 \ # 5 seconds
 --cudabacktrace=true \ # CUDA
    --output=report \
    ./my_app

# PyTorch
nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    -o training_report \
    python train.py
```

Key parameter descriptions:
- `--trace`: Specifies the types of APIs to trace; `cuda` traces CUDA runtime/driver API, `nvtx` traces user markers
- `--delay` / `--duration`: Controls the collection time window to avoid capturing irrelevant initialization phases
- `--cudabacktrace=true`: Allows viewing the CPU call stack that initiates CUDA calls in the timeline

### 2.2 Two Operation Modes

**Profile Mode** (default):
- Directly launches the target program and collects data
- Suitable for short-duration tasks or services that can be restarted
- `nsys profile ./app`

**Launch Mode**:
- Starts the target program first, then attaches to collect data
- Suitable for long-running services or scenarios where restarting is difficult
- `nsys launch --trace=cuda ./app` → In another terminal, `nsys start` / `nsys stop`

### 2.3 NVTX Markers: Custom Analysis Intervals

NVTX (NVIDIA Tools Extension) allows inserting named markers in code, displayed as colored intervals in the timeline:

```python
# PyTorch use NVTX
import torch

with torch.cuda.nvtx.range("forward_pass"):
    output = model(input)

with torch.cuda.nvtx.range("backward_pass"):
    loss.backward()

with torch.cuda.nvtx.range("optimizer_step"):
    optimizer.step()
```

```cpp
// CUDA C++ use NVTX
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("matrix_multiply");
kernel<<<grid, block>>>(args);
nvtxRangePop();
```

Core value of NVTX: Correlates high-level semantics (e.g., "forward propagation", "gradient synchronization") with underlying CUDA operations, quickly identifying which logical phase is causing the bottleneck.

### 2.4 Timeline Analysis Key Points

In the Nsight Systems GUI, focus on the following patterns:

**1. Kernel Launch Gaps**:
- Blank areas between two kernels
- Possible causes: excessive CPU-side processing time, Python GIL lock, kernel launch overhead
- Solution: Use CUDA Graph for batched submission, reduce Python layer operations

**2. Memory Copy and Compute Not Overlapping**:
- HtoD (Host to Device) and DtoH are serialized with kernel execution
- Solution: Use multiple Streams + asynchronous copy to implement pipelining

**3. Synchronization Point Blocking**:
- `cudaDeviceSynchronize()` or `cudaStreamSynchronize()` causes CPU waiting
- Solution: Reduce unnecessary synchronization, use events (cudaEvent) for lightweight synchronization

**4. SM Warp Occupancy Panel**:
- Yellow sections indicate the ratio of active warps, gray indicates idle
- A low ratio indicates insufficient SM utilization
- Combine with the Tensor Active metric to determine Tensor Core utilization

---

## 3. Nsight Compute in Practice

### 3.1 Basic Commands

```bash
# analysis( kernel)
ncu --set detailed -o report ./my_app

# analysis kernel
ncu --set detailed --kernel-name "regex:gemm" -o report ./my_app

# recommendedcompilation(keep)
nvcc -O3 -lineinfo -arch=sm_90 kernel.cu -o kernel
```

`--set detailed` collects complete performance metrics, including all panels such as Roofline, Occupancy, Memory, Compute, and Scheduler.

### 3.2 Roofline Analysis

The Roofline model is the primary tool for determining kernel performance bottlenecks:

```
 ┌────── compute (Peak FLOPS)
FLOPS        │     /
(performance) │ / ← memory = peakbandwidth
             │   /
             │  /  ● Compute Bound Kernel
             │ /
             │/ ● Memory Bound Kernel
             └──────────────────────────
                Arithmetic Intensity
                (FLOP/Byte)
```**How to Interpret**:
- **Y-axis**: Actual achieved FLOPS
- **X-axis**: Arithmetic Intensity, i.e., the number of operations per byte of data
- **Left side of the diagonal**: Memory Bound — optimize by reducing memory access or improving cache hit rate
- **Right side of the diagonal**: Compute Bound — optimize by reducing instruction count or improving instruction throughput
- **Distance from the roofline**: Reflects the amount of room for optimization

**Real-world Case** (comparison of three kernels on an RTX 2080 Ti):
- kernel_A: High Intensity → Compute Bound, SM resource utilization near 100%
- kernel_B: Also Compute Bound, but FLOPS drops due to low Occupancy (6.25%)
- kernel_C: Low Intensity → Memory Bound, memory access efficiency is the bottleneck

### 3.3 Occupancy Analysis

Occupancy = Number of active warps / Theoretical maximum warps per SM

**Limiting Factor Troubleshooting** (line chart in the ncu Occupancy panel):

| Limiting Resource | Common Cause | Solution |
|----------|---------|---------|
| Shared Memory | Each block allocates a large amount of shared memory | Reduce shared memory usage, use dynamic allocation |
| Registers | Excessive register usage per thread | Reduce local variables, `--maxrregcount` limit registers |
| Block Size | Too few threads | Increase threads per block |
| Block Count | Number of blocks per SM hits hardware limit | Increase block size to reduce block count |

**Case Study**:
kernel_B has an Occupancy of only 6.25% (2 warps / 32 max warps). The reason is that each block is forcibly allocated 48KB of shared memory (the SM limit on Volta architecture), which means each SM can only accommodate 1 block, and each block has only 64 threads (2 warps). The GPU does not have enough warps to hide memory latency, and performance drops directly.

### 3.4 Memory Workload Analysis

**Key Metrics**:

| Metric | Meaning | Healthy Range |
|------|------|---------|
| L1 Sectors/Req | Number of sectors actually utilized per request | Ideally 32 (128B / 4B) |
| L2 Hit Rate | L2 cache hit rate | Higher is better |
| Global Load Efficiency | Global load efficiency | Close to 100% |
| Shared Memory Bank Conflict | Shared memory bank conflicts | 0 is ideal |

**Consequences of Uncoalesced Access** (real-world case):

```cpp
// kernel_C coalescedaccess
const int stride = 16;
int strided_idx = threadIdx.x * stride + ...;
A[idx] = B[strided_idx] + B[strided_idx];
```

Each thread's address for loading B is spaced 16×8=128 bytes apart, which exactly equals the L2 cache line size (128B). Result:
- L2 Sectors/Req = 1 (only 4/128 = 3.1% of the cache line bandwidth utilized per request)
- L2 Hit Rate appears high (because each access hits a different cache line)
- However, **effective bandwidth** is extremely low — this is the "most inefficient approach"

Fix: Reorganize the data layout so that adjacent threads access adjacent addresses (coalesced access).

---

## 4. Warp Stall Cause Analysis

When a warp is not in the Eligible state, it is in the Stall state. Nsight Compute provides detailed stall cause classifications.

### 4.1 Core Stall Reason Reference Table

| Stall Code | Cause | Solution |
|-----------|------|---------|
| `stalled_long_scoreboard` | Long-latency waits related to L1TEX (global memory loads not yet completed) | Reduce global memory accesses, improve cache hit rate, use shared memory |
| `stalled_lg_throttle` | Excessive local/global memory operations queued | Coalesce memory accesses, reduce register spilling |
| `stalled_math_pipe_throttle` | All warps simultaneously contending for the same math pipeline | Increase active warp count to hide latency, mix different instruction types |
| `stalled_short_scoreboard` | Waits related to MIO (shared memory, special function units, etc.) | Reduce shared memory accesses, use registers instead |
| `stalled_barrier` | Synchronization waits caused by `__syncthreads()` | Reduce unnecessary synchronization, use `__syncwarp()` instead |
| `stalled_branch_resolving` | Excessive conditional branches | Reduce branch divergence |
| `stalled_no_instructions` | Instruction cache miss or kernel too small | Increase workload, reduce branches |
| `stalled_not_selected` | Ready but not selected by the scheduler | Lower active warp count to improve cache locality |
| `stalled_mio_throttle` | MIO pipeline (special functions/shared memory/branches) congested | Balance instruction-to-data ratio |
| `stalled_tex_throttle` | Excessive texture/surface L1 accesses | Switch to global memory |
| `stalled_wait` | Fixed execution dependency latency | Loop unrolling, use fast math options |
| `stalled_imc_miss` | Constant cache miss | Access as few constant locations as possible within the same warp |
| `stalled_drain` | Large amount of data written back at kernel completion | Coalesce store operations |
| `stalled_sleeping` | Thread actively sleeping/yielding | Reduce unnecessary sleep calls |
| `stalled_membar` | Memory barrier waiting | Reduce unnecessary synchronization |

### 4.2 Stall Cause Summary by Category

**Memory Access Related**:
- Global memory not satisfying coalescing, addresses cannot be aligned
- Cache miss (local/constant/global memory)
- Shared memory bank conflict
- Register spilling leading to local memory usage**Compute Pipeline Related**:
- MIO pipeline (shared memory + MUFU + branches) congestion
- Math pipeline (CUDA Core / Tensor Core) saturation
- Instruction and data imbalance

**Inter-Warp Divergence**:
- Uneven workload distribution across warps within a block
- Branch divergence causing too few active threads within a warp

**Synchronization Operations**:
- Bucket brigade effect of `__syncthreads()`
- Suggestion: If threads in a warp do not depend on other warps, use `__syncwarp()` directly

### 4.3 Warp Issue Efficiency Analysis

The Warp State panel in Nsight Compute displays the warp issue efficiency distribution:

```
Eligible cycles low -> warp high
│
├── Selected (execute)
├── Eligible  -> stalled_not_selected
└── Stalled
 ├── Long Scoreboard (globalmemorywait)
 ├── Short Scoreboard (shared memorywait)
 ├── Barrier (synchronouswait)
 ├── Math Throttle (compute)
    └── ...
```

**Note**: A large proportion of eligible warps does not necessarily mean good performance; the computation time within each warp must still be considered holistically.

---

## 5. Practical Tips and Common Pitfalls

### 5.1 Impact of Compilation Options

```bash
# recommended profiling compilation
nvcc -O3 -lineinfo -arch=sm_90 kernel.cu

# English note
# -O3 : highoptimization(do not -O0 profiling, resultactualperformance)
# -lineinfo : keeprow, source-SASS (performance)
# -arch : ,
```

### 5.2 Common Performance Analysis Misconceptions

**Misconception 1**: GPU utilization at 100% = performance is already optimal

The GPU utilization shown by nvidia-smi only indicates "whether a kernel was executing during the sampling period." Even if only 1 SM is working, or even if it is only doing memory reads/writes without computation, 100% may still be displayed. MFU (Model FLOPS Utilization) is the metric that measures true computational efficiency. Real-world case: In a certain LLM training scenario, GPU utilization was 100% but MFU was only 20%. After optimization through kernel fusion, MFU increased to 38%, and training speed improved by 4x.

**Misconception 2**: Only looking at Occupancy

High occupancy does not always mean high performance. Excessively high occupancy can actually lead to:
- Reduced registers and cache available per warp
- Increased cache thrashing
- Sometimes lowering occupancy can actually improve cache hit rate

**Misconception 3**: Ignoring kernel launch overhead

On small-scale problems, kernel launch latency may be the primary bottleneck. On the B200, for a kernel operating on 10KB of data, combining CUDA Graph + PDL + early exit can achieve a 3x speedup.

### 5.3 Analysis Checklist

1. **nsys Global Check**:
   - [ ] Are there gaps between kernels?
   - [ ] Is memory copy overlapping with computation?
   - [ ] Are there unnecessary synchronization points?
   - [ ] Is SM Warp Occupancy uniform?

2. **ncu Hotspot Kernel Check**:
   - [ ] Roofline position: compute-bound or memory-bound?
   - [ ] What is the occupancy limiting factor?
   - [ ] Are memory accesses coalesced?
   - [ ] What is the primary warp stall reason?
   - [ ] What is the Tensor Core utilization (if applicable)?

---

## 6. eGPU: Next-Generation GPU Performance Analysis Framework

eGPU is an observability framework that offloads eBPF bytecode to the GPU through dynamic PTX injection, representing a new direction in GPU performance analysis:

**Core Features**:
- Dynamically add/modify/remove performance probes within a running GPU kernel without restarting the kernel
- Compile eBPF bytecode into PTX fragments and inject them via a "trampoline" mechanism
- CPU-GPU shared memory region enables zero-copy eBPF map access

**Comparison with Traditional Tools**:

| Feature | CUPTI/NVBit | eGPU |
|------|------------|------|
| Instrumentation Overhead | 5.2% | 1.8% |
| Dynamic Injection | Not supported (requires restart) | Supported (runtime injection) |
| Kernel Interrupt | Required | Not required |
| PTX Injection Latency | - | < 20μs |

**Practical Use Cases**:
- **MemTrace**: Real-time tracking of GPU memory access patterns, detecting bank conflicts and uncoalesced accesses (overhead < 3%)
- **Fair-Share Scheduling**: Dynamically adjusting GPU resource allocation in multi-tenant environments, achieving a Jain's fairness index of 0.96

---

## Related Documents

- [NCU Performance Analysis Guide](ncu-profiling-guide.md) — Detailed Nsight Compute metrics and analysis methodology
- [Profiling Tools by Architecture](../kernel-opt/profiling-tools-by-arch.md) — Profiling tool support for different GPU architectures
- [Occupancy Tuning](../kernel-opt/occupancy-tuning-by-arch.md) — Occupancy calculation and tuning strategies by architecture
- [NVIDIA Compute Capabilities](../kernel-opt/nvidia-compute-capabilities.md) — SM resource limits (registers/shared memory/warp count)
- [PTX Programming Model](ptx-programming-model.md) — Understanding PTX instructionsʰ to assist with SASS-level analysis
