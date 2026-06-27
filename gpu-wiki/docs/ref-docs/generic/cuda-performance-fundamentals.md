# Community CUDA Performance Optimization Basics

A compilation of practical knowledge on CUDA programming fundamentals and performance optimization from the Zhihu community. Supplements real-world experience and common pitfalls not covered in [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) and [GPU Execution Model and Thread Optimization](gpu-execution-model.md).

> **Source Note**: This article synthesizes core knowledge from approximately 49 related articles on Zhihu, after deduplication, filtering, and structured organization.

---

## 1. Memory Access Optimization

### 1.1 Coalesced Access

Coalesced access is the most critical aspect of CUDA performance tuning. When a warp (32 threads) simultaneously accesses contiguous, aligned global memory addresses, the hardware can combine these requests into very few memory transactions.

**Key Rules**:
- **Row-major access**: Two-dimensional arrays in C/CUDA are stored in row-major order, with `threadIdx.x` being the fastest-changing dimension (equivalent to the "column" index), so using `threadIdx.x` to index the innermost dimension achieves coalesced access.
- **Alignment requirements**: The starting address of a memory transaction should be an even multiple of the cache granularity (32B or 128B). Memory allocated via `cudaMalloc` is automatically aligned to 256B or 512B.
- **L1 cache affects granularity**: Granularity is 128B when L1 cache is enabled, and 32B when disabled (L2 only).
```c
// Coalesced access (high performance): threadIdx.x indexes innermost dimension
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];

// Non-coalesced access (low performance): threadIdx.x indexes outer dimension
int idx = blockIdx.x * threadIdx.x + blockDim.x;
float val = data[idx];
```
**Measured Data**: With L1 disabled, misaligned access (offset=1) is about 35% slower than aligned access (15.0ms vs 11.1ms, 16M-element vector addition).

**Pitfall**: Coalesced access is a warp-level behavior, not a single-thread behavior. Individual threads can access non-contiguous addresses, as long as the addresses accessed by all 32 threads in a warp on the same memory instruction are contiguous.

### 1.2 Vectorized Load/Store

Using vector types like `float2`/`float4`/`int4`, a single instruction can load 64-bit or 128-bit data, corresponding to PTX instructions `LDG.E.64`/`LDG.E.128`.

**Benefits**:
- Reduces instruction count (float4 reduces loop iterations by a factor of 4)
- Lowers instruction scheduling overhead and latency
- Improves bandwidth utilization

```c
__global__ void vectorized_copy(float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < n) {
        // One LDG.E.128 instruction loads 4 floats
        float4 data = reinterpret_cast<float4*>(&in[idx])[0];
        data.x *= 2.0f; data.y *= 2.0f;
        data.z *= 2.0f; data.w *= 2.0f;
        reinterpret_cast<float4*>(&out[idx])[0] = data;
    }
}
```

**Alignment Constraints**:
- `float2` requires 8-byte alignment, `float4` requires 16-byte alignment
- `reinterpret_cast<int2*>(d_in + 1)` is invalid (misaligned), `reinterpret_cast<int2*>(d_in + 2)` is valid
- Struct size must be a power of two bytes (1/2/4/8/16B), otherwise the compiler may add padding

**Trade-off**: Vectorized loads increase register pressure. If a kernel is already register-limited or has very low parallelism, vectorization may not be suitable.

### 1.3 Data Layout: AoS vs SoA

GPUs strongly prefer the SoA (Structure of Arrays) layout:

```c
// AoS (bad): Non-contiguous access within warp
struct Point3D { float x, y, z; };
Point3D points[N];
float xCoord = points[i].x;  // Stride of 12B, cannot coalesce

// SoA (good): Contiguous access within warp
struct Points {
    float xCoords[N], yCoords[N], zCoords[N];
};
float xCoord = pts.xCoords[i];  // Continuous access, perfect coalescing
```

For scenarios requiring simultaneous access to multiple fields of a struct, consider ASTA (Array of Structures of Tiled Arrays) as a compromise.

---

## 2. Shared Memory Usage Tips

### 2.1 Bank Conflict

Shared Memory is divided into **32 banks**, each 4 bytes wide. When multiple threads within the same warp access different addresses in the same bank, the accesses are serialized.

**Key Insights**:
- Bank conflicts only occur between threads within the same warp; different warps do not conflict.
- Broadcast mechanism: All threads accessing the same address is not a conflict; the hardware broadcasts automatically.
- Effective bandwidth = peak bandwidth / conflict degree (32-way conflict is worst, bandwidth drops to 1/32).

**Common Conflict Scenarios and Solutions**:

| Scenario | Conflict Cause | Solution |
|------|---------|------|
| Column-major matrix access | Stride equals row width (worst when it's a multiple of 32) | Padding: `__shared__ float tile[32][33]` |
| Strided access with stride=2 | 2-way conflict | Swizzle: XOR-based address mapping |
| Matrix transpose | Column-major on write | Padding or Swizzle |

**Pitfall**: If Padding or Swizzle is used to resolve bank conflicts, it may break pointer alignment, making vectorized loads/stores (float4) impossible. A specific swizzle layout must be designed arbeited for vectorization.

### 2.2 Tiling

Tiling is the most classic application of shared memory. Taking matrix multiplication `[H,K] * [K,W]` as an example:
- Naive implementation: Each output element reads global memory `2K` times, total I/O is `H * W * 2K`
- Tiling implementation: Total I/O reduced to `H * W * 2K / TILE_SIZE`, a reduction factor of `1/TILE_SIZE`**The Necessity of Double `__syncthreads()`**:
```c
for (int tile = 0; tile < numTiles; tile++) {
    // Load tile to shared memory
    M_tile[ir][ic] = M[...];
    N_tile[ir][ic] = N[...];
    __syncthreads();   // Ensure all threads have loaded before computing

    for (int k = 0; k < TILE_SIZE; k++)
        res += M_tile[ir][k] * N_tile[k][ic];
    __syncthreads();   // Ensure all threads have finished computing before loading next tile
    // If missing second sync, fast threads may overwrite tile data prematurely
}
```

### 2.3 Async Copy

Starting with Ampere (SM 80+), the `cp.async` instruction supports moving data directly from Global Memory to Shared Memory, bypassing registers:
- Does not consume registers, which can improve occupancy
- Computation and data movement are truly parallel at the hardware level
- Manual loop unrolling is not required to find the optimal movement pattern

Hopper (SM 90+) introduces TMA (Tensor Memory Accelerator), a DMA engine independent of SMs that supports multi-dimensional tensor layouts and automatic bounds checking, achieving far greater efficiency than `cp.async`.

---

## 3. Occupancy and Resource Balance

### 3.1 The Essence of Occupancy

Occupancy = Number of active warps per SM / Maximum supported warps per SM. Its fundamental purpose is to **hide latency**, not to pursue 100%.

**Key Insights**:
- The performance gain from increasing from 60% to 100% is often very limited
- Lower occupancy can still sufficiently hide latency when ILP (Instruction-Level Parallelism) is adequate
- Reducing occupancy can bring benefits: more registers per thread, fewer register spills, and reduced resource contention

**Hardware Limits Using A100 as an Example**:

| Resource | Limit |
|------|------|
| Max threads per SM | 2048 |
| Max blocks per SM | 32 |
| Registers per SM | 65536 32-bit |
| Shared memory per SM | Max 164 KB |

### 3.2 Register Pressure

Registers are allocated per thread block, and the allocation granularity is rounded up to steps of **256 registers per warp**. Using just 1 extra register per thread can cross the boundary to the next multiple of 256, amplifying the actual consumption.

**Calculation Example** (Compute Capability 7.0):
- 37 registers per thread + 128 threads/block = 12 blocks can reside, occupancy 75%
- 37 registers per thread + 320 threads/block = 4 blocks can reside, occupancy 63%

**Register Spilling**: Variables that exceed the limit are spilled to local memory (which physically resides in global memory), increasing access latency by hundreds of times.

CUDA 13.0 New Feature: Supports redirecting spill data to shared memory (`enable_smem_spilling` pragma), achieving a 5-10% performance improvement in QUDA library tests.

### 3.3 Block Size Selection Principles

- Block size should be a **multiple of warp size (32)**
- Block size should be a **factor of the SM's maximum thread count** (e.g., a factor of 1536 such as 512 is better than 1024)
- **128-256 threads per block** is a good starting point for general scenarios
- Kernels that frequently call `__syncthreads()` should use smaller blocks
- The `cudaOccupancyMaxPotentialBlockSize` API can provide suggested values (but this is only a starting point, not the destination)

### 3.4 Conflicts Between Optimization Strategies

Various optimization strategies often constrain each other:
- Maximizing occupancy vs. controlling cache thrashing: too many threads compete for cache resources
- Increasing shared memory vs. occupancy: SM shared memory is finite
- Thread coarsening vs. occupancy: coarsening increases resource consumption per thread

**Core Principle**: Identify the current bottleneck first, then choose the corresponding optimization direction. If the bottleneck is control flow divergence, pursuing memory tiling optimization is targeting the wrong problem.

---

## 4. Warp Execution and Branch Optimization

### 4.1 Warp Divergence

When 32 threads within the same Warp execute different branch paths, the GPU must serialize execution of each branch (execution cost = sum of all branch costs).

**How to Determine**: Do not just check whether the code contains `if`; instead, check whether **adjacent 32 threads execute the same branch**. For example:
```c
if (tid < N)  // When N is much larger than 32, warps in front all take the same branch, no divergence
    c[tid] = a[tid] + b[tid];
```

**Profile Tools**:
```bash
nvcc --ptx demo.cu  # Check if branch instructions are generated
```

**Optimization Techniques**:

1. **Use predication instead of branching**:
```c
// Has divergence
if (idx % 2 == 0) output[idx] += 3;
else output[idx] = idx;

// No divergence (compiler uses predicate instructions)
int val = idx % 2 == 0 ? output[idx] + 3 : idx;
output[idx] = val;
```

2. **Reduce Optimization**: Replace `tid % (2*s) == 0` with `tid < s`, so that the first N warps execute the same branch, reducing the number of divergent warps.

3. **Data Reordering**: Group threads that execute the same branch into the same warp.

### 4.2 Warp Shuffle

Threads within a Warp can directly exchange register data through the `__shfl_sync` series of primitives, bypassing shared memory with lower latency. Typically used for warp-level reductions.

### 4.3 Warp Stall Causes

| Stall Type | Cause | Countermeasure |
|-----------|------|------|
| Memory Stall | Waiting for Global Memory (200-400 cycles) | Increase occupancy, prefetch |
| Sync Stall | Waiting for `__syncthreads()` | Reduce sync points, thread coarsening |
| Execution Dependency | Waiting for previous instruction result | ILP, loop unrolling |
| Instruction Overhead | Address arithmetic, loop control | Loop unrolling, template specialization |## 5. Kernel Launch Overhead and Optimization

### 5.1 Launch Overhead

For each kernel launch, the CPU must compute execution parameters, maintain pointer addresses, validate Grid/Block dimensions, and push instructions into the GPU hardware queue. Typical overhead is **3-10 us**.

**Disaster scenario**: If kernel execution time is only 1 us, then 90% of the time is spent on "calling the GPU to notify it to work" rather than the actual work itself. In inference scenarios, lightweight models (such as YOLO26n) still have hundreds of kernels even after operator fusion, and at 30-60 FPS, launch overhead scales exponentially.

### 5.2 CUDA Graph

CUDA Graph takes a "snapshot" of the kernel topology and parameters of the entire execution flow into an executable graph, after which only a single `cudaGraphLaunch` call is needed:

```c
// Capture phase
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execution phase (repeated calls, very low overhead)
cudaGraphLaunch(graphExec, stream);
```

**Measured results**: On Jetson Orin Nano, CUDA Graph reduces CPU utilization by **over 80%**.

**Limitations**:
- Unfriendly to dynamic shapes (the graph structure must be fixed)
- Multi-threaded capture conflicts (requires a global mutex)
- Pre-allocated GPU memory overhead

### 5.3 Kernel Fusion

Merging multiple small kernels into one large kernel reduces:
- Launch count
- Global memory reads and writes of intermediate results
- Driver scheduling overhead

Typical fusions: Conv + BN + ReLU, merging multiple element-wise operations, softmax + matmul within Attention.

### 5.4 Persistent Kernel

Keeps the kernel running continuously on the GPU, polling for new tasks to avoid repeated launches. Suitable for latency-sensitive inference scenarios.

---

## 6. Common Performance Pitfalls and Debugging Methods

### 6.1 Implicit Type Promotion Trap

**Case**: The `10e-3` in `while (tmp > 10e-3)` is a `double` literal. The compiler inserts `cvt.f64.f32` type promotion instructions, resulting in a **50x performance degradation** (7ms -> 128us).

```c
// Slow (implicit double promotion)
while (tmp > 10e-3) { ... }    // 10e-3 is double

// Fast (explicit float)
float theta = 10e-3f;          // Note the f suffix
while (tmp > theta) { ... }
```

**Lesson**: Always use the `f` suffix to mark float literals (`1.0f`, `0.001f`) to avoid costly float-to-double conversions on the GPU.

### 6.2 Overuse of `cudaDeviceSynchronize`

When `cudaDeviceSynchronize` accounts for more than 50% in Nsight Systems, it usually indicates a serious synchronization bottleneck. Common causes:
- Residual debugging `torch.cuda.synchronize()` in the code
- Implicit synchronization calls such as `.cpu()`, `.numpy()`, `.item()`
- Frequent D2H data transfers

### 6.3 High CPU Usage Caused by Spin-Wait

The CUDA driver uses a spin-wait strategy by default, where CPU threads continuously poll while waiting for the GPU. On weak CPU platforms (such as Jetson), this can severely increase CPU usage. You can switch to blocking wait via `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)`, but the effect varies by scenario.

### 6.4 Identifying Register Spilling

Add `-Xptxas -v` during compilation to view register usage and spilling status:
```
ptxas info: 176 bytes stack frame, 176 bytes spill stores, 176 bytes spill loads
ptxas info: Used 255 registers
```

If there are spill stores/loads, spilling has occurred. Solutions:
- Reduce the number of per-thread variables
- Use `--maxrregcount` to limit register usage
- Reduce the number of threads per block to free up registers
- CUDA 13.0+: Use `enable_smem_spilling` to redirect spills to shared memory

### 6.5 Hidden Cost of Integer Division

Integer division and modulus on the GPU typically require **10+ instructions**. However, if the divisor is a compile-time constant, it only needs 2-3 instructions. You can "hardcode" commonly used divisors via template parameters, or use switch-case statements differentiable paths for predetermined constants.

### 6.6 Profiling Methodology

**Tool Selection**:
- **Nsight Systems (nsys)**: System-level performance analysis, viewing the global timeline, CPU-GPU interactions, and kernel overlap
- **Nsight Compute (ncu)**: Kernel-level analysis, examining occupancy, memory bandwidth, and warp stall causes

**Common nsys Commands**:
```bash
nsys profile \
  --trace cuda,osrt,nvtx,cudnn,cublas \
  --gpu-metrics-device=all \
  --duration=60 --delay=120 \
  --cuda-memory-usage true \
  --output profile_log \
  torchrun train.py
```

**Analysis Workflow**: Analysis Summary -> Timeline View -> System Status View -> Expert System View -> Event View, drilling down from macro to micro.

**NVTX Markers**:
```python
import nvtx
@nvtx.annotate("data_loading", color="blue")
def load_data(): ...

with nvtx.annotate("backward_pass", color="red"):
    loss.backward()
```

---

## 7. Key Numbers Every CUDA Developer Must Know

### 7.1 Memory Hierarchy Latency

| Storage Type | Latency (GPU cycles) | Physical Location | Notes |
|-------------|---------------------|------------------|-------|
| Registers | 0–1 | Inside SM | Be aware of RAW dependency latency ~20+ cycles |
| Shared Memory / L1 | ~20–30 | Inside SM | Bank conflicts increase latency |
| L2 Cache | ~200 | GPU-wide shared | 32B sector granularity |
| Global Memory (HBM) | ~400–800+ | On-board | Primary bottleneck |

### 7.2 Bandwidth Numbers

| Interconnect Type | Bandwidth |
|-------------------|-----------|
| PCIe Gen4 x16 | ~32 GB/s one-way |
| PCIe Gen5 x16 | ~64 GB/s one-way |
| NVLink (H100) | ~900 GB/s aggregate |
| HBM3 (H100) | ~3,350 GB/s |

PCIe offers only **2–4%** of GPU memory bandwidth. Conclusion: keep computation on the GPU whenever possible—even if the GPU is suboptimal at certain steps, it is still more cost-effective than a PCIe round trip.

### 7.3 Execution Parameters

| Parameter | Value |
|-----------|-------|
| Warp Size | 32 threads |
| Warp Schedulers / SM | 4 (Volta and later) |
| Global Memory Transaction Granularity | 32B (sector) / 128B (cache line) |
| Kernel Launch Overhead | ~3–10 µs |
| Max Resident Threads per SM (A100) | 2048 |
| Max Resident Blocks per SM (A100) | 32 |
| Shared Memory Banks | 32, 4B per bank |

### 7.4 Key Roofline Model Computation

```
Ridge Point = Peak GFLOPS / Memory bandwidth (GB/s)
```

For example, GPU peak 10 TFLOPS + bandwidth 500 GB/s → Ridge Point = 20 FLOPs/Byte.

- Arithmetic Intensity < Ridge Point: **Memory Bound** (ReLU: ~0.125 FLOPs/Byte)
- Arithmetic Intensity > Ridge Point: **Compute Bound** (large matrix multiplications)

### 7.5 Optimization Checklist

When writing CUDA code, check each item:

1. **Are warps fully utilized?** Is the block size a multiple of 32? Are there tail effects?
2. **Are memory accesses coalesced?** Do threads within a warp access contiguous, aligned addresses?
3. **Are bank conflicts avoided?** Do shared memory strided accesses stride by multiples of 32?
4. **Are small data transfers using PCIe?** Can computation be kept on the GPU?
5. **Is the kernel large enough?** Is execution time significantly greater than launch overhead?
6. **Is occupancy reasonable?** Are there enough warps to hide latency?
7. **Are there unnecessary synchronizations?** Can `.cpu()`/`.item()` and similar be removed?
8. **Are the data types correct?** Are there any implicit float→double promotions?

---

## 8. Performance Optimization Methodology

### 8.1 Optimization Stages

1. **Stage Zero**: Identify the performance bottleneck before starting to optimize. Do not waste effort on insignificant parts.
2. **Stage One**: Scope and refine input requirements. Generality often conflicts with high performance—make trade-offs early.
3. **Stage Two**: Streamline algorithmic complexity and computation volume. Eliminate common subexpressions, hoist loop-invariant code, and remove redundant computations.
4. **Stage Three**: Estimate bottlenecks based on performance models and adjust accordingly.

### 8.2 Bottleneck Diagnosis

Use a profiler to compare measured values against peak values:
- **Bandwidth Bottleneck**: bandwidth near peak, compute far from peak → compute simple operations on the fly and reduce loads/stores.
- **Compute Bottleneck**: compute near peak, bandwidth far from peak → cache reusable results for reuse.
- **Insufficient Latency Hiding**: all metrics far from peak → increase occupancy or ILP, restructure the program.

### 8.3 Recomputation vs. Memory Access Heuristics

"Because global memory latency is high, recomputation is preferable to reading"—this is a common misconception. If memory read/write units are not busy and there are sufficient parallel tasks, global memory access latency is easily hidden, and the compiler will also push `LDG` instructions forward as much as possible. Rule of thumb: **values computable in a few instructions need not be stored; values requiring higher overhead are better stored in most cases**.

### 8.4 When to Apply Thread Coarsening

Thread coarsening makes a single thread process multiple elements, essentially trading TLP (thread-level parallelism) for ILP (instruction-level parallelism):

| Scenarios Suitable for Coarsening | Effect After Coarsening |
|-----------------------------------|--------------------------|
| Redundant computations exist (multiple threads repeat the same intermediate step) | Eliminates redundancy |
| Memory Bound + multiple threads reading the same data | One read, multiple reuses |
| Heavy `__syncthreads()` synchronization overhead | Reduces synchronization frequency |
| Register-rich (low per-thread usage) | Fully utilizes registers |

**Costs**: reduced occupancy, loss of transparent scalability (coarsening factor must be tuned per GPU), and possible register spilling.

---

## Related Documents

- [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) – Systematic theory of memory hierarchy
- [GPU Execution Model and Thread Optimization](gpu-execution-model.md) – Thread model and scheduling mechanisms
- [GPU Instruction-Level Optimization](gpu-instruction-optimization.md) – PTX/SASS instruction-level optimization techniques
- [GPU Application-Level Optimization](gpu-application-optimization.md) – Application-level optimization strategies
