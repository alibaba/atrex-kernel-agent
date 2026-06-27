# Memory-Bound Kernel Optimization: Hierarchical Reduction

The performance bottleneck for memory-bound kernels (such as RMSNorm, Softmax, Cross Entropy, and elementwise operations) is HBM bandwidth rather than compute capacity. This article explains how to leverage the GPU's 4-level memory hierarchypei for hierarchical reduction to achieve throughput near the theoretical peak (Speed-of-Light, SOL).

> This article is DSL-agnostic general GPU optimization knowledge. For CuTeDSL-specific implementation, see [QuACK Reduction Kernels](../cutedsl/quack-reduction-kernels.md)

---

## 1. Memory-Bound vs Compute-Bound

### 1.1 Detection Method: Arithmetic Intensity

**Arithmetic Intensity (AI)** = FLOPs / Bytes transferred

```
             AI < threshold
                  |
     +------------+------------+
     |                         |
  Memory-Bound            Compute-Bound
 (bandwidthperformance) (performance)
```

**Roofline inflection point for H100:**
- FP32: AI = 989.4 TFLOPS / 3.35 TB/s ~= 295 FLOPs/byte
- BF16 Tensor Core: AI = 1978.9 TFLOPS / 3.35 TB/s ~= 590 FLOPs/byte

**AI for typical kernels:**

| Kernel | FLOPs per element | Bytes per element | AI | Classification |
|--------|-------------------|-------------------|------|------|
| Elementwise (ReLU) | 1 | 2R + 2W = 4 (BF16) | 0.25 | Memory-bound |
| RMSNorm | ~5 | 2R + 2W = 4 (BF16) | 1.25 | Memory-bound |
| Softmax | ~5 | 2R + 2W = 4 (BF16) | 1.25 | Memory-bound |
| GEMM (M=N=K=4096) | 2K = 8192 | ~2 (BF16, amortized) | ~4096 | Compute-bound |

The AI of a reduction kernel is O(1), independent of the reduction dimension N, and is always memory-bound.

### 1.2 Performance Ceiling

For memory-bound kernels, the theoretical maximum throughput is determined by HBM bandwidth:

```
 = total_bytes / HBM_bandwidth

: Softmax, M=16K, N=131K, FP32
  read  = 16K * 131K * 4B = 8 GB
  write = 16K * 131K * 4B = 8 GB
  total = 16 GB
 = 16 GB / 3.35 TB/s = 4.78 ms
```

| GPU | HBM Bandwidth | 90% SOL Target |
|-----|---------|-------------|
| A100 (HBM2e) | 2.0 TB/s | 1.8 TB/s |
| H100 (HBM3) | 3.35 TB/s | 3.0 TB/s |
| B200 (HBM3e) | 8.0 TB/s | 7.2 TB/s |

---

## 2. Speed-of-Light (SOL) Analysis

### 2.1 Measuring with NCU

Use NVIDIA Nsight Compute to obtain the actual amount of DRAM transfer:

```bash
ncu --set full -k "kernel_name" ./your_program
```

> For more NCU usage, see [NCU Profiling Guide](ncu-profiling-guide.md)

Key metrics:

| Metric | Meaning |
|--------|------|
| `dram__bytes_read.sum` | Actual number of bytes read from HBM |
| `dram__bytes_write.sum` | Actual number of bytes written to HBM |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Percentage of peak bandwidth achieved |
| `lts__t_sectors_*.sum` | Number of L2 cache sector accesses |
| `smsp__sass_inst_executed_op_ldl_pred_on.sum` | Number of LDL (local load) instructions; non-zero indicates register spilling |

### 2.2 Calculating SOL%

```
achieved_bandwidth = (dram__bytes_read.sum + dram__bytes_write.sum) / kernel_time
SOL% = achieved_bandwidth / peak_bandwidth * 100%
```

**Real case (H100, Softmax, M=16K, N=131K, FP32):**
- After optimization: 3.01 TB/s = 89.7% SOL
- torch.compile: ~1.89 TB/s = 56.4% SOL (due to 2 GMEM reads)
- Liger (N=65K): ~2.0 TB/s = 59.7% SOL (due to register spilling)

### 2.3 Cost of Multiple Passes

Each additional GMEM read effectively halves the effective bandwidth:

```
1-pass: bandwidth = HBM_bandwidth (100%)
2-pass: bandwidth = HBM_bandwidth / 2 (50%)
3-pass: bandwidth = HBM_bandwidth / 3 (33%)
```

**Rule**: Avoid multiple GMEM reads at all costs. Use online algorithms or cluster reduction to achieve single-pass execution.

---

## 3. Coalesced Memory Access

### 3.1 Principle

GPU GMEM access is performed in units of 128 bytes (= 1 cache line). When 32 threads in a warp simultaneously access contiguous memory, the requests can be coalesced into the minimum number of cache line requests:

### 3.2 Vectorized Load/Store

Each thread loads multiple consecutive elements at once, increasing the access granularity per thread:

```c
// CUDA C++ example
// scalarload: 1 instruction per 4 bytes
float val = input[tid];

// vectorload: 1 instruction per 16 bytes (128 bits)
float4 vals = reinterpret_cast<float4*>(input)[tid];
// load 4 FP32

// BF16 vector: 1 instruction per 16 bytes
// 8 BF16 = 128 bits
```

**Vector width selection:**

| dtype | Element Size | 128-bit Vector Width |
|-------|-------------|---------------------|
| FP32 | 4 bytes | 4 elements |
| BF16/FP16 | 2 bytes | 8 elements |
| FP8 | 1 byte | 16 elements |

### 3.3 Asynchronous Load

Hopper+ supports asynchronous GMEM -> SMEM copy, bypassing registers:

```c
// CUDA C++ (cp.async)
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem_addr));
asm volatile("cp.async.commit_group;");
asm volatile("cp.async.wait_group 0;");
```

> See [Async Copy](nvidia-ptx-sync-and-async.md) for details.

---

## 4. 4-Level Hierarchical Reduction

```
                    +-----------------------+
                    |    Global Memory      |  ~400ns, 3.35 TB/s (H100)
                    |      (HBM DRAM)       |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
            Level 4 | Distributed Shared    |  ~30-50ns, SM-to-SM fabric
                    |   Memory (DSMEM)      |  Hopper+ only, up to 16 blocks
                    +-----------+-----------+
                                |
                    +-----------v-----------+
            Level 3 |   Shared Memory       |  ~20ns, ~20 TB/s
                    |      (SMEM)           |  Per-SM, 192-256 KB (H100)
                    +-----------+-----------+
                                |
                    +-----------v-----------+
            Level 2 |   Warp Shuffle        |  ~5ns, ~100 TB/s
                    |   (Registers)         |  32 threads, no memory access
                    +-----------+-----------+
                                |
                    +-----------v-----------+
            Level 1 |  Thread Registers     |  ~1ns, >100 TB/s
                    |   (Local reduce)      |  Zero sync overhead
                    +-----------+-----------+
```

### 4.1 Level 1 -- Thread-Local Reduction (Registers)

Each thread performs reduction on all elements it holds, completed entirely in registers with zero synchronization overhead.

```c
// CUDA C++ example: thread-local reduction
// Assuming each thread holds 8 BF16 values (after 128-bit vectorized load)
float thread_sum = 0.0f;
for (int i = 0; i < elems_per_thread; i++) {
    thread_sum += values[i] * values[i];  // Example: sum of squares for RMSNorm
}
```

**Cost**: 0 cycles synchronization, pure ALU operations.

### 4.2 Level 2 -- Warp Shuffle Reduction (Between Registers)

32 threads within the same warp exchange register values via shuffle instructions, requiring no memory access:

```c
// CUDA C++ example: butterfly warp reduction (summation)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;  // All 32 threads obtain the same reduction result
}

// Maximum value version
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}
```

**Butterfly Pattern Illustration (8 threads simplified):**
```
Step 1 (offset=4):  T0<->T4  T1<->T5  T2<->T6  T3<->T7
Step 2 (offset=2):  T0<->T2  T1<->T3  T4<->T6  T5<->T7
Step 3 (offset=1):  T0<->T1  T2<->T3  T4<->T5  T6<->T7

32 threads: 5 steps (offset = 16, 8, 4, 2, 1)
```

**Cost**: ~5 cycles per step (1 SHFL + 1 ALU), ~25 cycles for 5 steps.

**Partial Warp Reduction**: When `threads_per_row < 32`, you can shuffle only within subgroups:

```c
// Reduce only within 8-thread subgroups
float warp_reduce_sum_partial(float val, int width) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```**Butterfly Pattern Illustration (8 threads simplified):**
```
Step 1 (offset=4):  T0<->T4  T1<->T5  T2<->T6  T3<->T7
Step 2 (offset=2):  T0<->T2  T1<->T3  T4<->T6  T5<->T7
Step 3 (offset=1):  T0<->T1  T2<->T3  T4<->T5  T6<->T7

32 : 5 steps (offset = 16, 8, 4, 2, 1)
```

**Cost**: ~5 cycles per step (1 SHFL + 1 ALU), ~25 cycles for 5 steps.

**Partial Warp Reduction**: When `threads_per_row < 32`, you can shuffle only within subgroups:

```c
// 8 reduction
float warp_reduce_sum_partial(float val, int width) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 4.3 Level 3 -- Block SMEM Reduction (Shared Memory)

When a thread block has multiple warps participating in the same row reduction, SMEM exchange is needed:

```c
// CUDA C++ example: block reduction
__shared__ float reduction_buffer[MAX_WARPS_PER_ROW];

__device__ float block_reduce_sum(float val) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int warps_per_row = blockDim.x / 32;  // Simplified: assuming all warps in the same row

    // Step 1: warp reduction
    val = warp_reduce_sum(val);

    // Step 2: lane 0 of each warp writes to SMEM
    if (lane == 0) {
        reduction_buffer[warp] = val;
    }
    __syncthreads();  // ~20 cycles

    // Step 3: first warp reads back and reduces
    float block_val = 0.0f;
    if (lane < warps_per_row) {
        block_val = reduction_buffer[lane];
    }
    block_val = warp_reduce_sum(block_val);

    return block_val;  // All threads obtain the same result
}
```

**Cost**: 1 `__syncthreads` (~20 cycles) + SMEM read/write (~10 cycles each).

### 4.4 Level 4 -- Cluster DSMEM Reduction (Distributed Shared Memory)

The Thread Block Cluster introduced with Hopper (SM90)+ allows adjacent SMs to directly read and write each other's SMEM via DSMEM, bypassing GMEM entirely.

> See [Thread Block Cluster](../../../kernel-opt/nvidia/common/thread-block-cluster.md)

```c
// CUDA C++ conceptual illustration (actual requires PTX or CuTeDSL)
// Assuming cluster_size = 4, each block completes block-level reduction

// Step 1: Write this block's reduction result to other blocks' SMEM (via DSMEM)
// Use mapa instruction to get remote SMEM address
uint32_t remote_smem = __mapa(local_smem_ptr, peer_block_rank);
// Use st.async to write asynchronously
asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
             :: "r"(remote_smem), "f"(my_val), "r"(remote_mbar));

// Step 2: Wait for all blocks in cluster to finish writing
asm volatile("mbarrier.try_wait ..."); // Wait for mbarrier

// Step 3: Read all block values from local SMEM's reduction buffer and reduce
float cluster_val = 0.0f;
for (int i = 0; i < cluster_size * warps_per_row; i++) {
    cluster_val += reduction_buffer[i];  // All blocks' data now in local SMEM
}
```

**Why it's better than GMEM atomics:**
- DSMEM latency ~30-50ns vs GMEM ~400ns
- No GMEM contention (atomic contention)
- SM-to-SM fabric is a dedicated network, does not consume HBM bandwidth

**Cluster Size Selection:**

| cluster_n | Equivalent SMEM | Applicable N (BF16) | Applicable N (FP32) |
|-----------|-----------------|---------------------|---------------------|
| 1 | 192-256 KB | <= 16K | <= 32K |
| 2 | 384-512 KB | <= 32K | <= 64K |
| 4 | 768 KB - 1 MB | <= 64K | <= 128K |
| 8 | 1.5-2 MB | <= 128K | <= 256K |
| 16 | 3-4 MB | <= 256K | <= 512K |

### 4.5 When to Use Which Level

```
The size of N determines the reduction level needed:

N <= 32 elements:
  Thread reduction (each thread handles <= 1 element)
  -> Only warp shuffle needed

N <= 4K elements (BF16):
  Thread reduction + Warp shuffle
  -> 128 threads, each thread ~32 elements, 4 warps
  -> Block reduction (via SMEM)

N <= 16K elements (BF16):
  Thread reduction + Warp shuffle + Block reduction
  -> 256 threads, each thread ~64 elements, 8 warps
  -> Single SM registers sufficient

N = 16K ~ 64K elements (BF16):
  Need cluster (cluster_n = 2~4)
  -> Otherwise register spilling or multiple GMEM reads required

N >= 64K elements (BF16):
  Must use cluster (cluster_n >= 4)
  -> Without cluster, performance drops ~50% (register spilling)
  -> cluster_n=16 can handle up to 256K elements
```## 5. DSMEM Cross-SM Reduction (Hopper/Blackwell)

### 5.1 DSMEM Concepts

Distributed Shared Memory is an inter-SM shared memory channel introduced in the Hopper architecture. Thread blocks within the same cluster can directly access the SMEM of other blocks:

```
  SM 0           SM 1           SM 2           SM 3
+--------+    +--------+    +--------+    +--------+
| SMEM 0 |<-->| SMEM 1 |<-->| SMEM 2 |<-->| SMEM 3 |
| 256 KB |    | 256 KB |    | 256 KB |    | 256 KB |
+--------+    +--------+    +--------+    +--------+
     ^              ^              ^              ^
     |              |              |              |
     +--------------+--------------+--------------+
                SM-to-SM Network (DSMEM fabric)
                    Cluster (size = 4)
```

### 5.2 Synchronization Mechanism

DSMEM reduction uses mbarrier (memory barrier) for coordination:

```
Timeline:

Block 0:  [reduce] -> [st.async to all] -> [mbar wait] -> [read local buf] -> [reduce]
Block 1:  [reduce] -> [st.async to all] -> [mbar wait] -> [read local buf] -> [reduce]
Block 2:  [reduce] -> [st.async to all] -> [mbar wait] -> [read local buf] -> [reduce]
Block 3:  [reduce] -> [st.async to all] -> [mbar wait] -> [read local buf] -> [reduce]

mbar block st.async complete
-> block reduction_buffer block
-> warp reduce obtain cluster reductionresult
```

### 5.3 DSMEM in Persistent Kernels

In persistent kernels (where the same kernel processes multiple batch rows), double mbarriers are needed:
- **Full barrier**: released after producer finishes writing, consumer waits
- **Empty barrier**: released after consumer finishes reading, producer waits (to avoid overwriting unread data)

```
Phase 0:  Producer writes to buffer[0] -> full_mbar[0] arrives
          Consumer waits full_mbar[0] -> reads buffer[0] -> empty_mbar[0] arrives
Phase 1:  Producer waits empty_mbar[0] -> writes to buffer[1] -> full_mbar[1] arrives
          Consumer waits full_mbar[1] -> reads buffer[1] -> empty_mbar[1] arrives
(repeat with phase flipping)
```

---

## 6. Online Algorithms

### 6.1 Online Softmax

Traditional 3-pass softmax:
```
Pass 1: max_x = max(x_i)               // 1 GMEM read
Pass 2: sum_exp = sum(exp(x_i - max_x)) // 1 GMEM read (!)
Pass 3: y_i = exp(x_i - max_x) / sum_exp // 1 GMEM read + 1 write
```

Online softmax (Milakov & Gimelshein, 2018) merges pass 1+2:
```
// running max running sum
max_prev = -inf
sum_prev = 0

for each x_i:
    max_new = max(max_prev, x_i)
 // fix sum: previous sum max_prev compute, requiresfix
    sum_new = sum_prev * exp(max_prev - max_new) + exp(x_i - max_new)
    max_prev = max_new
    sum_prev = sum_new

// 1 GMEM read (pass 1+2) + 1 read+write (pass 3)
// 1 complete GMEM read = 33%
```

**Cross-thread online merging**: when two threads/warps/blocks each have (max_a, sum_a) and (max_b, sum_b):
```
max_final = max(max_a, max_b)
sum_final = sum_a * exp(max_a - max_final) + sum_b * exp(max_b - max_final)
```

**Implementation trick**: pack (max, sum) into a single `int64_t` for transmission (concatenating two `float` values), so that warp shuffle and DSMEM store only need to transfer one value.

### 6.2 Online RMS

RMSNorm is naturally online:
```
// sum of squares +, blockreduction
sum_sq = 0
for each x_i:
    sum_sq += x_i * x_i
rstd = rsqrt(sum_sq / N + eps)
```

No online trick is needed, but cluster reduction remains important: to avoid spilling due to data that cannot all fit in registers.

### 6.3 When Online Is Beneficial

| Scenario | Online | Two-pass | Recommendation |
|------|--------|----------|------|
| Softmax forward only | 1R + 1R/W | 2R + 1R/W | Online (33% faster) |
| Softmax forward + backward | Requires exp(x) | Naturally gets exp(x) | Two-pass (needs to retain intermediate results) |
| Large N + cluster | Register insufficient idealized (max, sum) pairs | Both reductions in registers | Two-pass + cluster |
| Small N (<=16K) | Either | Either | Little difference |

---

## 7. NCU Profiling for Memory-Bound Kernels

### 7.1 Key Metrics

```bash
# complete profile
ncu --set full --kernel-name "softmax" --launch-count 1 ./program

# memory
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  lts__t_sectors_srcunit_tex_op_read.sum,\
  smsp__sass_inst_executed_op_ldl_pred_on.sum \
  ./program
```

### 7.2 Diagnostic Checklist

| Symptom | Possible Cause | Solution |
|------|---------|---------|
| DRAM bandwidth < 70% peak | Unaligned/Uncoalesced access | Check vectorized load, ensure 128-bit alignment |
| DRAM read bytes >> theoretical | Multiple GMEM reads | Use online algorithm or cluster reduction |
| LDL instructions > 0 | Register spilling | Reduce per-thread data; use cluster to share |
| L2 hit rate abnormally high | Data repeatedly read in L2 | Check for unnecessary GMEM round-trips |
| SMEM bank conflict | Unaligned SMEM access | Use swizzle or adjust access pattern |

### 7.3 SOL Calculation Script

```python
def compute_sol(dram_read_bytes, dram_write_bytes, kernel_time_ns, peak_bandwidth_TBps):
    total_bytes = dram_read_bytes + dram_write_bytes
    achieved_TBps = total_bytes / (kernel_time_ns * 1e-9) / 1e12
    sol_pct = achieved_TBps / peak_bandwidth_TBps * 100
    return achieved_TBps, sol_pct

# example: H100 softmax, M=16K, N=131K, FP32
achieved, sol = compute_sol(
    dram_read_bytes=16384 * 131072 * 4,   # 8 GB
    dram_write_bytes=16384 * 131072 * 4,  # 8 GB
    kernel_time_ns=5320000,               # 5.32 ms
    peak_bandwidth_TBps=3.35
)
# achieved = 3.01 TB/s, sol = 89.7%
```

---

## 8. Performance Data (QuACK Benchmarks)

The following data is from H100 80GB HBM3 (peak 3.35 TB/s), batch size 8K-32K.

### 8.1 RMSNorm (Model Memory Throughput, TB/s)

| N | QuACK (BF16) | torch.compile | Liger | cuDNN |
|------|------|---------------|-------|-------|
| 4K | ~3.0 | ~2.5 | ~2.8 | ~2.5 |
| 16K | ~3.0 | ~2.5 | ~2.8 | ~2.5 |
| 65K | ~3.0 | ~2.0 | ~2.0 | - |
| 131K | ~3.0 | ~2.0 | N/A | - |
| 262K | ~3.0 | ~2.0 | N/A | - |

### 8.2 Softmax (Model Memory Throughput, TB/s)

| N | QuACK (FP32) | torch.compile | Liger |
|------|------|---------------|-------|
| 4K | ~3.0 | ~2.8 | ~3.0 |
| 16K | ~3.0 | ~2.5 | ~3.0 |
| 65K | ~3.0 | ~2.0 | ~2.0 |
| 131K | ~3.01 | ~1.89 | N/A |
| 262K | ~3.0 | ~1.9 | N/A |

### 8.3 Key Observations

1. **N >= 65K is the watershed**: At this point, a single SM's registers + SMEM are insufficient to hold all data
   - Without cluster: register spilling -> throughput drops ~50%
   - With cluster: DSMEM extends effective SMEM -> maintains ~90% SOL

2. **torch.compile bottleneck**: The generated Triton kernel, even using online softmax, still requires 2 GMEM reads (second pass computing exp(x-max)/sum), with an effective bandwidth ceiling of 2/3 of peak

3. **Liger bottleneck**: At N=65K, significant register spilling occurs (LDL instructions observable in NCU), attempting to fit all 65K elements into a single SM's registers

4. **cluster_n=16 limit**: Theoretically can handle 16 * 16K = 256K elements (BF16); QuACK benchmarks confirm sustained ~3 TB/s at N=262K

---

## Rules of Thumb

1. **128-bit vectorization**: All GMEM loads/stores must use 128-bit vectors (FP32: float4, BF16: 8 elements)
2. **Top-down reduction**: Registers -> warp shuffle -> SMEM -> DSMEM, passing only reduced scalars at each level
3. **Online first**: Use online algorithms to avoid multiple GMEM reads
4. **Cluster is the savior for 65K+**: At N >= 65K, cluster reduction delivers ~50% performance improvement
5. **Monitor LDL**: LDL instructions in NCU indicate register spilling — reduce per-SM data volume
6. **90% SOL is a realistic target**: Well-optimized memory-bound kernels can consistently achieve ~3 TB/s on H100

## References

- [QuACK Blog: Getting Memory-bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md) -- Core reference
- [Thread Block Cluster](../../../kernel-opt/nvidia/common/thread-block-cluster.md) -- Cluster and DSMEM details
- [PTX Synchronization and Asynchronous Operations](nvidia-ptx-sync-and-async.md) -- mbarrier, cp.async instructions
- [NCU Profiling Guide](ncu-profiling-guide.md) -- Performance analysis tool
- [QuACK CuTeDSL Implementation Details](../cutedsl/quack-reduction-kernels.md) -- CuTeDSL specific code
- Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018
- CUDA C++ Best Practices Guide, "Coalesced Access to Global Memory"
