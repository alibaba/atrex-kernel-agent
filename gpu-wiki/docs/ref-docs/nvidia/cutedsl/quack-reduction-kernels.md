# QuACK: CuTeDSL Memory-Bound Reduction Kernels

QuACK (Quirky Assortment of CuTe Kernels) is a high-performance memory-bound kernel library built on CuTeDSL, covering common LLM operators such as RMSNorm, Softmax, Cross Entropy, and TopK. Its core design philosophy leverages the GPU's 4-level memory hierarchy to implement hierarchical reduction, achieving approximately 90% HBM bandwidth utilization on the H100.

> For general hierarchical reduction principles (DSL-agnostic), see [Hierarchical Reduction](../common/hierarchical-reduction-memory-bound.md)

Source: [github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack) (Wentao Guo, Ted Zadouri, Tri Dao)

---

## 1. ReductionBase Framework

All reduction kernels inherit from the `ReductionBase` base class, which uniformly manages the infrastructure for 4-level reduction.

### 1.1 Base Class Interface

```python
class ReductionBase:
    def __init__(self, dtype, N, stage, reduction_dtype=Float32):
 self.dtype = dtype # inputdatatype
 self.N = N # reduction dimensionsize
 self.stage = stage # reduction buffer stage (stagereduction)
 self.reduction_dtype = reduction_dtype # reductionresulttype
```

Subclasses must implement two methods:
- `_threads_per_row()`: Returns the number of threads used per row based on N (controls the granularity of warp/block reduction)
- `_set_cluster_n()`: Determines the cluster size (1/2/4/8/16) based on N and dtype

### 1.2 TV-Layout and Vectorized Loading

The base class constructs a Thread-Value Layout via `_get_tiled_copy()`, ensuring 128-bit aligned coalesced access:

```python
def _get_tiled_copy(self, vecsize: int = 1):
    threads_per_row = self._threads_per_row()
 num_threads = self._num_threads # 128 (N<=16K) or 256
    num_blocks_N = ceil_div(N // vecsize, threads_per_row * cluster_n)
    tiler_mn = (num_threads // threads_per_row,
                vecsize * num_blocks_N * threads_per_row)
    tiled_copy = copy_utils.tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize)
    return tiled_copy, tiler_mn, threads_per_row
```

- `vecsize` is calculated based on the maximum dtype width, ensuring 128-bit transfers per load/store
  - BF16: vecsize=8 (8 x 16b = 128b)
  - FP32: vecsize=4 (4 x 32b = 128b)
- `tiler_mn[0]` = number of batch rows processed per block
- `tiler_mn[1]` = number of reduction columns processed per block

### 1.3 Reduction Buffer and DSMEM Allocation

```python
def _allocate_reduction_buffer_and_mbar(self, smem, tv_layout, is_persistent=False):
    # reduction_buffer shape: (num_warps // warps_per_row, (warps_per_row, cluster_n), stage)
    reduction_buffer = smem.allocate_tensor(
        self.reduction_dtype,
        self._get_reduction_buffer_layout(tv_layout, self.cluster_n), ...)
    if self.cluster_n > 1:
        mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage * (2 if is_persistent else 1))
    else:
        mbar_ptr = None
    return reduction_buffer, mbar_ptr
```
- When `cluster_n > 1`, an additional mbarrier is allocated for DSMEM synchronization
- Persistent kernels (e.g., RMSNormBackward) use double mbarriers (full + empty) to implement a producer-consumer pipeline

### 1.4 Cluster Initialization

```python
def _initialize_cluster(self, tidx, mbar_ptr, num_warps, is_persistent=False):
    if self.cluster_n > 1:
        if tidx < self.stage:
            cute.arch.mbarrier_init(mbar_ptr + tidx, 1)  # full barrier
            if is_persistent:
                cute.arch.mbarrier_init(mbar_ptr + self.stage + tidx, num_warps * cluster_n)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
```### 1.5 Unified `row_reduce` Interface

All kernel calls use `row_reduce()` globalization to complete 4-level reduction:

```python
def row_reduce(x, op, threads_per_row, reduction_buffer, mbar_ptr, init_val=0.0, ...):
 # Level 1: Thread reduction(registerreduction)
    val = x.reduce(op, init_val=init_val, reduction_profile=0)

 # Level 2: Warp reduction(shuffle reduction)
    val = cute.arch.warp_reduction(val, warp_op, threads_in_group=min(threads_per_row, 32))

 # Level 3+4: Block/Cluster reduction(SMEM/DSMEM reduction)
    if warps_per_row > 1 or cluster_n > 1:
        val = block_or_cluster_reduce(val, warp_op, reduction_buffer, mbar_ptr, ...)
    return val
```

### 1.6 Cluster Size Selection Strategy

Taking RMSNorm BF16 as an example, typical thresholds for `_set_cluster_n()`:

| N (BF16) | cluster_n | Per SM | Total Processed |
|----------|-----------|-------------|---------|
| <= 16K | 1 | 16K | 16K |
| <= 32K | 2 | 16K | 32K |
| <= 64K | 4 | 16K | 64K |
| <= 128K | 8 | 16K | 128K |
| > 128K | 16 | 16K | 256K |

For FP32, thresholds are doubled (since each element takes twice the space).

---

## 2. RMSNorm

File: `quack/rmsnorm.py`

### 2.1 Forward

The `RMSNorm` class inherits from `ReductionBase`, with stage=1 (RMSNorm) or stage=2 (LayerNorm).

**Algorithm Flow:**

1. **Load**: GMEM -> SMEM (cp_async) -> Registers (autovec_copy)
2. **Optional Residual Add**: `x = x + residual` (fused residual add)
3. **Reduction**: `sum_sq_x = row_reduce(x * x, ADD)`
4. **Compute rstd**: `rstd = rsqrt(sum_sq_x / N + eps)`
5. **Normalize + Affine Transform**: `y = x * rstd * weight + bias`
6. **Store Back**: Registers -> GMEM

**LayerNorm Variant** requires 2 reductions:
1. First reduction computes the mean: `sum_x = row_reduce(x, ADD)`, `mean = sum_x / N`
2. Second reduction computes the variance: `sum_sq = row_reduce((x - mean)^2, ADD)`

**Register Pressure Management** -- When N is large, data may not all fit in registers:
- `reload_from = None`: N <= 8K (RMSNorm) / 16K (LayerNorm), data remains in registers
- `reload_from = "smem"`: After reduction, reload from SMEM into registers
- `reload_from = "gmem"`: Reload from GMEM (slowest but supports the largest N)

**Weight Load Latency Optimization** (`delay_w_load`):
- Default `False`: Preload weights during cp_async wait to hide latency
- Can be set to `True`: Load weights only after reduction completes, reducing register pressure

### 2.2 Backward

`RMSNormBackward` uses a persistent kernel design:

```python
class RMSNormBackward(ReductionBase):
    def __init__(self, dtype, N):
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16K else "smem"
```

**Key Features:**
- **Persistent kernel**: Multiple batch rows are processed within a single kernel launch via a loop, avoiding repeated launch overhead
- **Double buffering**: `stage=2`, `sX`, and `sdO` each allocate 2 buffers for alternating prefetch
- **Producer-consumer mbarrier**: In persistent mode, uses two sets of barriers: full and empty
- **dW accumulation**: Each block accumulates partial dW, performs intra-block reduction via SMEM, and then merges on the host side using `dw_partial.sum(dim=0)`

**Gradient Computation:**
```
x_hat = x * rstd
wdy = dout * weight
mean_xhat_wdy = row_reduce(x_hat * wdy, ADD) / N
dx = (wdy - x_hat * mean_xhat_wdy) * rstd
```

### 2.3 Fused Variants

Fusions supported by `RMSNormFunction`:
- **Fused residual add**: `x = x + residual` is completed in the forward pass, avoiding a separate elementwise kernel
- **Residual output**: Outputs `residual_out = x + residual` before normalization for use by the next layer
- **Mixed dtype**: Input, output, and weights can be different dtypes (any combination of bf16/fp16/fp32)

---

## 3. Softmax

File: `quack/softmax.py`

### 3.1 Forward

The `Softmax` class supports two modes:

**Two-pass (`online_softmax=False`):**
1. First reduction: `max_x = row_reduce(x, MAX)` -- stage 0
2. Compute `exp_x = exp2(x * log2e - max_x * log2e)` (using `exp2` instead of `exp`, which is faster in hardware)
3. Second reduction: `denom = row_reduce(exp_x, ADD)` -- stage 1
4. Output: `y = exp_x * rcp_approx(denom)` (using approximate reciprocal to avoid division)**Online softmax (`online_softmax=True`, default):**

`online_softmax_reduce` packs max and sum into a single `Int64` (concatenation of two `Float32`), completing in a single pass:

```python
def online_softmax_reduce(x, threads_per_row, reduction_buffer, mbar_ptr, ...):
 # Thread-level: compute max sum_exp
    max_x = warp_reduction(x.reduce(MAX, ...), fmax, ...)
    exp_x = exp2(x * log2e - max_x * log2e)
    sum_exp_x = warp_reduction(exp_x.reduce(ADD, ...), add, ...)

 # Block/Cluster level: Int64 row
 # warp (max_x, sum_exp_x) Int64 write reduction buffer
    reduction_buffer[row, col] = f32x2_to_i64(max_x, sum_exp_x)
    # ...barrier...
 # , online coalesced:
    # max_final = max(max_1, max_2, ...)
    # sum_final = sum_1 * exp(max_1 - max_final) + sum_2 * exp(max_2 - max_final) + ...
```

**Numerical stability:**
- OOB elements filled with `-inf` (transparent to max, exp(-inf)=0 transparent to sum)
- Uses `exp2` + `log2(e)` multiplication instead of `exp`, leveraging the GPU's exp2 hardware unit
- Uses `rcp_approx` approximate reciprocal instead of division

### 3.2 Backward

Gradient computation for `SoftmaxBackward`:

```
dot = row_reduce(dy * y, ADD) # reduction
dx = y * (dy - dot)              # elementwise
```

Only requires a 1-stage reduction buffer.

---

## 4. Cross Entropy

File: `quack/cross_entropy.py`

### 4.1 Forward

The `CrossEntropy` class inherits from `ReductionBase`, with the core being the fusion of log-softmax + NLL loss.

**Algorithm:**
1. Load logits `x` into registers
2. Load target index, read `target_logit = x[row, target]`
3. Compute log-sum-exp:
   - Online mode: single pass to obtain `max_x` and `denom`
   - Two-pass mode: separately reduce max and sum (two-pass is used when backpropagation is needed, because `exp_x` is required)
4. `loss = max_x + log(denom) - target_logit`
5. **Optional fused backward**: If `mdX is not None`, gradients are computed directly in the forward kernel:
   ```
   probs = exp_x / denom
   dx[j] = probs[j]              # for j != target
   dx[target] = probs[target] - 1  # for j == target
   ```

**Decision logic:**
```python
# ifrequires dx, use two-pass(requires exp_x); otherwise online
cross_entropy_op = CrossEntropy(dtype, N, online_softmax=not has_dx)
```

**`ignore_index` support**: When `target == ignore_index`, `loss = 0`, `dx = 0`.

### 4.2 Backward

`CrossEntropyBackward` is a standalone kernel (does not use ReductionBase), supporting splitting the N dimension by blocks:

```python
# block 16K column
probs = exp2(x * log2e - lse * log2e)
grad = where(col == target, probs - 1.0, probs) * dloss
```

### 4.3 Fused Linear + Cross Entropy

`linear_cross_entropy.py` provides a chunked implementation:
- Splits the batch dimension by chunk_size (default 4096)
- Each chunk: GEMM(x_chunk, weight) -> cross_entropy_fwd -> dx_chunk
- dW is accumulated across chunks, with the last chunk's dW deferred to backward

---

## 5. TopK

Files: `quack/topk.py`, `quack/sort/bitonic_sort.py`, `quack/sort/sorting_networks.py`

### 5.1 Bitonic Sort Network

TopK implementation is based on a bitonic sort network, completed entirely within registers:

**Hierarchical structure:**
- N <= 64: Uses pre-generated optimal sorting networks (minimizing the number of compare-and-swap operations)
  - N=2: 1 CE, depth 1
  - N=4: 5 CEs, depth 3
  - N=8: 19 CEs, depth 6
  - N=16: 60 CEs, depth 10
  - N=32: 185 CEs, depth 14
  - N=64: 521 CEs, depth 21
- N=128: Recursive bitonic sort

### 5.2 TopK Algorithm

```python
def bitonic_topk(arr, k, ascending=False, warp_width=32):
 # 1. arr by k , sort
    topk_vals = arr[:k]
    bitonic_sort(topk_vals)

 # 2. coalesced: bitonic_topk_merge keep top-k
    for i in range(1, n // k):
        other_vals = arr[i*k : (i+1)*k]
        bitonic_sort(other_vals)
        bitonic_topk_merge(topk_vals, other_vals)

 # 3. warp coalesced: passed shuffle_sync_bfly data
    for i in range(log2(warp_width)):
        other_vals = shuffle_sync_bfly(topk_vals, offset=1<<i)
        bitonic_topk_merge(topk_vals, other_vals)
```**Index Encoding Trick**: Encode the column index into the lowest `log2(N)` bits of the float. After sorting, both the value and index are obtained simultaneously, avoiding extra key-value pair storage.

**Constraints**: N <= 4096, k <= 128. Both N and k must be powers of 2.

### 5.3 Optional Fused Softmax

TopK forward supports optional fused softmax:
```python
if self.softmax:
 max_val = shuffle_sync(topk_vals[0], offset=0, ...) # sort, max
    exp_x = exp2(topk_vals * log2e - max_val * log2e)
    denom = warp_reduction_sum(exp_x.reduce(ADD, ...))
    topk_vals = exp_x * rcp_approx(denom)
```

### 5.4 Backward

`TopKBackward` uses a scatter strategy:
1. Load `dvalues` and `indices`
2. If fused softmax, compute softmax backward first: `grads = vals * (dvals - dot(dvals, vals))`
3. Scatter grad by indices into the SMEM output buffer
4. Read back from SMEM and write to GMEM

---

## 6. Performance Characteristics

### 6.1 Benefits of Cluster Reduction

Based on benchmarks on H100 HBM3 (3.35 TB/s peak):

| N | Without cluster | With cluster | Improvement |
|---|-----------|-----------|------|
| <= 16K | ~3.0 TB/s | Not needed | - |
| 32K | ~2.8 TB/s | ~3.0 TB/s | ~7% |
| 65K | ~2.0 TB/s (register spilling!) | ~3.0 TB/s | ~50% |
| 131K | ~1.9 TB/s | ~3.0 TB/s | ~58% |
| 262K | Worse | ~3.0 TB/s | >50% |

**Key Insight**: When N >= 65K, a single SM's registers+SMEM cannot hold all the data. Without a cluster, this leads to register spilling (LDL instructions observable in SASS), causing throughput to plummet. Clustering combines multiple SMs' SMEM into DSMEM, avoiding spilling.

### 6.2 NCU Profile Metrics

QuACK softmax (batch=16K, N=131K, FP32, cluster_n=4, 256 threads):
- DRAM throughput: 3.01 TB/s = 89.7% peak
- Effectively uses DSMEM (visible in NCU memory workload chart)

### 6.3 Comparison with Baselines

| Kernel | QuACK | torch.compile | Liger |
|--------|-------|---------------|-------|
| Softmax (N=131K, FP32) | ~3.01 TB/s | ~1.89 TB/s | Not supported |
| RMSNorm (N=131K, BF16) | ~3.0 TB/s | ~2.0 TB/s | Not supported |
| Cross Entropy (N=131K) | ~3.0 TB/s | ~2.0 TB/s | Not supported |

**torch.compile Issue**: The generated Triton kernel uses online softmax for the first pass, but still requires a second pass to load data from GMEM followers the final result. Two GMEM reads result in an effective bandwidth of approximately 2/3 of the theoretical peak.

**Liger Issue**: At N=65K, significant register spilling occurs (LDL instructions visible in NCU SASS profile), causing throughput to plummet from ~3 TB/s to ~2 TB/s.

### 6.4 Online Algorithm vs Two-pass

| Algorithm | GMEM Read Count | Use Case |
|------|-------------|---------|
| Online softmax | 1 time | Forward-only (does not need exp_x) |
| Two-pass | 1 time (with cluster) / 2 times (without cluster) | Needs backward (needs to retain exp_x) |

QuACK's cluster reduction enables two-pass to also require only 1 GMEM read: after data is loaded into registers, both reductions are performed entirely within registers+SMEM+DSMEM.

---

## 7. Architecture Design Summary

```
ReductionBase
 |-- _get_tiled_copy # 128-bit vector TV-Layout
 |-- _set_cluster_n # : by N cluster size
 |-- _threads_per_row # : by N
 |-- _allocate_reduction_buffer_and_mbar # SMEM/DSMEM
 |-- _initialize_cluster # mbarrier
  |
  |-- RMSNorm               (stage=1, online, fused residual)
  |-- RMSNormBackward        (stage=2, persistent kernel, dW accumulation)
  |-- Softmax               (stage=1 online / stage=2 two-pass)
  |-- SoftmaxBackward        (stage=1, dot product reduction)
  |-- CrossEntropy           (stage=1 online / stage=2 two-pass, fused dx)
  |-- TopKBackward           (stage=1, scatter-based)

TopK (independent, ReductionBase)
 |-- bitonic_topk # register bitonic sort + warp shuffle
 |-- optimal_sort # sorting network
```

## Reference

- [QuACK Blog: Getting Memory-bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
- [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
- [Hierarchical Reduction General Principles](../common/hierarchical-reduction-memory-bound.md)
