# Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide

> Applicable Architecture: AMD gfx942 (CDNA3, MI300 Series)
> Framework Version: Triton Gluon (Experimental)

---

## 1. Module Import

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.experimental.gluon.language as ttgl

# AMD CDNA3
from triton.experimental.gluon.language.amd import cdna3
from triton.experimental.gluon.language.amd import warp_pipeline_stage
```

---

## 2. API Reference

### 2.1 Data Movement

#### `cdna3.buffer_load`

Load from global memory to registers using a scalar base pointer + offset tensor. Saves registers compared to `gl.load` (1 scalar pointer vs N 64-bit pointer tensors).

```python
ttgl.amd.cdna3.buffer_load(
 ptr, # scalarpointer
 offsets, # int32 offsettensor
 mask=None, # optional bool mask
 other=None, # optionaldefault( mask)
 cache=None, # cache, "cs"
)
```

#### `cdna3.buffer_store`

Store from registers to global memory using a scalar base pointer + offset tensor.

```python
ttgl.amd.cdna3.buffer_store(
 stored_value, # storetensor
 ptr, # scalarpointer
 offsets, # int32 offsettensor
 mask=None, # optional bool mask
 cache=None, # cache
)
```

**Usage Example**:

```python
# highload A matrix (M×K, K contiguous)
load_layout_a = ttgl.BlockedLayout(
    size_per_thread=[1, 8],
    threads_per_warp=[8, 8],
    warps_per_cta=[8, 1],
    order=[1, 0],
)
offs_am = gl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, load_layout_a))
offs_ak = gl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, load_layout_a))
offs_a = (pid_m * BLOCK_M + offs_am)[:, None] * stride_am + offs_ak[None, :]
k_mask = offs_ak[None, :] < K
a = ttgl.amd.cdna3.buffer_load(a_ptr, offs_a, mask=k_mask)
```

---

### 2.2 Matrix Computation

#### `cdna3.mfma`

Compute `a * b + acc` using AMD MFMA matrix cores.

```python
ttgl.amd.cdna3.mfma(
 a, # , requires DotOperandLayout
 b, # , requires DotOperandLayout
 acc, # , requires AMDMFMALayout
)
```

**Layout Configuration**:

```python
# layout
mma_layout = ttgl.amd.AMDMFMALayout(
    version=3,                   # gfx942
 instr_shape=[16, 16, 16], # MFMA (M, N, K)
 transposed=True, # result, dot
 warps_per_cta=[2, 4], # 8 warp: 2 M, 4 N
)

# layout
dot_a = ttgl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=4)
dot_b = ttgl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=4)
```

**Supported `instrShape` for gfx942**:

| instrShape | Description |
|-----------|------|
| `[16, 16, K]` | General-purpose choice, highest K-dimension throughput |
| `[32, 32, K]` | Suitable for large tiles, reduces instruction count |
| `[4, 64, K]` | Narrow M, wide N |
| `[64, 4, K]` | Wide M, narrow N |

**`k_width` Selection**:

| Data Type | k_width |
|---------|---------|
| bf16 / fp16 | 4 |
| fp8 | 8 |
| fp32 | 1 |

---

### 2.3 Shared Memory Management

#### Allocation

```python
smem = gl.allocate_shared_memory(
 element_ty, # type, gl.bfloat16
 shape, # , [NUM_BUFFERS, BLOCK_M, BLOCK_K]
 layout, # SharedLayout, SwizzledSharedLayout
 value=None, # optional
)
```

#### Store / Load

```python
smem.store(tensor_value) # registerwrite shared memory
result = smem.load(layout=target_layout) # shared memory register
```

#### Indexing / Slicing (Zero-copy Subviews)

```python
sub = smem.index(i) # 0 i
sub = smem.slice(start, length, dim=0) # dim
```#### SwizzledSharedLayout (Eliminate Bank Conflicts)

```python
shared_layout = ttgl.SwizzledSharedLayout(
 vec=4, # 4 swizzle
 per_phase=1, # bit 1
 max_phase=16, # 16 bit
 order=[1, 0], # K contiguous
)
```

**Parameter Selection Guide**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `vec` | vec × element_bytes should be 4 or 8 bytes | bf16: 4, fp32: 2 |
| `max_phase` | Larger value reduces bank conflicts, but increases swizzle overhead | 8 or 16 |
| `order` | Matrix A (M×K): `[1,0]`; Matrix B (K×N): `[0,1]` | Choose based on the contiguous dimension |

---

### 2.4 Instruction Scheduling Control

#### `cdna3.sched_barrier(mask)`

Insert a scheduling barrier, `mask` controls which instruction types cannot cross.

| mask | Description |
|------|------|
| `0` | No instructions can cross (strictest) |

#### `cdna3.sched_group_barrier(mask, size, group_id)`

Group scheduling barrier for precise control of instruction interleaving.

| mask | Instruction Type |
|------|---------|
| `0x008` | MFMA |
| `0x020` | VMEM (global memory) |
| `0x100` | LDS_READ |
| `0x200` | DS_WRITE |

```python
# example: 2 VMEM 1 MFMA
ttgl.amd.cdna3.sched_group_barrier(0x020, 2, 0) # 2 VMEM
ttgl.amd.cdna3.sched_group_barrier(0x008, 1, 0) # 1 MFMA
```

#### `cdna3.s_set_prio(prio)`

Set the current wave priority.

| prio | Description |
|------|------|
| 0 | Lowest (default) |
| 1 | Medium-low |
| 2 | Medium-high |
| 3 | Highest |

#### `cdna3.s_barrier()`

Synchronize all waves within the threadgroup (equivalent to `__syncthreads()`).

#### `cdna3.iglp_opt(mask)`

Instruction group-level parallelism optimization (experimental).

| mask | Strategy |
|------|------|
| 0 | Interleave DS and MFMA (small GEMM) |
| 1 | Single-wave small GEMM |
| 2 | Interleave TRANS and MFMA (attention) |
| 3 | Interleave TRANS and MFMA (no predecessor interleaving) |

---

### 2.5 Warp Pipeline Stage (Automatic Pingpong)

Mark pipeline stage boundaries; the compiler automatically inserts `cond_barrier` + scheduling instructions.

```python
from triton.experimental.gluon.language.amd import warp_pipeline_stage

with warp_pipeline_stage("load"):
    a = ttgl.amd.cdna3.buffer_load(a_ptr, offs_a)

with warp_pipeline_stage("compute"):
    acc = ttgl.amd.cdna3.mfma(a, b, acc)
```

> **Note**: `amdg.cond_barrier` is automatically inserted by the compiler's BlockPingpong pass and cannot be called directly in gluon.

---

### 2.6 Warp ID Query

```python
wid = ttgl.amd.cdna3.warp_id # returns int32
```

Used Marker for manually implementing warp distribution or conditional logic.

---

### 2.7 Buffer Atomic Operations

```python
ttgl.amd.cdna3.buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None)
ttgl.amd.cdna3.buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None)
```

**Supported Data Types**: `float16, float32, bfloat16, float64, int32, int64, uint32, uint64`

---

### 2.8 Common Gluon Interfaces (Available on All Architectures)

| Interface | Description |
|------|------|
| `gl.barrier()` | Thread synchronization within a CTA (equivalent to `ttg.barrier local`) |
| `gl.barrier(cluster=True)` | Cross-cluster synchronization |
| `gl.allocate_shared_memory()` | Allocate shared memory |
| `gl.convert_layout(value, layout)` | Layout conversion |
| `gl.arange(start, end, layout)` | Generate a sequence tensor |
| `gl.full(shape, value, dtype, layout)` | Generate a constant tensor |
| `gl.warp_specialize()` | Warp specialization (different warps execute different tasks) |
| `gl.program_id(axis)` | Get program ID |
| `gl.cdiv(a, b)` | Ceiling division |## 3. Layout Type Quick Reference

| Layout Type | Usage | Typical Configuration |
|---------|------|---------|
| `BlockedLayout` | Global memory load/store | `sizePerThread=[1,8], order=[1,0]` |
| `AMDMFMALayout` | MFMA accumulator | `version=3, instrShape=[16,16,16]` |
| `DotOperandLayout` | MFMA operand | `k_width=4` (bf16) |
| `SliceLayout` | Reduction dimension index | `SliceLayout(dim, parent)` |
| `SwizzledSharedLayout` | Shared memory | `vec=4, perPhase=1, maxPhase=16` |
| `PaddedSharedLayout` | Shared memory with padding | Usedega eliminate special bank conflicts |

---

## 4. High-Performance GEMM Optimization Tips

### 4.1 Data Loading Optimization

- **Use `buffer_load` instead of `gl.load`**: Saves registers, scalar base pointer + offset mode
- **Make the `sizePerThread` contiguous dimension as large as possible**: e.g., `[1, 8]`, to fully leverage 128-bit vector loads
- **Match `order` to the memory layout**: For matrix A, use `[1,0]` when K is contiguous; for matrix B, use `[0,1]` when N is contiguous

### 4.2 Shared Memory Optimization

- **SwizzledSharedLayout eliminates bank conflicts**: `vec × element_bytes` matches the bank width
- **Multi-buffering hides latency**: `allocate_shared_memory(dtype, [NUM_BUFFERS, M, K], layout)`
- **K-dimension subslice enables pingpong**: Split K=64 into 4×16, alternating between dot and memory operations

### 4.3 Pingpong Scheduling (Compute-Bound Scenarios)

```python
# manual pingpong mode
gl.barrier()
ttgl.amd.cdna3.sched_barrier(0)
ttgl.amd.cdna3.s_set_prio(1)
acc = ttgl.amd.cdna3.mfma(a, b, acc)
ttgl.amd.cdna3.s_set_prio(0)
gl.barrier()
ttgl.amd.cdna3.sched_barrier(0)
# ... buffer_load local_store ...
```

**Applicable conditions**:
- `num_warps=8`, two warp groups execute alternately
- Tile size ≥ 128×128×64 (bf16)
- Compute-bound scenarios

### 4.4 PID Reordering (Load Balancing for Multi-XCD GPUs)

On multi-XCD GPUs, the default round-robin PID allocation can lead to high cross-XCD communication overhead and uneven load distribution. A three-step combination of 1D grid + XCD remapping + Grouped ordering is required.

**XCD configurations for different GPUs**:

| GPU | Number of XCDs | CU/XCD | Total CUs | NUM_XCDS |
|-----|---------|--------|-------|----------|
| MI300X | 8 | 38 | 304 | 8 |
| **MI308X** | **4** | **20** | **80** | **4** |
| MI250X | 2 | 52 | 104 | 2 |

**Step 1: Switch to a 1D grid**

```python
# ❌ 2D grid: none XCD mapping
matmul_kernel[grid_m, grid_n](...)

# ✅ 1D grid: pid mapping
matmul_kernel[grid_m * grid_n](...)
```

**Step 2: XCD remapping (`remap_xcd`)**

Assign adjacent PIDs to the same XCD to reduce cross-XCD communication:

```python
@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 4):
 """mapping PID XCD PID contiguous"""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid
```

**Step 3: Grouped ordering (`pid_grid`) — L2 cache optimization**

```python
@triton.jit
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 8):
 """ 1D pid mapping 2D grid, grouped ordering"""
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n
```**Complete usage example**:

```python
@triton.heuristics({
    "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
                           * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
})
@gluon.jit
def matmul_kernel(..., GRID_MN: tl.constexpr):
    pid = gl.program_id(axis=0)
    pid = remap_xcd(pid, GRID_MN, NUM_XCDS=4)  # MI308X: 4 XCDs
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=8)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # ...
```

**Expected impact**: 10-20% performance improvement, depending on how well the grid size matches the number of XCDs. The effect is more pronounced when the grid size is small (< 1024 blocks).

**GROUP_SIZE_M selection**:

| GROUP_SIZE_M | Effect |
|-------------|------|
| 1 | No grouping, simple row-major |
| 4 | Suitable for small to medium grids |
| **8** | **Recommended default**, suitable for most GEMMs |
| 16 | Suitable for very large grids |

---

## 5. Features Not Supported on gfx942

| Feature | Required Architecture |
|------|---------|
| TDM (Tensor Data Mover) | gfx1250 |
| WMMA instructions | RDNA3/4, gfx1250 |
| gfx1250 mbarrier | gfx1250 |
| gfx1250 async_copy (global↔shared) | gfx1250 |
| gfx1250 cluster barrier | gfx1250 |
| cdna4 buffer_load_to_shared | gfx950 (CDNA4) |
| cdna4 mfma_scaled | gfx950 (CDNA4) |

---

## 6. Common Parameter Configuration Reference

| Parameter | Small GEMM | Medium GEMM | Large GEMM |
|------|---------|---------|---------|
| BLOCK_M | 128 | 256 | 256 |
| BLOCK_N | 128 | 128 | 256 |
| BLOCK_K | 64 | 64 | 64 |
| num_warps | 4 | 8 | 8 |
| num_stages | 1-2 | 2 | 2 |
| instrShape | [16,16,16] | [16,16,16] | [16,16,16] |
| warps_per_cta | [2,2] | [4,2] | [2,4] |
| Pingpong | No | Yes | Yes |
| GROUP_SIZE_M | 4 | 4 | 4 |
```
