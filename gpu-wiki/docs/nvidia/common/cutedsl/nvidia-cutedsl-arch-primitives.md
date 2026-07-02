# CuTeDSL Architecture Primitives (NVIDIA General)


**Last updated**: 2026-06-30

## Thread/Block/Grid/Cluster Indexing

```python
from cutlass.cute.arch import (
    thread_idx, block_dim, block_idx, grid_dim,           # CTA/Grid
    cluster_idx, cluster_dim, cluster_size,                # Cluster
    block_in_cluster_idx, block_in_cluster_dim,            # CTA in Cluster
    block_idx_in_cluster,                                  # linearized
    lane_idx, warp_idx,                                    # Warp/Lane
)

# returns Tuple[Int32, Int32, Int32] or Int32
tidx = thread_idx()       # (x, y, z)
bid = block_idx()         # (x, y, z)
lid = lane_idx()          # 0-31
wid = warp_idx()          # warp index within CTA
cid = block_idx_in_cluster()  # linearized CTA ID within cluster
```

---

## Synchronization Primitives

### Thread/Warp Synchronization

```python
from cutlass.cute.arch import sync_threads, sync_warp, barrier, barrier_arrive

sync_threads()                  # __syncthreads()
sync_warp(mask=0xFFFFFFFF)      # warp-level synchronization
barrier(barrier_id=1, number_of_threads=128)       # named barrier
barrier_arrive(barrier_id=1, number_of_threads=128)  # arrive only

cluster_arrive(aligned=True)    # cluster barrier arrive
cluster_arrive_relaxed()        # relaxed semantics
cluster_wait()                  # Wait for all CTAs
```

### Cluster Synchronization

```python
from cutlass.cute.arch import cluster_arrive, cluster_arrive_relaxed, cluster_wait

cluster_arrive(aligned=True)    # cluster barrier arrive
cluster_arrive_relaxed # relaxed
cluster_wait # wait CTA
```

---

## Fence Operations

### Acquire-Release Fence

```python
from cutlass.cute.arch import (
    fence_acq_rel_cta,       # CTA scope
    fence_acq_rel_cluster,   # Cluster scope
    fence_acq_rel_gpu,       # GPU scope
    fence_acq_rel_sys,       # System scope
)
```

### Proxy Fence

```python
from cutlass.cute.arch import fence_proxy

fence_proxy(
    kind='async',           # 'alias'|'async'|'async.global'|'async.shared'|'tensormap'|'generic'
    space='cta',            # 'cta'|'cluster'|None
)
```

---

## mbarrier Operations

```python
from cutlass.cute.arch import (
    mbarrier_init,                    # Initialize mbarrier
    mbarrier_init_fence,              # Initialize fence
    mbarrier_arrive,                  # Arrive
    mbarrier_arrive_and_expect_tx,    # Arrive + set transfer byte expectation
    mbarrier_expect_tx,              # Set transfer expectation (no arrive)
    mbarrier_wait,                   # Blocking wait
    mbarrier_try_wait,               # Non-blocking wait → Boolean
    mbarrier_conditional_try_wait,   # Conditional non-blocking wait
)

# English note
mbarrier_init(mbar_ptr, count=128)
mbarrier_arrive(mbar_ptr)
mbarrier_arrive_and_expect_tx(mbar_ptr, bytes=1024)
mbarrier_wait(mbar_ptr, phase=0)

# CTA (cluster )
mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=1)
# peer_cta_rank mbar pointerconversion CTA SMEM
```

---

## Warp Primitives

### Elect One

```python
from cutlass.cute.arch import elect_one

with elect_one():
 # warp execute
    pass
```

### Make Warp Uniform

```python
from cutlass.cute.arch import make_warp_uniform

uniform_val = make_warp_uniform(value) # compilation: value warp
```

with elect_one():
    # Only one thread in warp executes this code
    pass

uniform_val = make_warp_uniform(value)  # Compiler hint: value uniform within warp

ballot = vote_ballot_sync(pred, mask=0xFFFFFFFF)  # → Int32, per-thread predicate bits
any_true = vote_any_sync(pred)                     # → Boolean
all_true = vote_all_sync(pred)                     # → Boolean
uniform = vote_uni_sync(pred)                      # → Boolean, all threads same?

### Warp Reduction

```python
from cutlass.cute.arch import warp_redux_sync

result = warp_redux_sync(
    value,
    kind='add',              # Int32/Uint32: 'add'|'and'|'max'|'min'|'or'|'xor'
    mask_and_clamp=0xFFFFFFFF,
)

result = warp_redux_sync(
    value,
    kind='fmax',             # Float32: 'fmax'|'fmin'
    abs=True,                # Optional: take absolute value
    nan=True,                # Optional: NaN handling
)
```

---

## SMEM Allocation

```python
from cutlass.cute.arch import alloc_smem, get_dyn_smem, get_dyn_smem_size

# English note
ptr = alloc_smem(element_type=cutlass.Float32, size_in_elems=1024, alignment=128)

# Dynamic allocation (need to specify smem size at kernel launch)
ptr = get_dyn_smem(element_type=cutlass.Float16, alignment=128)
size = get_dyn_smem_size()  # Returns dynamic SMEM bytes
```

---

## Atomic Operations

All atomics share a common pattern, returning the old value:

```python
from cutlass.cute.arch import (
    atomic_add, atomic_and, atomic_or, atomic_xor,
    atomic_max, atomic_min, atomic_exch, atomic_cas,
    atomic_max_float32,
)

old = atomic_add(ptr, val, sem='relaxed', scope='gpu')
old = atomic_cas(ptr, cmp=expected, val=desired, sem='acq_rel', scope='cta')
old = atomic_max_float32(ptr, value, positive_only=True)  # Non-negative f32 max
```

**sem options:** `'relaxed'` | `'release'` | `'acquire'` | `'acq_rel'`
**scope options:** `'gpu'` | `'cta'` | `'cluster'` | `'sys'`

---

## Load/Store Operations

### Store

```python
from cutlass.cute.arch import store

store(ptr, val,
    cop='wb',           # 'wb'|'cg'|'cs'|'wt' (cache write policy)
    ss=None,            # None=global, 'cta'=shared::cta, 'cluster'=shared::cluster
    sem='release',
    scope='gpu',
)
```

### Load

```python
from cutlass.cute.arch import load

val = load(ptr, dtype=cutlass.Float32,
    cop='ca',               # 'ca'|'cg'|'cs'|'lu'|'cv' (cache read policy)
    level1_eviction_priority='evict_normal',
    level_prefetch_size='size_128b',  # 'size_64b'|'size_128b'|'size_256b'
    sem='acquire',
    scope='gpu',
)
```

**Cache Policy Quick Reference:**

| cop (Load) | Meaning |
|------------|---------|
| `ca` | Cache at all levels (default) |
| `cg` | Cache at global level only |
| `cs` | Cache streaming (do not evict other data) |
| `lu` | Last use (hint that no subsequent access will occur) |
| `cv` | Cache volatile (bypass L1 cache) |

| cop (Store) | Meaning |
|-------------|---------|
| `wb` | Write-back (default) |
| `cg` | Cache at global level |
| `cs` | Cache streaming |
| `wt` | Write-through |

---

## Async Copy Operations

```python
from cutlass.cute.arch import (
 cp_async_commit_group, # cp.async
 cp_async_wait_group, # wait cp.async (n=pending )
 cp_async_bulk_commit_group, # cp.async.bulk
 cp_async_bulk_wait_group, # wait bulk
)

cp_async_commit_group # cp.async
cp_async_wait_group(0) # wait pending group complete
cp_async_bulk_wait_group(1, read=True) # wait ≤1 pending
```

---

## Register Management

```python
from cutlass.cute.arch import (
    warpgroup_reg_alloc,         # Allocate registers (warpgroup level)
    warpgroup_reg_dealloc,       # Release registers
    setmaxregister_increase,     # Dynamically increase register limit
    setmaxregister_decrease,     # Dynamically decrease register limit
)

warpgroup_reg_alloc(reg_count=128)     # Allocate 128 registers
warpgroup_reg_dealloc(reg_count=64)    # Release 64 registers
setmaxregister_increase(reg_count=255) # Increase limit
setmaxregister_decrease(reg_count=32)  # Decrease limit (release to other warps)
```## Type Conversion Built-in Functions

```python
from cutlass.cute.arch import (
    cvt_i8x4_to_f32x4,          # int8x4 → float32x4
    cvt_i8x2_to_f32x2,          # int8x2 → float32x2
    cvt_i8_bf16,                 # int8 → bfloat16
    cvt_i8x2_to_bf16x2,         # int8x2 → bfloat16x2
    cvt_i8x4_to_bf16x4,         # int8x4 → bfloat16x4
    cvt_f32x2_bf16x2,           # float32x2 → bfloat16x2
    cvt_i8_bf16_intrinsic,       # Fast int8→bf16 vector conversion
    cvt_i4_bf16_intrinsic,       # int4→bf16 (optional shuffle reorder)
    prmt,                        # Byte permutation permute
)

# fastvectorconversion
result = cvt_i8_bf16_intrinsic(vec_i8, length=4)
result = cvt_i4_bf16_intrinsic(vec_i4, length=8, with_shuffle=True)
```

### Math Built-ins

```python
from cutlass.cute.arch import fmax, rcp_approx, exp2, popc

max_val = fmax(a, b)        # float max
recip = rcp_approx(a)       # Approximate reciprocal
e2 = exp2(a)                # 2^a
bits = popc(value)          # Population count
```

---

## Cluster Launch Control (CLC)

SM90+ supports dynamically canceling unlaunched clusters:

```python
from cutlass.cute.arch import issue_clc_query, clc_response

issue_clc_query(mbar_ptr, clc_response_ptr) # cluster

# result
cta_x, cta_y, cta_z, status = clc_response(result_addr)
# If cancellation successful, return coordinates of first CTA in cancelled cluster
```

---

## SM100 (Blackwell) Extension Overview

The Blackwell architecture introduces the following CuTeDSL extensions (detailed API in the `cutlass.cute.nvgpu.tcgen05` module):

### TMEM (Tensor Memory)

Blackwell's exclusive on-chip memory, used for accumulators:

```python
from cutlass.cute.arch import alloc_tmem, dealloc_tmem, retrieve_tmem_ptr

alloc_tmem(num_columns, smem_ptr_to_write_address, arch='sm_100')
dealloc_tmem(tmem_ptr, num_columns)
```

### tcgen05 MMA

5th-generation Tensor Core:

| MMA Type | Data Type | Characteristics |
|----------|---------|------|
| `MmaTF32Op` | TF32 | Standard |
| `MmaF16BF16Op` | FP16/BF16 | Same as Hopper |
| `MmaI8Op` | INT8 | Integer MMA |
| `MmaFP8Op` | FP8 | E4M3/E5M2 |
| `MmaMXF8Op` | MX FP8/FP6/FP4 | Block-scaled |
| `MmaMXF4Op` | MX FP4 | Block-scaled, no major mode |
| `MmaMXF4NVF4Op` | MX FP4 NV | Block-scaled + scale factor type |

### 2CTA Instructions

Blackwell supports two CTAs collaborating to execute MMA/Copy (`CtaGroup.TWO`), controlled by the SM100 utility function via the `use_2cta_instrs` parameter.

### SM100 SmemLayoutAtomKind

New `MN_SW128_32B` (128B swizzle + 32B base), totaling 9 swizzle modes.


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CuTeDSL Programming Model](cutedsl-programming-model.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](cutlass-cute-fundamentals.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
