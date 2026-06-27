# API Mapping Reference (NVIDIA Hopper)

**Last Updated**: 2026-03-18
**Target Architecture**: NVIDIA Hopper (sm_90, H20/H100/H200)
**Verification Status Legend**: ✅ Verified | ⚠️ Pending Verification | ❌ Falsified

> **⚠️ TTGIR Field Names ≠ Gluon Parameter Names**: `versionMajor/versionMinor`→`version=[major,minor]`, `warpsPerCTA`→`warps_per_cta`, etc. See `layouts.md` for the full mapping.

> **⚠️ `num_stages`**: Set to 1 in the Gluon launcher. If the Triton source code uses `num_stages > 1`, you need to reference the TTGIR and manually implement multi-level pipelining in the Gluon kernel (using `async_copy`).

---

## Import Statements

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
 async_copy, # asynchronous DMA: global -> shared memory
 fence_async_shared, # wgmma must fence
 warpgroup_mma, # asynchronousmatrixmultiplication
 warpgroup_mma_wait, # wait wgmma complete
)
```

---

## Program Control

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.program_id(axis)` | `gl.program_id(axis)` | ✅ | Identical |
| `tl.num_programs(axis)` | `gl.num_programs(axis)` | ✅ | Identical |

---

## Tensor Creation

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.arange(start, end)` | `gl.arange(start, end, layout=...)` | ✅ | **Must specify layout** |
| `tl.zeros(shape, dtype)` | `gl.zeros(shape, dtype, layout=...)` | ✅ | **Must specify layout** |
| `tl.full(shape, value, dtype)` | `gl.full(shape, value, dtype, layout=...)` | ✅ | **Must specify layout** |
| `tl.zeros_like(input)` | `gl.zeros_like(input, layout=...)` | ✅ | Optional layout |

---

## Memory Access (⚠️ Significant Differences from AMD)

| Triton | Gluon (Hopper) | Verification Status | Notes |
|--------|----------------|---------|------|
| `tl.load(ptr, mask, other)` | `gl.load(ptr_tensor, mask, other)` | ✅ | ptr_tensor = base_ptr + offset |
| `tl.store(ptr, value, mask)` | `gl.store(ptr_tensor, value, mask)` | ✅ | ptr_tensor = base_ptr + offset |
| `tl.make_block_ptr(...)` | ❌ **Prohibited** | ❌ | Manually compute offsets |

**Key Difference**: Hopper uses `gl.load`/`gl.store` for 2D block access (pointer arithmetic), rather than AMD's `buffer_load`/`buffer_store` (scalar ptr + offset tensor).

```python
# Hopper: pointer tensor = base_ptr + offset_tensor
ptr_tensor = base_ptr + gl.cast(offsets, gl.int32)
val = gl.load(ptr_tensor, mask=mask, other=0.0)
gl.store(ptr_tensor, value, mask=mask)

# AMD: scalar ptr + separate offset tensor
val = gl.amd.cdna3.buffer_load(ptr=base_ptr, offsets=offsets_i32, mask=mask, other=0.0)
```

---

## Asynchronous Memory Transfer (Hopper-specific)

| API | Purpose | Verification Status | Notes |
|-----|------|---------|------|
| `async_copy.async_copy_global_to_shared(smem, ptr_tensor, mask=mask)` | CP_ASYNC DMA: global → shared (bypasses registers) | ✅ | **Performance-critical**: 50%+ faster than gl.load+smem.store |
| `async_copy.commit_group()` | Commit a group of async copy operations | ✅ | Call after all async_copy calls |
| `async_copy.wait_group(num_outstanding=N)` | Wait for async copy to complete | ✅ | N=0 means wait for all to complete |

**Source**: `triton.experimental.gluon.language.nvidia.hopper.async_copy` (ported from Ampere)

---

## Shared Memory Management

| Operation | Gluon API | Verification Status | Notes |
|------|-----------|---------|------|
| Allocate smem (with initial value) | `gl.allocate_shared_memory(dtype, shape, layout, value=data)` | ✅ | For temporary buffers |
| Pre-allocate smem (no write) | `gl.allocate_shared_memory(dtype, [depth, ...], layout)` | ✅ | For persistent pipeline buffers |
| Index buffer slot | `smem.index(i)` | ✅ | Corresponds to TTGIR `memdesc_index` |
| Write to slot in-place | `smem.index(i).store(data)` | ✅ | Corresponds to TTGIR `local_store` |
| Read from slot | `smem.index(i).load(layout=...)` | ✅ | Hopper: DotOperandLayout not required |## Matrix Multiplication (⚠️ Major Differences from AMD)

| Triton | Gluon (Hopper) | Validation Status | Notes |
|--------|----------------|---------|------|
| `tl.dot(a, b, acc)` | `warpgroup_mma(a_smem, b_smem, acc, is_async=True)` | ✅ | Operands must be in shared memory |
| — | `fence_async_shared()` | ✅ | **Must be called before wgmma** |
| — | `warpgroup_mma_wait(num_outstanding=0, deps=(result,))` | ✅ | **Must wait after wgmma** |

**Key Differences**:
- AMD MFMA: Operands are loaded from smem into **registers** via `DotOperandLayout`, then `mfma` is executed
- Hopper wgmma: Operands remain in **shared memory** (`NVMMASharedLayout`), and wgmma reads directly from smem

```python
# Hopper wgmma completemode
h_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BV], nvmma_shared_layout, value=data)
fence_async_shared()
acc = warpgroup_mma(w_smem, h_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

---

## Synchronization Operations (Hopper-specific)

| API | Purpose | Validation Status | Notes |
|-----|------|---------|------|
| `fence_async_shared()` | shared memory fence | ✅ | Called after writing to smem and before wgmma reads smem |
| `warpgroup_mma_wait(num_outstanding=0, deps=(result,))` | Wait for wgmma to complete | ✅ | Pass the tensor to wait on in deps |

---

## Math Operations (Same as AMD)

| Triton | Gluon | Validation Status | Notes |
|--------|-------|---------|------|
| `tl.exp(x)` | `gl.exp(x)` | ✅ | Same |
| `tl.where(c, x, y)` | `gl.where(c, x, y)` | ✅ | Same |
| `tl.cast(x, dtype)` | `gl.cast(x, dtype)` | ✅ | Same |
| `x.to(dtype)` | `x.to(dtype)` | ✅ | Same |

See `../../amd/common/porting_rules.md` for a complete list of math operations.

---

## Layout Conversion

| Operation | Gluon API | Validation Status | Notes |
|------|-----------|---------|------|
| Convert distributed layout | `gl.convert_layout(data, target_layout)` | ✅ | Used for mma → blocked and other conversions |

---

## Forbidden APIs (Hopper)

| API | Reason |
|-----|------|
| `gl.amd.cdna3.buffer_load` | AMD-specific, will crash on NVIDIA |
| `gl.amd.cdna3.buffer_store` | AMD-specific |
| `gl.amd.cdna3.mfma` | AMD-specific |
| `gl.amd.AMDMFMALayout` | AMD-specific |
| `gl.DotOperandLayout` | Not needed for Hopper wgmma |
| `tl.libdevice.*` | Use `gl.*` equivalent |
| `tl.make_block_ptr` | Calculate offset manually |
