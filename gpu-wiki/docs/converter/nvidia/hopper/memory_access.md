# Memory Access Patterns (NVIDIA Hopper)

## Standard 2D Block Access

### Triton (BEFORE)
```python
p = tl.make_block_ptr(base, (M, N), (stride_m, stride_n), (row_off, col_off), (BLOCK_M, BLOCK_N), (1, 0))
val = tl.load(p, boundary_check=(0, 1))
```

### Gluon Hopper (AFTER)
```python
# Step 1: Select layout (extract from TTGIR)
load_layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8],
    threads_per_warp=[16, 2],   # Note: 16 × 2 = 32 (NVIDIA warp size)
    warps_per_cta=[4, 1],
    order=[1, 0]
)

# Step 2: Create index (using SliceLayout)
row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, load_layout))
col_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, load_layout))

# Step 3: Expand dimensions
row_2d = gl.expand_dims(row_idx, axis=1)   # [BLOCK_M, 1]
col_2d = gl.expand_dims(col_idx, axis=0)   # [1, BLOCK_N]

# Step 4: Compute offset
offsets = (row_off + row_2d) * stride_m + (col_off + col_2d) * stride_n

# Step 5: Create mask
mask = ((row_off + row_2d) < M) & ((col_off + col_2d) < N)

# Step 6: Construct pointer tensor + load
#    ⚠️ Hopper key difference: pointer arithmetic (base_ptr + offset_tensor)
ptr_tensor = base + gl.cast(offsets, gl.int32)
val = gl.load(ptr_tensor, mask=mask, other=0.0)
```

---

### ⚠️ Key Differences from AMD

| Aspect | AMD CDNA3 | Hopper |
|------|-----------|--------|
| 2D Load API | `gl.amd.cdna3.buffer_load(ptr=base, offsets=offs, ...)` | `gl.load(base + gl.cast(offs, gl.int32), ...)` |
| 2D Store API | `gl.amd.cdna3.buffer_store(stored_value=val, ptr=base, offsets=offs, ...)` | `gl.store(base + gl.cast(offs, gl.int32), val, ...)` |
| ptr Parameter | Scalar pointer + separate offset tensor | Pointer tensor (base_ptr + offset_tensor) |
| offset Type | `gl.int32` | `gl.int32` |

```python
# AMD:
val = gl.amd.cdna3.buffer_load(ptr=base_ptr, offsets=gl.cast(offs, gl.int32), mask=mask, other=0.0)
gl.amd.cdna3.buffer_store(stored_value=val, ptr=base_ptr, offsets=gl.cast(offs, gl.int32), mask=mask)

# Hopper:
val = gl.load(base_ptr + gl.cast(offs, gl.int32), mask=mask, other=0.0)
gl.store(base_ptr + gl.cast(offs, gl.int32), val, mask=mask)
```

---

## 1D Block Access

```python
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = (start + idx) * stride
mask = (start + idx) < bound
val = gl.load(base_ptr + gl.cast(offsets, gl.int32), mask=mask, other=0.0)
```

---

## 1D Vector Access (e.g. gate value)

```python
# TTGIR slice layout: #ttg.slice<{dim = 1, parent = #mma}>
slice_layout: gl.constexpr = gl.SliceLayout(1, mma)
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = gl.cast((start + idx) * stride, gl.int32)
mask = (start + idx) < bound
val = gl.load(base_ptr + offsets, mask=mask, other=0.0)
# gl.expand_dims 2D compute
val_2d = gl.expand_dims(val, axis=1)  # [BLOCK, 1]
```

---

## Scalar Access

```python
val = gl.load(ptr + offset)
gl.store(ptr + offset, val)
```

Scalar access is exactly the same on AMD and Hopper, both using `gl.load` / `gl.store`.

---

## Store Patterns

```python
# Compute offsets (same as load)
offsets = row_2d * stride_m + col_2d

# ⚠️ If data is in MMA layout, need to convert_layout to BlockedLayout first
val_blocked = gl.convert_layout(val_mma, blocked_layout)

# store
gl.store(base_ptr + gl.cast(offsets, gl.int32), val_blocked, mask=mask)
```**Note**: The data of `gl.store` must be in a layout suitable for global memory writes (typically `BlockedLayout`). If the accumulator is in `NVMMADistributedLayout` (MMA layout), it needs to be converted via `gl.convert_layout` first.

---

## Memory Access for async_copy Pipelines

For data that needs to enter shared memory and participate in pipelines (such as wTPLACEHOLDER_000005__), do not use `gl.load`, but use `async_copy_global_to_shared` instead:

```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy

# Construct pointer tensor
ptr_tensor = base_ptr + gl.cast(offsets, gl.int32)

# Direct DMA to shared memory (bypasses registers)
async_copy.async_copy_global_to_shared(smem.index(slot), ptr_tensor, mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
```

**When to use `gl.load` vs `async_copy`**:

| Scenario | Use API | Reason |
|------|---------|------|
| Data needs to enter shared memory (wgmma operands, persistent pipeline buffers) | `async_copy_global_to_shared` | CP_ASYNC DMA, bypasses registers, high bandwidth |
| Data used directly in registers (v, g, and other non-wgmma data) | `gl.load` | Direct to register, no smem needed |
| Scalar/1D simple access | `gl.load` / `gl.store` | Simple scenarios |

---

## Key Points

1. **Offsets must be gl.int32**
2. **Hopper uses pointer arithmetic**: `base_ptr + gl.cast(offsets, gl.int32)` (not scalar ptr + separate offsets)
3. **Layout must be extracted from TTGIR**
4. **Do not use tl.make_block_ptr**
5. **mask dimensions must match offsets**
6. **convert_layout is required before Store**: MMA → BlockedLayout
7. **Use async_copy for pipeline data, gl.load for non-pipeline data**
