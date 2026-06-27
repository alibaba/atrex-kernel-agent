# Memory Access Patterns

## Standard 2D Block Access

### Triton (BEFORE)
```python
p = tl.make_block_ptr(base, (M, N), (stride_m, stride_n), (row_off, col_off), (BLOCK_M, BLOCK_N), (1, 0))
val = tl.load(p, boundary_check=(0, 1))
```

### Gluon (AFTER)
```python
# Step 1: Select layout (extract from TTGIR)
load_layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[8, 2],
    threads_per_warp=[8, 8],
    warps_per_cta=[1, 4],
    order=[0, 1]
)

# Step 2: Create indices
row_idx = gl.arange(0, BLOCK_M, layout=load_layout)
col_idx = gl.arange(0, BLOCK_N, layout=load_layout)

# Step 3: Expand dimensions
row_2d = gl.expand_dims(row_idx, axis=1)
col_2d = gl.expand_dims(col_idx, axis=0)

# Step 4: Calculate offset
offsets = (row_off + row_2d) * stride_m + (col_off + col_2d) * stride_n

# Step 5: Create mask
mask = ((row_off + row_2d) < M) & ((col_off + col_2d) < N)

# Step 6: Type conversion
offsets_i32 = gl.cast(offsets, gl.int32)

# Step 7: buffer_load
val = gl.amd.cdna4.buffer_load(
    ptr=base,
    offsets=offsets_i32,
    mask=mask,
    other=0.0
)
```

---

## 1D Block Access

```python
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = (start + idx) * stride
mask = (start + idx) < bound
offsets_i32 = gl.cast(offsets, gl.int32)
val = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets_i32, mask=mask, other=0.0)
```

---

## 1D Vector Access (e.g., gate values)

```python
# TTGIR corresponding slice layout: #ttg.slice<{dim = 1, parent = #mma}>
slice_layout: gl.constexpr = gl.SliceLayout(1, mma)
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = gl.cast((start + idx) * stride, gl.int32)
mask = (start + idx) < bound
val = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
# Use gl.expand_dims to expand to 2D for subsequent computation
val_2d = gl.expand_dims(val, axis=1)  # [BLOCK, 1]
```

---

## Scalar Access

```python
val = gl.load(ptr + offset)
```

---

## Store Pattern

```python
# Calculate offsets (same as above)
offsets_i32 = gl.cast(offsets, gl.int32)

# buffer_store
gl.amd.cdna4.buffer_store(
    stored_value=val,
    ptr=base,
    offsets=offsets_i32,
    mask=mask
)
```

---

## Async Copy Directly to Shared Memory

CDNA4 supports hardware asynchronous copy (DMA), which can transfer data from global memory **directly** to shared memory, **bypassing registers**.

### Imports

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy
```

### buffer_load_to_shared (recommended)

```python
# Step 1: Pre-allocate shared memory (without value=)
smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], shared_layout)

# Step 2: Calculate offsets and mask (same as standard buffer_load)
offsets_i32 = gl.cast(offsets, gl.int32)
mask = (row_idx < M) & (col_idx < N)

# Step 3: async copy global → shared (DMA, bypassing registers)
async_copy.buffer_load_to_shared(
    dest=smem,           # shared memory descriptor
    ptr=base,            # scalar base pointer
    offsets=offsets_i32,  # int32 offset tensor
    mask=mask,
    other=0.0
)

# Step 4: Submit and wait for completion
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# Step 5: Load from shared memory to registers
data = smem.load(layout=dot_op)
# Or use load_shared_relaxed to avoid redundant sync barriers:
data = async_copy.load_shared_relaxed(smem, layout=dot_op)
```

### global_load_to_shared (Alternative)

```python
# Use pointer tensor instead of scalar base pointer + offset
ptr_tensor = base + gl.cast(offsets, gl.int32)
async_copy.global_load_to_shared(
    dest=smem,
    ptr=ptr_tensor,
    mask=mask,
    other=0.0
)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
```

### ⚠️ Selection Recommendations

| Method | Register Pressure | Hardware OOB Protection | Recommended |
|------|-----------|-------------|--------|
| `buffer_load_to_shared` | Low (scalar base pointer + 32-bit offset) | ✅ Supported | ⭐ **Preferred** |
| `global_load_to_shared` | High (64-bit pointer tensor) | ❌ Not supported | Use only when 64-bit addressing is needed |
| `gl.amd.cdna4.buffer_load` + `smem.store` | Highest (via registers) | ✅ Supported | ❌ Poor performance in pipeline scenarios |

### Hardware Constraints

- The `size_per_thread * bits_per_element` in `offsets` layout must be **128** or **32**
- Writes to `dest` must be coalesced
- If `dest` has swizzle, it can only swizzle within warp boundaries

---

## Key Points

1. **Offsets must be gl.int32**
2. **Layout must be extracted from TTGIR**
3. **Do not use tl.make_block_ptr**
4. **Mask dimensions must match offsets**
5. **Prefer `buffer_load_to_shared` in pipeline scenarios** — bypasses registers, yields better performance
6. **Do not interleave async_copy with ordinary buffer_load/store** — forces serialization, degrades performance
