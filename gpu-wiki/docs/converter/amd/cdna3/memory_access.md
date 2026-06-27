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
val = gl.amd.cdna3.buffer_load(
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
val = gl.amd.cdna3.buffer_load(ptr=base, offsets=offsets_i32, mask=mask, other=0.0)
```

---

## 1D Vector Access (e.g. gate values)

```python
# TTGIR corresponding slice layout: #ttg.slice<{dim = 1, parent = #mma}>
slice_layout: gl.constexpr = gl.SliceLayout(1, mma)
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = gl.cast((start + idx) * stride, gl.int32)
mask = (start + idx) < bound
val = gl.amd.cdna3.buffer_load(ptr=base, offsets=offsets, mask=mask, other=0.0)
# Use gl.expand_dims to expand to 2D for subsequent computation
val_2d = gl.expand_dims(val, axis=1)  # [BLOCK, 1]
```

---

## Scalar Access

```python
val = gl.load(ptr + offset)
```

---

## Store Patterns

```python
# Calculate offsets (same as above)
offsets_i32 = gl.cast(offsets, gl.int32)

# buffer_store
gl.amd.cdna3.buffer_store(
    stored_value=val,
    ptr=base,
    offsets=offsets_i32,
    mask=mask
)
```

---

## Key Points

1. **Offsets must be gl.int32**
2. **Layout must be extracted from TTGIR**
3. **Do not use tl.make_block_ptr**
4. **mask dimensions must match offsets**
