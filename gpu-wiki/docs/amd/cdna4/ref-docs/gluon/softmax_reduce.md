---
pattern: softmax_reduce
sub_patterns: []
pitfalls: [9]
---

# Softmax / Reduction / Element-wise Optimization Guide

> This document focuses on kernels **without matrix multiplication** (pure reduction, element-wise, softmax).
> For the general ISA optimization checklist, see `common_optimizations.md`.

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | Reduction (sum/max), element-wise (mul/add), softmax |
| **Main Loop Structure** | Iterate along a single dimension or process each element in parallel |
| **Inter-iteration Dependencies** | Cross-thread dependencies for reduction; no dependencies for element-wise |
| **ISA Signatures** | `buffer_load_dwordx4`, `buffer_store_dwordx4`, `v_add_f32`, `v_max_f32`, `v_exp_f32` |

**Identification Criteria**:
- No mfma/mfma_scaled instructions
- Pure element-wise operations or reduction
- Typically Memory Bound

---

## Bottleneck Characteristics

### Extremely Low Tile AI

```
Tile_FLOPs = BM × BN (per element op)
Tile_Bytes = (BM × BN) × element_size × 2 (read + write)
Tile_AI    = 1 / (2 × element_size) ≈ 0.25 FLOPs/Byte (bf16)
```

MI355X dense BF16 Ridge Point ≈ 312.5. **Tile AI << Ridge Point → Memory Bound**.

**Conclusion**: Operators of this type are almost always Memory Boundaker. Optimization should focus on maximizing bandwidth utilization.

---

## Optimization Priority

| Priority | Optimization Item | Notes |
|--------|--------|------|
| ⭐⭐⭐ | §3.0 Order Check | Coalesced memory access is a prerequisite |
| ⭐⭐⭐ | §3.1 dwordx4 load/store | Maximize bandwidth |
| ⭐⭐ | §3.9 Small Matrix Tuning | When grid blocks < num_CUs |
| ⭐ | XCD Remap | Load balancing for large grids |
| — | warp_pipeline_stage | Not applicable (no MFMA) |

---

## Typical Optimization Path

```
§3.0 order check -> §3.1 dwordx4 load/store -> §3.9 (if small matrix) -> XCD remap
```

---

## Element-wise Special Considerations

### Fusing Multiple Element-wise Operations

If possible, fuse multiple element-wise operations into a single kernel:
- Reduce HBM round-trips
- Improve arithmetic intensity

```python
# ❌ separate: A->B->C ( HBM )
B = exp(A)
C = B * scale

# ✅ : A->C ( HBM )
C = exp(A) * scale
```

### Using tl.assume to Assist the Compiler

```python
tl.assume(stride > 0) # boundarycheck
```

---

## Reduction Special Considerations

### Warp-level Reduction

Use hardware-supported warp-level primitives:
- `gl.reduce_sum()`
- `gl.reduce_max()`

Avoid implementing cross-lane shuffles manually.

### Shared Memory Reduction

For block-level reduction:
1. Perform warp-level reduction first
2. Write results to shared memory
3. A single warp performs the final reduction

```python
# Step 1: warp reduce
val_warp = gl.reduce_sum(val, axis=-1)

# Step 2: write to smem (only lane 0 of each warp)
if lane_id % 64 == 0:
    smem[warp_id].store(val_warp)

# Step 3: final reduce by single warp
gl.barrier()
if warp_id == 0:
    final = gl.reduce_sum(smem.load(...))
```

---

## Softmax Special Considerations

### Online Softmax

Use the online softmax algorithm to avoid overflow:
- Maintain running max and running sum
- Rescale the accumulator at each step

```python
m_i = -float('inf')
d_i = 0.0
acc = gl.zeros(...)

for i in range(num_blocks):
    x = load_block(i)
    m_new = gl.max(x)
    d_new = gl.sum(gl.exp(x - m_new))

    alpha = gl.exp(m_i - m_new)
    acc = acc * alpha + gl.exp(x - m_new)
    d_i = d_i * alpha + d_new
    m_i = m_new

output = acc / d_i
```

### Causal Mask Optimization

For softmax in causal attention:
- Skip mask computation for full blocks
- Only apply masking to the tail block

See `mla_decode.md` OPT-3.

---

## References

- General Optimization: `common_optimizations.md`
- MLA Decode: `mla_decode.md`
- Lessons Learned: `pitfalls.md`
