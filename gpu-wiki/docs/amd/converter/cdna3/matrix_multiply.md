# Matrix Multiplication Patterns


**Last updated**: 2026-06-30

## Standard MFMA Pattern

### Triton (BEFORE)
```python
accumulator = tl.dot(a, b, accumulator)
```

### Gluon (AFTER)
```python
# Step 1: (Optional, depends on TTGIR implementation) Allocate shared memory
a_smem = gl.allocate_shared_memory(
    element_ty=a.dtype,
    shape=[BLOCK_M, BLOCK_K],
    layout=shared_layout,
    value=a
)

b_smem = gl.allocate_shared_memory(
    element_ty=b.dtype,
    shape=[BLOCK_K, BLOCK_N],
    layout=shared2_layout,
    value=b
)

# Step 2: Load to register (using DotOperandLayout)
# If not using shared_memory
a_dot = gl.convert_layout(a, layout=dot_op0) # DotOperandLayout(operand_index=0, parent=mma, k_width=4)
b_dot = gl.convert_layout(b, layout=dot_op1)  # DotOperandLayout(operand_index=1, parent=mma, k_width=4)
# If using shared_memory
a_dot = a_smem.load(dot_op0)  # DotOperandLayout(operand_index=0, parent=mma, k_width=4)
b_dot = b_smem.load(dot_op1)  # DotOperandLayout(operand_index=1, parent=mma, k_width=4)

# Step 3: MFMA
acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc_init)
```

---

## Layout Selection

Extracted from TTGIR:
- `#shared` → `SwizzledSharedLayout`
- `#dot_op<operand_index=0>` → `DotOperandLayout(operand_index=0, ...)`
- `#mma` → `AMDMFMALayout`

### Example
```python
# TTGIR: #shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
shared_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])

# TTGIR: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8]}>
mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    version=3,
    instr_shape=[32, 32, 8],
    warps_per_cta=[2, 2]
)

# TTGIR: #blocked = #ttg.blocked<{{sizePerThread = [8, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
blocked_layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[8, 2],
    threads_per_warp=[8, 8],
    warps_per_cta=[1, 4],
    order=[0, 1]
)
```

---

## Complete Matmul Loop

```python
# Initialize accumulator
accumulator = gl.zeros((BLOCK_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mma_layout)

# Loop K dimension
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Load A and B
    # ... (using memory access pattern)

    # Allocate to shared memory
    a_smem = gl.allocate_shared_memory(a.dtype, [BLOCK_M, BLOCK_K], shared_layout, value=a)
    b_smem = gl.allocate_shared_memory(b.dtype, [BLOCK_K, BLOCK_N], shared2_layout, value=b)

    # Load to register
    a_dot = a_smem.load(dot_op0)
    b_dot = b_smem.load(dot_op1)

    # MFMA
    accumulator = gl.amd.cdna3.mfma(a_dot, b_dot, accumulator)

# Convert and store
c = accumulator.to(gl.float16)
gl.amd.cdna3.buffer_store(c, c_ptr, c_offsets, mask=c_mask)
```

---

## Key Takeaways

1. **All layouts must be extracted from TTGIR**
2. **DotOperandLayout must be used**
3. **MFMA inputs must be float16/bfloat16**
4. **Use float32 for the accumulator**
5. **k_width is extracted from #ttg.dot_op in TTGIR**

## Related

- **Cross-Architecture Reference**: [CDNA4 Matrix Multiplication](../cdna4/matrix_multiply.md) | [Hopper Matrix Multiplication](../../../nvidia/hopper/converter/hopper/matrix_multiply.md) (wgmma)
- **ISA Reference**: [AMD MFMA Programming Guide](../../common/amd-mfma-matrix-cores.md)
- **Downstream Optimization**: [CDNA3 GEMM Optimization](../../gluon/gfx942/pattern_overview.md)
