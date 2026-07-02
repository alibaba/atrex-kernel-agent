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
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc_init)
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

# TTGIR: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 8]}>
mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4,
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
    accumulator = gl.amd.cdna4.mfma(a_dot, b_dot, accumulator)

# Convert and store
c = accumulator.to(gl.float16)
gl.amd.cdna4.buffer_store(c, c_ptr, c_offsets, mask=c_mask)
```

---

## Scaled MFMA (CDNA4 Exclusive)

CDNA4 supports hardware-accelerated matrix multiplication for OCP Microscaling Formats (MX), enabling scaled matrix multiplication directly on low-precision formats.

### Import

```python
from triton.experimental.gluon.language.amd.cdna4 import get_mfma_scale_layout
```

### Supported Formats

| Format | Description |
|------|------|
| `e2m1` | 4-bit floating point (FP4), 2-bit exponent, 1-bit mantissa |
| `e4m3` | 8-bit floating point (FP8), 4-bit exponent, 3-bit mantissa |
| `e5m2` | 8-bit floating point (FP8), 5-bit exponent, 2-bit mantissa |

### Basic Pattern

```python
# Step 1: Layout definition
mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4,
    instr_shape=[32, 32, 16],  # instrShape related to format
    warps_per_cta=[2, 2]
)
dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=8)
dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=8)

# Step 2: Calculate scale layout
a_scale_layout: gl.constexpr = get_mfma_scale_layout(dot_op0, [BLOCK_M, BLOCK_K // group_size])
b_scale_layout: gl.constexpr = get_mfma_scale_layout(dot_op1, [BLOCK_K // group_size, BLOCK_N])

# Step 3: Load data and scale
a_dot = a_smem.load(dot_op0)
b_dot = b_smem.load(dot_op1)
a_scale = gl.amd.cdna4.buffer_load(ptr=scale_a_ptr, offsets=scale_a_off, mask=scale_a_mask, other=0.0)
b_scale = gl.amd.cdna4.buffer_load(ptr=scale_b_ptr, offsets=scale_b_off, mask=scale_b_mask, other=0.0)

# Step 4: Convert scale to correct layout
a_scale = gl.convert_layout(a_scale, layout=a_scale_layout)
b_scale = gl.convert_layout(b_scale, layout=b_scale_layout)

# Step 5: Scaled MFMA
acc = gl.amd.cdna4.mfma_scaled(
    a_dot,          # operand A (low precision)
    a_scale,        # A's scale tensor (or None)
    "e4m3",         # A's format
    b_dot,          # operand B (low precision)
    b_scale,        # B's scale tensor (or None)
    "e4m3",         # B's format
    acc             # float32 accumulator
)
```

### get_mfma_scale_layout Usage

```python
# Calculate scale tensor's distributed layout based on DotOperandLayout and scale shape
# Scale shape is typically [M, K // group_size] or [K // group_size, N]
scale_layout = get_mfma_scale_layout(dot_operand_layout, scale_shape)

# Parameters:
#   dot_operand_layout: DotOperandLayout — corresponding operand's layout
#   scale_shape: List[int] — scale tensor's shape
# Returns:
#   DistributedLinearLayout — scale's distributed layout
```

### ⚠️ Important Notes

1. **`mfma_scaled` is only available on CDNA4 (gfx950) hardware**
2. **a_scale / b_scale can be None** — indicates no scaling for the corresponding operand
3. **The format string must be one of `"e2m1"`, `"e4m3"`, `"e5m2"`**
4. **The accumulator still uses float32**
5. **The scale layout must be computed via `get_mfma_scale_layout`** — do not construct it manually

---

## Key Points

1. **All Layouts must be extracted from TTGIR**
2. **DotOperandLayout must be used**
3. **MFMA inputs must be float16/bfloat16** (standard mfma); low-precision formats use mfma_scaled
4. **Accumulator uses float32**
5. **k_width is extracted from TTGIR's #ttg.dot_op**
6. **AMDMFMALayout version=4** (CDNA4), distinct from version=3 (CDNA3)
7. **Scaled MFMA is a CDNA4-exclusive capability** — neither CDNA3 nor Hopper supports it


## Related

- [API Mapping Reference (CDNA4 / MI355X)](api_mapping.md)
- [Common Errors & Solutions (CDNA4 / gfx950)](common_pitfalls.md)
- [Triton → Gluon Conversion Guide (AMD CDNA4)](conversion-guide.md)
- [CDNA4 Layout Mapping (Triton → Gluon)](layouts.md)
- [Memory Access Patterns](memory_access.md)
- [Matrix Multiplication Patterns](../cdna3/matrix_multiply.md)
- [Matrix Multiplication Patterns (NVIDIA Hopper wgmma)](../../../nvidia/hopper/converter/hopper/matrix_multiply.md)
- [AMD MFMA Matrix Core Programming Guide](../../common/amd-mfma-matrix-cores.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
