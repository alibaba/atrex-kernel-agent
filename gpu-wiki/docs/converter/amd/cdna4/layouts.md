## Layouts

Do not fabricate layouts. All layouts must be extracted from TTGIR via `tools/extract_ttgir.py`.

### TTGIR → Gluon Field Name Mapping (⚠️ Do not directly copy TTGIR)

| TTGIR Field | Gluon Python Parameter |
|-----------|------------------|
| `isTransposed` | `transposed` |
| `sizePerThread` | `size_per_thread` |
| `threadsPerWarp` | `threads_per_warp` |
| `warpsPerCTA` | `warps_per_cta` |
| `instrShape` | `instr_shape` |
| `kWidth` | `k_width` |
| `opIdx` | `operand_index` |

---

### BlockedLayout
Distributed layout used for global memory load/store. The product of `threads_per_warp` must be 64, and the product of `warps_per_cta` must equal num_warps.

TTGIR:
```
#blocked = #ttg.blocked<{sizePerThread = [8, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
```
Gluon:
```python
blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 2], threads_per_warp=[8, 8], warps_per_cta=[1, 4], order=[0, 1])
```

1D example (for element-wise kernels):
```python
layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[4], threads_per_warp=[64], warps_per_cta=[4], order=[0])
```

---

### AMDMFMALayout
Accumulator layout used for MMA instructions.

**CDNA4 (`version=4`) new features** (compared to CDNA3 `version=3`):
- New `instr_shape` combinations: `[16, 16, 32]` and `[32, 32, 16]` (in addition to the existing `[16, 16, 16]` and `[32, 32, 8]`)
- New optional parameter `tiles_per_warp`: controls the number of MFMA tiles covered per warp, default `[1, 1]`
- New optional parameter `element_bitwidth`: must be 32 or 64, specifies the operand bit width

> **Note**: The `AMDMFMALayout` class itself remains unchanged. CDNA4 and CDNA3 use the same class, distinguished by the `version` field.

TTGIR:
```
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
```
Gluon:
```python
mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], warps_per_cta=[4, 1], transposed=True)
```

Verification example (verified via Python):
```python
# Valid CDNA4 MFMA layout
mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1])

# With optional parameters
mma64: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=False, warps_per_cta=[2, 2], tiles_per_warp=[1, 1], element_bitwidth=64)
```

---

### SwizzledSharedLayout
Used for shared memory to accelerate matmul computation.

TTGIR:
```
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
```
Gluon:
```python
shared_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
```

---

### SliceLayout
A slice view of another layout along a dimension, used to create a 1D index via `gl.arange` and then `gl.expand_dims` to 2D.

TTGIR:
```
#ttg.slice<{dim = 1, parent = #blocked}>
#ttg.slice<{dim = 0, parent = #mma}>
```
Gluon:
```python
slice_blocked_dim1: gl.constexpr = gl.SliceLayout(1, blocked)
slice_mma_dim0: gl.constexpr = gl.SliceLayout(0, mma)
```

Typical usage — constructing a 2D offset for buffer_load/buffer_store:
```python
row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, target_layout))
col_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, target_layout))
row_2d = gl.expand_dims(row_idx, axis=1)   # [BLOCK_M, 1]
col_2d = gl.expand_dims(col_idx, axis=0)   # [1, BLOCK_N]
```
`target_layout` should match the layout of the load/store target (blocked layout for load, mma layout for store).

### DotOperandLayoutggyAML
Operand layout for MFMA matrix multiplication. Used when loading from shared memory to register.

TTGIR:
```
#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
```
gluon:
```python
dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)
```

`k_width` Common values: bf16/f16 → 4, f32 → 2, f8 → 8. Obtained from the `kWidth` field of TTGIR.

`operand_index=0` is used for the left operand (A) of matrix multiplication, and `operand_index=1` is used for the right operand (B).

---

### Dynamic Layout Calculation (⚠️ Important)

When a kernel uses dynamic `num_warps` (via `@triton.heuristics`) or dynamic tile sizes (via `@triton.autotune`), you cannot use a fixed layout extracted from a single TTGIR. You must dynamically compute the layout at kernel runtime using constexpr parameters.

#### 1D Layout Dynamic Calculation

```python
@gluon.jit
def kernel(X, BLOCK_SIZE: gl.constexpr, NUM_WARPS: gl.constexpr):
    # size_per_thread = BLOCK_SIZE / (threads_per_warp × warps_per_cta)
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK_SIZE // (64 * NUM_WARPS)],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0])
```

**Constraint**: `BLOCK_SIZE ≥ 64 × NUM_WARPS`, otherwise `size_per_thread` would be less than 1. A minimum tile size must be guaranteed in the heuristic.

#### 2D Layout Dynamic Calculation

2D is more complex and must simultaneously satisfy:
- `spt[i] × tpw[i] × wpc[i] = TILE_DIM[i]` (per dimension)
- `tpw[0] × tpw[1] = 64` (AMD warp size)
- `wpc[0] × wpc[1] = NUM_WARPS`
- All values ≥ 1 and powers of 2

**Recommended strategy: allocate all threads and warps to the larger dimension**:

```python
# Assume TILE_N is the larger dimension (guaranteed by heuristic that TILE_N ≥ 64 × NUM_WARPS)
layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[TILE_N // (64 * NUM_WARPS), TILE_K],
    threads_per_warp=[64, 1],
    warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0])
```

#### Common Valid 2D Layout Reference Table

| TILE_DIM0 | TILE_DIM1 | num_warps | tpw | wpc | spt |
|:---------:|:---------:|:---------:|:---:|:---:|:---:|
| 1024 | 1 | 16 | [64,1] | [16,1] | [1,1] |
| 1024 | 8 | 16 | [64,1] | [16,1] | [1,8] |
| 256 | 32 | 4 | [64,1] | [4,1] | [1,32] |
| 4 | 256 | 4 | [1,64] | [1,4] | [4,1] |
| 1 | 8192 | 16 | [1,64] | [1,16] | [1,8] |

See errors 14 and 15 in `common_pitfalls.md` for details.
