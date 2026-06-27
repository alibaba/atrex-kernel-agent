## Layouts

Do not fabricate Layouts. All Layouts must be extracted from TTGIR via `tools/extract_ttgir.py`.

### TTGIR → Gluon Field Name Mapping (⚠️ Do not copy TTGIR directly)

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
Distributed layout for global memory load/store. The product of `threads_per_warp` must be 64, and the product of `warps_per_cta` must equal num_warps.

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
Accumulator layout for MMA instructions.

TTGIR:
```
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
```
Gluon (⚠️ Note `isTransposed` → `transposed`, all parameters are required):
```python
mma: gl.constexpr = gl.amd.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], warps_per_cta=[4, 1], transposed=True)
```

---

### SwizzledSharedLayout
Used for shared memory, accelerating matmul computation.

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
A slice view of another layout along a specified dimension, used to `gl.arange` create a 1D index and then `gl.expand_dims` to 2D.

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
`target_layout` should match the layout of the load/store target (use blocked layout for loads, mma layout for stores).

---

### DotOperandLayout
Operand layout for MFMA matrix multiplication. Used when loading from shared memory to registers.

TTGIR:
```
#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
```
Gluon:
```python
dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)
```

Common values for `k_width`: bf16/f16 → 4, f32 → 2, f8 → 8. Obtain from the `kWidth` field in TTGIR.

`operand_index=0` is used for the left operand (A) of matrix multiplication, `operand_index=1` for the right operand (B).

---

### Dynamic Layout Computation (⚠️ Important)

When a kernel uses dynamic `num_warps` (via `@triton.heuristics`) or dynamic tile sizes (via `@triton.autotune`), you cannot use a fixed Layout extracted from a single TTGIR dump. Layouts must be computed dynamically inside the kernel using constexpr parameters.

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

**Constraint**: `BLOCK_SIZE ≥ 64 × NUM_WARPS`, otherwise `size_per_thread` will be less than 1. The heuristic must guarantee the minimum tile size.

#### 2D Layout Dynamic Calculation

2D is more complex impression and must simultaneously satisfy:
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

For details, see errors 14 and 15 in `common_pitfalls.md`.
