# Hopper Layout Mapping (Triton → Gluon)

Mapping Triton tensor layouts to Gluon layouts on NVIDIA Hopper (sm_90).


**Last updated**: 2026-07-01

## Layouts (NVIDIA Hopper)

**Target Architecture**: NVIDIA Hopper (sm_90)

Do not fabricate layouts. All layouts must be extracted from TTGIR via `tools/extract_ttgir.py`.

### TTGIR → Gluon Field Name Mapping (⚠️ Do not copy TTGIR directly)

| TTGIR Field | Gluon Python Parameter |
|-----------|------------------|
| `versionMajor`, `versionMinor` | `version=[major, minor]` |
| `sizePerThread` | `size_per_thread` |
| `threadsPerWarp` | `threads_per_warp` |
| `warpsPerCTA` | `warps_per_cta` |
| `instrShape` | `instr_shape` |
| `swizzlingByteWidth` | `swizzle_byte_width` |
| `elementBitWidth` | `element_bitwidth` |
| `transposed` | `transposed` |

### ⚠️ Key Differences Between Hopper and AMD

| Property | NVIDIA Hopper (sm_90) | AMD CDNA3 (MI300) |
|------|----------------------|-------------------|
| `threads_per_warp` product | **32** | 64 |
| MMA Accumulator Layout | `NVMMADistributedLayout` | `AMDMFMALayout` |
| MMA Operand Source | **shared memory** (`NVMMASharedLayout`) | register (`DotOperandLayout`) |
| Shared Memory Layout | `NVMMASharedLayout` or `SwizzledSharedLayout` | `SwizzledSharedLayout` |
| TTGIR Target Identifier | `cuda:90` | `hip:gfx942` |

---

### BlockedLayout
Distributed layout used for global memory load/store. The product of `threads_per_warp` must be **32** (NVIDIA warp size).

TTGIR:
```
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
```
Gluon (note: 8 × 4 = 32):
```python
blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 1], threads_per_warp=[8, 4], warps_per_cta=[1, 4], order=[0, 1])
```

More Hopper BlockedLayout examples:
```python
# [BT, 64] dataload ( w matrix)
blocked3: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0])

# [64, BV] datastore
blocked2: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[16, 2], warps_per_cta=[4, 1], order=[1, 0])

# [64, BV] final state store
blocked1: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 4], threads_per_warp=[8, 4], warps_per_cta=[4, 1], order=[1, 0])
```

---

### NVMMADistributedLayout (Hopper MMA Accumulator)
The **accumulator layout** used for wgmma instructions. Equivalent to AMD's `AMDMFMALayout`.

TTGIR:
```
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
```
Gluon (⚠️ `versionMajor/versionMinor` → `version=[major, minor]`):
```python
mma: gl.constexpr = gl.NVMMADistributedLayout(
    version=[3, 0],
    warps_per_cta=[4, 1],
    instr_shape=[16, 16, 16],
)
```

**Key Differences**:
- AMD `version` is a scalar (e.g., `version=3`), while Hopper `version` is a list `[major, minor]`
- wgmma operands are read directly from shared memory, so no `DotOperandLayout` is needed

---

### NVMMASharedLayout (Hopper wgmma Operands)
The **shared memory operand layout** used for wgmma instructions. wgmma requires both operands to be in shared memory and must use this layout.

TTGIR:
```
# Non-transposed (LHS operand: w[BT, 64])
shared_w: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128, element_bitwidth=16, transposed=False)

# Transposed (RHS operand: k[64, BT] requires transposed read)
shared_k: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128, element_bitwidth=16, transposed=True)

# Small swizzle (small matrix RHS: v_new[BT, BV] where BV=16)
shared_v: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=32, element_bitwidth=16, transposed=False)
```

Gluon:
```python
# Non-transposed (LHS operand: w[BT, 64])
shared_w: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128, element_bitwidth=16, transposed=False)

# Transposed (RHS operand: k[64, BT] requires transposed read)
shared_k: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128, element_bitwidth=16, transposed=True)

# Small swizzle (small matrix RHS: v_new[BT, BV] where BV=16)
shared_v: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=32, element_bitwidth=16, transposed=False)
```

**Parameter Selection Rules**:

| Parameter | Selection Basis |
|------|---------|
| `swizzle_byte_width` | `min(128, TILE_DIM × element_bytes)`. bf16 example: 16 cols→32, 32 cols→64, ≥64 cols→128. Extracted from TTGIR |
| `element_bitwidth` | bf16/f16 → `16`; f32 → `32`; f8 → `8` |
| `transposed` | Row-major → `False`; column-major read required → `True`. Check the `transposed` field in TTGIR |

---

### SwizzledSharedLayout (General)
Used for shared memory in non-wgmma scenarios. For example, use `gl.load` to smem and then temporary storage via `allocate_shared_memory(value=...)`.

TTGIR:
```
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
```
Gluon:
```python
shared1: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 2, order=[1, 0])
```

---

### SliceLayout
A sliced view of another layout along a certain dimension, used to create a 1D index via `gl.arange` and then `gl.expand_dims` to 2D.

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

Typical usage — constructing a 2D offset:
```python
row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, target_layout))
col_idx = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, target_layout))
row_2d = gl.expand_dims(row_idx, axis=1)   # [BLOCK_M, 1]
col_2d = gl.expand_dims(col_idx, axis=0)   # [1, BLOCK_N]
```
`target_layout` should match the layout of the load/store target.

---
```python
# AMD (requires DotOperandLayout):
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)
acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc)

# Hopper (no DotOperandLayout needed):
# Operands are directly smem descriptors
fence_async_shared()
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```
---

### Hopper Legal 2D BlockedLayout Reference Table

`threads_per_warp` product = 32 (NVIDIA warp size)

| TILE_DIM0 | TILE_DIM1 | num_warps | tpw | wpc | spt |
|:---------:|:---------:|:---------:|:---:|:---:|:---:|
| 64 | 64 | 4 | [8, 4] | [1, 4] | [8, 1] |
| 64 | 16 | 4 | [16, 2] | [4, 1] | [1, 8] |
| 64 | 64 | 4 | [4, 8] | [4, 1] | [1, 8] |
| 64 | 4 | 4 | [8, 4] | [4, 1] | [1, 1] |

**Constraint Formula**:
```
spt[i] × tpw[i] × wpc[i] = TILE_DIM[i] (dimension)
tpw[0] × tpw[1] = 32                       (Hopper warp size)
wpc[0] × wpc[1] = num_warps
 ≥ 1 2
```


## Related

- [API Mapping Reference (NVIDIA Hopper)](api_mapping.md)
- [Common Errors and Solutions (NVIDIA Hopper)](common_pitfalls.md)
- [Triton → Gluon Conversion Guide (NVIDIA Hopper)](conversion-guide.md)
- [Matrix Multiplication Patterns (NVIDIA Hopper wgmma)](matrix_multiply.md)
- [Memory Access Patterns (NVIDIA Hopper)](memory_access.md)
- [CDNA3 Layout Mapping (Triton → Gluon)](../../../../amd/converter/cdna3/layouts.md)
- [CDNA4 Layout Mapping (Triton → Gluon)](../../../../amd/converter/cdna4/layouts.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../common/triton/triton-tile-ir-beyond-simt.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../../RELATIONS.md)
