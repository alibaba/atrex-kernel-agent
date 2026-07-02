# API Mapping Reference (CDNA4 / MI355X)

**Last updated**: 2026-03-28

> **⚠️ TTGIR Field Names ≠ Gluon Parameter Names**: `isTransposed`→`transposed`, `sizePerThread`→`size_per_thread`, `warpsPerCTA`→`warps_per_cta`, etc. See `layouts.md` for the full mapping.

> **⚠️ `num_stages`**: Set to 1 in the Gluon launcher. If the Triton source uses `num_stages > 1`, you should refer to TTGIR and manually implement multi-stage pipelining in the Gluon kernel. CDNA4 can use hardware async_copy GET for pipelining.

> **⚠️ Inheritance**: CDNA4 inherits all CDNA3 APIs via `from ..cdna3 import *`. `buffer_load`, `buffer_store`, and `mfma` are the exact same function objects. `buffer_atomic_*` is redefined in CDNA4 with additional support for `bf16`'s `fadd` operations. CDNA4-specific APIs: `mfma_scaled`, `async_copy`, `get_mfma_scale_layout`.

---

## Program Control

| Triton | Gluon | Notes |
|--------|-------|-------|
| `tl.program_id(axis)` | `gl.program_id(axis)` | Identical |
| `tl.num_programs(axis)` | `gl.num_programs(axis)` | Identical |

---

## Tensor Creation

| Triton | Gluon | Notes |
|--------|-------|-------|
| `tl.arange(start, end)` | `gl.arange(start, end, layout=...)` | **Must specify layout** |
| `tl.zeros(shape, dtype)` | `gl.zeros(shape, dtype, layout=...)` | **Must specify layout** |
| `tl.full(shape, value, dtype)` | `gl.full(shape, value, dtype, layout=...)` | **Must specify layout** |
| `tl.zeros_like(input)` | `gl.zeros_like(input, layout=...)` | Layout optional |

---

## Memory Access

| Triton | Gluon | Notes |
|--------|-------|-------|
| `tl.load(ptr, mask, other)` | `gl.load(ptr, mask, other)` | Scalar/simple access |
| `tl.load(ptr, mask, other)` | `gl.amd.cdna4.buffer_load(ptr, offsets, mask, other, cache)` | 2D block access, inherited from CDNA3 |
| `tl.store(ptr, value, mask)` | `gl.store(ptr, value, mask)` | Scalar/simple access |
| `tl.store(ptr, value, mask)` | `gl.amd.cdna4.buffer_store(stored_value, ptr, offsets, mask, cache)` | 2D block access, inherited from CDNA3 |
| `tl.make_block_ptr(...)` | ❌ **Prohibited** | ❌ | Manually compute offset |

### buffer_store Signature

```python
gl.amd.cdna4.buffer_load(
    ptr,        # pointer to scalar: global memory scalar base pointer
    offsets,    # tensor: int32/uint32 offset tensor
    mask=None,  # tensor, optional: mask
    other=None, # tensor or scalar, optional: default value for masked elements
    cache=None  # str, optional: cache modifier
)
```

### buffer_load Signature

```python
gl.amd.cdna4.buffer_store(
    stored_value,  # tensor: tensor to store
    ptr,           # pointer to scalar: global memory scalar base pointer
    offsets,       # tensor: int32/uint32 offset tensor
    mask=None,     # tensor, optional: mask
    cache=None     # str, optional: cache modifier
)
```

---

## Shared Memory Management

| Operation | Gluon API | Notes |
|-----------|-----------|-------|
| Allocate SMEM | `gl.allocate_shared_memory(dtype, shape, layout, value=data)` | For temporary buffers |
| Pre-allocate SMEM (no write) | `gl.allocate_shared_memory(dtype, [depth, ...], layout)` | For persistent buffers |
| Index into buffer slot | `smem.index(i)` | Corresponds to TTGIR `memdesc_index` |
| In-place write to slot | `smem.index(i).store(data)` | Corresponds to TTGIR `local_store`, **does not allocate new memory** |
| Read from slot | `smem.index(i).load(layout=dot_op)` | Corresponds to TTGIR `local_load` |

---

## Matrix Multiplication

| Triton | Gluon | Notes |
|--------|-------|-------|
| `tl.dot(a, b, acc)` | `gl.amd.cdna4.mfma(a, b, acc)` | Inherited from CDNA3, standard matrix multiply |
| *(No Triton equivalent)* | `gl.amd.cdna4.mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)` | **CDNA4 only**, OCP microscaling format matrix multiply |### mfma Signature (Inherited from CDNA3)

```python
gl.amd.cdna4.mfma(
    a,    # tensor: operand A, layout must be DotOperandLayout
    b,    # tensor: operand B, layout must be DotOperandLayout
    acc   # tensor: accumulator, layout must be AMDMFMALayout
)
# Computation: a @ b + acc
```

### mfma_scaled Signature (CDNA4 Exclusive)

```python
gl.amd.cdna4.mfma_scaled(
    a,         # tensor: operand A, layout must be DotOperandLayout
    a_scale,   # Optional[tensor]: A's scaling factor, layout obtained from get_mfma_scale_layout
    a_format,  # str: A's format, allowed values: "e2m1", "e4m3", "e5m2"
    b,         # tensor: operand B, layout must be DotOperandLayout
    b_scale,   # Optional[tensor]: B's scaling factor, layout obtained from get_mfma_scale_layout
    b_format,  # str: B's format, allowed values: "e2m1", "e4m3", "e5m2"
    acc        # tensor: accumulator, layout must be AMDMFMALayout
)
# Computation: (a * a_scale) @ (b * b_scale) + acc
# Uses OCP Microscaling Formats (MX) specification
```

**Layout Constraints**:
- `a.type.layout` must be `DotOperandLayout`, and its `parent` must match the `AMDMFMALayout` of `acc`
- `b.type.layout` must be `DotOperandLayout`, and its `parent` must match the `AMDMFMALayout` of `acc`
- `a_format` and `b_format` only support: `"e2m1"`, `"e4m3"`, `"e5m2"`

---

## Async Copy (CDNA4 Exclusive)

> **CDNA4 Exclusive Feature**: Hardware DMA asynchronous copy that moves data directly from global memory to shared memory, bypassing registers. This is a key capability of the CDNA4 pipeline.

| Operation | Gluon API | Notes |
|------|-----------|------|
| Async buffer load to smem | `async_copy.buffer_load_to_shared(dest, ptr, offsets, mask, other, cache_modifier)` | **Recommended**, low register pressure, hardware OOB mask |
| Async global load to smem | `async_copy.global_load_to_shared(dest, ptr, mask, other, cache_modifier)` | High register pressure, no hardware OOB mask |
| Submit async operation group | `async_copy.commit_group()` | Submits currently outstanding async operations |
| Wait for async operations to complete | `async_copy.wait_group(num_outstanding=0)` | Blocks until the number of outstanding groups ≤ num_outstanding |
| Relaxed load from smem | `async_copy.load_shared_relaxed(smem, layout)` | Prevents the compiler from inserting unnecessary waits |

### buffer_load_to_shared Signature (Recommended)

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy

async_copy.buffer_load_to_shared(
    dest,             # shared_memory_descriptor: destination shared memory descriptor
    ptr,              # pointer to scalar: global memory scalar base pointer
    offsets,          # tensor: int32/uint32 offset tensor
    mask=None,        # tensor, optional: mask, supports hardware OOB mask
    other=None,       # tensor or scalar, optional: default value for masked elements
    cache_modifier="" # str: cache modifier, default ""
)
```

**Advantages**: Uses a scalar base pointer + 32-bit offset, resulting in low register pressureega and supporting hardware out-of-bounds masking. Prefer this API.

**Hardware Constraints**:
- The size_per_thread × bits_per_element of the `offsets` layout must be 128 or 32 (128 recommended for best performance)
- Writes to `dest` must be coalesced
- If `dest` has swizzle, it must only swizzle within warp boundaries

### global_load_to_shared Signature

```python
async_copy.global_load_to_shared(
    dest,             # shared_memory_descriptor: destination shared memory descriptor
    ptr,              # pointer tensor: pointer tensor pointing to global memory (non-scalar)
    mask=None,        # tensor, optional: mask
    other=None,       # tensor or scalar, optional: default value for masked elements
    cache_modifier="" # str: cache modifier, default ""
)
```

**Note**: Uses pointer tensors (64-bit indexing), resulting in high register pressure and no hardware out-of-bounds mask. Use only when a 64-bit addressing range is required.

**Hardware Constraints**:
- The size_per_thread × bits_per_element of the `ptr` layout must be 128 or 32 (128 recommended)
- The `ptr` layout must be `BlockedLayout` or `SliceLayout`
- Writes to `dest` must be coalesced
- If `dest` has swizzle, it must only swizzle within warp boundaries
- Interleaving with `ttgl.load/store` or `buffer_load/store` will degrade performance

### commit_group signature

```python
async_copy.commit_group()
# No parameters. Submit all currently outstanding async operations as a group.
```

### wait_group signature

```python
async_copy.wait_group(
    num_outstanding=0  # int: allowed number of outstanding groups, default 0 (wait for all to complete)
)
# Blocks until the number of outstanding commit groups ≤ num_outstanding
# Note: Unsubmitted async operations will also be waited for, even if num_outstanding=0
```

### load_shared_relaxed signature

```python
async_copy.load_shared_relaxed(
    smem,    # shared_memory_descriptor: shared memory descriptor
    layout   # DistributedLayout: target tensor layout
)
# Returns: tensor, data loaded from shared memory
# Hints compiler to skip unnecessary waits, for data already synchronized via wait_group
```

### Typical usage patterns

```python
# 1. Pre-allocate shared memory
smem = gl.allocate_shared_memory(dtype, [num_stages, M, K], shared_layout)

# 2. Asynchronously load to shared memory
async_copy.buffer_load_to_shared(smem.index(stage), ptr, offsets, mask)
async_copy.commit_group()

# 3. Wait for completion
async_copy.wait_group(num_outstanding=0)

# 4. Read from shared memory (relaxed mode, avoid redundant waits)
data = async_copy.load_shared_relaxed(smem.index(stage), dot_op_layout)
```

---

## Scale Layout (CDNA4 only)

| Operation | Gluon API | Notes |
|------|-----------|------|
| Get scale factor layout | `gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape)` | Used with `mfma_scaled` |

### get_mfma_scale_layout signature

```python
gl.amd.cdna4.get_mfma_scale_layout(
    dot_operand_layout,  # DotOperandLayout: dot operand layout
    shape                # List[int]: scaling tensor shape
)
# Returns: DistributedLinearLayout, scaling factor layout
# Usage: scale_layout = gl.amd.cdna4.get_mfma_scale_layout(a_dot_layout, scale_shape)
#        a_scale = gl.load(scale_ptr, layout=scale_layout)

# Internal logic:
#   - Extract operand_index from dot_operand_layout
#   - Extract instr_shape, tiles_per_warp, warps_per_cta from its parent (AMDMFMALayout)
```

**Constraint**: `dot_operand_layout.parent` must be an instance of `AMDMFMALayout`.

---

## Math Operations

| Triton | Gluon | Notes |
|--------|-------|------|
| `tl.exp(x)` | `gl.exp(x)` | Same |
| `tl.sin(x)` | `gl.sin(x)` | Same |
| `tl.cos(x)` | `gl.cos(x)` | Same |
| `tl.sqrt(x)` | `gl.sqrt(x)` | Same |
| `tl.rsqrt(x)` | `gl.rsqrt(x)` | Same |
| `tl.log(x)` | `gl.log(x)` | Same |
| `tl.abs(x)` | `gl.abs(x)` | Same |

---

## Type Conversion

| Triton | Gluon | Notes |
|--------|-------|------|
| `tl.cast(x, dtype)` | `gl.cast(x, dtype)` | Same |
| `x.to(dtype)` | `x.to(dtype)` | Same (tensor method) |

---

## Atomic Operations

| Triton | Gluon | Notes |
|--------|-------|------|
| `tl.atomic_max(...)` | `gl.amd.cdna4.buffer_atomic_max(ptr, offsets, value, mask, sem, scope)` | Returns pre-op value |
| `tl.atomic_min(...)` | `gl.amd.cdna4.buffer_atomic_min(ptr, offsets, value, mask, sem, scope)` | Returns pre-op value |
| `tl.atomic_add(...)` | `gl.amd.cdna4.buffer_atomic_add(ptr, offsets, value, mask, sem, scope)` | **Additionally supports bf16 fadd** (vs cdna3) |
| `tl.atomic_and(...)` | `gl.amd.cdna4.buffer_atomic_and(ptr, offsets, value, mask, sem, scope)` | int32/int64 only |
| `tl.atomic_or(...)` | `gl.amd.cdna4.buffer_atomic_or(ptr, offsets, value, mask, sem, scope)` | int32/int64 only |
| `tl.atomic_xor(...)` | `gl.amd.cdna4.buffer_atomic_xor(ptr, offsets, value, mask, sem, scope)` | int32/int64 only |
| `tl.atomic_xchg(...)` | `gl.amd.cdna4.buffer_atomic_xchg(ptr, offsets, value, mask, sem, scope)` | int32/int64 only |### `buffer_atomic` Common Signature

```python
gl.amd.cdna4.buffer_atomic_<op>(
    ptr,        # pointer to scalar: global memory scalar base pointer
    offsets,    # tensor: int32/uint32 offset tensor
    value,      # tensor: operand
    mask=None,  # tensor, optional: mask, elements with mask[i]==0 do not execute atomic operation
    sem=None,   # str, optional: memory semantics, default acq_rel
    scope=None  # str, optional: memory synchronization scope, default "gpu" (corresponds to "agent" on AMDGPU)
)
# Returns: tensor, value in global memory before operation (pre-op value)
```

**Supported Data Types**: `float16`, `float32`, `bfloat16`, `float64`, `int32`, `int64`, `uint32`, `uint64`

**CDNA4 vs CDNA3 Differences**: CDNA4's `buffer_atomic_add` additionally supports `bfloat16`'s `fadd` operation.

---

## Disallowed APIs

| Triton | Gluon | Reason |
|--------|-------|--------|
| `tl.libdevice.*` | ❌ | CUDA-specific |
| `tl.pointer` | ❌ | Use PyTorch tensor type hints |

---

## Adding a New Mapping

When encountering a new API, add it to this table in the following format:

```markdown
| Triton | Gluon | Verification Status | Notes |
|--------|-------|----------------|------|
| `tl.xxx` | `gl.yyy` | ⚠️ Pending hardware verification | Source: Official documentation |
```

**Verification Status Update Process**:
1. Check official documentation to confirm the mapping relationship
2. Write minimal test code tem prove the verification
3. Pass the functional test
4. Update to ✅ Verified after verification on CDNA4 hardware


## Related

- [Common Errors & Solutions (CDNA4 / gfx950)](common_pitfalls.md)
- [Triton → Gluon Conversion Guide (AMD CDNA4)](conversion-guide.md)
- [CDNA4 Layout Mapping (Triton → Gluon)](layouts.md)
- [Matrix Multiplication Patterns](matrix_multiply.md)
- [Memory Access Patterns](memory_access.md)
- [API Reference Table](../cdna3/api_mapping.md)
- [API Mapping Reference (NVIDIA Hopper)](../../../nvidia/hopper/converter/hopper/api_mapping.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
