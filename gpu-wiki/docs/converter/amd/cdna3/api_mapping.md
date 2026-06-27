# API Reference Table

**Last updated**: 2026-03-03
**Verification Status Legend**: ✅ Verified | ⚠️ Pending Verification | ❌ Falsified

> **⚠️ TTGIR field names ≠ Gluon parameter names**: `isTransposed`→`transposed`, `sizePerThread`→`size_per_thread`, `warpsPerCTA`→`warps_per_cta`, etc. See `layouts.md` for the complete mapping.

> **⚠️ `num_stages`**: Set to 1 in the Gluon launcher. If the Triton source code uses `num_stages > 1`, you need to refer to the TTGIR and manually implement multi-stage pipelining in the Gluon kernel.

---

## Program Control

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.program_id(axis)` | `gl.program_id(axis)` | ✅ | Exactly the same |
| `tl.num_programs(axis)` | `gl.num_programs(axis)` | ✅ | Exactly the same |

---

## Tensor Creation

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.arange(start, end)` | `gl.arange(start, end, layout=...)` | ✅ | **Must specify layout** |
| `tl.zeros(shape, dtype)` | `gl.zeros(shape, dtype, layout=...)` | ✅ | **Must specify layout** |
| `tl.full(shape, value, dtype)` | `gl.full(shape, value, dtype, layout=...)` | ✅ | **Must specify layout** |
| `tl.zeros_like(input)` | `gl.zeros_like(input, layout=...)` | ✅ | Layout optional |

---

## Memory Access

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.load(ptr, mask, other)` | `gl.load(ptr, mask, other)` | ✅ | Simple access |
| `tl.load(ptr, mask, other)` | `gl.amd.cdna3.buffer_load(...)` | ✅ | 2D block access |
| `tl.store(ptr, value, mask)` | `gl.store(ptr, value, mask)` | ✅ | Simple access |
| `tl.store(ptr, value, mask)` | `gl.amd.cdna3.buffer_store(...)` | ✅ | 2D block access |
| `tl.make_block_ptr(...)` | ❌ **Prohibited** | ❌ | Manually computing offsets |

---

## Shared Memory Management

| Operation | Gluon API | Verification Status | Notes |
|------|-----------|---------|------|
| Allocate smem (with initial value) | `gl.allocate_shared_memory(dtype, shape, layout, value=data)` | ✅ | For temporary buffers |
| Pre-allocate smem (without write) | `gl.allocate_shared_memory(dtype, [depth, ...], layout)` | ✅ | For persistent pipeline buffers |
| Index buffer slot | `smem.index(i)` | ✅ | Corresponds to TTGIR `memdesc_index` |
| In-place write to slot | `smem.index(i).store(data)` | ✅ | Corresponds to TTGIR `local_store`, **does not allocate new memory** |
| Read from slot | `smem.index(i).load(layout=dot_op)` | ✅ | Corresponds to TTGIR `local_load` |

---

## Matrix Multiplication

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.dot(a, b, acc)` | `gl.amd.cdna3.mfma(a, b, acc)` | ✅ | cdna3 mfma-specific matrix multiply |

---

## Math Operations

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.exp(x)` | `gl.exp(x)` | ✅ | Same |
| `tl.sin(x)` | `gl.sin(x)` | ✅ | Same |
| `tl.cos(x)` | `gl.cos(x)` | ✅ | Same |
| `tl.sqrt(x)` | `gl.sqrt(x)` | ✅ | Same |
| `tl.rsqrt(x)` | `gl.rsqrt(x)` | ✅ | Same |
| `tl.log(x)` | `gl.log(x)` | ✅ | Same |
| `tl.abs(x)` | `gl.abs(x)` | ✅ | Same |

---

## Type Conversion

| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.cast(x, dtype)` | `gl.cast(x, dtype)` | ✅ | Same |
| `x.to(dtype)` | `x.to(dtype)` | ✅ | Same (tensor method) |

---

## Prohibited APIs

| Triton | Gluon | Reason |
|--------|-------|------|
| `tl.libdevice.*` | ❌ | CUDA-specific |
| `tl.pointer` | ❌ | Use PyTorch tensor type hints instead |

---

## Adding New Mappings

When encountering a new API, add it to this table in the following format:

```markdown
| Triton | Gluon | Verification Status | Notes |
|--------|-------|---------|------|
| `tl.xxx` | `gl.yyy` | ⚠️ Pending Verification | Source: Official documentation |
```

**Verification status update process**:
1. Check official documentation to confirm the mapping relationship
2. Write minimal test code Terminology validation
3. Functional test passed
4. Update status to ✅ Verified
