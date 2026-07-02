# CDNA4 Pipeline Conversion (Triton → Gluon)

Converting Triton software-pipelined loops (num_stages > 1) to Gluon on AMD CDNA4 (gfx950), including hardware async-copy DMA.


**Last updated**: 2026-07-01

## Pipeline Conversion Mode (AMD CDNA4, num_stages > 1)

### Background

In Triton, when `num_stages > 1`, the compiler automatically generates a software pipeline in TTGIR. Gluon does not automatically generate a pipeline and must be implemented manually.

**CDNA4 Pipeline Core**: Use **hardware async copy (DMA)** Fake-asynchronous transfer from global to shared memory, **bypassing registers**. This is conceptually similar to Hopper's CP_ASYNC but uses AMD's buffer descriptor model. It is fundamentally different from CDNA3's manual `buffer_load` + `smem.store` pipeline.

### Imports

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy
```

---

### CDNA4 async_copy API

| API | Function | Description |
|-----|----------|------|
| `async_copy.buffer_load_to_shared(dest, ptr, offsets, mask, other)` | DMA: global → shared (scalar base pointer + offset) | ⭐ **Preferred**, low register pressure, supports hardware out-of-bounds protection |
| `async_copy.global_load_to_shared(dest, ptr, mask, other)` | DMA: global → shared (pointer tensor) | Alternative, high register pressure, does not support hardware out-of-bounds protection |
| `async_copy.commit_group()` | Commit a batch of async copy operations | Call after all async copy calls |
| `async_copy.wait_group(num_outstanding=N)` | Wait until ≤ N copy transactions remain outstanding | Must be called before consuming smem data |
| `async_copy.load_shared_relaxed(smem, layout)` | Load from smem, avoiding redundant sync barriers | Replaces `smem.load(layout)` to reduce overhead |

---

### ⚠️ Performance Critical: async_copy vs Two-Step Transfer

| Transfer Method | Path | Description |
|-----------------|------|------|
| `async_copy.buffer_load_to_shared` | global → shared (DMA) | ⭐ Hardware DMA, bypasses registers |
| `gl.amd.cdna4.buffer_load` + `smem.store` | global → register → shared | ❌ Two-step transfer, poor performance |

**async_copy must be used**. Two-step transfer cannot meet the performance requirements of `benchmark.py`.

---

### ✅ Correct Approach: async_copy + Double-Buffer Pipeline

```python
# 1. Pre-allocate persistent shared memory (double-buffered, depth=2)
smem_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, BLOCK_K], shared_layout)
smem_b = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_K, BLOCK_N], shared_layout)

# 2. PROLOGUE: async_copy prefetch iter 0 to slot 0
a_offsets_i32 = gl.cast(a_offsets, gl.int32)
b_offsets_i32 = gl.cast(b_offsets, gl.int32)

async_copy.buffer_load_to_shared(
    smem_a.index(0),
    a_ptr,
    a_offsets_i32,
    mask=a_mask,
    other=0.0
)
async_copy.buffer_load_to_shared(
    smem_b.index(0),
    b_ptr,
    b_offsets_i32,
    mask=b_mask,
    other=0.0
)

# Submit and wait for prologue to complete
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# ============================================================
# 3. MAIN LOOP: compute slot (i-1)%2 + prefetch to slot i%2
# ============================================================
for i in range(1, loop_n):
    # --- Launch next batch of prefetch (DMA) ---
    next_a_offsets_i32 = gl.cast(next_a_offsets, gl.int32)
    next_b_offsets_i32 = gl.cast(next_b_offsets, gl.int32)

    async_copy.buffer_load_to_shared(
        smem_a.index(i % 2),
        a_ptr,
        next_a_offsets_i32,
        mask=next_a_mask,
        other=0.0
    )
    async_copy.buffer_load_to_shared(
        smem_b.index(i % 2),
        b_ptr,
        next_b_offsets_i32,
        mask=next_b_mask,
        other=0.0
    )
    async_copy.commit_group()
# --- Load from previous slot to registers and compute ---
# Method 1: load_shared_relaxed (recommended — avoid redundant sync barriers)
a_dot = async_copy.load_shared_relaxed(smem_a.index((i - 1) % 2), layout=dot_op0)
b_dot = async_copy.load_shared_relaxed(smem_b.index((i - 1) % 2), layout=dot_op1)
# Method 2: Regular smem load (also works)
# a_dot = smem_a.index((i - 1) % 2).load(layout=dot_op0)
# b_dot = smem_b.index((i - 1) % 2).load(layout=dot_op1)

acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)

# --- Wait for prefetch to complete ---
async_copy.wait_group(num_outstanding=0)

# 4. EPILOGUE: Handle last slot
a_dot = smem_a.index((loop_n - 1) % 2).load(layout=dot_op0)
b_dot = smem_b.index((loop_n - 1) % 2).load(layout=dot_op1)
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
```---

### ❌ Wrong Approach: buffer_load + smem.store (Two-Step Transfer)

```python
# ❌ Wrong: Two-step transfer (global → register → shared)
data = gl.amd.cdna4.buffer_load(ptr=base, offsets=offsets_i32, mask=mask, other=0.0)
smem.index(slot).store(data)

# ✅ Correct: async_copy DMA (global → shared, bypassing registers)
async_copy.buffer_load_to_shared(smem.index(slot), base, offsets_i32, mask=mask, other=0.0)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
```

---

### buffer_load_to_shared vs global_load_to_shared

| Method | Pointer Model | Register Pressure | Hardware Out-of-Bounds Protection | Recommendation |
|------|---------|-----------|-------------|--------|
| `buffer_load_to_shared(dest, ptr, offsets, ...)` | Scalar base pointer + int32 offset tensor | Low | ✅ Supported | ⭐ **Preferred** |
| `global_load_to_shared(dest, ptr_tensor, ...)` | 64-bit pointer tensor | High | ❌ Not supported | Only use when 64-bit addressing is required |

**Prefer `buffer_load_to_shared`** for better performance and lower register pressure.

---

### Two Types of Shared Memory Usage

| Type | Allocation Method | Data Source | Scenario |
|------|---------|---------|------|
| **Persistent buffer** (reused across iterations) | `allocate_shared_memory(dtype, [depth, ...], layout)` without `value=` | `async_copy.buffer_load_to_shared` | Data requiring pipeline such as a, b |
| **Temporary buffer** (single use) | `allocate_shared_memory(dtype, shape, layout, value=data)` with `value=` | Written from registers | Data immediately consumed such as intermediate results → mfma |

---

### Shared Memory Budget

MI355X (CDNA4) hardware limit: **163840 bytes** (160KB)

Common buffer sizes:
- `[2, 64, 64] × bf16` = 16384 bytes (16KB)
- `[2, 128, 128] × bf16` = 65536 bytes (64KB)
- `[2, 64, 16] × bf16` = 4096 bytes (4KB)
- `[64, 16] × bf16` = 2048 bytes (2KB)

Compared to CDNA3 (64KB), CDNA4's 160KB shared memory allows for larger tiles or deeper pipelines (num_stages=3+).

---

### Hardware Constraints

1. **`offsets` layout's `size_per_thread * bits_per_element` must be 128 or 32**
   - 128 bits is recommended for optimal performance
2. **Writes to `dest` (shared memory) must be coalesced**
3. **If `dest` has swizzle, it can only swizzle within warp boundaries**
4. **async_copy operations complete in order with normal `buffer_load`/`buffer_store`** — Interleaving them forces serialization, severely degrading performance

---

### ⚠️ Pipeline Notes

1. **Only use async_copy for data going into shared memory**. Data going directly into registers still uses `gl.amd.cdna4.buffer_load`.
2. **`commit_group()` must be called after all `buffer_load_to_shared` / `global_load_to_shared`**.
3. **`wait_group(num_outstanding=0)` must be called before consuming smem data**.
4. **`load_shared_relaxed` can replace `smem.load`** — Hints the compiler to skip unnecessary synchronization barriers, reducing overhead.
5. **Do not interleave normal buffer_load/store between async_copy operations** — They share the same execution pipeline, and interleaving causes all operations to execute sequentially.

---

### ⚠️ Pipelining Is Mandatory, Not an Optional Optimization

If the Triton source code uses `num_stages > 1`, you **must** implement pipelining in Gluon. Skipping pipelining leads to:
- Performance regression (all global memory loads become synchronous blocking)
- `benchmark.py` will definitely fail (ratio > 1.15)

### Checklist for Determining Whether Pipelining Is Needed

1. Check the `num_stages` parameter in the Triton wrapper → If > 1, pipelining is needed
2. Check if TTGIR contains `ttg.local_alloc : () -> !ttg.memdesc<Nx...>` → N > 0 indicates persistent smem is present
3. Check if TTGIR contains the `ttg.memdesc_index` + `ttg.local_store` pattern → Pipeline writes
4. Check if TTGIR contains the `amd.pipeliner_part = "prologue"` annotation → Compiler-marked prologue data

## Related

- **Cross-Architecture Comparison**: [CDNA3 Pipeline](../cdna3/pipeline.md) (Pure software) | [Hopper Pipeline](../../../nvidia/hopper/converter/hopper/pipeline.md) (CP_ASYNC)
- **🔴 Architecture Differences**: This document uses `async_copy.buffer_load_to_shared` hardware DMA, which CDNA3 lacks and must use software-based approaches instead
- **Layout Dependency**: [CDNA4 Layouts](layouts.md)
