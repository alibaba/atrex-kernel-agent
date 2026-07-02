# Hopper Pipeline Conversion (Triton → Gluon)

Converting Triton software-pipelined loops (num_stages > 1) to Gluon on NVIDIA Hopper (sm_90), including cp.async staging.


**Last updated**: 2026-07-01

## Pipeline Conversion Mode (NVIDIA Hopper, num_stages > 1)

### Background

When ``num_stages > 1`` is set in Triton, the compiler automatically generates software pipelining in TTGIR. Gluon does not generate pipelining automatically; it must be implemented manually.

**The core of Hopper pipelining**: Use **CP_ASYNC DMA** (``async_copy_global_to_shared``) to implement true global → shared memory asynchronous transfer, **bypassing registers**.

### Imports

```python
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,          # contains async_copy_global_to_shared, commit_group, wait_group
    fence_async_shared,  # must call before wgmma
    warpgroup_mma,       # async matrix multiply
    warpgroup_mma_wait,  # wait for wgmma completion
)
```

---

### Hopper async_copy API

| API | Purpose | AMD Equivalent |
|-----|---------|---------------|
| ``async_copy.async_copy_global_to_shared(smem_slot, ptr_tensor, mask=mask)`` | CP_ASYNC DMA: global → shared (bypasses registers) | ``buffer_load`` + ``smem.store`` |
| ``async_copy.commit_group()`` | Submit a batch of async copy operations | None |
| ``async_copy.wait_group(num_outstanding=N)`` | Wait until ≤ N copy groups are outstanding | None |

**Source**: ``triton.experimental.gluon.language.nvidia.hopper.async_copy`` (re-exported from Ampere module ``triton.experimental.gluon.language.nvidia.ampere.async_copy``)

---

### ⚠️ Performance Critical: async_copy vs gl.load + smem.store

| Transfer Method | Path | Gluon/Triton Ratio | Result |
|-----------------|------|--------------------|--------|
| ``gl.load`` + ``smem.store`` | global → register → shared | 1.50 (50% slower) | ❌ Failed |
| ``async_copy_global_to_shared`` | global → shared (DMA) | 1.02-1.09 | ✅ Passed |

**Must use async_copy**. The two-step transfer using ``gl.load`` + ``smem.store`` simply cannot meet performance requirements.

---

### ✅ Correct Approach: async_copy + Double-Buffer Pipeline

```python
# ============================================================
# 1. Pre-allocate persistent shared memory (double-buffered, depth=2)
# ============================================================
smem_w1 = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 64], shared_w)
smem_k1 = gl.allocate_shared_memory(gl.bfloat16, [2, 64, BT], shared_k)

# ============================================================
# 2. PROLOGUE: async_copy prefetch iter 0 to slot 0
# ============================================================
# Construct pointer tensor: base_ptr + offset_tensor (int32)
w_off = wr2 * stride_w + wc2
w_mask = (wr2 < T) & (wc2 < K)
async_copy.async_copy_global_to_shared(
    smem_w1.index(0),
    w + gl.cast(w_off, gl.int32),
    mask=w_mask
)

k_off = kr2 + kc2 * stride_k
k_mask = (kr2 < K) & (kc2 < T)
async_copy.async_copy_global_to_shared(
    smem_k1.index(0),
    k + gl.cast(k_off, gl.int32),
    mask=k_mask
)

# Data not passing through smem still uses gl.load to registers
pf_v = gl.load(v + gl.cast(v_off, gl.int32), mask=v_mask, other=0.0)

# Submit and wait for prologue completion
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# ============================================================
# 3. MAIN LOOP: compute cur_slot + prefetch to nxt_slot
# ============================================================
for i_t in range(NT):
    cur_slot = i_t % 2
    nxt_slot = (i_t + 1) % 2

    # --- Consume current slot ---
    async_copy.wait_group(num_outstanding=0)  # Ensure data ready

    # wgmma: requires fence → mma → wait three steps
    h_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BV], shared_v, value=b_h.to(gl.bfloat16))
    fence_async_shared()
    b_v_new = warpgroup_mma(smem_w1.index(cur_slot), h_smem, b_v_new, is_async=True)
    b_v_new = warpgroup_mma_wait(num_outstanding=0, deps=(b_v_new,))

    # dot(k, v_new): k also reads from persistent smem
    vn_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BV], shared_v, value=b_v_new_bf16)
    fence_async_shared()
    b_h = warpgroup_mma(smem_k1.index(cur_slot), vn_smem, b_h, is_async=True)
    b_h = warpgroup_mma_wait(num_outstanding=0, deps=(b_h,))

    # --- Prefetch next iteration to nxt_slot ---
    if i_t < NT - 1:
        next_w_off = ((i_t + 1) * BT + wr2) * stride_w + wc2
        next_w_mask = (((i_t + 1) * BT + wr2) < T) & (wc2 < K)
        async_copy.async_copy_global_to_shared(
            smem_w1.index(nxt_slot),
            w + gl.cast(next_w_off, gl.int32),
            mask=next_w_mask
        )

        # Data not passing through smem uses gl.load
        pf_v = gl.load(v + gl.cast(next_v_off, gl.int32), mask=next_v_mask, other=0.0)

        async_copy.commit_group()

# No explicit epilogue needed — last iteration handled in loop, no prefetch
```

### ❌ Wrong Approach: gl.load + smem.store

```python
# ❌ Wrong: two-step transfer (global → register → shared)
data = gl.load(ptr + gl.cast(offsets, gl.int32), mask=mask, other=0.0)
smem.index(slot).store(data)

# ✅ Correct: CP_ASYNC DMA (global → shared, bypasses registers)
async_copy.async_copy_global_to_shared(smem.index(slot), ptr + gl.cast(offsets, gl.int32), mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
```

---

### Two Types of Shared Memory Usage

| Type | Allocation Method | Data Source | Use Case |
|------|---------|---------|------|
| **Persistent buffer** (reused across iterations) | `allocate_shared_memory(dtype, [depth, ...], layout)` without `value=` | `async_copy_global_to_shared` | Pipeline data such as w, k |
| **Temporary buffer** (single use) | `allocate_shared_memory(dtype, shape, layout, value=data)` with `value=` | Written from registers | Immediately consumed data such as h→wgmma, v_new→wgmma |

---

### Shared Memory Budget

H20/H100 hardware limit: **65536 bytes** (64KB)

Common buffer sizes:
- `[2, 64, 64] × bf16` = 16384 bytes (16KB)
- `[2, 64, 16] × bf16` = 4096 bytes (4KB)
- `[64, 16] × bf16` = 2048 bytes (2KB)

4 double-buffered [64, 64] bf16 buffers = 4 × 16KB = 64KB = **exactly full**. Temporary smem is allocated after the previous one is consumed, so they are not active simultaneously.

---

### ⚠️ Pipeline Notes

1. **Only use async_copy for data entering shared memory**. Data going directly into registers (v, g) still uses `gl.load`.
2. **`commit_group()` must be called after all `async_copy_global_to_shared` calls**.
3. **`wait_group(num_outstanding=0)` must be called before consuming smem data**.
4. **`fence_async_shared()` must be called before wgmma reads from smem** (even if smem was written via `allocate_shared_memory(value=...)`).
5. **Variable scope**: Variables defined inside `if i_t < NT - 1:` are not visible outside the block. All prefetch logic must be completed within the same conditional block.
6. **The ptr parameter of async_copy** is a pointer tensor (`base_ptr + offset_tensor`), not a scalar ptr.

---

### ⚠️ Pipelining is Required, Not an Optional Optimization

If the Triton source code uses `num_stages > 1`, you **must** implement pipelining in Gluon. Skipping pipelining results in:
- Performance regression of 80%+ (even worse on Hopper without pipelining)
- `benchmark.py` will definitely fail

**Measured Data** (chunk_gdn kernel, H20, T=9418, K=128, BT=64, BV=16):

| Implementation Method | Gluon/Triton Ratio | Benchmark Result |
|----------|------------------|----------------|
| No pipelining | 1.85 (85% slower) | ❌ Failed |
| gl.load + smem.store pipelining | 1.50 (50% slower) | ❌ Failed |
| async_copy CP_ASYNC pipelining | 1.02-1.09 | ✅ Passed |

### Determining Whether Pipelining is Needed

1. Check the `num_stages` parameter in the Triton wrapper → if > 1, pipelining is needed
2. Check whether TTGIR contains `ttg.local_alloc : () -> !ttg.memdesc<Nx...>` → presence indicates persistent smem
3. Check whether TTGIR contains the `ttg.memdesc_index` + `ttg.local_store` pattern

## Related

- **Cross-Architecture Reference**: [CDNA3 Pipeline](../../../../amd/converter/cdna3/pipeline.md) (pure software) | [CDNA4 Pipeline](../../../../amd/converter/cdna4/pipeline.md) (hardware DMA)
- **🔴 Architecture Difference**: This document uses CP_ASYNC DMA, with a different API than CDNA4 DMA (`async_copy_global_to_shared` vs `buffer_load_to_shared`)
- **ISA Reference**: [PTX Sync and Async Operations](../../../common/ptx/nvidia-ptx-sync-and-async.md) — Underlying PTX instructions for CP_ASYNC
- **Layout Dependency**: [Hopper Layouts](layouts.md)
