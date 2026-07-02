# CDNA3 Pipeline Conversion (Triton → Gluon)

Converting Triton software-pipelined loops (num_stages > 1) to Gluon on AMD CDNA3 (gfx942).


**Last updated**: 2026-07-01

## Pipeline Conversion Mode (num_stages > 1)

### Background

When `num_stages > 1` is set in Triton, the compiler automatically generates a software pipeline in TTGIR:
- **prologue**: Prefetch the first batch of data to shared memory before the loop
- **main loop**: Compute the current slot's data + prefetch the next batch into another slot
- **epilogue**: Process the last iteration (no prefetch needed)

Key characteristics in TTGIR:
```
%buf = ttg.local_alloc : () -> !ttg.memdesc<Nx64x64xbf16, #shared, #smem, mutable>  # N=depth
%slot = ttg.memdesc_index %buf[%idx]         # Index specific slot
ttg.local_store %data, %slot                 # In-place write
%val = ttg.local_load %slot                  # Read from slot
```

depth = num_stages - 1 (e.g., num_stages=2 → depth=1, num_stages=3 → depth=2)

---

### TTGIR → Gluon Mapping

| TTGIR | Gluon | Description |
|-------|-------|------|
| `ttg.local_alloc : () -> !ttg.memdesc<Nx...>` | `gl.allocate_shared_memory(dtype, [N, ...], layout)` | Pre-allocate shared memory with depth, **without** `value=` |
| `ttg.memdesc_index %buf[%idx]` | `smem.index(idx)` | Index a specific buffer slot |
| `ttg.local_store %data, %slot` | `smem.index(idx).store(data)` | Write in-place to a slot |
| `ttg.local_load %slot` | `smem.index(idx).load(layout=dot_op)` | Load from a slot into registers |

---

### ✅ Correct Approach: Pre-allocate Shared Memory + index/store/load for Pipelining

```python
# 1. Pre-allocate persistent shared memory (depth=N, corresponds to TTGIR's memdesc<Nx...>)
#    Without value= parameter, only allocate space
smem_a = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, BLOCK_K], shared_layout)
smem_b = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_N, BLOCK_K], shared_layout)

# 2. PROLOGUE: prefetch first batch of data to slot 0
a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=mask_a)
b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=mask_b)
smem_a.index(0).store(a)
smem_b.index(0).store(tl.trans(b))  # If transpose is needed

# 3. MAIN LOOP: compute slot (i-1)%N + prefetch to slot i%N
for i in range(1, loop_n):
    # Launch next batch of prefetch (async buffer_load)
    next_a = gl.amd.cdna3.buffer_load(...)
    next_b = gl.amd.cdna3.buffer_load(...)

    # Load from previous slot to registers and compute
    a_dot = smem_a.index((i - 1) % 2).load(layout=dot_a_layout)
    b_dot = smem_b.index((i - 1) % 2).load(layout=dot_b_layout)
    acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc)

    # Write prefetch result to current slot (in-place overwrite, no new memory allocation)
    smem_a.index(i % 2).store(next_a)
    smem_b.index(i % 2).store(tl.trans(next_b))

# 4. EPILOGUE: process last slot
a_dot = smem_a.index((loop_n - 1) % 2).load(layout=dot_a_layout)
b_dot = smem_b.index((loop_n - 1) % 2).load(layout=dot_b_layout)
acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc)
```

---

### ❌ Common Mistake: Overwriting with allocate_shared_memory(value=...)

```python
# Wrong: allocate_shared_memory allocates new smem on each loop iteration
for i_t in range(NT - 1):
    w_dot = w_smem.load(dot_op0)
    ...
    w_smem = gl.allocate_shared_memory(..., value=next_w)  # ← Allocates new physical memory!
    # Old smem and new smem both alive → shared memory OOM
```

**To overwrite existing shared memory, you must use `smem.index(i).store(data)` rather than re-allocating.**

---

### Two Types of Shared Memory Usage

| Type | Allocation Method | Write Method | Scenario |
|------|---------|---------|------|
| **Persistent buffer** (reused across iterations) | `allocate_shared_memory(dtype, [depth, ...], layout)` without `value=` | `.index(i).store(data)` | Data that needs pipeline across iterations, such as w and k |
| **Temporary buffer** (single-use) | `allocate_shared_memory(dtype, shape, layout, value=data)` with `value=` | No overwrite needed | Immediately consumed data, such as h→dot and v_new→dot |### Shared Memory Budget

MI300 series hardware limit: **65536 bytes** (64KB)

Common buffer sizes:
- `[1, 64, 64] × bf16` = 8192 bytes (8KB)
- `[2, 64, 64] × bf16` = 16384 bytes (16KB)
- `[64, 16] × bf16` = 2048 bytes (2KB)

Persistent buffers use pre-allocation (including depth dimension), and temporary buffers use `value=` for on-demand allocation. Ensure the total active usage stays under 64KB.

---

### ⚠️ Pipelining Is Mandatory, Not Optional

If the Triton source code has `num_stages > 1`, you **must** implement manual pipelining in Gluon. Skipping pipelining will result in:
- 40-70% performance regression (all global memory loads become synchronous blocking)
- `benchmark.py` will definitely fail (ratio > 1.15)

**Measured Data** (chunk_gdn kernel, T=9418, K=128, BT=64, BV=16):

| Implementation | Gluon/Triton Ratio | Benchmark Result |
|----------|------------------|----------------|
| No pipelining | 1.497 (50% slower) | ❌ Failed |
| Three-stage pipelining | 1.114 (11% slower) | ✅ Passed |

The remaining ~11% gap comes from Triton compiler's `in_thread_transpose` optimization (gfx942 only). The Gluon compiler does not automatically run this pass, but you can express equivalent semantics via the combination of `convert_layout` + `allocate_shared_memory` + `smem.load`, which the compiler will automatically recognize and insert the equivalent operations.

### Checklist for Determining Whether Pipelining Is Needed

1. Check the `num_stages` parameter in the Triton wrapper → if > 1, pipelining is needed
2. Check if there is `ttg.local_alloc : () -> !ttg.memdesc<Nx...>` in TTGIR → N > 0 indicates persistent smem
3. Check if there is a `ttg.memdesc_index` + `ttg.local_store` pattern in TTGIR → pipelined writes
4. Check if there is a `amd.pipeliner_part = "prologue"` annotation in TTGIR → compiler-marked prologue data

## Related

- **Cross-Architecture Reference**: [CDNA4 Pipeline](../cdna4/pipeline.md) (hardware DMA) | [Hopper Pipeline](../../../nvidia/hopper/converter/hopper/pipeline.md) (CP_ASYNC)
- **🔴 Architecture Differences**: This article covers pure software pipelining (buffer_load→register→smem). CDNA4 and Hopper use hardware DMA to bypass registers; code is not portable between them.
- **Downstream Optimization**: [CDNA3 Warp Pipeline Stage](../../gluon/gfx942/warp_pipeline_stage.md) — optimizes the pipeline code produced by this article
- **Layout Dependencies**: [CDNA3 Layouts](layouts.md)
