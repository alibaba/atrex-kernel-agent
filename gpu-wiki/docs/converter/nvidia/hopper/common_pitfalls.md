# Common Errors and Solutions (NVIDIA Hopper)

**Last Updated**: 2026-03-18
**Target Architecture**: NVIDIA Hopper (sm_90)

---

## Error 1: Using AMD-Specific APIs Causes LLVM Crash

**Symptoms**:
```
LLVM ERROR: amdg.buffer_load created with unregistered dialect
```
or
```
LLVM ERROR: Operation created with unregistered dialect 'amdg'
```

**Cause**: AMD-specific APIs (`gl.amd.cdna3.buffer_load`, `gl.amd.cdna3.buffer_store`, `gl.amd.cdna3.mfma`) were used on an NVIDIA GPU.

**Solution**: Detect the TTGIR target and use the corresponding Hopper APIs:

| AMD API (Forbidden) | Hopper Replacement |
|---------------------|--------------------|
| `gl.amd.cdna3.buffer_load(ptr=, offsets=, mask=)` | `gl.load(base_ptr + gl.cast(offsets, gl.int32), mask=, other=)` |
| `gl.amd.cdna3.buffer_store(stored_value=, ptr=, offsets=, mask=)` | `gl.store(base_ptr + gl.cast(offsets, gl.int32), value, mask=)` |
| `gl.amd.cdna3.mfma(a, b, acc)` | `warpgroup_mma(a_smem, b_smem, acc, is_async=True)` |
| `gl.amd.AMDMFMALayout(...)` | `gl.NVMMADistributedLayout(...)` |
| `gl.DotOperandLayout(...)` | Not needed (wgmma reads directly from smem) |

**Prevention**: Run `extract_ttgir.py` before conversion and check the `ttg.target` field:
- `cuda:90` → Use Hopper APIs
- `hip:gfx942` → Use AMD APIs

---

## Error 2: Using gl.load + smem.store Instead of async_copy Leads to Poor Performance

**Symptom**:
```
benchmark.py reports Gluon/Triton ratio > 1.15 (e.g., 1.50, i.e., 50% slower)
```

**Cause**: The pipeline uses `gl.load` (global → register) + `smem.store` (register → shared) as a two-step transfer, instead of `async_copy_global_to_shared` (global → shared direct DMA). The two-step transfer adds an extra register transit, resulting in low bandwidth utilization.

**Solution**: Use CP_ASYNC DMA:
```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy

# ❌ Wrong: two-step transfer (50%+ slower)
data = gl.load(ptr + gl.cast(offsets, gl.int32), mask=mask, other=0.0)
smem.index(slot).store(data)

# ✅ Correct: CP_ASYNC DMA (bypasses registers)
async_copy.async_copy_global_to_shared(
    smem.index(slot),
    ptr + gl.cast(offsets, gl.int32),
    mask=mask
)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
```

**Measured Performance Comparison** (chunk_gdn kernel, H20, K=128):

| Transfer Method | Gluon/Triton Ratio | Result |
|----------------|-------------------|--------|
| `gl.load` + `smem.store` | 1.50 (50% slower) | ❌ Failed |
| `async_copy_global_to_shared` | 1.02-1.09 | ✅ Passed |

**Note**: Only use async_copy for data that needs to go into shared memory (e.g., w, k matrices). Data that goes directly into registers (e.g., v, g) should still use `gl.load`.
---

## Error 3: Missing fence_async_shared Causes Undefined Behavior

**Symptom**: Incorrect results at runtime (precision verification failure) or occasional crashes.

**Cause**: Before wgmma reads operands from shared memory, `fence_async_shared()` must be called to ensure that previous writes to smem (via `allocate_shared_memory(value=...)` or `smem.store()`) have completed.

**Solution**:
```python
# ❌ Wrong: missing fence
h_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BV], shared_v, value=data)
acc = warpgroup_mma(w_smem, h_smem, acc, is_async=True)  # ← may read stale data

# ✅ Correct: fence ensures smem writes are visible
h_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BV], shared_v, value=data)
fence_async_shared()  # ← must be called before wgmma
acc = warpgroup_mma(w_smem, h_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

**Rule**: **Every time after writing new data to smem and before wgmma reads that smem**, `fence_async_shared()` must be called once.

---

## Error 4: Missing warpgroup_mma_wait Causes Data Race

**Symptom**: Incorrect results at runtime or GPU hang.

**Cause**: `warpgroup_mma(..., is_async=True)` is an asynchronous operation, and the returned accumulator may not have finished computing. Before using the accumulator subsequently, `warpgroup_mma_wait` must be called.**Solution**:
```python
# ❌ Wrong: using result directly after async wgmma
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
result = -acc + some_value  # ← acc may not have finished computing

# ✅ Correct: wait for wgmma completion
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))  # ← wait for completion
result = -acc + some_value  # ← safe to use
```

**`deps` parameter**: You must pass the tensor tuple that needs to be waited on, e.g., `deps=(acc,)`. This ensures that the compiler does not reorder subsequent operations before the wait.

---

## Error 5: Variable Scope Issue Inside Runtime Conditional Blocks

**Symptom**:
```
NameError: name 'pf_v' is not defined
```
or compilation error indicating undefined variable.

**Cause**: In Gluon, variables defined inside a runtime conditional block (e.g., `if i_t < NT - 1:`) are not visible outside the block. This differs from Python's conventional scoping rules.

**Solution**: For all variables assigned inside a conditional block, ensure that any related usage also resides within the same conditional block:

```python
# ❌ Wrong: pf_v assigned inside conditional block but used outside
if i_t < NT - 1:
    pf_v = gl.load(v + gl.cast(vo_n, gl.int32), mask=vmk_n, other=0.0)
# Used in next iteration → may be undefined

# ✅ Correct: all prefetch logic within same conditional block
if i_t < NT - 1:
    pf_v = gl.load(v + gl.cast(vo_n, gl.int32), mask=vmk_n, other=0.0)
    # ... all logic using pf_v for prefetch/assignment goes here ...
    async_copy.commit_group()
```

**Alternatively**: Pre-define variables outside the loop Insure , ensuring that all branches assign values.

---

## Error 6: Shared Memory Exceeded (OOM)

**Symptom**:
```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 67584, Hardware limit: 65536
```

**Cause**: The H20 per-block shared memory limit is **64KB** (65536 bytes). Common scenarios that exceed this limit:
- 4 double-buffered [64, 64] bf16 smem = 4 × 2 × 64 × 64 × 2 = 65536 bytes = exactly full
- Plus temporary smem for `allocate_shared_memory(value=...)` within the loop → exceeded

**Solution**:
```python
# ❌ Wrong: allocate_shared_memory in loop allocates new physical memory each time
for i_t in range(NT):
    w_dot = w_smem.load(...)
    next_w = gl.load(...)
    w_smem = gl.allocate_shared_memory(..., value=next_w)  # ← new allocation

# ✅ Correct: pre-allocate + index/store in-place overwrite
smem_w = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 64], shared_w)  # pre-allocate
for i_t in range(NT):
    # consume current slot
    async_copy.wait_group(num_outstanding=0)
    fence_async_shared()
    acc = warpgroup_mma(smem_w.index(i_t % 2), ...)
    # prefetch to next slot
    async_copy.async_copy_global_to_shared(smem_w.index((i_t+1) % 2), ...)
```

**Note**: Temporary smem (`allocate_shared_memory(value=...)`) is used for data consumed immediately (e.g., h → wgmma). As long as the previous temporary smem has been consumed (wgmma_wait completed), it will not be simultaneously active with the new temporary smem.

---

## Error 7: wgmma Operand Uses SwizzledSharedLayout

**Symptom**: Compilation error or runtime crash.

**Cause**: wgmma requires operands to use `NVMMASharedLayout` and cannot use `SwizzledSharedLayout`.

**Solution**:
```python
# ❌ Wrong: wgmma operands use SwizzledSharedLayout
shared_layout = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
a_smem = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout, value=a)
acc = warpgroup_mma(a_smem, b_smem, acc)  # ← crash

# ✅ Correct: wgmma operands use NVMMASharedLayout
nvmma_layout = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, transposed=False)
a_smem = gl.allocate_shared_memory(gl.bfloat16, [64, 64], nvmma_layout, value=a)
fence_async_shared()
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
```

---

## Error 8: `eviction_policy` Parameter Not Supported in Gluon

**Symptom**:
```
TypeError: load() got an unexpected keyword argument 'eviction_policy'
```
**Solution**: Directly remove the `eviction_policy` parameter:

## Error 9: Incorrect threads_per_warp Product

**Symptoms**:
```
LLVM ERROR: product of threads_per_warp must be 32
```

**Cause**: The NVIDIA warp size is 32, not 64. A threads_per_warp product of 64 copied from AMD TTGIR or AMD examples is invalid on Hopper.

**Solution**: Ensure that the threads_per_warp product for all BlockedLayout is 32:
```python
# ❌ Wrong (AMD warp size = 64)
blocked = gl.BlockedLayout(size_per_thread=[4], threads_per_warp=[64], warps_per_cta=[4], order=[0])

# ✅ Correct (NVIDIA warp size = 32)
blocked = gl.BlockedLayout(size_per_thread=[4], threads_per_warp=[32], warps_per_cta=[4], order=[0])
```

---

## Error 10: LSP False Positives (Must Ignore)

The following LSP errors in Triton/Gluon code are **expected behavior** and **must be ignored**:

| Error Pattern | Reason |
|----------|------|
| `Type "NVMMADistributedLayout" is not assignable to declared type "constexpr"` | The `gl.constexpr` type annotation is not understood by the static checker |
| `Type "NVMMASharedLayout" is not assignable to declared type "constexpr"` | Same as above |
| `Type "BlockedLayout" is not assignable to declared type "constexpr"` | Same as above |
| `Type "SliceLayout" is not assignable to declared type "constexpr"` | Same as above |
| `"constexpr" is not assignable to "DistributedLayout"` | Gluon's constexpr is passed to layout parameters |

**Judgment Criteria**: Only trust the results of `check_syntax.py` and `validate.py`, not LSP diagnostics.

---

## Debugging Tips

### 1. Confirm GPU Architecture
```python
import torch
print(torch.cuda.get_device_properties(0).name) # H20/H100/H200
```

### 2. Check TTGIR Target
```bash
python tools/extract_ttgir.py kernel.py -o output.ttgir
grep "ttg.target" output.ttgir # "cuda:90"
```

### 3. Compare Precision
```python
max_diff = (output_triton - output_gluon).abs().max()
print(f"Max difference: {max_diff}")
```

---

## Quick Checklist

See the full verification checklist in `../../amd/common/verification_guide.md`.
