# Triton → Gluon Conversion Guide (NVIDIA Hopper)

> **This guide is exclusively for NVIDIA Hopper (sm_90, H20/H100/H200) targets.**
> For AMD GPUs, use related guides: `converter/amd/cdna3/conversion-guide.md` (CDNA3/MI300) or `converter/amd/cdna4/conversion-guide.md` (CDNA4/MI355X)
>
> Shared utilities and reference docs are available in `converter/amd/common/`.

## ⚠️ Critical Pitfalls (Must Read)

The following are the most common pitfalls encountered in real-world Hopper target conversions. **Keep these in mind before starting any conversion**:

### 1. Do Not Use AMD-Specific APIs
Using `gl.amd.cdna3.*` on NVIDIA GPUs will result in `LLVM ERROR: unregistered dialect`. Always use the corresponding Hopper API.

| Forbidden (AMD) | Use Instead (Hopper) |
|-----------------|----------------------|
| `gl.amd.cdna3.buffer_load(ptr=, offsets=)` | `gl.load(base_ptr + gl.cast(offsets, gl.int32))` |
| `gl.amd.cdna3.buffer_store(stored_value=, ptr=, offsets=)` | `gl.store(base_ptr + gl.cast(offsets, gl.int32), value)` |
| `gl.amd.cdna3.mfma(a, b, acc)` | `warpgroup_mma(a_smem, b_smem, acc, is_async=True)` |
| `gl.amd.AMDMFMALayout(...)` | `gl.NVMMADistributedLayout(...)` |
| `gl.DotOperandLayout(...)` | Not needed (wgmma reads directly from smem) |

### 2. `num_stages` and async_copy Pipelining
Gluon does not automatically generate pipelines. If the Triton source code `num_stages > 1`, you **must** manually implement the pipeline.

**The core of Hopper pipelining**: Use `async_copy_global_to_shared` (CP_ASYNC DMA) to transfer data directly from global memory to shared memory, **bypassing registers**.

**⚠️ Using `gl.load` + `smem.store` instead of async_copy will be 50%+ slower!**

```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy

# ✅ Correct: CP_ASYNC DMA (bypasses registers)
async_copy.async_copy_global_to_shared(smem.index(slot), ptr + gl.cast(offsets, gl.int32), mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# ❌ Wrong: two-step transfer (50%+ slower)
data = gl.load(ptr + gl.cast(offsets, gl.int32), mask=mask, other=0.0)
smem.index(slot).store(data)
```

See detailed patterns at: `pipeline.md`

### 3. The wgmma Three-Step Pattern
Hopper matrix multiplication (wgmma) requires a strict three-step sequence:

```python
# Step 1: fence — ensure smem writes complete
fence_async_shared()
# Step 2: async wgmma — operands must be in NVMMASharedLayout shared memory
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
# Step 3: wait — wait for wgmma completion
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

Omitting any step leads to undefined behavior (incorrect results or crashes).

See detailed patterns at: `matrix_multiply.md`

### 4. Hopper Warp Size = 32
The NVIDIA warp size is 32 (not AMD's 64). In all `BlockedLayout`, the product of `threads_per_warp` must equal 32.

### 5. Never Fabricate Layouts
All layouts must be extracted from TTGIR via `tools/extract_ttgir.py`. See `layouts.md` for details.

---

## Core Knowledge

### 1. Imports

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,           # CP_ASYNC DMA: global → shared
 fence_async_shared, # wgmma fence
 warpgroup_mma, # asynchronousmatrixmultiplication
 warpgroup_mma_wait, # wait wgmma complete
)
```

### 2. API Mapping (Hopper-Specific)

| Triton | Gluon (Hopper) | Notes |
|--------|----------------|-------|
| `tl.dot` | `warpgroup_mma` + `fence_async_shared` + `warpgroup_mma_wait` | Operands in shared memory |
| `tl.load` (2D block) | `gl.load(base_ptr + gl.cast(offsets, gl.int32))` | Pointer arithmetic |
| `tl.store` (2D block) | `gl.store(base_ptr + gl.cast(offsets, gl.int32), val)` | Pointer arithmetic |
| `tl.load` (pipeline data) | `async_copy.async_copy_global_to_shared` | CP_ASYNC DMA |For common API mapping, refer to the shared guide's `../../amd/common/porting_rules.md`.
For the complete Hopper mapping table, see: `api_mapping.md`

### 3. Conversion Modes

- **Memory Access Mode**: `memory_access.md`
- **Matrix Multiply Mode**: `matrix_multiply.md`
- **Pipeline Mode (num_stages>1)**: `pipeline.md`
- **Layout Reference**: `layouts.md`

---

## Conversion Workflow

```
1. Read Triton code
2. Call extract_ttgir.py -o output.ttgir to extract Layout information from TTGIR
3. Confirm TTGIR target is cuda:90 (NVIDIA Hopper)
4. ⚠️ Check num_stages value in original Triton code:
   - num_stages == 1 → set num_stages=1 in wrapper, no pipeline needed
   - num_stages > 1 → set num_stages=1 in wrapper, but **must** manually implement pipeline in Gluon kernel using async_copy, see pipeline.md
   - ⚠️ Even with pipeline, if using gl.load+smem.store instead of async_copy, performance will be 50%+ worse
5. Check api_mapping.md / shared porting_rules.md → if mapping exists → convert directly
6. No mapping → call lookup_api.sh to query
7. After conversion, execute four verification steps sequentially (see shared guide)
```

---

## Reference Documents

| Document | Location | Content |
|------|------|------|
| `api_mapping.md` | This guide | Hopper API quick reference table |
| `layouts.md` | This guide | Hopper Layout types and TTGIR mapping |
| Conversion pattern notes | This guide | Hopper conversion modes (memory access, wgmma, async_copy pipelines) |
| `common_pitfalls.md` | This guide | Common Hopper errors and solutions |
| `../../../amd/common/porting_rules.md` | Shared guide | Common API reference (with Hopper annotations) |
| `../../amd/common/learning_guide.md` | Shared guide | Unknown API learning process |
| `../../amd/common/verification_guide.md` | Shared guide | Validation methods and failure handling |

---

## Version Information

**Guide Version**: v1.0
**Last Updated**: 2026-03-18
**Target Architecture**: NVIDIA Hopper (sm_90, H20/H100/H200)
**Design Principle**: Static knowledge + real-time learning capability
