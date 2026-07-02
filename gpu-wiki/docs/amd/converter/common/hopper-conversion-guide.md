# Triton → Gluon Conversion Guide (NVIDIA Hopper)

> **This guide is specific to NVIDIA Hopper (sm_90, H20/H100/H200) targets.**
> For AMD GPUs, use related guides: `converter/amd/cdna3/conversion-guide.md` (CDNA3/MI300) or `converter/amd/cdna4/conversion-guide.md` (CDNA4/MI355X)
>
> Common utilities and reference documentation are in `converter/amd/common/`.

## ⚠️ Critical Pitfalls (Must Read)

The following are the most common pitfalls in actual Hopper target conversions. **Be sure to remember these before starting the conversion**:

### 1. Do Not Use AMD-Specific APIs
Using `gl.amd.cdna3.*` on an NVIDIA GPU will result in `LLVM ERROR: unregistered dialect`. You must use the corresponding Hopper APIs.

| Prohibited (AMD) | Use (Hopper) |
|------------------|--------------|
| `gl.amd.cdna3.buffer_load(ptr=, offsets=)` | `gl.load(base_ptr + gl.cast(offsets, gl.int32))` |
| `gl.amd.cdna3.buffer_store(stored_value=, ptr=, offsets=)` | `gl.store(base_ptr + gl.cast(offsets, gl.int32), value)` |
| `gl.amd.cdna3.mfma(a, b, acc)` | `warpgroup_mma(a_smem, b_smem, acc, is_async=True)` |
| `gl.amd.AMDMFMALayout(...)` | `gl.NVMMADistributedLayout(...)` |
| `gl.DotOperandLayout(...)` | Not needed (wgmma reads directly from shared memory) |

### 2. `num_stages` and async_copy Pipelining
Gluon does not automatically generate pipelines. If the Triton source `num_stages > 1`, you **must** manually implement the pipeline.

**Core of Hopper pipelining**: Use `async_copy_global_to_shared` (CP_ASYNC DMA) to transfer data directly from global memory to shared memory, **bypassing registers**.

**⚠️ Using `gl.load` + `smem.store` instead of async_copy will be 50%+ slower!**

```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy

# ✅ Correct: CP_ASYNC DMA (bypasses registers)
async_copy.async_copy_global_to_shared(smem.index(slot), ptr + gl.cast(offsets, gl.int32), mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# ❌ Wrong: Two-step transfer (50%+ slower)
data = gl.load(ptr + gl.cast(offsets, gl.int32), mask=mask, other=0.0)
smem.index(slot).store(data)
```

See detailed patterns at: `../../nvidia/hopper/pipeline.md`

### 3. wgmma Three-Step Pattern
Hopper matrix multiplication (wgmma) requires a strict three-step sequence:

```python
# Step 1: fence — ensure smem write is complete
fence_async_shared()
# Step 2: asynchronous wgmma — operands must be in shared memory with NVMMASharedLayout
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
# Step 3: wait — wait for wgmma to complete
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

Omitting any step will result in undefined behavior (incorrect results or crash).

See detailed patterns at: `../../nvidia/hopper/matrix_multiply.md`

### 4. Hopper warp size = 32
The NVIDIA warp size is 32 (not AMD's 64). The product of all `BlockedLayout` and `threads_per_warp` must equal 32.

### 5. Strictly Forbidden to Fabricate Layouts
All layouts must be extracted from TTGIR via `tools/extract_ttgir.py`. See `../../nvidia/hopper/layouts.md` for details.

---

## Core Knowledge

### 1. Import Statements

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,           # CP_ASYNC DMA: global → shared
    fence_async_shared,   # fence before wgmma
    warpgroup_mma,        # asynchronous matrix multiplication
    warpgroup_mma_wait,   # wait for wgmma completion
)
```

### 2. API Mapping (Hopper Specific)

| Triton | Gluon (Hopper) | Notes |
|--------|----------------|-------|
| `tl.dot` | `warpgroup_mma` + `fence_async_shared` + `warpgroup_mma_wait` | Operands are in shared memory |
| `tl.load` (2D block) | `gl.load(base_ptr + gl.cast(offsets, gl.int32))` | Pointer arithmetic |
| `tl.store` (2D block) | `gl.store(base_ptr + gl.cast(offsets, gl.int32), val)` | Pointer arithmetic |
| `tl.load` (pipeline data) | `async_copy.async_copy_global_to_shared` | CP_ASYNC DMA |For general API mapping, see the shared guide at `porting_rules.md`.
For the complete Hopper mapping table, see: `../../nvidia/hopper/api_mapping.md`

### 3. Conversion Modes

- **Memory Access Mode**: `../../nvidia/hopper/memory_access.md`
- **Matrix Multiply Mode**: `../../nvidia/hopper/matrix_multiply.md`
- **Pipeline Mode (num_stages>1)**: `../../nvidia/hopper/pipeline.md`
- **Layout Reference**: `../../nvidia/hopper/layouts.md`

---

## Conversion Flow

```
1. Read Triton code
2. Call extract_ttgir.py -o output.ttgir to get Layout information from TTGIR
3. Confirm TTGIR target is cuda:90 (NVIDIA Hopper)
4. ⚠️ Check the value of num_stages in the original Triton code:
   - num_stages == 1 → Set num_stages=1 in wrapper, no pipeline needed
   - num_stages > 1 → Set num_stages=1 in wrapper, but **must** manually implement pipeline in Gluon kernel
     using async_copy, see ../../nvidia/hopper/pipeline.md
   - ⚠️ Even with pipeline, if using gl.load+smem.store instead of async_copy, performance will still be 50%+ worse
5. Check api_mapping.md / shared porting_rules.md → Has mapping → Convert directly
6. No mapping → Call lookup_api.sh to query
7. After conversion, execute four verifications in sequence (see shared guide)
```

---

## Reference Documents

| Document | Location | Content |
|------|------|------|
| `../../nvidia/hopper/api_mapping.md` | This guide | Hopper API Quick Reference Table |
| `../../nvidia/hopper/layouts.md` | This guide | Hopper Layout Types and TTGIR Mapping |
| Conversion pattern notes | This guide | Hopper Conversion Modes (memory access, wgmma, async_copy pipelines) |
| `../../nvidia/hopper/common_pitfalls.md` | This guide | Hopper Common Errors and Solutions |
| `../porting_rules.md` | Shared guide | General API Reference (includes Hopper annotations) |
| `learning_guide.md` | Shared guide | Unknown API Learning Process |
| `verification_guide.md` | Shared guide | Verification Methods and Failure Handling |

---

## Version Information

**Guide Version**: v1.0
**Last updated**: 2026-03-18
**Target Architecture**: NVIDIA Hopper (sm_90, H20/H100/H200)
**Design Principle**: Static knowledge + real-time learning capability


## Related

- [HSACO Offline Launcher Guide](hsaco-offline-launcher.md)
- [Real-time Learning Guide](learning_guide.md)
- [General Gluon Operations Porting Rules (Applicability of each section is annotated: [Common] = General, [AMD CDNA3] = AMD-specific, [Hopper] = NVIDIA Hopper-specific)](porting_rules.md)
- [Verification Guide](verification_guide.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
