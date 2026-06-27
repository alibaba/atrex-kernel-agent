# Triton → Gluon Conversion Guide (AMD CDNA4)

> **This guide is specific to AMD CDNA4 (MI355X, gfx950) targets.**
> For CDNA3 (MI300 series), please use the related guide: `converter/amd/cdna3/conversion-guide.md`
> For NVIDIA GPUs, please use the related guide: `converter/nvidia/hopper/conversion-guide.md`
>
> Common utilities and reference documentation are in `converter/amd/common/`.

## ⚠️ Critical Pitfalls (Must Read)

The following are the most common pitfalls encountered during actual CDNA4 target conversions. **Make sure to remember these before starting any conversion**:

### 1. `num_stages` and async_copy Hardware Pipelining
The **`num_stages`** parameter in the Gluon launcher is **ineffective** (Gluon compiler does not automatically perform multi-stage pipelining).
However, CDNA4 features **hardware DMA async_copy**, which is fundamentally different from CDNA3's pure software pipelining. If the Triton source uses `num_stages > 1`, you **must** use `async_copy.buffer_load_to_shared` (or `async_copy.global_load_to_shared`) to transfer data directly from global memory to shared memory, **bypassing registers**.

**⚠️ Using `gl.amd.cdna4.buffer_load` + `smem.store` instead of async_copy will be 40-60% slower!**

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy

# ✅ Correct: Hardware DMA (global → shared, bypasses registers)
async_copy.buffer_load_to_shared(smem.index(slot), ptr, offsets, mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
# When loading from smem, use load_shared_relaxed to avoid redundant synchronization
data = async_copy.load_shared_relaxed(smem.index(slot), layout=dot_op)

# ❌ Wrong: Two-step transfer (40-60% slower)
data = gl.amd.cdna4.buffer_load(ptr=ptr, offsets=offsets, mask=mask)
smem.index(slot).store(data)
```

**Ping-pong Pipeline Pattern** (async_copy only):
```python
# Pre-allocate double buffer (depth=2 corresponds to num_stages=3)
smem = gl.allocate_shared_memory(gl.bfloat16, [2, M, K], shared_layout)

# PROLOGUE: Asynchronously prefetch first two iterations
async_copy.buffer_load_to_shared(smem.index(0), ptr, offsets_0, mask=mask_0)
async_copy.commit_group()
async_copy.buffer_load_to_shared(smem.index(1), ptr, offsets_1, mask=mask_1)
async_copy.commit_group()

# MAIN LOOP: compute slot[i%2] + prefetch to slot[(i+2)%2]
for i in range(NT - 2):
    async_copy.wait_group(num_outstanding=1)  # Wait for slot[i%2] to be ready
    data = async_copy.load_shared_relaxed(smem.index(i % 2), layout=dot_op)
    acc = gl.amd.cdna4.mfma(data, other, acc)
    async_copy.buffer_load_to_shared(smem.index(i % 2), ptr, offsets_next, mask=mask_next)
    async_copy.commit_group()

# EPILOGUE: Process last two iterations
async_copy.wait_group(num_outstanding=0)
# ... compute remaining slots ...
```

**Key Points**:
- `buffer_load_to_shared` is preferred over `global_load_to_shared` (lower register pressure + hardware bounds checking), use it preferentially
- `load_shared_relaxed` skips redundant synchronization when reading from smem, use together with async_copy
- gfx950 compiler **disables** `in_thread_transpose` pass and kpack=1, so ping-pong double buffer pipelining is the only efficient approach
- **⚠️ Do not use `allocate_shared_memory(value=...)` to overwrite persistent smem — that allocates new physical memory and causes OOM**

For detailed patterns, see: `pipeline.md`

### 2. TTGIR and Gluon Parameter Name Differences
Field names in TTGIR IR differ from parameter names in the Gluon Python API. Do not copy them directly:

| TTGIR Field | Gluon Python Parameter |
|-----------|------------------|
| `isTransposed = true` | `transposed=True` |
| `sizePerThread` | `size_per_thread` |
| `threadsPerWarp` | `threads_per_warp` |
| `warpsPerCTA` | `warps_per_cta` |
| `instrShape` | `instr_shape` |

### 3. Strictly Forbidden to Fabricate Layouts
All Layouts must be extracted from TTGIR via `tools/extract_ttgir.py`. CDNA4 uses `AMDMFMALayout(version=4)` and supports new `instr_shape` such as `[16,16,32]` and `[32,32,16]`, but it is still **forbidden to guess arbitrarily**. For details, see `layouts.md`.

### 4. CDNA4-Specific Constraints
- The product of `threads_per_warp` must be **64** (AMD warp size)
- Matrix multiplication must use `gl.amd.cdna4.mfma`, which requires `DotOperandLayout`
- 2D memory access uses `gl.amd.cdna4.buffer_load` / `buffer_store` (scalar ptr + offset tensor)
- **Note**: Although `cdna4` inherits all APIs from `cdna3` (`from ..cdna3 import *`), you must use the `gl.amd.cdna4.*` prefix in your code, and using `gl.amd.cdna3.*` is **prohibited**
- LDS capacity of 160 KB/CU (CDNA3 has only 64 KB), enabling larger tiles and deeper pipelines

### 5. CDNA4 Exclusive: `mfma_scaled` (FP4/FP6 Block-Scaled Matrix Multiplication)
The CDNA4-exclusive `mfma_scaled` supports low-precision matrix multiplication in OCP Microscaling (MX) format. **CDNA3 does not support this operation.**

```python
# mfma_scaled signature
acc = gl.amd.cdna4.mfma_scaled(
    a,              # operand A (DotOperandLayout)
    a_scale,        # A's scaling factor (tensor or None)
    a_format,       # A's format: "e2m1" (FP4), "e4m3" (FP8), "e5m2" (FP8)
    b,              # operand B (DotOperandLayout)
    b_scale,        # B's scaling factor (tensor or None)
    b_format,       # B's format: "e2m1", "e4m3", "e5m2"
    acc,            # accumulator (AMDMFMALayout)
)

# Get scaling factor layout (constexpr helper function)
scale_layout = gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape)
```

**Supported Formats**:
- `e2m1`: FP4 (4-bit), highest throughput
- `e4m3`: FP8 E4M3 (8-bit)
- `e5m2`: FP8 E5M2 (8-bit)

### 6. E8M0 Scale Conversion Pattern (Tested and Verified)
The E8M0 format scale (int8) **cannot be passed directly** to `scaled_upcast_fp4`. It must be converted to a bf16 shifted scale:

```python
# ✅ Correct: e8m0 i8 → bf16 shifted scale (extui→shli(7)→bitcast)
q_pe_scales_i16 = gl.cast(q_pe_scales_reshaped, gl.int16)      # zero-extend i8→i16
q_pe_scales_shifted = q_pe_scales_i16 << 7                      # shl 7: place exp in bf16 exp field
q_pe_scales_bf16 = gl.cast(q_pe_scales_shifted, gl.bfloat16, bitcast=True)

# ❌ Wrong: Directly passing int8 scale (CDNA4 does not support useShiftedScale=true i8 path)
q_pe_bf16 = gl.amd.cdna4.scaled_upcast_fp4(q_pe_fp4_dot, q_pe_scales_raw, gl.bfloat16, 1)
```

**Why shift by 7**: BF16 mantissa width - 1 = 7. This shift places the unbiased exponent of e8m0 into the exponent field of BF16.

---

## Core Knowledge

### 1. Import Statements

```python
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
# CDNA4 exclusive: Hardware async_copy
from triton.experimental.gluon.language.amd.cdna4 import async_copy
```

### 2. API Mapping (AMD CDNA4-Specific)

| Triton | Gluon (CDNA4) | Notes |
|--------|---------------|-------|
| `tl.dot` | `gl.amd.cdna4.mfma` | Requires shared memory + DotOperandLayout |
| `tl.dot` (low-precision MX) | `gl.amd.cdna4.mfma_scaled` | CDNA4 exclusive, FP4/FP8 block-scaled |
| `tl.load` (2D block) | `gl.amd.cdna4.buffer_load` | Scalar ptr + offset tensor |
| `tl.store` (2D block) | `gl.amd.cdna4.buffer_store` | Scalar ptr + offset tensor |
| `tl.load` (pipeline) | `async_copy.buffer_load_to_shared` | Hardware DMA, global → shared |
| `tl.load` (pipeline, 64-bit) | `async_copy.global_load_to_shared` | Tensor ptr, flexible but high register pressure |
| — | `async_copy.commit_group` | Commit an async operation group |
| — | `async_copy.wait_group` | Wait for async operations to complete |
| — | `async_copy.load_shared_relaxed` | Load from smem, skip redundant synchronization |
| — | `gl.amd.cdna4.scaled_upcast_fp4` | FP4 fused upcast + scale |
| — | `gl.fp4_to_fp` | FP4 → bf16/fp32 hardware conversion |
| `tl.abs` | `tl.abs` | Absolute value |
| `tl.max` | `tl.max` | Maximum value (supports axis) |
| `tl.exp` | `gl.exp` | Exponential |
| `tl.log` | `tl.log` | Logarithm |
| `tl.trans` | `tl.trans` | Transpose |

For common API mappings, refer to `../common/porting_rules.md`.
For the complete AMD CDNA4 mapping table, see: `api_mapping.md`

### 3. Conversion Patterns

- **Memory Access Pattern**: `memory_access.md`
- **Matrix Multiplication Pattern**: `matrix_multiply.md`
- **Pipeline Pattern (num_stages>1)**: `pipeline.md`
- **Layout Reference**: `layouts.md`

---

## Conversion Workflow

```
1. Read Triton code
2. Call extract_ttgir.py -o output.ttgir to get Layout information from TTGIR
3. Confirm TTGIR target is hip:gfx950 (AMD CDNA4)
4. ⚠️ Check the value of num_stages in the original Triton code:
   - num_stages == 1 → Set num_stages=1 in wrapper, no pipeline needed
   - num_stages > 1 → Set num_stages=1 in wrapper, but **must** manually implement pipeline in Gluon kernel
     using async_copy.buffer_load_to_shared, see pipeline.md
   - ⚠️ Using buffer_load+smem.store instead of async_copy will be 40-60% slower
5. Check `api_mapping.md` / `../common/porting_rules.md` → Has mapping → Convert directly
6. No mapping → Call lookup_api.sh to query
7. After conversion, execute four verifications in sequence (see shared guide)
```

---

## Reference Documentation

| Document | Location | Content |
|------|------|------|
| `api_mapping.md` | This guide | CDNA4 API reference quick lookup table |
| `layouts.md` | This guide | CDNA4 Layout types and TTGIR mapping |
| Conversion pattern notes | This guide | CDNA4 conversion patterns (memory access, matrix multiplication, async_copy pipeline) |
| `common_pitfalls.md` | This guide | CDNA4 common errors and solutions |
| `../common/porting_rules.md` | Shared guide | Common API reference (with AMD annotations) |
| `../common/learning_guide.md` | Shared guide | Unknown API learning process |
| `../common/verification_guide.md` | Shared guide | Verification methods and failure handling |

---

## Version Information

**Guide Version**: v1.1
**Last Updated**: 2026-03-28
**Target Architecture**: AMD CDNA4 (MI355X, gfx950)
**Design Principle**: Static knowledge + real-time learning capability
