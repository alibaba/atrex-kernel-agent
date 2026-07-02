# Common Errors & Solutions (CDNA4 / gfx950)

**Last updated**: 2026-03-21

---

## Error 1: Mask argument cannot be block type

**Symptom**:
```
ValueError: Mask argument cannot be block type, got <...>
```

**Cause**: Mask dimensions do not match offsets dimensions

**Solution**:
```python
# ❌ Wrong
mask = offs_m[:, None] < M  # 2D mask but offsets is 1D

# ✅ Correct
mask = (offs_m < M)[:, None]  # 1D first then expand
```

**Prevention**:
- Ensure the mask shape matches offsets
- First create a 1D mask, then expand with expand_dims

---

## Error 2: Layout Undefined

**Symptom**:
```
TypeError: arange() missing required argument: 'layout'
```

**Cause**: Forgot to specify a layout

**Solution**:
```python
# ❌ Wrong
idx = gl.arange(0, BLOCK_SIZE)

# ✅ Correct
layout: gl.constexpr = gl.BlockedLayout(...)
idx = gl.arange(0, BLOCK_SIZE, layout=layout)
```

---

## Error 3: Shared Memory Exceeded (OOM)

**Symptom**:
```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 180224, Hardware limit: 163840
```

**Cause**: Using `allocate_shared_memory(value=...)` to overwrite smem in a loop, causing both old and new buffers to be simultaneously active. CDNA4 has 160KB (163840 bytes) LDS per CU, which is significantly larger than CDNA3 (64KB), but caution is still required.

**Solution**: For persistent buffers reused across iterations, use `smem.index(i).store()` for in-place overwriting:
```python
# ❌ Wrong: allocate_shared_memory allocates new physical memory each time
for i_t in range(NT - 1):
    w_dot = w_smem.load(dot_op0)
    next_w = gl.amd.cdna4.buffer_load(...)
    w_smem = gl.allocate_shared_memory(..., value=next_w)  # ← New allocation, old smem not released

# ✅ Correct: pre-allocate + index/store in-place overwrite
smem_w = gl.allocate_shared_memory(gl.bfloat16, [depth, BT, 64], shared_layout)  # Pre-allocate
smem_w.index(0).store(first_w)  # prologue
for i in range(1, loop_n):
    w_dot = smem_w.index((i-1) % depth).load(layout=dot_op0)  # Consume
    next_w = gl.amd.cdna4.buffer_load(...)
    smem_w.index(i % depth).store(next_w)  # In-place overwrite, no new memory allocation
```

**Note**: CDNA4 has 160KB LDS (CDNA3 only 64KB), allowing larger shared memory buffer allocations, but unnecessary duplicate allocation should still be avoided.

---

## Error 4: Compilation Takes Too Long

**Symptom**: Compilation exceeds 5 minutes

**Cause**: Block size is too large or layout is too complex

**Solution**:
- Reduce BLOCK_SIZE_M, BLOCK_SIZE_N
- Use simpler layouts
- Reduce the number of constexpr values

---

## Error 5: MFMA Input Type Error

**Symptom**:
```
TypeError: mfma() inputs must be float16 or bfloat16
```

**Cause**: MFMA does not support int32 input

**Solution**:
```python
# ❌ Wrong
a = gl.zeros((M, K), dtype=gl.int32)
acc = gl.amd.cdna4.mfma(a, b, acc)

# ✅ Correct
a = gl.zeros((M, K), dtype=gl.float16)
acc = gl.amd.cdna4.mfma(a, b, acc)
```

---

## Error 6: Shared Memory Not Loaded Correctly (Missing DotOperandLayout)

**Symptom**:
```
RuntimeError: Shared memory load failed
```

**Cause**: DotOperandLayout was not used

**Solution**:
```python
# ❌ Wrong
a_dot = a_smem.load()  # Missing layout

# ✅ Correct
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)
```

---

## Error 7: Function Signature Mismatch

**Symptom**:
```
TypeError: missing required argument 'X'
```

**Cause**: Function definition does not match call parameters

**Solution**:
- Check all function definitions
- Ensure parameter names, counts, and order are exactly consistent
- Use the extract_function_signatures tool Chandler to verify

---

## Error 8: Static Analysis Reports Massive LSP False Positives

**Symptom**:
After modifying Gluon code, static analysis may report 20+ LSP errors similar to the following:
```
Type "AMDMFMALayout" is not assignable to declared type "constexpr"
Type "BlockedLayout" is not assignable to declared type "constexpr"
"constexpr" is not assignable to "DistributedLayout"
```

**Cause**: The static type checker does not understand Triton/Gluon's `gl.constexpr` type annotation system. These are **completely normal false positives** and do not affect compilation or execution.**Solution**:
1. **Ignore all LSP errors related to `constexpr`** — they will always exist and cannot be fixed
2. **Keep related source changes together** — when changing layout-heavy code, rewrite the affected block coherently rather than making many tiny edits
3. **Do not delete comments one by one** — if batch cleanup of comments is needed, rewrite the affected section as a whole

```python
# These patterns trigger LSP false positives, but are completely correct in Gluon:
mma: gl.constexpr = gl.amd.AMDMFMALayout(...)      # LSP reports error, but correct
blocked: gl.constexpr = gl.BlockedLayout(...)       # LSP reports error, but correct
dot_op0: gl.constexpr = gl.DotOperandLayout(...)    # LSP reports error, but correct
```

**How to distinguish real errors from false positives**: Only trust the results of `check_syntax.py` and `validate.py`, do not trust LSP diagnostics.

---

## Error 9: `gluon.constexpr` does not exist, use `gl.constexpr` instead

**Symptoms**:
```
AttributeError: module 'triton.experimental.gluon' has no attribute 'constexpr'
```

**Cause**: `constexpr` is defined in the `gluon.language` (i.e., `gl`) module, not in the `gluon` module itself. Some older examples use `gluon.constexpr`, but this attribute does not exist at runtime.

**Solution**:
```python
# ❌ Wrong
BLOCK_SIZE: gluon.constexpr

# ✅ Correct
BLOCK_SIZE: gl.constexpr

# Note: Difference between the two import methods:
from triton.experimental import gluon          # gluon module — for @gluon.jit
from triton.experimental.gluon import language as gl  # gl module — for gl.constexpr, gl.load etc.
```

**Note**: The difference between the two import methods:
```python
from triton.experimental import gluon          # gluon module — for @gluon.jit
from triton.experimental.gluon import language as gl  # gl module — for gl.constexpr, gl.load etc.
```

---

## Error 10: Dynamic `num_warps` causes invalid Layout parameters
**Symptoms**:
```
LLVM lowering failed
# or
RuntimeError: expected size per thread * bits per element to be 128 or 32
```
**Cause**: When the kernel dynamically computes `num_warps` (e.g., 4/8/16) via `@triton.heuristics`, the `warps_per_cta` in the Layout must match. However, Layout is constexpr and needs to be determined at compile time. If a fixed `warps_per_cta` extracted from TTGIR is used directly, it will cause an error when it does not match the actual `num_warps`.

**Solution**: Pass `NUM_WARPS` as an additional `gl.constexpr` parameter to the kernel, and compute the Layout dynamically inside the kernel:

```python
# ❌ Wrong: Layout warps_per_cta hardcoded
@gluon.jit
def kernel(X, BLOCK_SIZE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4], threads_per_warp=[64],
        warps_per_cta=[8],  # ← Hardcoded as 8, but actually could be 4 or 16
        order=[0])

# ✅ Correct: Dynamically compute Layout via constexpr parameters
@gluon.jit
def kernel(X, BLOCK_SIZE: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK_SIZE // (64 * NUM_WARPS)],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0])

# Launcher must pass both num_warps and NUM_WARPS
kernel[(grid,)](x_ptr, BLOCK_SIZE=block_size,
               num_warps=num_warps, num_stages=1,
               NUM_WARPS=num_warps)  # ← Pass additionally

# Working with @triton.heuristics: If num_warps is dynamically computed by heuristics,
# add an additional heuristic to set NUM_WARPS:
@triton.heuristics({
    "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    "num_warps": lambda args: 4 if args["TILE_N"] < 2048 else (8 if args["TILE_N"] < 4096 else 16),
    "NUM_WARPS": lambda args: args["num_warps"],  # ← Copy num_warps value to NUM_WARPS constexpr
})
```

**The Launcher side must pass both `num_warps` and `NUM_WARPS`**:
```python
kernel[(grid,)](x_ptr, BLOCK_SIZE=block_size,
               num_warps=num_warps, num_stages=1,
               NUM_WARPS=num_warps)  # ← Pass additionally
```

**Coordination with `@triton.heuristics`**: If `num_warps` is dynamically computed by heuristics, an additional heuristic needs to be added to set `NUM_WARPS`:
```python
@triton.heuristics({
    "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    "num_warps": lambda args: 4 if args["TILE_N"] < 2048 else (8 if args["TILE_N"] < 4096 else 16),
    "NUM_WARPS": lambda args: args["num_warps"],  # ← Copy num_warps value to NUM_WARPS constexpr
})
```

---

## Error 11: Small TILE size causes illegal 1D BlockedLayout

**Symptoms**:
```
LLVM ERROR: size_per_thread[0] computed to 0
# or
triton.compiler.errors.CompilationError: invalid layout parameters
```

**Cause**: 1D `BlockedLayout` requires `size_per_thread * threads_per_warp * warps_per_cta == TILE_SIZE`, and all values must be powers of 2 ≥ 1. For AMD GPUs, `threads_per_warp` is fixed at 64. When `TILE_SIZE < 64 * num_warps` (e.g., TILE_N=16, num_warps=4), `size_per_thread = 16 / (64 * 4) = 0`, making the Layout illegal.**Typical scenario**: Using `triton.next_power_of_2(N)` to compute TILE_N, when N is very small (e.g., N=10 → TILE_N=16).

**Solution**: Ensure that the minimum TILE size meets the Layout constraint in the heuristic:

```python
# ❌ Dangerous: TILE_N may be smaller than 64 * num_warps
def heur_tile_n(args):
    return triton.next_power_of_2(args["N"])

# ✅ Safe: Ensure minimum TILE_N = 64 * max_warps = 256
def heur_tile_n(args):
    raw = triton.next_power_of_2(args["N"])
    return max(raw, 256)  # Ensure TILE_N ≥ 64 * 4 (minimum num_warps)
```

**Key constraint formula** (1D):
```
TILE_SIZE ≥ threads_per_warp × warps_per_cta = 64 × num_warps
```

| num_warps | Minimum TILE_SIZE |
|:---------:|:-----------------:|
| 4 | 256 |
| 8 | 512 |
| 16 | 1024 |

**Increasing TILE_SIZE does not affect correctness** — excess elements will be filtered out by the mask, only wasting some registers.

---

## Error 12: 2D BlockedLayout parameter combination conflict

**Symptoms**:
```
triton.compiler.errors.CompilationError: invalid layout for 2D tensor
# or
LLVM ERROR: product of threads_per_warp must be 64
```

**Cause**: 2D `BlockedLayout` must simultaneously satisfy all of the following constraints:
```
spt[0] × tpw[0] × wpc[0] = TILE_DIM0     # dim0 coverage
spt[1] × tpw[1] × wpc[1] = TILE_DIM1     # dim1 coverage
tpw[0] × tpw[1] = 64                      # AMD warp size
wpc[0] × wpc[1] = num_warps               # total warps
All values must be powers of 2 ≥ 1
```

When the two tile dimensions are drastically disproportionate (e.g., TILE_N=1, TILE_K=8192), it is difficult to find a combination that satisfies all constraints.

**Solution**:

# When TILE_DIM0 >= 64 * NUM_WARPS:
layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[TILE_N // (64 * NUM_WARPS), TILE_K],
    threads_per_warp=[64, 1],
    warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0])

# Strategy B — Ensure larger dimension is large enough via heuristic
def heur_tile_n(args):
    raw = triton.cdiv(8192, args["TILE_K"])
    # Ensure TILE_N is large enough based on expected num_warps
    tile_size = raw * args["TILE_K"]
    if tile_size >= 4096:
        return max(raw, 1024)   # When num_warps=16
    elif tile_size >= 2048:
        return max(raw, 512)    # When num_warps=8
    else:
        return max(raw, 256)    # When num_warps=4

# Strategy C — Restrict the autotune configuration range
# ❌ May produce illegal layout autotune configurations
configs = [triton.Config({"TILE_N": n}) for n in [32, 64, 128, 256, 512, 1024]]

# ✅ Keep only configurations that can generate legal layouts
configs = [triton.Config({"TILE_N": n}) for n in [256, 512, 1024]]**Common valid 2D layout reference table**:

| TILE_N | TILE_K | num_warps | tpw | wpc | spt |
|:------:|:------:|:---------:|:---:|:---:|:---:|
| 1024 | 1 | 16 | [64,1] | [16,1] | [1,1] |
| 1024 | 8 | 16 | [64,1] | [16,1] | [1,8] |
| 256 | 32 | 4 | [64,1] | [4,1] | [1,32] |
| 4 | 256 | 4 | [1,64] | [1,4] | [4,1] |
| 1 | 8192 | 16 | [1,64] | [1,16] | [1,8] |

---

## Error 13: `extract_ttgir.py` only extracts the Layout of a single kernel

**Symptoms**: When running `extract_ttgir.py` on a multi-kernel file (e.g., an operator containing forward + backward), only the Layout information of one kernel is obtained.

**Cause**: `extract_ttgir.py` stops after encountering the first compilable kernel entry point impressions and does not automatically extract all kernels.

**Solution**: Create independent temporary entry files for each kernel and extract TTGIR separately:

```bash
# Create independent driver script for each kernel
# kernel_fwd_driver.py:
#   Import forward kernel from original file
#   Call it with appropriate parameters

python tools/extract_ttgir.py kernel_fwd_driver.py -o /tmp/fwd.ttgir
python tools/extract_ttgir.py kernel_bwd_driver.py -o /tmp/bwd.ttgir
```

**If kernels share the same BLOCK_SIZE and num_warps**, then their BlockedLayout is usually also the same Bain, allowing reuse of the same Layout. However, if kernels have different tile dimensions or warp counts, **they must be extracted separately**.---

## Error 14: The `eviction_policy` parameter is not supported in Gluon

**Symptoms**:
```
TypeError: load() got an unexpected keyword argument 'eviction_policy'
```

**Cause**: Triton's `tl.load(..., eviction_policy="evict_last")` is used to hint the L2 cache eviction policy. Gluon's `gl.load` and `gl.amd.cdna4.buffer_load` do not support this parameter.

**Solution**: Simply remove the `eviction_policy` parameter:
```python
# Triton
val = tl.load(ptr + offs, mask=mask, eviction_policy="evict_last")

# Gluon — Remove directly
val = gl.load(ptr + offs, mask=mask, other=0.0)
```

**Performance Impact**: No observable impact on the vast majority of kernels. `eviction_policy` is only meaningful under extreme cache pressure scenarios.

---

## Debugging Tips

### 1. Print Intermediate Results
```python
print("Intermediate value:", value)
print("Shape:", value.shape)
print("Dtype:", value.dtype)
```

### 2. Compare Triton/Gluon
```python
print("Triton output:", output_triton)
print("Gluon output:", output_gluon)
print("Difference:", (output_triton - output_gluon).abs().max())
```

### 3. Check Layout
```python
print("Layout:", idx.layout)
print("Shape:", idx.shape)
```

---

# CDNA4-Specific Errors

The following errors are specific to the CDNA4 (gfx950) architecture and do not apply to CDNA3 (gfx942).

---

## Error 15: In-thread transpose is disabled on gfx950

**Symptoms**:
```
Compilation fails or produces incorrect matrix multiplication results
```

**Cause**: The Triton compiler disables the in-thread transpose optimization for gfx950 (this pass is only enabled on gfx942/CDNA3). In CDNA3, the compiler can rearrange data in registers via in-thread transpose to match the MFMA input layout, but the CDNA4 compiler does not perform this operation.

**Solution**:
- Ensure data has the correct layout for MFMA consumption **before** storing it into shared memory
- Use `DotOperandLayout` to specify the correct operand layout when loading from shared memory
- Do not rely on the compiler to automatically transpose register data

```python
# ✅ Correct: Explicitly specify DotOperandLayout, ensure shared memory load directly produces correct layout
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
dot_op1 = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)  # Load directly with correct layout
b_dot = b_smem.load(dot_op1)
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
```

---

## Error 16: kpack=1 is enforced on gfx950

**Symptoms**:**Symptoms**:
```
Compiler internal error involving kpack parameter
```
**Cause**: The gfx950 compiler has a hard constraint that `kpack == 1` (`assert self.kpack == 1`). CDNA3 (gfx942) allows kpack > 1 to pack multiple K-dimension elements Kurdish order to improve MFMA throughput, but CDNA4 does not support this optimization.

**Solution**:
- Use `k_width` in `DotOperandLayout` instead of relying on kpack packing
- Do not copy kpack-related optimization logic from CDNA3 conversion code

```python
# ❌ Wrong: Assuming kpack > 1 (migrated from CDNA3 code)
# CDNA3 might use kpack=2 to pack fp16 elements
dot_op = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)  # Implies kpack=2

# ✅ Correct: CDNA4 kpack fixed at 1
dot_op = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)  # kpack=1 compatible
```

**Note**: This does not affect functional correctness (the compiler will assert failure), but extra care should be taken when reusing CDNA3 layout parameters.

---

## Error 17: Ping-pong scheduling is only active when using async_copy

**Symptoms**:
```
benchmark.py reports Gluon/Triton ratio > 1.15, but functional verification passes
```

**Cause**: CDNA4's ping-pong scheduling optimization (alternating between two sets of shared memory buffers to enable load/compute overlap) is only activated by the compiler when using `gl.amd.cdna4.async_copy.buffer_load_to_shared` or `gl.amd.cdna4.async_copy.global_load_to_shared`. Using regular `gl.amd.cdna4.buffer_load` will not trigger this optimization.

**Solution**: For kernels that require pipelining, use the async_copy API instead of regular buffer_load:

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy

# ❌ Regular buffer_load — does not trigger ping-pong scheduling
data = gl.amd.cdna4.buffer_load(ptr=base_ptr, offsets=offs, mask=mask, other=0.0)
smem.index(i).store(data)

# ✅ async_copy — triggers ping-pong scheduling, data directly from global → shared
async_copy.buffer_load_to_shared(
    dest=smem.index(i),
    ptr=base_ptr,
    offsets=offs,
    mask=mask,
    other=0.0
)
async_copy.commit_group()
# ... execute computation ...
async_copy.wait_group(num_outstanding=0)
data_dot = async_copy.load_shared_relaxed(smem.index(i), layout=dot_op)
```

## When to Use async_copy: When the Triton source uses `num_stages > 1`, or when the kernel requires load/compute overlap.

---

## Error 18: async_copy Layout Constraint — size_per_thread × bits_per_element Must Be 128 or 32

**Symptoms**:
```
LLVM lowering failed
# or
RuntimeError: expected size per thread * bits per element to be 128 or 32
```

**Cause**: The underlying hardware instructions for `async_copy.buffer_load_to_shared` and `async_copy.global_load_to_shared` require that the data width per thread be strictly 128 bits or 32 bits. That is: `size_per_thread[dim] × bits_per_element` must equal 128 or 32.

**Solution**: Adjust the `size_per_thread` of `BlockedLayout` to satisfy the constraint:

```python
# Common dtype bits_per_element:
#   float32 = 32 bits → size_per_thread = 4 (128-bit) or 1 (32-bit)
#   float16 = 16 bits → size_per_thread = 8 (128-bit) or 2 (32-bit)
#   bfloat16 = 16 bits → size_per_thread = 8 (128-bit) or 2 (32-bit)
#   int8 = 8 bits → size_per_thread = 16 (128-bit) or 4 (32-bit)

# Common dtype bits_per_element:
#   float32 = 32 bits → size_per_thread = 4 (128-bit) or 1 (32-bit)
#   float16 = 16 bits → size_per_thread = 8 (128-bit) or 2 (32-bit)
#   bfloat16 = 16 bits → size_per_thread = 8 (128-bit) or 2 (32-bit)
#   int8 = 8 bits → size_per_thread = 16 (128-bit) or 4 (32-bit)

# ❌ Wrong: float16 + size_per_thread=4 → 4×16=64 bits (not 128 or 32)
layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[4, 1],
    threads_per_warp=[16, 4],
    warps_per_cta=[4, 1],
    order=[1, 0])

# ✅ Correct: float16 + size_per_thread=8 → 8×16=128 bits
layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[8, 1],
    threads_per_warp=[8, 8],
    warps_per_cta=[4, 1],
    order=[1, 0])
```

**Recommendation**: Always use 128-bit width (i.e., `size_per_thread × bits = 128`), which is the hardware-optimal transfer granularity.

**Reference Table**:

| dtype | bits | size_per_thread (128-bit, recommended) | size_per_thread (32-bit) |
|:-----:|:----:|:--------------------------------------:|:------------------------:|
| float32 | 32 | 4 | 1 |
| float16 | 16 | 8 | 2 |
| bfloat16 | 16 | 8 | 2 |
| int8 | 8 | 16 | 4 |
| fp8 | 8 | 16 | 4 |

---

## Error 19: Mixing async_copy with Regular buffer_load/store Causes Performance Degradation

**Symptoms**:
```
benchmark.py reports Gluon/Triton ratio > 1.15, but functional verification passes
```

**Cause**: On CDNA4, `async_copy.buffer_load_to_shared` / `async_copy.global_load_to_shared` and regular `gl.amd.cdna4.buffer_load` / `gl.amd.cdna4.buffer_store` still complete **in-order**. Although async_copy itself is asynchronous (global → shared bypasses registers), if regular buffer_load/store operations are interleaved between async_copy and wait_group, the hardware forces a wait for async_copy to complete before executing subsequent memory operations, completely negating the asynchronous advantage.

**Solution**: **Strictly separate** async_copy operations from regular memory operations:

```python
# ❌ Wrong: async_copy and buffer_load interleaved
async_copy.buffer_load_to_shared(dest=smem_a.index(i), ptr=a_ptr, offsets=a_offs, mask=a_mask)
async_copy.commit_group()
# Mixing regular buffer_load before wait — forces serialization!
other_data = gl.amd.cdna4.buffer_load(ptr=b_ptr, offsets=b_offs, mask=b_mask, other=0.0)
async_copy.wait_group(num_outstanding=0)

# ✅ Correct: all async_copy concentrated, regular operations after wait
async_copy.buffer_load_to_shared(dest=smem_a.index(i), ptr=a_ptr, offsets=a_offs, mask=a_mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
# Do regular operations after wait completes
a_dot = async_copy.load_shared_relaxed(smem_a.index(i), layout=dot_op0)
other_data = gl.amd.cdna4.buffer_load(ptr=b_ptr, offsets=b_offs, mask=b_mask, other=0.0)
```

**Best Practices**:
- Whenever possible, load all data that requires prefetching via async_copy
- If certain data must be loaded via regular buffer_load (such as scalars or unaligned accesses), place it after `wait_group`
- Between `commit_group()` and `wait_group()`, only place compute operations (MFMA, etc.) and no memory operations

---

## Quick Checklist

See the complete verification checklist in `../common/verification_guide.md`.


## Related

- [API Mapping Reference (CDNA4 / MI355X)](api_mapping.md)
- [Triton → Gluon Conversion Guide (AMD CDNA4)](conversion-guide.md)
- [CDNA4 Layout Mapping (Triton → Gluon)](layouts.md)
- [Matrix Multiplication Patterns](matrix_multiply.md)
- [Memory Access Patterns](memory_access.md)
- [Common Errors and Solutions](../cdna3/common_pitfalls.md)
- [Common Errors and Solutions (NVIDIA Hopper)](../../../nvidia/hopper/converter/hopper/common_pitfalls.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
