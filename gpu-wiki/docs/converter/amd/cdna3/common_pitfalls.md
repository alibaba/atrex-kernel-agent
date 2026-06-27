# Common Errors and Solutions

**Last updated**: 2026-03-12

---

## Error 1: Mask argument cannot be block type

**Symptom**:
```
ValueError: Mask argument cannot be block type, got <...>
```

**Cause**: Mask dimensions do not match the offsets dimensions

**Solution**:
```python
# ❌ Wrong
mask = offs_m[:, None] < M  # 2D mask but offsets is 1D

# ✅ Correct
mask = (offs_m < M)[:, None]  # 1D first then expand
```

**Prevention**:
- Ensure the mask shape matches the offsets
- Create a 1D mask first, then expand with expand_dims

---

## Error 2: Unsupported ptr type

**Symptom**:
```
ValueError: Unsupported ptr type for buffer_load
```

**Cause**: An unsupported pointer type was used

**Solution**:
```python
# ❌ Wrong
ptr = base_ptr + offsets  # tensor pointer

# ✅ Correct
val = gl.amd.cdna3.buffer_load(
    ptr=base_ptr,  # scalar pointer
    offsets=offsets,
    mask=mask,
    other=0.0
)
```

---

## Error 3: Numerical precision mismatch

**Symptom**:
```
torch.allclose failed, max difference: 0.05
```

**Cause**: Incorrect accumulator dtype

**Solution**:
# ❌ Wrong
acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float16)

# ✅ Correct
acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32)

# After computation is complete
acc = acc.to(gl.float16)  # Convert for storage
---

## Error 4: Layout undefined

**Symptom**:
```
TypeError: arange() missing required argument: 'layout'
```

**Cause**: Forgot to specify the layout

**Solution**:
```python
# ❌ Wrong
idx = gl.arange(0, BLOCK_SIZE)

# ✅ Correct
layout: gl.constexpr = gl.BlockedLayout(...)
idx = gl.arange(0, BLOCK_SIZE, layout=layout)
```

---

## Error 5: Shared Memory limit exceeded (OOM)

**Symptom**:
```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 67584, Hardware limit: 65536
```

**Cause**: Using `allocate_shared_memory(value=...)` to overwrite smem in a loop, causing both old and new buffers to be active simultaneously.

**Solution**: For persistent buffers reused across iterations, use `smem.index(i).store()` for in-place overwriting:
```python
# ❌ Wrong: allocate_shared_memory allocates new physical memory each time
for i_t in range(NT - 1):
    w_dot = w_smem.load(dot_op0)
    next_w = gl.amd.cdna3.buffer_load(...)
    w_smem = gl.allocate_shared_memory(..., value=next_w)  # ← New allocation, old smem not released

# ✅ Correct: pre-allocate + index/store in-place overwrite
smem_w = gl.allocate_shared_memory(gl.bfloat16, [depth, BT, 64], shared_layout)  # Pre-allocate
smem_w.index(0).store(first_w)  # prologue
for i in range(1, loop_n):
    w_dot = smem_w.index((i-1) % depth).load(layout=dot_op0)  # Consume
    next_w = gl.amd.cdna3.buffer_load(...)
    smem_w.index(i % depth).store(next_w)  # In-place overwrite, no new memory allocation
```

See `pipeline.md`.

---

## Error 6: Compilation takes too long

**Symptom**: Compilation exceeds 5 minutes

**Cause**: Block size too large or layout too complex

**Solution**:
- Reduce BLOCK_SIZE_M, BLOCK_SIZE_N
- Use simpler layouts
- Reduce the number of constexpr values

---

## Error 6: Incorrect MFMA input type

**Symptom**:
```
TypeError: mfma() inputs must be float16 or bfloat16
```

**Cause**: MFMA does not support int32 input

**Solution**:
```python
# ❌ Wrong
a = gl.zeros((M, K), dtype=gl.int32)
acc = gl.amd.cdna3.mfma(a, b, acc)

# ✅ Correct
a = gl.zeros((M, K), dtype=gl.float16)
acc = gl.amd.cdna3.mfma(a, b, acc)
```

---

## Error 7: Shared memory not loaded correctly

**Symptom**:
```
RuntimeError: Shared memory load failed
```

**Cause**: DotOperandLayout is not used

**Solution**:
```python
# ❌ Wrong
a_dot = a_smem.load()  # Missing layout

# ✅ Correct
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)
```

---

## Error 8: Function signature mismatch

**Symptom**:
```
TypeError: missing required argument 'X'
```

**Cause**: Function definition does not match call parameters

**Solution**:
- Check all function definitions
- Ensure parameter names, count, and order are fully consistent
- Use the extract_function_signatures tool to verify

---

## Error 9: Static Analysis Reports Many LSP False Positives

**Symptoms**:
After modifying Gluon code, static analysis may report 20+ LSP errors similar to the following:
```
Type "AMDMFMALayout" is not assignable to declared type "constexpr"
Type "BlockedLayout" is not assignable to declared type "constexpr"
"constexpr" is not assignable to "DistributedLayout"
```

**Cause**: The static type checker does not understand Triton/Gluon's `gl.constexpr` type annotation system. These are **completely normal false positives** and do not affect compilation or execution.

**Solution**:
1. **Ignore all LSP errors involving `constexpr`** — they will always exist and cannot be fixed
2. **Keep related source changes together** — when changing layout-heavy code, rewrite the affected block coherently rather than making many tiny edits
3. **Do not delete comments line by line** — if batch cleanup of comments is needed, rewrite the affected section as a whole

```python
# These patterns trigger LSP false positives, but are completely correct in Gluon:
mma: gl.constexpr = gl.amd.AMDMFMALayout(...)      # LSP reports error, but correct
blocked: gl.constexpr = gl.BlockedLayout(...)       # LSP reports error, but correct
dot_op0: gl.constexpr = gl.DotOperandLayout(...)    # LSP reports error, but correct
```

**How to distinguish true and false errors**: Only trust the results of `check_syntax.py` and `validate.py`, do not trust LSP diagnostics.

---

## Debugging Tips

### 1. Print intermediate results
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

## Error 10: Skipping pipeline implementation leads to substandard performance

**Symptoms**:
```
benchmark.py reports Gluon/Triton ratio > 1.15 (e.g., 1.50, i.e., 50% slower)
```

**Cause**: The Triton source code uses `num_stages > 1` (e.g., `num_stages=2`), and the Triton compiler automatically generates software pipelining (load/compute overlap). During conversion, only `num_stages=1` is set but an equivalent pipeline is not manually implemented in the Gluon kernel, causing all global memory loads to become synchronous and blocking.

**Typical scenario**: Seeing `ttg.local_alloc : () -> !ttg.memdesc<Nx...>` and `ttg.memdesc_index` in TTGIR means the original Triton used pipelining.

**Solution**: Implement a three-stage pipeline (prologue + main loop + epilogue):
```python
# 1. Pre-allocate persistent smem (without value=)
smem = gl.allocate_shared_memory(dtype, [1, M, K], shared_layout)

# 2. PROLOGUE: prefetch iter 0 data
data_0 = gl.amd.cdna3.buffer_load(...)
smem.index(0).store(data_0)
pf_reg = gl.amd.cdna3.buffer_load(...)  # Data bypassing smem

# 3. MAIN LOOP (0 to NT-2): compute current + prefetch next iteration
for i_t in range(NT - 1):
    next_data = gl.amd.cdna3.buffer_load(...)    # Issue load for next iteration
    next_reg = gl.amd.cdna3.buffer_load(...)

    dot = smem.index(0).load(layout=dot_op)      # Consume current smem
    acc = gl.amd.cdna3.mfma(dot, other, acc)
    # ... use pf_reg for computation ...

    smem.index(0).store(next_data)               # Overwrite smem
    pf_reg = next_reg                            # Pass register data

# 4. EPILOGUE: Last iteration, no prefetch
dot = smem.index(0).load(layout=dot_op)
acc = gl.amd.cdna3.mfma(dot, other, acc)
```

**Performance comparison** (actual case: chunk_gdn kernel, T=9418, K=128):

| Implementation | Gluon time | Triton time | Ratio | Result |
|---------------|-----------|------------|-------|--------|
| No pipeline (num_stages=1) | 0.999 ms | 0.667 ms | 1.497 | ❌ Failed |
| Manual pipeline (prologue/loop/epilogue) | 0.726 ms | 0.651 ms | 1.114 | ✅ Passed |

**Verification method**: Before conversion, you must check the value of the `num_stages` parameter in the Triton wrapper. If > 1, you **must** implement pipelining and cannot skip it.

For more details, see `pipeline.md`.

---

## Error 11: Reporting Completion Without Running Benchmark Verification

**Symptom**: After passing functional verification (`validate.py`), the report directly states "conversion complete", but `benchmark.py` has not been run.

**Consequence**: Kernels that are functionally correct but 40-70% slower in performance are delivered as finished products.

**Rule**: The completion criterion for conversion tasks is **all four verifications passed**:
1. `check_syntax.py` — Syntax ✅
2. Compile and run — No exceptions ✅
3. `validate.py` — Accuracy ✅
4. **`benchmark.py` — Performance ✅ (cannot be skipped)**

**Passing only the first three = the task is not complete.** Benchmark must be run and the ratio confirmed to be within the 85%-115% range.

---

## ❌ Wrong
BLOCK_SIZE: gluon.constexpr

# ✅ Correct
BLOCK_SIZE: gl.constexpr

# Note: Difference between the two import methods:
from triton.experimental import gluon          # gluon module — for @gluon.jit
from triton.experimental.gluon import language as gl  # gl module — for gl.constexpr, gl.load etc.
---

## ❌ Wrong: Layout warps_per_cta hardcoded
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
---

## Error 14: Small TILE Size Causes Illegal 1D BlockedLayout

**Symptom**:
```
LLVM ERROR: size_per_thread[0] computed to 0
# or
triton.compiler.errors.CompilationError: invalid layout parameters
```

**Cause**: 1D `BlockedLayout` requires `size_per_thread * threads_per_warp * warps_per_cta == TILE_SIZE`, and all values must be powers of 2 ≥ 1. For AMD GPUs, `threads_per_warp` is fixed at 64. When `TILE_SIZE < 64 * num_warps` (e.g., TILE_N=16, num_warps=4), `size_per_thread = 16 / (64 * 4) = 0`, making the Layout illegal.

**Typical Scenario**: Using `triton.next_power_of_2(N)` to compute TILE_N, when N is very small (e.g., N=10 → TILE_N=16).**Solution**: In the heuristic, ensure the minimum TILE size meets the Layout constraints:

```python
# ❌ Dangerous: TILE_N may be smaller than 64 * num_warps
def heur_tile_n(args):
    return triton.next_power_of_2(args["N"])

# ✅ Safe: Ensure minimum TILE_N = 64 * max_warps = 256
def heur_tile_n(args):
    raw = triton.next_power_of_2(args["N"])
    return max(raw, 256)  # Ensure TILE_N ≥ 64 * 4 (minimum num_warps)
```

**Key Constraint Formula** (1D):
```
TILE_SIZE ≥ threads_per_warp × warps_per_cta = 64 × num_warps
```

| num_warps | Minimum TILE_SIZE |
|:---------:|:-------------:|
| 4 | 256 |
| 8 | 512 |
| 16 | 1024 |

**Increasing TILE_SIZE does not affect correctness** — excess elements will be filtered out by the mask, only wasting some registers.

---

## Error 15: Conflicting 2D BlockedLayout Parameter Combinations

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
All values must be ≥ 1 powers of 2
```

When the two tile dimensions are significantly disproportionate (e.g., TILE_N=1, TILE_K=8192), it is difficult to find a combination that satisfies all constraints.

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
**Strategy C — Restrict the autotune configuration range**:
```python
# ❌ May produce illegal layout autotune configurations
configs = [triton.Config({"TILE_N": n}) for n in [32, 64, 128, 256, 512, 1024]]

# ✅ Keep only configurations that can generate legal layouts
configs = [triton.Config({"TILE_N": n}) for n in [256, 512, 1024]]
```

**Common Valid 2D Layout Reference Table**:

| TILE_N | TILE_K | num_warps | tpw | wpc | spt |
|:------:|:------:|:---------:|:---:|:---:|:---:|
| 1024 | 1 | 16 | [64,1] | [16,1] | [1,1] |
| 1024 | 8 | 16 | [64,1] | [16,1] | [1,8] |
| 256 | 32 | 4 | [64,1] | [4,1] | [1,32] |
| 4 | 256 | 4 | [1,64] | [1,4] | [4,1] |
| 1 | 8192 | 16 | [1,64] | [1,16] | [1,8] |

---

## Error 16: `extract_ttgir.py` Only Extracts the Layout of a Single Kernel

**Symptoms**: For multi-kernel files (e.g., operators containing forward + backward), after running `extract_ttgir.py`, only the Layout information of a single kernel is obtained.

**Cause**: `extract_ttgir.py` stops after encountering the first compilable kernel entry point presets and does not automatically extract all kernels.

**Solution**: Create independent temporary entry files for each kernel and extract the TTGIR separately:

```bash
# Create independent driver script for each kernel
# kernel_fwd_driver.py:
#   Import forward kernel from original file
#   Call it with appropriate parameters

python tools/extract_ttgir.py kernel_fwd_driver.py -o /tmp/fwd.ttgir
python tools/extract_ttgir.py kernel_bwd_driver.py -o /tmp/bwd.ttgir
```

**If kernels share the same BLOCK_SIZE and num_warps**, their BlockedLayout is typically also the same, and a single Layout can be reused. However, if kernels have different tile dimensions or warp counts, **they must be extracted separately**.

## Error 17: `eviction_policy` parameter is not supported in Gluon

**Symptom**:
```
TypeError: load() got an unexpected keyword argument 'eviction_policy'
```

**Cause**: Triton's `tl.load(..., eviction_policy="evict_last")` is used to hint at the L2 cache eviction policy. Gluon's `gl.load` and `gl.amd.cdna3.buffer_load` do not support this parameter.

**Solution**: Simply remove the `eviction_policy` parameter:
```python
# Triton
val = tl.load(ptr + offs, mask=mask, eviction_policy="evict_last")

# Gluon — Remove directly
val = gl.load(ptr + offs, mask=mask, other=0.0)
```

**Performance Impact**: No observable impact for the vast majority of kernels. `eviction_policy` is only meaningful in extreme cache pressure scenarios.

---

## Quick Checklist

See the full verification checklist in `../common/verification_guide.md`.
