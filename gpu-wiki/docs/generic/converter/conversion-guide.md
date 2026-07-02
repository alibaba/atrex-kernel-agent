# PyTorch → Triton Conversion Guide

## Applicability

This guide covers:
- "Convert this PyTorch code to a Triton kernel"
- "Help me implement this torch operator with Triton"
- "Generate a high-performance Triton kernel to replace this PyTorch code"
- "Accelerate this nn.Module forward function using Triton"

## ⚠️ Critical Pitfalls (Must Read)

The following are the most common pitfalls in PyTorch → Triton conversion. **Be sure to remember these before starting conversion**:

### 1. Full Conversion — All Operations Must Be Implemented in Triton
The PyTorch code provided by the user must be **fully converted to Triton**, without selectively skipping any operations. This includes:
- `nn.Linear` / `torch.matmul` → Use `tl.dot` to implement tiled matmul
- element-wise ops → Fuse into the same kernel or separate kernels
- reduction ops → Implement using `tl.sum` / `tl.max`, etc.

**Do not** leave any operations to be executed in PyTorch (e.g., `torch.nn.functional.linear`). The goal of conversion is to produce a pure Triton implementation.

### 2. Operator Fusion Is the Core Value
Fuse as many steps as possible into a single Triton kernel to reduce kernel launch overhead and intermediate tensor memory costs.
Typical fusion scenarios:
- matmul + bias + activation → Perform fused epilogue after the matmul K-loop
- scale + add + clamp + reduction → Single kernel

### 3. Reduction Operations Require Special Handling
Reduction operations like `torch.logsumexp`, `torch.softmax`, and `torch.sum(dim=)` require:
- Correct `tl.max` / `tl.sum` reduction
- Numerical stability handling (e.g., the log-sum-exp trick)
- Correct axis parameter

### 4. Numerical Precision
- Always use `tl.float32` for accumulators, converting back to the target precision only at the end
- `torch.clamp` → Use `tl.minimum(tl.maximum(x, min_val), max_val)` or a combination with `tl.where`
- Avoid precision loss caused by intermediate FP16/BF16 accumulation

### 5. `tl.math.tanh` and `tl.math.log1p` Do Not Exist
**Verified through actual practice**: In the current Triton version, `tl.math.tanh` and `tl.math.log1p` **do not exist**. Calling them directly will result in a compilation error `AttributeError`. They must be manually implemented:

```python
# ❌ Wrong: tl.math.tanh does not exist
tanh = tl.math.tanh(x)

# ✅ Correct: Implement tanh using sigmoid
tanh = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

# ❌ Wrong: tl.math.log1p does not exist
log1p = tl.math.log1p(x)

# ✅ Correct: Implement manually
log1p = tl.where(x > 0.001, tl.log(1.0 + x), x - x * x / 2.0)
```

**Affected activation functions**: All implementations that use tanh/log1p, such as Mish, Softplus, and GELU, need to be replaced.

### 6. nn.Linear's Weight Shape Is `[out_features, in_features]`
PyTorch `nn.Linear(in, out)` weight has shape `[out, in]`, computing `y = x @ weight.T + bias`.
When loading weight tiles in Triton tiled matmul, note that:
- Weight is stored as `[N, K]` (N=out, K=in)
- `tl.dot` requires `a[M,K] × b[K,N]`
- Therefore, you must `tl.dot(x_tile, tl.trans(w_tile), acc)` or transpose indices at load time

```python
# weight shape: [N, K], loading via [offs_n, offs_k] gives [BLOCK_N, BLOCK_K]
# tl.dot needs [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
# So must transpose w_tile:
w_tile = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                 mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
acc = tl.dot(a_tile, w_tile.T, acc, input_precision="ieee")  # Transpose!
```

### 7. `tl.dot` Does Not Default to IEEE FP32 Precision — Must Be Explicitly Specified
`tl.dot(a, b, acc)` may default to a lower-precision path (e.g., TF32), and FP32 matmul errors may be significant. When full precision is required, you must explicitly specify `input_precision="ieee"`.

```python
# ❌ Default precision may have large errors
acc = tl.dot(a, b, acc)

# ✅ Explicitly specify IEEE precision
acc = tl.dot(a, b, acc, input_precision="ieee")
```

Errors will be further amplified by subsequent reductions (e.g., exp in logsumexp).

### 8. Scalar Results from `tl.store` Must Also Include a Mask
Even for scalar reduction outputs (one value per row), `tl.store` must still include a mask; otherwise, `check_syntax.py` will raise an error:

```python
# ❌ Wrong: missing mask
tl.store(out_ptr + row, result)

# ✅ Correct: mask
tl.store(out_ptr + row, result, mask=row < M)
```

---

## Core Knowledge

### 1. Import Statements

```python
import torch
import triton
import triton.language as tl
```

### 2. Kernel Structure

```python
@triton.jit
def my_kernel(
# Pointer parameters
    x_ptr, w_ptr, bias_ptr, out_ptr,
    # Size parameters
    M, N, K,
    # Stride parameters
    stride_xm, stride_xk, stride_wn, stride_wk,
    stride_om, stride_on,
    # Compile-time constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    pid = tl.program_id(0)
    # ... kernel body ...
```

### 3. API Mapping (PyTorch → Triton)

For the complete mapping table, see: `api_mapping.md`
API not in the mapping table → Check `porting_rules.md`

---

## Output File Naming Rules (Mandatory)

Output file names must be based on the original PyTorch file name with suffixes appended:

| File | Naming Rule | Example |
|------|---------|------|
| Triton Conversion Result | `{original_filename}_triton.py` | `22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish_triton.py` |
| PyTorch Reference Implementation | `{original_filename}_ref.py` | `22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish_ref.py` |

The original file name = the full file name with the `.py` suffix removed. **Do not simplify or abbreviate the file name arbitrarily** (e.g., do not write it as `22_triton.py`).

---
1. Analyze PyTorch code
   - Identify computation graph: all operations must be converted to Triton
   - Analyze which operations can be fused into the same kernel
   - Determine input/output tensor shape, dtype, stride

2. Determine output filenames
   - Generate _triton.py and _ref.py filenames according to naming rules

3. Design Triton kernel
   - Decide tiling strategy: BLOCK_SIZE_M, BLOCK_SIZE_N, etc.
   - Decide grid dimensions and program ID mapping
   - Choose num_warps and num_stages

4. Implement kernel + wrapper
   - kernel: GPU function decorated with @triton.jit
   - wrapper: Python function responsible for allocating output tensors, computing grid, launching kernel

5. Check api_mapping.md / porting_rules.md → Has mapping → Convert directly

6. No mapping → Check Triton official documentation

7. After conversion, execute the following verifications in sequence, **all must pass to be considered complete**:
   a. check_syntax.py → Syntax verification
   b. Run triton code → Compilation verification
   c. validate.py → Functional verification (accuracy)
   d. benchmark.py → **Performance verification (must pass, cannot be skipped)**

### ⚠️ Completion Criteria (Blocking Requirements)

The completion criteria for a conversion task are that **all four verifications must pass**:

| Verification Item | Tool | Passing Condition | Can Be Skipped |
|--------|------|----------|------------|
| Syntax Verification | `check_syntax.py` | No errors | ❌ Cannot be skipped |
| Compilation Verification | Run directly | No exceptions | ❌ Cannot be skipped |
| Functional Verification | `validate.py` | Precision passes | ❌ Cannot be skipped |
| **Performance Verification** | **`benchmark.py`** | **Utilization meets target** | **❌ Cannot be skipped** |

**Passing only functional verification without passing performance verification = Task not complete.**

### Performance Evaluation Criteria

Performance evaluation is based on **hardware utilization**, not simple time comparison. The following must be calculated:

1. **Compute Utilization**
   ```
 actual TFLOPS = FLOPs / elapsed time(seconds)
 utilization = actual TFLOPS / peak TFLOPS × 100%
   ```

2. **Bandwidth Utilization**
   ```
 actualbandwidth = data(Bytes) / elapsed time(seconds)
 bandwidthutilization = actualbandwidth / peakbandwidth × 100%
   ```

3. **Comprehensive Assessment**
   Determine the bottleneck based on the operator type:
   - **Compute-bound** (e.g., matmul): Primarily based on compute utilization
   - **Memory-bound** (e.g., element-wise, reduction): Primarily based on bandwidth utilization

   For detailed hardware specifications, see: `hardware-specs/`

### Performance Passing Conditions

Use empirical judgment based on operator type and scale:

| Operator Type | Typical AI Range | Bottleneck Type | Passing Criteria |
|----------|-------------|---------|---------|
| Pure element-wise (add, mul, activation) | < 1 FLOPs/Byte | Memory-bound | Bandwidth utilization ≥ 50% |
| Reduction (sum, max, softmax) | 1-10 FLOPs/Byte | Memory-bound | Bandwidth utilization ≥ 40% |
| Fused ops (matmul+activation) | Depends on matrix size | Depends on AI | Corresponding utilization ≥ 40% |
| Large matrix matmul (M,N,K > 1024) | > 100 FLOPs/Byte | Compute-bound | Compute utilization ≥ 50% |

**Note**: The above values are empirical references. Actual evaluation should consider operator specifications, data types, and hardware characteristics comprehensively. The key is to ensure that hardware resources are being utilized reasonably.

### Validation Utilities

| Tool | Purpose | When to Invoke |
|------|------|----------|
| `check_syntax.py` | Syntax verification | After conversion is complete |
| `validate.py` | Functional verification | After syntax passes |
| `benchmark.py` | **Performance verification** | **Must run after functional verification passes** |

### Source Editing Strategy

**Core principle: keep source edits coherent and easy to review.**

Recommended approach:

| Scenario | Strategy |
|------|------|
| First-time Triton code generation | Create the complete file in one coherent pass |
| Batch modifications needed | Rewrite the affected function or file as a whole, rather than making many tiny line edits |
| Fixing a single bug | Make the smallest clear source change that addresses the defect |

---

## Performance Evaluation Method

### Roofline Analysis Method

```
Arithmetic Intensity (AI) = FLOPs / Bytes_transferred

If AI < Peak TFLOPS / Peak Bandwidth:
    → Memory Bound → Optimization Focus: Reduce memory access, fuse operators, increase data reuse
Else:
    → Compute Bound → Optimization Focus: Improve instruction throughput, increase parallelism
```

## Conversion Patterns

### Pattern 1: Pure Element-wise Fusion

```python
# PyTorch (multiple kernel launches)
h = torch.zeros(K, V)
for t in range(T):
    h = decay * h + k[t].T @ v[t]  # Each iteration is a separate kernel launch

# Triton (single kernel launch, fusing all operations)
@triton.jit
def fused_kernel(...):
    h = tl.zeros((K, V), dtype=tl.float32)
    for t in range(T):
        k_t = tl.load(k_ptr + t * stride_k, ...)
        v_t = tl.load(v_ptr + t * stride_v, ...)
        h = decay * h + k_t.T @ v_t  # All in one kernel
```

### Pattern 2: Reduction (e.g., logsumexp)

```python
# PyTorch
result = torch.logsumexp(x, dim=1, keepdim=True)

# Triton — each program processes one row
@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, M, N, stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=float('-inf'))
    max_val = tl.max(x, axis=0)  # Numerical stability
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    result = max_val + tl.log(sum_exp)
    tl.store(out_ptr + row, result)
```

### Pattern 3: Activation Functions

```python
# Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
@triton.jit
def mish(x):
    return x * tl.math.tanh(tl.math.log1p(tl.exp(x)))

# Softplus: ln(1 + exp(x)), numerically stable version
@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.math.log1p(tl.exp(x)))
```

### Pattern 4: Tiled Matmul (nn.Linear → tl.dot)

```python
# PyTorch: y = x @ weight.T + bias  (nn.Linear)
# Triton: tiled GEMM with K-loop, FP32 accumulator

@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                     mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(a, b.T, acc, input_precision="ieee")  # a:[M,K] x b.T:[K,N]

    # fused bias add
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    acc = acc + bias[None, :]

    # fused epilogue (scale, activation, etc.) can be directly appended here
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=mask)
```

**Key Points**:
- The weight shape of `nn.Linear(input_size, hidden_size)` is `[hidden_size, input_size]`
- Therefore `y = x @ weight.T + bias`, in Triton, after loading the weight, perform `.T` or adjust the indices
- Fused epilogue (scale, clamp, activation, etc.) can be directly appended after the K-loop
- Always use `tl.float32` for the accumulator
- The optimal values of `BLOCK_K` and `num_warps` depend on the target GPU architecture

### Pattern 5: 2D Tiling (Processing 2D Tensors)

```python
@triton.jit
def kernel_2d(
    x_ptr, out_ptr,
    M, N, stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptr + offs, mask=mask)
    # ... operations ...
    tl.store(out_ptr + offs, result, mask=mask)
```

### Pattern 6: Recurrent State — Manual K-Block Register-Resident

**Applicable Scenarios**: Kernels for RNN, SSM (Mamba), Delta Rule, Linear Attention, etc., that need to accumulate a state matrix `h[K, V]` across time steps.

**Core Problem**: The state `h` has a lifetime spanning the entire chunk loop (potentially hundreds of iterations) and must remain resident in registers. However, `h` is typically `[K, V]` (e.g., 128×128), and placing the entirety in registers would exceed the budget, causing spills.

**Solution**: Manually split along the K dimension into independent variables of a fixed width of 64, using variable names instead of indices.

```python
# PyTorch original approach:
# h: [K, V], each chunk does v_new = u - w @ h, then h += k^T @ v_new
# PyTorch original approach:
# h: [K, V], each chunk does v_new = u - w @ h, then h += k^T @ v_new
for i_t in range(NT):
    b_v_new = -w_chunk @ h + u_chunk    # [BT, V]
    h = h * decay + k_chunk.T @ b_v_new  # [K, V]

# Triton: Manually split h along K dimension into independent register variables
# Each b_hX is [64, BV], BV is further narrowed via grid sharding
b_h1 = tl.zeros([64, BV], dtype=tl.float32)
if K > 64:
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
if K > 128:
    b_h3 = tl.zeros([64, BV], dtype=tl.float32)
if K > 192:
    b_h4 = tl.zeros([64, BV], dtype=tl.float32)

for i_t in range(NT):
    # w @ h K-block accumulation: each b_hX independently participates in dot
    b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
    p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
    b_v_new += tl.dot(tl.load(p_w, boundary_check=(0, 1)), b_h1.to(w_dtype))
    if K > 64:
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
        b_v_new += tl.dot(tl.load(p_w, boundary_check=(0, 1)), b_h2.to(w_dtype))
    # ... b_h3, b_h4 follow the same pattern ...

    b_v_new = -b_v_new + tl.load(p_u, boundary_check=(0, 1))

    # k^T @ v_new K-block accumulation
    b_v_for_dot = b_v_new.to(k_dtype)
    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
    b_h1 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v_for_dot)
    if K > 64:
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
        b_h2 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v_for_dot)
    # ... b_h3, b_h4 follow the same pattern ...
```

# Why cannot use K-loop:
# - Triton does not support dynamic slicing on register tensors (b_h[i*64:(i+1)*64, :] unavailable)
# - If h is placed in shared memory for indexing, each access incurs additional 20-30 cycles latency
# - If register budget is exceeded, spills occur, each HBM round-trip costs ~400 cycles
# - Manual unrolling allows compiler to allocate registers precisely, achieving zero spills and zero indirect addressing throughout loop

# Reducing Register Pressure with Grid Sharding: V dimension is split via grid into extremely narrow BV (e.g., 16), keeping total b_hX per program manageable:
```python
# 4 × [64, 16] × 4 bytes (FP32) = 16 KB registers, acceptable
# 4 × [64, 128] × 4 bytes = 128 KB, far exceeding register budget
def grid(meta):
    return (triton.cdiv(V, meta["BV"]), N * H)  # BV=16 → V/16 programs
```
Columns along V dimension are completely independent (no reduction), so sharding incurs zero additional overhead.

---

## Optimization Strategies

### General Strategies (Applicable to All Backends)

#### 1. `@triton.heuristics` — Zero-Overhead Conditional Compilation [🔴 High Priority]

Convert Python-layer `None` checks into compile-time constants. The Triton compiler completely eliminates branches whose conditions are not satisfied, avoiding the need to maintain multiple kernel variants. Applicable to any kernel with optional parameters or optional computation paths.

#### 2. `do_not_specialize` — Avoid Recompilation Due to Dynamic Shapes [🔴 High Priority]

Dynamic dimensions such as sequence length T will trigger JIT recompilation every time they change. Use `do_not_specialize` to mark them as runtime parameters.

```python
@triton.jit(do_not_specialize=["T"])
def my_kernel(..., T, ...):
    # T is a runtime parameter, one compilation serves all lengths
    # H, K, V, BT, etc. remain as tl.constexpr, retaining compiler optimization space
```

**When to use**: When a parameter changes frequently between invocations (variable-length sequences within a batch, dynamic padding) and does not affect the tiling strategy.
**When not to use**: Parameters that affect BLOCK_SIZE or grid shape should remain constexpr.

#### 3. Grid Sharding to Reduce Register Pressure [🔴 High Priority]

When a kernel requires a large amount of register-resident data (e.g., recurrent state), use grid-dimension sharding to reduce the data volume per program.

```python
# State h is [K, V], V=128. If BV=128, each program needs 4×64×128×4B = 128KB → register explosion
# BV=16 reduces it to 4×64×16×4B = 16KB → manageable
def grid(meta):
    return (triton.cdiv(V, meta["BV"]), N * H)  # BV=16 → 8 programs share the V dimension
```

**Applicability**: The sharded dimension must have no reduction dependencies between slices (completely independent).
**Key tradeoff**: Smaller BV → lower register pressure, but higher program count. Sufficient programs are needed to fully occupy the GPU.

#### 4. Mixed-Precision Strategy — Long-term FP32 + Short-term BF16 [🔴 High Priority]

States accumulated across loop iterations use FP32 for precision, `tl.dot` inputs are cast to bf16 to utilize tensor cores, and results are accumulated back into FP32.
```python
b_h = tl.zeros([64, BV], dtype=tl.float32)  # Long-term state FP32

for i_t in range(NT):
    b_v_new += tl.dot(b_w, b_h.to(b_w.dtype))  # bf16 × bf16 → fp32 accumulation
    b_v_new = b_v_new.to(k.dtype.element_ty)     # Intermediate result converted to bf16
    b_h += tl.dot(b_k, b_v_new)                  # bf16 × bf16 → fp32 accumulation
```

**Note**: When `tl.dot` inputs are already bf16, `input_precision="ieee"` is not needed (that parameter only prevents TF32 truncation for fp32 inputs).

#### 5. Pointer Offset Precomputation [🟡 Medium Priority]

Compute all base address offsets and stride constants once before the loop; use only linear offsets inside the loop.

```python
h += (boh * H + i_h) * K * V
v += (bos * H + i_h) * V
k += (bos * Hg + i_h // (H // Hg)) * K
stride_v = H * V
stride_h = H * K * V

for i_t in range(NT):
    p_h = tl.make_block_ptr(h + i_t * stride_h, ...)  # Only i_t * stride_h changes
```

#### 6. GQA Head Mapping Performed Inside the Kernel [🟡 Medium Priority]

In GQA (Grouped Query Attention), key heads are fewer than value heads. The mapping is done directly inside the kernel using integer division—no need тек for Python-layer `repeat_interleave` pre-expansion.

```python
# H = value heads, Hg = key heads, heads_per_group = H // Hg
k += (bos * Hg + i_h // (H // Hg)) * K  # i_h is value head index, automatically maps to corresponding key head
```

Saves memory and bandwidth by avoiding the creation of intermediate expanded tensors.

#### 7. Variable-Length Batch (cu_seqlens) [🟡 Medium Priority]

Use `cu_seqlens` + precomputed `chunk_offsets` to support variable-length sequences in a packed batch, with constant-time indexing inside the kernel.

```python
# Python side precomputation
chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)

# Kernel side
if IS_VARLEN:
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
else:
    bos, eos = i_n * T, i_n * T + T
    boh = i_n * NT
```

Combined with `@triton.heuristics`'s `IS_VARLEN` conditional compilation, zero overhead for equal-length scenarios.

#### 8. `tl.make_block_ptr` vs `offs+mask` — Pure Syntactic Sugar [🟢 Low Priority]

`tl.make_block_ptr` + `boundary_check` is an equivalent shorthand for `offs+mask`. **Experimentally verified (TTGIR comparison + benchmark), both are lowered to identical instruction sequences at the TTGIR level, with performance difference < 0.5%.**

| | make_block_ptr | offs+mask |
|---|---|---|
| Code size | Less (one line vs five lines) | More |
| Performance | Same | Same |
| TTGIR | Same `async_copy_global_to_local` | Same |
| Use case | Cleaner for frequent 2D block load/store | Required when non-rectangular mask is needed |

Choose based purely on readability preference. In load/store-intensive kernels (e.g., recurrent state with 8+ loads/stores per chunk), `make_block_ptr` can significantly reduce boilerplate code.

#### 9. Memory Access Optimization [🔴 High Priority]

- Ensure coalesced access: inner loop follows the contiguous dimension
- Use `tl.max_contiguous` and `tl.multiple_of` to hint the compiler
- Avoid strided access patterns

### Backend Tuning

Different GPU backends (NVIDIA, AMD, etc.) have significant differences in optimal values for parameters such as `num_warps`, `num_stages`, and block size. Please refer to the documentation for the corresponding hardware architecture for specific tuning recommendations.

---

## Learning Approach (When Encountering Unknown APIs)

```
1. Check `porting_rules.md` (complete API reference)
2. Check `api_mapping.md` (quick reference)
3. Check Triton official documentation: https://triton-lang.org/main/python-api/triton.language.html
4. Check Triton GitHub examples: https://github.com/triton-lang/triton/tree/main/python/tutorials
5. Experimental verification: Write small test code to check if compilation passes
6. Record reusable findings in `api_mapping.md` when they apply generally
```

---

## Verification Methods

### Level 1: Syntax Verification (Fastest)
```bash
python tools/check_syntax.py generated.py
```

### Level 2: Compilation Verification
```bash
python generated.py
```

### Level 3: Functional Verification (Most Reliable)
```bash
python tools/validate.py generated.py reference.py --var-name result_gold
```

### Level 4: Performance Verification (Final Acceptance)
```bash
python tools/benchmark.py generated.py reference.py
```

Performance evaluation is based on hardware utilization calculations, combined with operator type and hardware specifications for comprehensive judgment.

---

## Reference Documentation

| Document | Content |
|------|------|
| `porting_rules.md` | Complete conversion rules and pattern reference |
| `api_mapping.md` | PyTorch → Triton API quick reference lookup table |

---

## Version Information

**Guide Version**: v2.0
**Last updated**: 2026-03-18
**Design Principles**: Performance-first + Automated Verification + Hardware Utilization Driven
**v2.0 Changes**: Added Recurrent State conversion pattern (Pattern 6); restructured optimization strategies into general + backend-specific categories; added general strategies such as @triton.heuristics, do_not_specialize, Grid sharding, GQA in-kernel, VarLen batch; make_block_ptr experimentally verified as pure syntactic sugar
**v2.2 Changes**: Separated hardware-specific tuning strategies into corresponding architecture directories, generic retains only general Triton programming knowledge

## Related

- **Knowledge files for this guide**: [Conversion Rules](porting_rules.md) | [API Mapping](api_mapping.md)
- **Downstream Conversion**: Triton output can be further converted to Gluon — see architecture-specific conversion documents under `converter/amd/` and `converter/nvidia/`
- **Optimization Knowledge**: For optimization of converted Triton kernels, see corresponding architecture documents under `kernel-opt/`
