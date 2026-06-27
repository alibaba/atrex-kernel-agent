# PyTorch → Triton Complete Conversion Rules

## Core Principles

### 1. Full Conversion
**All operations** in the user-provided PyTorch code must be converted to Triton, including matmul (`nn.Linear`, `torch.matmul`). No operations may remain in PyTorch for execution.

### 2. Operator Fusion First
Fuse multiple operations into a single Triton kernel whenever possible to reduce GPU kernel launch overhead and intermediate tensor memory allocation/transfer. Typical fusion scenarios:
- matmul + bias + activation → fused epilogue after matmul K-loop
- scale + add + clamp + reduction → single kernel

### 3. Numerical Stability
All reduction operations (softmax, logsumexp, etc.) must implement numerically stable versions (subtract max value). Use `tl.float32` for accumulators.

---

## Kernel Structure Specification

### Basic Structure
```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    # Input pointers
    input_ptr,
    # Output pointers
    output_ptr,
    # Dimensions
    M, N,
    # Strides (in elements)
    stride_m, stride_n,
    # Scalar parameters
    scale_factor,
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Compute program ID
    pid = tl.program_id(0)

    # 2. Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # 3. Load data
    x = tl.load(input_ptr + offs, mask=mask)

    # 4. Compute
    y = x * scale_factor

    # 5. Store result
    tl.store(output_ptr + offs, y, mask=mask)
```

### Wrapper Function Structure
```python
def my_wrapper(input_tensor, scale_factor):
    # 1. Determine output shape and dtype
    output = torch.empty_like(input_tensor)

    # 2. Get dimensions and strides
    N = input_tensor.numel()

    # 3. Determine BLOCK_SIZE and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # 4. Launch kernel
    my_kernel[grid](
        input_tensor, output,
        N,
        scale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return output
```

---

## Memory Access Patterns

### 1D Contiguous Access
```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
x = tl.load(ptr + offs, mask=mask)
```

### 2D Tile Access
```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

# 2D offset
offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

x = tl.load(ptr + offs, mask=mask)
```

### Row-wise Processing (for reduction)
```python
# program row
row = tl.program_id(0)
offs = tl.arange(0, BLOCK_N)
mask = offs < N

x = tl.load(ptr + row * stride + offs, mask=mask, other=float('-inf'))
# or other=0.0, reduction type
```

---

## Reduction Patterns

### sum reduction (along an axis)
```python
# loadrowdata
row_data = tl.load(ptr + row * stride + tl.arange(0, BLOCK_N), mask=mask, other=0.0)
row_sum = tl.sum(row_data, axis=0)
```

### max reduction
```python
row_data = tl.load(ptr + row * stride + tl.arange(0, BLOCK_N), mask=mask, other=float('-inf'))
row_max = tl.max(row_data, axis=0)
```

### logsumexp (numerically stable)
```python
x = tl.load(ptr + row * stride + offs, mask=mask, other=float('-inf'))
max_val = tl.max(x, axis=0)
x_shifted = x - max_val
exp_x = tl.exp(x_shifted)
sum_exp = tl.sum(exp_x, axis=0)
result = max_val + tl.log(sum_exp)
```

### softmax (numerically stable)
```python
x = tl.load(ptr + row * stride + offs, mask=mask, other=float('-inf'))
max_val = tl.max(x, axis=0)
x_shifted = x - max_val
exp_x = tl.exp(x_shifted)
sum_exp = tl.sum(exp_x, axis=0)
softmax_out = exp_x / sum_exp
```

## Activation Implementation Reference

### ReLU
```python
y = tl.maximum(x, 0.0)
```

### GELU (tanh Approximation)
```python
# ⚠️ tl.math.tanh does not exist, implement using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
y = 0.5 * x * (1.0 + tanh_inner)
```

### SiLU (Swish)
```python
y = x * tl.sigmoid(x)
```

### Mish
```python
# ⚠️ tl.math.tanh and tl.math.log1p do not exist, must be implemented manually
# mish(x) = x * tanh(softplus(x))
# softplus(x) = log(1 + exp(x)), numerically stable version adds tl.where
# tanh(x) = 2 * sigmoid(2x) - 1
softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
tanh_sp = 2.0 * tl.sigmoid(2.0 * softplus_x) - 1.0
y = x * tanh_sp
```

### LeakyReLU
```python
y = tl.where(x >= 0, x, alpha * x)
```

---

## Grid Computation

### 1D Grid
```python
BLOCK = 1024
grid = (triton.cdiv(N, BLOCK),)
kernel[grid](...)
```

### 2D Grid
```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
kernel[grid](...)
```

### Row-wise Grid (Reduction)
```python
# program row
grid = (M,)
kernel[grid](...)
```

---

## Complete Triton API Reference

### Program Control
- `tl.program_id(axis)` — ID of the current program along the axis dimension
- `tl.num_programs(axis)` — Total number of programs along the axis dimension

### Memory Access
- `tl.load(ptr, mask=None, other=None)` — Load from global memory
- `tl.store(ptr, val, mask=None)` — Write to global memory

### Tensor Creation
- `tl.arange(start, end)` — Integer sequence in range [start, end)
- `tl.zeros(shape, dtype)` — Tensor filled with zeros
- `tl.full(shape, value, dtype)` — Tensor filled with a constant value

### Math Operations
- `tl.exp(x)`, `tl.exp2(x)`, `tl.log(x)`, `tl.log2(x)`
- `tl.sin(x)`, `tl.cos(x)`, `tl.sqrt(x)`, `tl.rsqrt(x)`
- `tl.abs(x)`, `tl.sigmoid(x)`, `tl.erf(x)`
- `tl.maximum(x, y)`, `tl.minimum(x, y)`
- `tl.where(cond, x, y)`

### tl.math Submodule
- `tl.math.tanh(x)`, `tl.math.log1p(x)`, `tl.math.erf(x)`
- `tl.math.floor(x)`, `tl.math.ceil(x)`
- `tl.math.pow(x, y)`, `tl.math.sqrt(x)`

### Reduction
- `tl.sum(x, axis=0)` — Sum along the specified axis
- `tl.max(x, axis=0)` — Max value along the specified axis
- `tl.min(x, axis=0)` — Min value along the specified axis

### Matrix Multiplication
- `tl.dot(a, b, acc=None)` — 2D tiled matrix multiplication

### Types
- `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`
- `tl.int8`, `tl.int16`, `tl.int32`, `tl.int64`
- `tl.uint8`, `tl.uint16`, `tl.uint32`, `tl.uint64`
- `tl.constexpr` — Compile-time constant

### Utility
- `tl.cdiv(x, y)` — Ceiling division
- `tl.multiple_of(x, vals)` — Alignment hint
- `tl.max_contiguous(x, vals)` — Contiguity hint
- `tl.expand_dims(x, axis)` — Expand dimensions

### Atomic Operations
- `tl.atomic_add(ptr, val, mask=None)`
- `tl.atomic_max(ptr, val, mask=None)`
- `tl.atomic_min(ptr, val, mask=None)`
- `tl.atomic_cas(ptr, cmp, val, mask=None)`

---

## Common Errors

### 1. Forgetting the Mask
```python
# ❌ Dangerous: potential out-of-bounds access
x = tl.load(ptr + offs)

# ✅ Correct: always use mask
x = tl.load(ptr + offs, mask=offs < N, other=0.0)
```

### 2. Numerically Unstable Reduction
```python
# ❌ Numerically unstable
result = tl.log(tl.sum(tl.exp(x), axis=0))

# ✅ Numerically stable (log-sum-exp trick)
max_val = tl.max(x, axis=0)
result = max_val + tl.log(tl.sum(tl.exp(x - max_val), axis=0))
```

### 3. Accumulator Precision
```python
# ❌ FP16 accumulation, poor precision
acc = tl.zeros((M, N), dtype=tl.float16)

# ✅ FP32 accumulation, convert back at end
acc = tl.zeros((M, N), dtype=tl.float32)
# ... computation ...
result = acc.to(tl.float16)
```

### 4. Forgetting to handle keepdim
```python
# PyTorch: torch.logsumexp(x, dim=1, keepdim=True)  → shape (M, 1)
# Triton: tl.sum returns scalar or 1D → need to manually handle broadcast or store logic
```

### 5. Missing mask in tl.store scalar result
```python
# ❌ Even if logically not out-of-bounds, will be rejected by check_syntax
tl.store(out_ptr + row, result)

# ✅ Always add mask
tl.store(out_ptr + row, result, mask=row < M)
```

### 6. nn.Linear weight transpose omission
```python
# nn.Linear(in_features, out_features) weight shape = [out_features, in_features] = [N, K]
# Computation: y = x @ weight.T + bias
# tl.dot requires a[M,K] x b[K,N], so weight tile transpose is needed

# ❌ Wrong: dot without transpose, dimension mismatch or result all wrong
acc = tl.dot(x_tile, w_tile, acc)  # w_tile is [BLOCK_N, BLOCK_K]

# ✅ Correct: transpose weight tile
acc = tl.dot(x_tile, tl.trans(w_tile), acc)  # [M,K] x [K,N]
```

### 7. tl.dot does not specify input_precision="ieee"
```python
# ❌ Default precision may use low-precision path (e.g., TF32), large error
acc = tl.dot(a_tile, b_tile, acc)

# ✅ For FP32 matmul needing full precision, specify IEEE precision
acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
```

### 8. Using non-existent tl.math API
```python
# ❌ The following APIs do not exist in current Triton version
tl.math.tanh(x)   # AttributeError
tl.math.log1p(x)  # AttributeError

# ✅ Implement manually
tanh_x = 2.0 * tl.sigmoid(2.0 * x) - 1.0  # tanh
log1p_x = tl.log(1.0 + x)                   # log1p
```
