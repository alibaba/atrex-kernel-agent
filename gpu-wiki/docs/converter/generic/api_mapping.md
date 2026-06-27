# API Mapping Reference: PyTorch → Triton

**Last Updated**: 2026-03-16
**Verification Status Legend**: ✅ Verified | ⚠️ Pending Verification | ❌ Not Applicable

---

## Element-wise Math Operations

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `x + y` | `x + y` | ✅ | Direct operator |
| `x - y` | `x - y` | ✅ | Direct operator |
| `x * y` | `x * y` | ✅ | Direct operator |
| `x / y` | `x / y` | ✅ | Direct operator |
| `x ** n` | `tl.math.pow(x, n)` | ✅ | Or manual expansion |
| `torch.abs(x)` | `tl.abs(x)` | ✅ | |
| `torch.neg(x)` | `-x` | ✅ | |
| `torch.exp(x)` | `tl.exp(x)` | ✅ | |
| `torch.exp2(x)` | `tl.exp2(x)` | ✅ | Natural logarithm |
| `torch.log(x)` | `tl.log(x)` | ✅ | |
| `torch.log2(x)` | `tl.log2(x)` | ✅ | |
| `torch.sqrt(x)` | `tl.sqrt(x)` | ✅ | 1/sqrt(x) |
| `torch.rsqrt(x)` | `tl.rsqrt(x)` | ✅ | |
| `torch.sin(x)` | `tl.sin(x)` | ✅ | |
| `torch.cos(x)` | `tl.cos(x)` | ✅ | |
| `torch.floor(x)` | `tl.math.floor(x)` | ✅ | ⚠️ `torch.ceil(x)` does not exist, must be implemented manually |
| `tl.math.ceil(x)` | `torch.erf(x)` | ✅ | |

---

## Special Math Functions

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.log1p(x)` | `tl.log(1.0 + x)` | ✅ | ⚠️ `tl.math.log1p` does not exist, must be implemented manually |
| `torch.expm1(x)` | `tl.exp(x) - 1.0` | ✅ | |
| `torch.fmod(x, y)` | `x % y` | ✅ | |
| `torch.maximum(x, y)` | `tl.maximum(x, y)` | ✅ | |
| `torch.minimum(x, y)` | `tl.minimum(x, y)` | ✅ | |

---

## Comparison & Logical Operations

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.where(cond, x, y)` | `tl.where(cond, x, y)` | ✅ | |
| `x > y` | `x > y` | ✅ | Returns bool tensor |
| `x < y` | `x < y` | ✅ | |
| `x >= y` | `x >= y` | ✅ | |
| `x == y` | `x == y` | ✅ | |
| `cond1 & cond2` | `cond1 & cond2` | ✅ | Bitwise AND |
| `cond1 \| cond2` | `cond1 \| cond2` | ✅ | Bitwise OR |

---

## Clamp / Clip

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.clamp(x, min, max)` | `tl.minimum(tl.maximum(x, min_val), max_val)` | ✅ | Combined implementation |
| `torch.clamp(x, min=val)` | `tl.maximum(x, val)` | ✅ | One-sided clamp |
| `torch.clamp(x, max=val)` | `tl.minimum(x, val)` | ✅ | One-sided clamp |
| `torch.relu(x)` | `tl.maximum(x, 0.0)` | ✅ | Equivalent to clamp(min=0) |

---

## Activation Functions

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.nn.functional.relu(x)` | `tl.maximum(x, 0.0)` | ✅ | |
| `torch.nn.functional.gelu(x)` | `0.5 * x * (1 + (2.0 * tl.sigmoid(2.0 * (0.7978845608 * (x + 0.044715 * x*x*x))) - 1.0))` | ✅ | ⚠️ tanh is implemented using sigmoid |
| `torch.nn.functional.silu(x)` | `x * tl.sigmoid(x)` | ✅ | SiLU = x * sigmoid(x) |
| `torch.nn.functional.mish(x)` | See expanded implementation below | ✅ | ⚠️ Both tanh and log1p must be implemented manually |
| `torch.nn.functional.softplus(x)` | `tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))` | ✅ | ⚠️ log1p is implemented using log(1+x) |
| `torch.nn.functional.leaky_relu(x, a)` | `tl.where(x >= 0, x, a * x)` | ✅ | |
| `torch.nn.functional.elu(x, alpha)` | `tl.where(x > 0, x, alpha * (tl.exp(x) - 1))` | ✅ | |

**Mish Full Implementation** (since `tl.math.tanh` and `tl.math.log1p` do not exist):
```python
# mish(x) = x * tanh(softplus(x))
softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
tanh_sp = 2.0 * tl.sigmoid(2.0 * softplus_x) - 1.0
mish_x = x * tanh_sp
```

## Reduction Operations

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.sum(x, dim)` | `tl.sum(x, axis=dim)` | ✅ | |
| `torch.max(x, dim)` | `tl.max(x, axis=dim)` | ✅ | Returns value only |
| `torch.min(x, dim)` | `tl.min(x, axis=dim)` | ✅ | Returns value only |
| `torch.mean(x, dim)` | `tl.sum(x, axis=dim) / N` | ✅ | Manually divide by element count |
| `torch.var(x, dim)` | Manual: `mean → (x-mean)^2 → sum / N` | ✅ | Requires two passes |
| `torch.logsumexp(x, dim)` | Manual: `max + log(sum(exp(x - max)))` | ✅ | Numerically stable version |
| `torch.softmax(x, dim)` | Manual: `exp(x-max) / sum(exp(x-max))` | ✅ | Numerically stable version |

---

## Type Casting

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `x.to(torch.float32)` | `x.to(tl.float32)` | ✅ | |
| `x.to(torch.float16)` | `x.to(tl.float16)` | ✅ | |
| `x.to(torch.bfloat16)` | `x.to(tl.bfloat16)` | ✅ | |
| `x.float()` | `x.to(tl.float32)` | ✅ | |
| `x.half()` | `x.to(tl.float16)` | ✅ | |
| `x.int()` | `x.to(tl.int32)` | ✅ | |

---

## Memory Access

| PyTorch Concept | Triton | Verification Status | Notes |
|-------------|--------|---------|------|
| `x[i]` (indexing) | `tl.load(ptr + offset, mask=mask)` | ✅ | |
| `x[i] = val` (assignment) | `tl.store(ptr + offset, val, mask=mask)` | ✅ | |
| `x[i:j]` (slicing) | `tl.load(ptr + tl.arange(0, N))` | ✅ | Implemented via arange |
| Broadcast | `x[:, None] + y[None, :]` | ✅ | Automatic broadcasting |

---

## Tensor Creation (inside kernel)

| Purpose | Triton | Verification Status | Notes |
|------|--------|---------|------|
| Index sequence | `tl.arange(0, N)` | ✅ | |
| All zeros | `tl.zeros((M, N), dtype=tl.float32)` | ✅ | |
| Fill with value | `tl.full((M,), value, dtype=tl.float32)` | ✅ | |

---

## Matrix Multiplication

| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.matmul(a, b)` | tiled K-loop + `tl.dot(a_tile, b_tile, acc)` | ✅ | 2D tiled GEMM |
| `nn.Linear(in, out)` | tiled K-loop + `tl.dot` + bias add | ✅ | weight shape [out, in], transpose required |
| `torch.mm(a, b)` | Same as matmul | ✅ | |
| `torch.bmm(a, b)` | Add batch dimension to pid | ✅ | grid includes batch dimension |

**Note**: `tl.dot(a, b)` requires a and b to be 2D tiles, with matching K dimensions. Accumulator uses FP32.

---

## Adding New Mappings

When encountering a new API, add it to this table in the following format:

```markdown
| PyTorch | Triton | Verification Status | Notes |
|---------|--------|---------|------|
| `torch.xxx` | `tl.yyy` | ⚠️ Pending Verification | Source: Official documentation |
```
