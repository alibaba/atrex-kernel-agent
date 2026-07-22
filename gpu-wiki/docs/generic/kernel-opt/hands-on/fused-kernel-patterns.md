# Fused Kernel Pattern

## Pattern: Cross-Entropy Loss Fusion

**Source**: `triton-tutorials/08-grouped-gemm.py`, `triton-kernels/`

Fuse softmax + log + nll_loss into a single kernel to avoid writing intermediate results back to HBM:

```python
@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr, ...):
    # 1. Load a row of logits
    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))

    # 2. Online max (numerical stability)
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val

    # 3. Compute log(sum(exp(x)))
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp)

    # 4. Get the logit corresponding to label
    label = tl.load(labels_ptr + pid)
    target_logit = tl.load(logits_ptr + pid * vocab_size + label)

    # 5. loss = log_sum_exp - (target_logit - max_val)
    loss = log_sum_exp - (target_logit - max_val)
    tl.store(loss_ptr + pid, loss)
```

## Pattern: Element-wise + Reduction Fusion

```python
# Anti-pattern: Write separately
x = relu(x)           # kernel 1: Read HBM → Write HBM
y = layernorm(x)      # kernel 2: Read HBM → Write HBM

# Correct: Fuse into single kernel
@triton.jit
def fused_relu_layernorm(x_ptr, y_ptr, ...):
    x = tl.load(x_ptr + offsets)
    x = tl.maximum(x, 0)  # relu in-register
    # layernorm in-register
    mean = tl.sum(x, axis=1) / N
    var = tl.sum((x - mean) ** 2, axis=1) / N
    y = (x - mean) / tl.sqrt(var + eps)
    tl.store(y_ptr + offsets, y)
```

**Fusion Criteria**:
- element-wise → element-wise: **always fuse**
- element-wise → reduction: **usually fuse** (same row)
- reduction → element-wise: **usually fuse** (e.g. layernorm = reduce + scale)
- matmul → element-wise epilogue: **depends on whether it exceeds register budget**

---

## Related Documents

- **Same Series**: [Online Softmax and Flash Attention](online-softmax-flash-attention.md) — specific implementation of softmax fusion
- **Same Series**: [Memory Access Optimization](memory-access-optimization.md) — reducing HBM access is the core goal of fusion
- **Prerequisite Knowledge**: [GPU Memory Hierarchy](../../ref-docs/gpu-memory-hierarchy.md) — understand the bandwidth differences between register vs shared memory vs HBM
- **Prerequisite Knowledge**: [GPU Instruction-Level Optimization](../../ref-docs/gpu-instruction-optimization.md) — arithmetic instruction throughput
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — overview of all patterns
