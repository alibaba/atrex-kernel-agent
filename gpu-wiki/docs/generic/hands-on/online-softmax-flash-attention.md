# Online Softmax and Flash Attention


**Last updated**: 2026-07-01

## Pattern: Online Softmax Avoids Two-Pass Computation

**Source**: `triton-tutorials/06-fused-attention.py`, `flash-attention/flash_attn_triton_*.py`

```python
# Traditional softmax: pass1 compute max, pass2 compute sum, pass3 normalize — 3 passes
# Online softmax: Single pass, update while computing

m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # running max
l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)            # running sum
acc = tl.zeros([BLOCK_M, BLOCK_N_V], dtype=tl.float32)      # running output

for j in range(0, num_kv_blocks):
    # Load K block, compute QK^T
    qk = tl.dot(q, tl.trans(k))

    # Update running max
    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)

    # Scale old accumulator
    alpha = tl.math.exp2((m_i - m_new) * log2e)  # exp2 is faster than exp
    acc *= alpha[:, None]
    l_i *= alpha

    # Compute softmax for current block
    p = tl.math.exp2((qk - m_new[:, None]) * log2e)
    l_i += tl.sum(p, axis=1)

    # Accumulate PV
    acc += tl.dot(p.to(v.dtype), v)
    m_i = m_new

# Final normalization
acc /= l_i[:, None]
```

**Key Techniques**:
- Use `exp2` instead of `exp`: GPUs have a `exp2` hardware instruction, which is approximately 2x faster than `exp`
- Precompute `log2e = 1.44269504` constants
- Always use `float32` for accumulators to avoid precision loss

## Pattern: Causal Mask Optimization

```python
# mask block(load K/V)
if CAUSAL:
    if start_n * BLOCK_N >= (start_m + 1) * BLOCK_M:
 continue # entireblock causal mask ,

# block mask
if CAUSAL:
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk, float('-inf'))
```

**Practical Experience**:
- Causal attention can skip approximately 50% of KV blocks
- Block-level skipping is far more efficient than element-wise masking
- For non-causal (full attention), do not add mask branches

---

## Related

- **Same Series**: [Cascade / State Merge](cascade-state-merge.md) — split-KV attention result merging
- **Same Series**: [Fused Kernel Patterns](fused-kernel-patterns.md) — softmax fusion approaches
- **Prerequisites**: [GPU Memory Hierarchy](../gpu-memory-hierarchy.md) — HBM bandwidth bottleneck as the motivation for Flash Attention optimization
- **Hopper Practical Guide**: [Hopper Optimization Guide](README.md) — Flash Attention implementation on SM90
- **AMD Practical Guide**: [AMD Optimization Guide](../../amd/common/hands-on/README.md) — Flash Attention implementation on AMD
- **Index**: [Triton Kernel Optimization Patterns Guide](README.md) — overview of all patterns
