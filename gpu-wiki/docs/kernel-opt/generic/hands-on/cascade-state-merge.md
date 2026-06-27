# Cascade / State Merge (FlashInfer Pattern)

## Pattern: Distributed Attention Result Merging

**Source**: `flashinfer/cascade.py`

```python
@triton.jit
def merge_state_kernel(v1, s1, v2, s2, v_out, s_out, ...):
    """Merge two partial attention results (for split-KV or distributed attention)"""
    # s1, s2 are their respective log-sum-exp values
    # v1, v2 are their respective attention outputs

    s_max = tl.maximum(s1, s2)
    exp1 = tl.exp(s1 - s_max)
    exp2 = tl.exp(s2 - s_max)

    # Weighted merge
    v_out = (v1 * exp1[:, None] + v2 * exp2[:, None]) / (exp1 + exp2)[:, None]
    s_out = s_max + tl.log(exp1 + exp2)
```

**Application Scenarios**:
- Split-KV attention: KV cache is sharded into multiple chunks, computed separately, then merged
- Distributed attention: Different GPUs compute different KV shards
- Cascade inference: KV caches for prefill and decode are managed separately

---

## Related Documents

- **Same Series**: [Online Softmax and Flash Attention](online-softmax-flash-attention.md) — Source of the log-sum-exp numerical stability technique
- **Same Series**: [Mamba / SSM State Management](mamba-ssm-state-management.md) — Another state merging pattern
- **Prerequisites**: [GPU Application-Level Optimization](../../../ref-docs/generic/gpu-application-optimization.md) — Distributed parallelization strategies
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — Overview of all patterns
