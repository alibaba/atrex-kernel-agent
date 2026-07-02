# MLA Decode (Multi-Latent Attention)


**Last updated**: 2026-07-01

## Pattern: MLA Inference Optimization

**Source**: `cutedsl/cutlass/mla/`, `cutedsl/flashinfer/mla_decode_*.py`

```python
# MLA: DeepSeek's attention variant
# KV cache uses low-rank compression: KV = [k_rope; W_uk @ c_kv]
# Reduces KV cache size (from 2 * n_heads * head_dim → latent_dim)

# Decode kernel features:
# 1. Concatenates RoPE key and decompressed KV
# 2. Batches multiple requests (grouped query attention)
# 3. Supports FP8/FP16 mixed precision

@cute.kernel
def mla_decode_kernel(q, kv_cache, rope_cache, out, ...):
    # Load latent KV
    c_kv = tl.load(kv_cache_ptr)

    # Decompress: k_nope = W_uk @ c_kv (computed online, avoids storing full K)
    k = concat(rope_key, matmul(W_uk, c_kv))

    # Standard attention
    s = matmul(q, k.T) * scale
    p = softmax(s)
    o = matmul(p, v)
```

**Practical Experience**:
- The bottleneck of MLA decode is KV cache reads (memory-bound)
- Block-scaled FP8 (MXF8) can halve the KV cache size again
- Blackwell's tcgen05 block-scaled MMA is a great fit for this scenario

---

## Related

- **Block-Scaled MMA**: [Block-Scaled MMA](block-scaled-mma.md) — MXF8 for KV cache compression
- **tcgen05 MMA and TMEM**: [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) — tcgen05 MMA basics
- **Hopper Hands-on**: [Hopper Optimization Hands-on](README.md) — Hopper MLA comparison
