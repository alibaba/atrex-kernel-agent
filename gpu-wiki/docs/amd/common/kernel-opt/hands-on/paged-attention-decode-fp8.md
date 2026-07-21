# Paged Attention Decode (FP8)

Paged Attention decode optimization pattern extracted from `reference-kernels/amd/`, combining FP8 quantization and KV Cache paged management.

---

## Pattern: KV Cache Paged Management + FP8 Quantization

**Source**: `cdna/flydsl/FlyDSL/pa_decode_fp8.py`

```python
@flyc.kernel
def paged_attention_decode_fp8(
    q, k_cache, v_cache, page_table,
    scale_k, scale_v,  # FP8 per-block scale
    output, ...
):
    # 1. Query page table to get KV cache physical page addresses
    page_indices = load_page_table(page_table, seq_id)

    # 2. Iterate over all KV pages
    m_i = -inf
    l_i = 0.0
    acc = zeros()

    for page_idx in page_indices:
        # Load FP8 KV + scale
        k_fp8 = load_page(k_cache, page_idx)
        v_fp8 = load_page(v_cache, page_idx)
        k_scale = load_scale(scale_k, page_idx)
        v_scale = load_scale(scale_v, page_idx)

        # Dequant + attention (CDNA3 requires manual dequant)
        k = dequant_fp8(k_fp8, k_scale)
        s = dot(q, k.T)

        # Online softmax
        m_new = max(m_i, max(s))
        p = exp2((s - m_new) * log2e)
        alpha = exp2((m_i - m_new) * log2e)
        acc = acc * alpha + dot(p, dequant_fp8(v_fp8, v_scale))
        l_i = l_i * alpha + sum(p)
        m_i = m_new

    output = acc / l_i
```

**Practical Experience**:
- FP8 KV cache reduces memory usage by 50% (vs FP16), critical for long-sequence inference
- CDNA3's FP8 uses the FNUZ format (non-standard), while CDNA4 uses the OCP standard format
- The bottleneck of paged attention is page table indirect addressing and non-contiguous memory access
- Page size is typically 16-64 tokens; too small increases page table overhead, too large wastes memory

---

## Related Documents

- **MFMA Instruction Selection**: [MFMA Instruction Selection and Usage](mfma-instruction-selection.md) — mfma_scale for FP8 GEMM
- **CDNA4 FP8 Hands-On**: [CDNA4 FP8 GEMM Optimization Hands-On](../../../cdna4/ref-docs/cdna4-fp8-gemm-optimization.md)
- **Generic Triton Patterns**: [Triton Optimization Pattern Hands-On](../../../../generic/kernel-opt/hands-on/README.md) — Flash Attention, online softmaxonn, online softmax
