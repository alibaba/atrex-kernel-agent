# Flash Attention Hopper Specialization

## Pattern: TMA + WGMMA-based FMHA

**Source**: `cutedsl/cutlass/fmha.py`, `cutedsl/flash-attention/`

```python
# The core differences between Hopper FMHA and the generic Triton version:
# 1. Q/K/V are loaded into shared memory via TMA (automatic padding/OOB handling)
# 2. Both QK^T and PV are computed with WGMMA (read directly from shared memory)
# 3. Softmax is performed in registers

# TMA loads Q (loaded only once)
cute.copy(tma_q, q_gmem, q_smem)

for kv_block in range(num_kv_blocks):
    # TMA loads K (automatic padding/OOB handling)
    cute.copy(tma_k, k_gmem[kv_block], k_smem)
    cute.copy(tma_v, v_gmem[kv_block], v_smem)

    # WGMMA: S = Q @ K^T (SS mode, both Q/K read from shared memory)
    s = cute.gemm(qk_mma, q_smem, k_smem)

    # Online softmax in registers
    m_new = cute.max(s, axis=1)
    p = cute.exp2((s - m_new) * log2e)

    # WGMMA: O += P @ V
    acc = cute.gemm(pv_mma, p_smem, v_smem, acc)
```

**Practical Experience**:
- Hopper FMHA is ~2x faster than the Ampere version, primarily due to TMA (reducing address computation) and WGMMA (reducing register pressure)
- WGMMA SS mode requires softmax results to be written back to shared memory before performing PV multiplication
- Pipeline depth is typically 2 (K/V double buffering)

---

## Related Documents

- **Fused Attention Topic**: [Hopper Fused Attention Optimization](../gluon/fused_attention.md) — prefill / paged attention
- **General Triton Patterns**: [Triton Optimization Patterns in Practice](../../../../generic/kernel-opt/hands-on/README.md) — Flash Attention basics
- **CuTeDSL SM90**: [CuTeDSL SM90 Specialized Features](../../ref-docs/cutedsl/hopper-cutedsl-sm90.md) — WGMMA details
- **Hardware Specifications**: [Hopper Hardware Specifications Table](../../hardware-specs/hardware_specs_hopper.md) — H100/H20 peak TFLOPS
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
