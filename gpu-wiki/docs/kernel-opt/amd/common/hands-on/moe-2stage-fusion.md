# MoE 2-Stage Fusion

MoE (Mixture of Experts) inference fusion optimization patterns extracted from `reference-kernels/amd/`.

---

## Pattern: Expert GEMM + SiLU Activation Fusion

**Source**: `cdna/flydsl/FlyDSL/moe_gemm_2stage.py`, `moe_blockscale_2stage.py`

```python
# Typical MoE inference pipeline:
# Stage 1: gate_proj = expert_W_gate @ x + expert_W_up @ x
# Stage 2: down_proj = expert_W_down @ (silu(gate_proj) * up_proj)

# Fusion optimization: Complete Stage 1 two GEMMs + SiLU in a single kernel
@flyc.kernel
def moe_fused_stage1(x, W_gate, W_up, out, expert_ids, ...):
    expert_id = expert_ids[block_id]

    # Load current expert weights
    w_gate = load_expert_weight(W_gate, expert_id)
    w_up = load_expert_weight(W_up, expert_id)

    # Two GEMMs share input x
    gate_out = mfma(x_tile, w_gate)
    up_out = mfma(x_tile, w_up)

    # SiLU fusion (completed in registers)
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # Implement sigmoid using v_exp2 + v_rcp
    gate_sigmoid = v_rcp(1.0 + v_exp2(-gate_out * LOG2E))
    result = gate_out * gate_sigmoid * up_out

    store(out, result)
```

**Practical Experience**:
- `v_exp2` and `v_rcp` are hardware instructions for AMD GPUs (1 cycle throughput)
- Implementing SiLU with `exp2 + rcp` is approximately 4x faster than `exp + div`
- Both GEMMs share the LDS buffer for x, reducing data loading by 50%
- The block-scaled version uses `mfma_scale` to further reduce data volume

---

## Pattern: Mixed MoE (Different Experts with Different Precisions)

```python
# mixed_moe_gemm_2stage.py
# Scenario: Some experts use FP8, others use FP16
# Dynamically select precision based on expert importance

expert_precision = load_precision_table(expert_id)
if expert_precision == FP8:
    acc = mfma_scale(a_fp8, b_fp8, acc, scale_a, scale_b)
else:
    acc = mfma(a_bf16, b_bf16, acc)
```

---

## Related Documents

- **MFMA Instruction Selection**: [MFMA Instruction Selection and Usage](mfma-instruction-selection.md)
- **Optimization Frameworks**: [AMD GPU Kernel Optimization Framework Overview](../../../../ref-docs/amd/common/amd-kernel-optimization-frameworks.md) — FlyDSL/CK/TileLang comparison
- **General Instruction Optimization**: [GPU Instruction-Level Optimization](../../../../ref-docs/generic/gpu-instruction-optimization.md)
- **CDNA4 FP8 Hands-on**: [CDNA4 FP8 GEMM Optimization Hands-on](../../../../ref-docs/amd/common/gfx950/cdna4-fp8-gemm-optimization.md)
