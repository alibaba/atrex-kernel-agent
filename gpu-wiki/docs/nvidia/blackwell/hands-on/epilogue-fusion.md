# Epilogue Fusion


**Last updated**: 2026-06-30

## Pattern: GEMM + Activation/Norm/Quantize Fusion

**Source**: `cutedsl/flashinfer/*_fusion.py`, `cutedsl/cutlass/`

```python
# Blackwell's 3-role warp specialization enables complex fusion in the epilogue

# Pattern 1: GEMM + SwiGLU
# y = silu(x @ W_gate) * (x @ W_up)
# Two GEMMs share input x, SwiGLU fusion is performed in the epilogue
gate_out = gemm(x, W_gate)  # GEMM 1
up_out = gemm(x, W_up)      # GEMM 2
# Epilogue fusion:
y = silu(gate_out) * up_out  # No intermediate results written back to HBM

# Pattern 2: GEMM + RMSNorm + FP4 Quantize
# Inference scenario: GEMM output directly goes through norm + quantization, reducing HBM reads/writes
out = gemm(x, W)
# Epilogue fusion:
out_normed = rmsnorm(out)
out_fp4, scale = fp4_quantize(out_normed)  # Directly output FP4 + scale
```

**Practical Experience**:
- The value of epilogue fusion depends on the size of the GEMM output: the larger the output, the more HBM reads and writes are avoided
- SwiGLU fusion requires executing two GEMMs simultaneously Dong, requiring sufficient shared memory and TMEM capacity
- RMSNorm+FP4Quant fusion is very common in LLM inference (required at every layer)
- Blackwell's epilogue warp has independent registers and execution flow, without affecting the GEMM of the next tile

---

## Related

- **Three-Role Warp Specialization**: [Three-Role Warp Specialization](three-role-warp-specialization.md) — Architectural foundation of epilogue warps
- **Block-Scaled MMA**: [Block-Scaled MMA](block-scaled-mma.md) — FP4 Quantize paired with block-scaled MMA
- **Hopper Practical Guide**: [Hopper Optimization Practices](README.md) — Hopper epilogue comparison
- **Generic Triton**: [Triton Optimization Patterns Practical Guide](../../../generic/hands-on/README.md) — Cross-architecture fusion patterns
