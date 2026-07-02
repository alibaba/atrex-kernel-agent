# Blackwell-GeForce Triton kernels

NVIDIA RTX PRO 5000 / 4000 (`sm_120`) Triton kernels.

| Sub-dir | Kernel | dtype | Hardware | Speedup |
|---|---|---|---|---|
| [`gdn_post/`](gdn_post/) | `fused_rmsnorm_gated` (vLLM `_deltanet_post` epilogue) | bf16 in / bf16 out | sm_120 (RTX PRO 5000 72GB) | norm+gate 13.01× kernel-only, **2.16× e2e deltanet_forward** |

## Related docs

- Optimisation report: [`docs/ref-docs/nvidia/triton/sm120/`](../../../../docs/nvidia/blackwell-geforce/triton/)
- Pitfalls: [`docs/pitfalls/nvidia/triton/`](../../../../docs/nvidia/blackwell-geforce/triton/pitfalls/)
- Sibling DSL on same hardware: [`../cutedsl/`](../cutedsl/)
