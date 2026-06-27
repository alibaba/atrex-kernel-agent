# vLLM Gated-DeltaNet post-processing (Triton on sm_120)

Drop-in fused kernels for the `_deltanet_post` block in vLLM's
GDN attention path on RTX PRO 5000 (`sm_120`).

| File | Replaces | Default config | Perf @ canonical [6144,32,128] bf16 |
|---|---|---|---|
| [`fused_rmsnorm_gated_pro5000.py`](fused_rmsnorm_gated_pro5000.py) | The eager `RMSNormGated + SiLU(z) gating` block (kept `scaled_fp4_quant` separate) | `BLOCK_M=2, num_warps=4, num_stages=3` | 1404.66 us → **107.96 us** (13.01×) kernel-only; **2.16× e2e** |

## Why `BLOCK_M=2` (not larger)

`BLOCK_M=8` is 4× slower at this shape; `BLOCK_M=16/32` are 20-27× slower.
See [`sm120-fused-rmsnorm-gated-pitfalls.md`](../../../../../docs/pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md#3-blockm-sweep-can-show-27-maxmin-spread-at-the-same-shape).

## Related docs

- Full V1 → V3 journey: [`sm120-fused-rmsnorm-gated-bf16-optimization.md`](../../../../../docs/ref-docs/nvidia/triton/sm120/sm120-fused-rmsnorm-gated-bf16-optimization.md)
- Pitfalls: [`sm120-fused-rmsnorm-gated-pitfalls.md`](../../../../../docs/pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md)
- CuTeDSL real-fusion attempt (deferred): [`v3-fa-fusion-deferred-plan.md`](../../../../../docs/ref-docs/nvidia/cutedsl/sm120/v3-fa-fusion-deferred-plan.md)
