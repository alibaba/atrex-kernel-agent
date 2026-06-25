# FlyDSL CDNA3 (gfx942) — Hardware-tuned reference kernels

FlyDSL kernel implementations on CDNA3 / MI308X.
Distinct from `reference-kernels/amd/cdna/flydsl/FlyDSL/`: those are **CDNA-generic** implementations, not tuned for a single SKU; each file in this directory is annotated with a specific hardware model, and is accompanied by optimization journeys under `docs/ref-docs/amd/flydsl/gfx942/` and troubleshooting notes under `docs/pitfalls/amd/flydsl/`.

| File | Hardware | Description | Optimization Journey |
|---|---|---|---|
| [flash_attn_func_mi308x.py](flash_attn_func_mi308x.py) | MI308X (gfx942) | bf16 + GQA flash attention forward, prefill-only. v15: ds_swizzle latency hiding (+4.6%). vs aiter `mha_batch_prefill_func` + `flash_attn_func` (fmha_v3) | [cdna3-flash-attention-bf16-gqa-optimization.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-gqa-optimization.md) |
| [flash_attn_func_nomask_mi308x.py](flash_attn_func_nomask_mi308x.py) | MI308X (gfx942) | bf16 flash attention forward **no-mask** path, head_dim=64 native. V6: `ds_bpermute_lgkm_sum` manual ATT scheduling, 60.19 TFLOPS; BF16 objective met. Historical CK row is FP16, not the BF16 target | [cdna3-flash-attention-bf16-nomask-isa-scheduling.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-nomask-isa-scheduling.md) |
| [flash_attn_func_mask_mi308x.py](flash_attn_func_mask_mi308x.py) | MI308X (gfx942) | bf16 flash attention forward with **bit-packed binary mask + LSE** (non-causal), head_dim=64 native. V10: **71.8 TFLOPS**, SHARE_KV_LDS + pk_fma + occupancy=4 + BHSD native layout. Supersedes V7 (50.6T, backed up as `.v7.bak`) | [V8-V10](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-mask-lse-optimization.md), [V0-V7](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-mask-optimization.md) |
| [chunk_gdn_flydsl_operator.py](chunk_gdn_flydsl_operator.py) | MI308X (gfx942) | Chunk-GDN standalone FlyDSL megakernel wrapper: input is precomputed `a/g_cumsum`, runs only the fused `recompute_w_u + fwd_h + fwd_o`; includes BDV64 hot path and BDV32 small-H path | [cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md) |
| [attn_bwd_dkdv_mi308x.py](attn_bwd_dkdv_mi308x.py) | MI308X | Attention backward dK+dV (bf16, arbitrary mask). V15: dO B-operand from LDS (6.52ms). lds_qt→lds_dot merge + strided dO reads. | [cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md) |
| [attn_bwd_dq_mi308x.py](attn_bwd_dq_mi308x.py) | MI308X | Attention backward dQ (bf16, arbitrary mask). V14: lds_kt eliminated via strided scalar reads (2.93ms, occupancy 7). | [cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md) |
| [flash_attn_bwd_flydsl_mi308x.py](flash_attn_bwd_flydsl_mi308x.py) | MI308X | Attention backward **API wrapper** (bf16, arbitrary mask). Bit-packed u32 mask + OOB guards + precomputed loop bounds. 11.7ms end-to-end, 3.0× vs PyTorch, 2.3× vs aiter CK-tile. | [cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md) |## Chunk-GDN Standalone Test

```bash
cd /root/gpu-wiki/reference-kernels/amd/cdna3/flydsl/FlyDSL

# Only run shape, chunk_offsets, and validation tests
CHUNK_GDN_RUN_GPU_TESTS=0 /opt/conda310/bin/python3.10 test_chunk_gdn_flydsl_operator.py

# Run GPU correctness tests in MI308/FlyDSL environment:
# dense hot path, BDV32 small-H, tail path, varlen all compared with PyTorch reference
CHUNK_GDN_RUN_GPU_TESTS=1 /opt/conda310/bin/python3.10 test_chunk_gdn_flydsl_operator.py
```

## Chunk-GDN 397B-TP2 rocprofv3 Baseline

Performance comparison uses the ported Triton back-half baseline:
`/root/gpu-wiki/reference-kernels/amd/cdna/triton/chunk_gdn/`, not
the PyTorch reference. The 397B-TP2 hot path shape is
`(B,Hg,H,K,V)=(1,8,32,128,128)`, and the comparison boundary is after precomputing `a/g_cumsum`,
specifically `recompute_w_u + fwd_h + fwd_o`.

`rocprofv3 --kernel-trace`, warmup 2 + target 5 iteration P50 results:

| T | Triton back-half P50 | FlyDSL megakernel P50 | Speedup |
|---:|---:|---:|---:|
| 4096 | 1048.282us | 609.561us | 1.720x |
| 16384 | 4113.662us | 2498.645us | 1.646x |
| 65536 | 16403.075us | 9974.661us | 1.644x |
| 200000 | 50805.507us | 30473.504us | 1.667x |

For detailed methodology and optimization insights, see
[cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md](../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md);
for MI308X porting pitfalls, see
[chunk-gdn-mi308x-wave-specialization-pitfalls.md](../../../../../docs/pitfalls/amd/flydsl/chunk-gdn-mi308x-wave-specialization-pitfalls.md).
