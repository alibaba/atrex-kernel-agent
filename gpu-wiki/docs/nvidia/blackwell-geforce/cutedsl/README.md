# nvidia/blackwell-geforce/cutedsl

Index for `nvidia/blackwell-geforce/cutedsl/`.

## Subdirectories

| Directory | Description |
|------|------|
| [pitfalls/](pitfalls/) | Vendoring `flash_attn.cute` on cutlass <4.5: API private-name rename trap … |

## Files

| File | Description |
|------|------|
| [sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md](sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md) | Stage 3 Closeout — Path-1 fused sigmoid·gate + NVFP4 quant on sm_120 |
| [sm120-gdn-chunk-fwd-bf16-neumann-optimization.md](sm120-gdn-chunk-fwd-bf16-neumann-optimization.md) | CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120 |
| [sm120-gdn-decode-cpasync-cache-mode.md](sm120-gdn-decode-cpasync-cache-mode.md) | SM120 GDN Decode: cp.async + GLOBAL Cache Quick Reference (kernel-opt) |
| [sm120-gdn-decode-fp32state-bf16qkv-optimization.md](sm120-gdn-decode-fp32state-bf16qkv-optimization.md) | CuteDSL GDN Decode (fp32 state, bf16 q/k/v) on sm_120 — Optimization Journey |
| [sm120-moe-data-prep-optimization.md](sm120-moe-data-prep-optimization.md) | SM120 INT32 MoE Data-Prep — Optimization Journey |
| [sm120-moe-data-prep.md](sm120-moe-data-prep.md) | SM120 MoE Data-Prep — Quick Reference |
| [sm120-nvfp4-inline-ptx-gemm.md](sm120-nvfp4-inline-ptx-gemm.md) | SM120 NVFP4 GEMM: CuTeDSL + inline PTX Pitfall Summary |
| [sm120-nvfp4-persistent-gemm-pro5000-optimization.md](sm120-nvfp4-persistent-gemm-pro5000-optimization.md) | SM120 NVFP4 Persistent GEMM (NVFP4×NVFP4, fp32 accum) on RTX PRO 5000 |
| [sm120-pipeline-tma-async-api-notes.md](sm120-pipeline-tma-async-api-notes.md) | cute-DSL `cutlass.pipeline.PipelineTmaAsync` API notes for sm_120 (4.4.2) |
| [v3-fa-fusion-deferred-plan.md](v3-fa-fusion-deferred-plan.md) | V3 deferred plan: cute-DSL FlashAttention forward + sigmoid·gate + NVFP4 quant single-kernel fusion (sm_120) |
