# nvidia/blackwell-geforce/pitfalls/cutedsl

Architecture-scoped knowledge index. The architecture is determined by the path; the role is the next directory level.

## Documents

- [Vendoring `flash_attn.cute` on cutlass <4.5: API private-name rename trap](cute-442-vendor-flash-attn-pitfalls.md)
- [GDN Chunk Forward Pitfalls (CuTeDSL, SM120)](gdn-chunk-fwd-pitfalls.md)
- [CuTeDSL GDN Decode on sm_120 — Pitfalls](gdn-decode-pitfalls.md)
- [cute-DSL NVFP4 GEMM pitfalls (sm_120, RTX PRO 5000)](nvfp4-gemm-pitfalls.md)
- [sm_120 trap: `vllm.vllm_flash_attn.flash_attn_varlen_func` has no fast path on Blackwell-Geforce](sm120-flash-attn-vllm-no-fast-path.md)
- [SM120 INT32 MoE Data-Prep — Pitfalls](sm120-moe-data-prep-pitfalls.md)
- [sm_120 ncu trap: l1tex__t_sector_hit_rate.pct includes ld.shared hits](sm120-ncu-l1-hit-rate-shared-pollution.md)
- [sm_120 cute 4.4.2 TMA + warp-spec implementation pitfalls](sm120-tma-warp-spec-pitfalls.md)
