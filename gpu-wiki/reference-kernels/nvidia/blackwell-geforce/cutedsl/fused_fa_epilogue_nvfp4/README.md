# Fused FA epilogue + NVFP4 quant (sm_120)

⚠ **TUNED FOR `sm_120` (NVIDIA RTX PRO 5000 / 4000 Blackwell-Geforce)**. NOT a generic-arch baseline. SM80-era MMA reuse, 99 KB SMEM cap, no TMA on NVFP4-SF byte stream (see pitfalls).

Path-1 epilogue: `flash_attn(Q,K,V) → x = attn_out * sigmoid(gate) → scaled_fp4_quant(x)`. The fused kernel here replaces the (`gate-mul + sigmoid` + standalone `scaled_fp4_quant`) two-kernel chain with a single CuTeDSL kernel that consumes (`attn_out`, `gate`) bf16, computes sigmoid·mul·amax·e4m3·e2m1 in registers, writes `x_fp4` (`stg.E.64`) + swizzled e4m3 SF (`stg.E.8`) to gmem.

**Producer (`flash_attn`) is NOT fused into this kernel** — that is the deferred V3 true-fusion plan in `docs/nvidia/blackwell-geforce/ref-docs/cutedsl/v3-fa-fusion-deferred-plan.md`, blocked on cluster cutlass-DSL ≥ 4.5.

---

## Files

| File | Purpose |
|------|---------|
| `fused_fa_epilogue_nvfp4_sm120.py` | Module-level `@cute.kernel` + `@cute.jit` launcher; sm_120 GDN-style. multi-row mode (1 thread per SF block, 256 threads × 2 rows / block, 440 blocks = 4 waves × 110 SMs). Includes `_bootstrap_cutedsl()` for clusters with `cutlass==0.1.0` ML-lib namespace pollution. |
| `cute_helpers_sm120.py` | Vendored NVFP4 PTX helpers from flashinfer datacenter blackwell (`bfloat2_max_abs_8`, `bfloat2x8_to_e2m1x16_packed`, `cvt_f32_to_e4m3`, `nvfp4_compute_output_scale`, `rcp_approx_ftz`, `get_ptr_as_int64`, `ld_global_v4_u32`, `st_global_u64`, `compute_sf_index_swizzled_128x4_gpu`) + custom `bfloat2_sigmoid_mul` (single inline_asm block folding sigmoid+mul on bf16x2). |
| `nvfp4_reference_pytorch.py` | PyTorch reference for NVFP4 quantize/dequantize per NVIDIA cuda formula. Used by `validate.py` since vllm dev206 has ABI mismatch with cluster torch. |
| `validate.py` | Multi-shape correctness: bit-exact `x_bs_lin`, `x_fp4` mismatch %, `rel_err(quant)`, e2e `dequant→bf16 GEMM rel_err`. |

---

## Performance (V_final, RTX PRO 5000 sm_120, ncu)

| SEQ_LEN | duration (us) | DRAM (GB/s) | % memcpy ceiling 1099 GB/s | speedup vs (sigmoid_mul + standalone scaled_fp4_quant) |
|---|---|---|---|---|
| 512  | 14.9  | 566  | 51.5% (L2-resident regime) | 0.90× |
| 1024 | 21.5  | 780  | 71.0% | 1.26× |
| 2048 | 42.7  | 787  | 71.6% | **3.41×** |
| 4096 | 72.0  | 933  | 84.9% | **6.82×** |
| **6144** (canonical) | **107.6** | **977** | **88.9%** (V0 Stage 2 ncu measured 91.9%) | **6.53×** |
| 8192 | 137.5 | 1031 | **93.8%** | **7.15×** |

V_final reaches **89-94% memcpy ceiling for SEQ_LEN ≥ 6144** = the standalone-quant box's physical upper bound. V1 (cp.async + LoadCacheMode.GLOBAL bypass L1) and V2 (TMA G2S + warp-spec) both hit the same wall within 1.4% — see optimization journey doc.

End-to-end Path-1 forward (vllm.flash_attn dispatcher + V_final fused-quant) is **1.43× – 1.85×** vs Stage 0 SDPA + standalone, capped by `vllm.vllm_flash_attn` having no fast path on sm_120 RTX PRO 5000.

---

## Related docs

- **Optimization journey**: [docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md)
- **Deferred true-fusion plan** (cutlass 4.5+): [docs/nvidia/blackwell-geforce/ref-docs/cutedsl/v3-fa-fusion-deferred-plan.md](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/v3-fa-fusion-deferred-plan.md)
- **PipelineTmaAsync API notes**: [docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-pipeline-tma-async-api-notes.md](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-pipeline-tma-async-api-notes.md)
- **Pitfalls**:
  - [TMA + warp-spec](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-tma-warp-spec-pitfalls.md)
  - [vendor flash_attn.cute on cutlass < 4.5](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/cute-442-vendor-flash-attn-pitfalls.md)  - [ncu L1/TEX hit rate counts ld.shared](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-ncu-l1-hit-rate-shared-pollution.md)
  - [vllm flash_attn no fast path on sm_120](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-flash-attn-vllm-no-fast-path.md)
- **Adjacent sm_120 optimizations**:
  - [SM120 NVFP4 GEMM (581 TFLOPS = 71% CUTLASS)](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-nvfp4-persistent-gemm-pro5000-optimization.md)
  - [SM120 GDN decode (100.8% memcpy ceiling)](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- **NVFP4 helpers source**: [reference-kernels/nvidia/blackwell/cutedsl/flashinfer/quantization_cute_dsl_utils.py](../../../blackwell/cutedsl/flashinfer/quantization_cute_dsl_utils.py) (datacenter Blackwell, but PTX cvt is arch-agnostic)
