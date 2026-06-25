# FlyDSL RDNA4 (gfx1250) Kernel Reference Implementations

Reference GPU kernel implementations for FlyDSL on the AMD RDNA4 (gfx1250) architecture, focusing on GEMM variants, WMMA instructions, and fused MoE.

> **inline_asm syntax**: See [`docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md`](../../../../../docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md).

---

| Kernel | Description | inline_asm Usage |
|--------|------|----------------|
| [gemm_common_gfx1250.py](gemm_common_gfx1250.py) | gfx1250 GEMM common utilities | — |
| [gemm_fp8fp4_gfx1250.py](gemm_fp8fp4_gfx1250.py) | gfx1250 FP8/FP4 mixed-precision GEMM | f-string constructed `s_prefetch_inst_pc_rel` × 10 (first launch I-cache warmup) |
| [moe_gemm_2stage_common_gfx1250.py](moe_gemm_2stage_common_gfx1250.py) | gfx1250 MoE GEMM two-stage common utilities | — |
| [moe_gemm_2stage_mxscale_gfx1250.py](moe_gemm_2stage_mxscale_gfx1250.py) | gfx1250 MoE GEMM two-stage MX-scale quantization (FP4 + scale) | prefetch + `s_setreg_imm32_b32 hwreg(26, 4, 1), 1` wave mode control |
| [moe_gemm_2stage_wmma_gfx1250.py](moe_gemm_2stage_wmma_gfx1250.py) | gfx1250 MoE GEMM two-stage WMMA path (BF16) | entry `s_setreg_imm32_b32 hwreg(26, 4, 1), 1` |
| [pipeline_utils.py](pipeline_utils.py) | gfx1250 GEMM pipeline tail-plan utilities | — |
| [rdna_f16_gemm.py](rdna_f16_gemm.py) | RDNA FP16 GEMM kernel | `s_wait_dscnt + s_wait_storecnt + s_barrier_signal/wait` complete split barrier sequence |
| [rdna_fp8_preshuffle_gemm.py](rdna_fp8_preshuffle_gemm.py) | RDNA FP8 preshuffle GEMM kernel | — |
| [wmma_gemm_gfx1250.py](wmma_gemm_gfx1250.py) | gfx1250 WMMA GEMM kernel | — |
