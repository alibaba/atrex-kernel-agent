# Blackwell CuTeDSL FlashInfer Kernels

CuTeDSL reference kernel implementations for the FlashInfer project on the Blackwell (SM100) architecture.

---

| Kernel | Description |
|--------|------|
| [add_rmsnorm_fp4quant.py](add_rmsnorm_fp4quant.py) | Fused Addition + RMSNorm + FP4 Quantization |
| [blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py](blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py) | Block-Scaled Contiguous Gather Grouped GEMM + SwiGLU Fusion |
| [blockscaled_contiguous_grouped_gemm_finalize_fusion.py](blockscaled_contiguous_grouped_gemm_finalize_fusion.py) | Block-Scaled Contiguous Gather Grouped GEMM Finalize Fusion |
| [custom_pipeline.py](custom_pipeline.py) | Custom Pipeline |
| [dense_blockscaled_gemm_sm100.py](dense_blockscaled_gemm_sm100.py) | Dense Block-Scaled GEMM SM100 |
| [fp4_common.py](fp4_common.py) | FP4 Common Utility Functions |
| [gemm_allreduce_two_shot.py](gemm_allreduce_two_shot.py) | GEMM All-Reduce Two-Shot Communication |
| [grouped_gemm_masked_blackwell.py](grouped_gemm_masked_blackwell.py) | Masked Grouped GEMM (Blackwell) |
| [mla_decode_fp16.py](mla_decode_fp16.py) | MLA Decode FP16 Implementation |
| [mla_decode_fp8.py](mla_decode_fp8.py) | MLA Decode FP8 Implementation |
| [moe_utils.py](moe_utils.py) | MoE Utility Functions |
| [mxfp4_quantize.py](mxfp4_quantize.py) | MXFP4 Quantization Kernel |
| [mxfp8_quantize.py](mxfp8_quantize.py) | MXFP8 Quantization Kernel |
| [nvfp4_quantize.py](nvfp4_quantize.py) | NVFP4 Quantization Kernel |
| [quantization_cute_dsl_utils.py](quantization_cute_dsl_utils.py) | Quantization CuTeDSL Utility Functions |
| [rmsnorm_fp4quant.py](rmsnorm_fp4quant.py) | RMSNorm + FP4 Quantization Fusion |
