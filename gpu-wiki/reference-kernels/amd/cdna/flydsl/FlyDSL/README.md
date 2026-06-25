# FlyDSL CDNA Kernel Reference Implementation

Reference implementations of GPU kernels using FlyDSL on AMD CDNA architectures (gfx942 / MI300X / MI308X), covering operators such as GEMM, Attention, Normalization, MoE, and Custom AllReduce.

> **inline_asm usage**: see [`docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md`](../../../../../docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md).

---

| Kernel | Description |
|--------|-------------|
| [__init__.py](__init__.py) | Python package entry point |
| [blockscale_preshuffle_gemm.py](blockscale_preshuffle_gemm.py) | Block-scale preshuffle GEMM kernel |
| [custom_all_reduce.py](custom_all_reduce.py) | Custom AllReduce Python shim (FlyDSL 1-stage / 2-stage algorithm interface) |
| [custom_all_reduce_kernel.py](custom_all_reduce_kernel.py) | Custom AllReduce FlyDSL kernel: cross-GPU signal protocol + `buffer_inv sc1` / `buffer_wbl2 sc0 sc1` cache control inline_asm |
| [example_01-vectorAdd.py](example_01-vectorAdd.py) | Example 01: Vector Addition |
| [example_02-tiledCopy.py](example_02-tiledCopy.py) | Example 02: Tiled Memory Copy |
| [example_03-tiledMma.py](example_03-tiledMma.py) | Example 03: Tiled MMA Matrix Multiply |
| [example_04-preshuffle_gemm.py](example_04-preshuffle_gemm.py) | Example 04: Preshuffle GEMM |
| [flash_attn_func.py](flash_attn_func.py) | Flash Attention kernel |
| [fused_moe_mixtral_bf16.py](fused_moe_mixtral_bf16.py) | Fully optimized Fused MoE BF16 implementation (Mixtral-7B, MI308X 89.9% peak) |
| [fused_rope_cache_kernel.py](fused_rope_cache_kernel.py) | Fused RoPE cache kernel |
| [hgemm_splitk.py](hgemm_splitk.py) | Half-precision GEMM Split-K kernel; includes `global_store_dword sc0 sc1` / `global_load_dword sc1` inline_asm with cache modifier (split-K counter protocol) |
| [kernels_common.py](kernels_common.py) | Kernel common utility functions |
| [layernorm_kernel.py](layernorm_kernel.py) | Layer Normalization kernel |
| [mfma_epilogues.py](mfma_epilogues.py) | MFMA epilogue handling |
| [mfma_preshuffle_pipeline.py](mfma_preshuffle_pipeline.py) | MFMA preshuffle pipeline |
| [mixed_moe_gemm_2stage.py](mixed_moe_gemm_2stage.py) | Mixed-precision MoE GEMM two-stage kernel |
| [moe_blockscale_2stage.py](moe_blockscale_2stage.py) | MoE block-scale two-stage kernel |
| [moe_gemm_2stage.py](moe_gemm_2stage.py) | MoE GEMM two-stage kernel |
| [pa_decode_fp8.py](pa_decode_fp8.py) | Paged Attention FP8 decode kernel |
| [preshuffle_gemm.py](preshuffle_gemm.py) | Preshuffle GEMM kernel |
| [rmsnorm_kernel.py](rmsnorm_kernel.py) | RMS Normalization kernel |
| [sage_attn_flydsl.py](sage_attn_flydsl.py) | SageAttention quantized Flash Attention (INT8 QK + FP8 V, MI308X, significantly outperforms Gluon) |
| [softmax_kernel.py](softmax_kernel.py) | Softmax kernel |
| [tensor_shim.py](tensor_shim.py) | Tensor shim compatibility layer |
