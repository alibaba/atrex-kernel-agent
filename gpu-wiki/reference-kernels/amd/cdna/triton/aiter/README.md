# aiter Triton Kernel Collection

AMD's official AI inference operator library [aiter](https://github.com/ROCm/aiter) Triton kernel implementations, targeting MI300X (gfx942) and MI350 (gfx950) hardware.

> **Note**: Most kernels depend on aiter framework utilities (`pid_grid`, `remap_xcd`, `make_kernel_repr`, etc.) and cannot be run independently, but the core `@triton.jit` algorithms are valuable as references.

---

## Attention

| File | Description |
|------|-------------|
| [pa_decode.py](attention/pa_decode.py) | Paged Attention Decode V1/V2, including FP8 KV cache variants (12 kernels) |
| [pa_prefill.py](attention/pa_prefill.py) | Paged Attention Prefill |
| [pa_mqa_logits.py](attention/pa_mqa_logits.py) | Paged Attention MQA Logits |
| [mla_decode_rope.py](attention/mla_decode_rope.py) | MLA Decode with Online RoPE (nope/pe separation) |
| [mha.py](attention/mha.py) | Multi-Head Attention (including FP8 quantization, Alibi, dropout) |
| [lean_atten.py](attention/lean_atten.py) | Lean Attention (Stream-K persistent + ping-pong scheduling) |
| [lean_atten_paged.py](attention/lean_atten_paged.py) | Lean Attention Paged variant |
| [unified_attention.py](attention/unified_attention.py) | Unified Attention (sliding window + sink token + binary search) |
| [unified_attention_sparse_mla.py](attention/unified_attention_sparse_mla.py) | Unified Attention Sparse MLA |
| [prefill_attention.py](attention/prefill_attention.py) | Context/Prefill Attention |
| [extend_attention.py](attention/extend_attention.py) | Extend Attention |
| [chunked_pa_prefill.py](attention/chunked_pa_prefill.py) | Chunked Paged Attention Prefill |
| [hstu_attention.py](attention/hstu_attention.py) | HSTU Attention |
| [pod_attention.py](attention/pod_attention.py) | Pod Attention |
| [fp8_mqa_logits.py](attention/fp8_mqa_logits.py) | FP8 MQA Logits |
| [mha_fused_bwd.py](attention/mha_fused_bwd.py) | MHA Fused Backward |
| [mha_onekernel_bwd.py](attention/mha_onekernel_bwd.py) | MHA One-Kernel Backward |

## Flash Attention (AMD)

| File | Description |
|------|-------------|
| [fwd_prefill.py](flash_attn_amd/fwd_prefill.py) | AMD-optimized Flash Attention Forward (Prefill) |
| [fwd_decode.py](flash_attn_amd/fwd_decode.py) | AMD-optimized Flash Attention Forward (Decode) |
| [bwd.py](flash_attn_amd/bwd.py) | AMD-optimized Flash Attention Backward |
| [interface_v2.py](flash_attn_amd/interface_v2.py) | Flash Attention v2 Interface |
| [interface_v3.py](flash_attn_amd/interface_v3.py) | Flash Attention v3 Interface |
| [utils.py](flash_attn_amd/utils.py) | Flash Attention Utility Functions |

## GEMM

### Basic

| File | Description |
|------|-------------|
| [gemm_a16w16.py](gemm/basic/gemm_a16w16.py) | FP16/BF16 Standard GEMM |
| [gemm_a16w16_atomic.py](gemm/basic/gemm_a16w16_atomic.py) | FP16 GEMM with atomic output |
| [gemm_a16w16_gated.py](gemm/basic/gemm_a16w16_gated.py) | FP16 Gated GEMM |
| [gemm_a8w8.py](gemm/basic/gemm_a8w8.py) | INT8 Activation x INT8 Weight GEMM |
| [gemm_a8w8_blockscale.py](gemm/basic/gemm_a8w8_blockscale.py) | A8W8 Block-Scale GEMM (including preshuffle variant) |
| [gemm_a8w8_per_token_scale.py](gemm/basic/gemm_a8w8_per_token_scale.py) | A8W8 Per-Token Scale GEMM |
| [gemm_a16w8_blockscale.py](gemm/basic/gemm_a16w8_blockscale.py) | A16W8 Block-Scale GEMM |
| [gemm_a16wfp4.py](gemm/basic/gemm_a16wfp4.py) | A16 x WFP4 GEMM |
| [gemm_a8wfp4.py](gemm/basic/gemm_a8wfp4.py) | A8 x WFP4 GEMM |
| [gemm_afp4wfp4.py](gemm/basic/gemm_afp4wfp4.py) | MXFP4 x MXFP4 GEMM (gfx950 tl.dot_scaled) |### Batched

| File | Description |
|------|------|
| [batched_gemm_bf16.py](gemm/batched/batched_gemm_bf16.py) | Batched BF16 GEMM |
| [batched_gemm_a8w8.py](gemm/batched/batched_gemm_a8w8.py) | Batched A8W8 GEMM |
| [batched_gemm_a16wfp4.py](gemm/batched/batched_gemm_a16wfp4.py) | Batched A16WFP4 GEMM |
| [batched_gemm_afp4wfp4.py](gemm/batched/batched_gemm_afp4wfp4.py) | Batched MXFP4 GEMM |
| [batched_gemm_a8w8_...prequant...py](gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py) | Batched A8W8 Per-Token-Group Pre-Quant |

### Fused

| File | Description |
|------|------|
| [fused_gemm_a8w8_blockscale_mul_add.py](gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py) | A8W8 GEMM + Mul + Add Fusion |
| [fused_gemm_a8w8_blockscale_split_cat.py](gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py) | A8W8 GEMM + Split + Cat Fusion |
| [fused_gemm_a8w8_blockscale_a16w16.py](gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py) | A8W8 + A16W16 Mixed Precision Fusion |
| [fused_gemm_afp4wfp4_mul_add.py](gemm/fused/fused_gemm_afp4wfp4_mul_add.py) | MXFP4 GEMM + Mul + Add |
| [fused_gemm_afp4wfp4_split_cat.py](gemm/fused/fused_gemm_afp4wfp4_split_cat.py) | MXFP4 GEMM + Split + Cat |
| [fused_gemm_afp4wfp4_a16w16.py](gemm/fused/fused_gemm_afp4wfp4_a16w16.py) | MXFP4 + A16W16 Mixed Precision Fusion |

### Feed Forward

| File | Description |
|------|------|
| [ff_a16w16_fused_gated.py](gemm/feed_forward/ff_a16w16_fused_gated.py) | Fused Gated FFN (gate+up+activation+down integrated) |
| [ff_a16w16_fused_ungated.py](gemm/feed_forward/ff_a16w16_fused_ungated.py) | Fused Ungated FFN |

## MoE

| File | Description |
|------|------|
| [moe_op.py](moe/moe_op.py) | MoE Basic GEMM |
| [moe_op_gemm_a8w8.py](moe/moe_op_gemm_a8w8.py) | MoE A8W8 GEMM (including MXFP4, SwiGLU, XCD swizzle) |
| [moe_op_gemm_a8w8_blockscale.py](moe/moe_op_gemm_a8w8_blockscale.py) | MoE A8W8 Block-Scale GEMM |
| [moe_op_gemm_a4w4.py](moe/moe_op_gemm_a4w4.py) | MoE A4W4 GEMM |
| [moe_op_gemm_a8w4.py](moe/moe_op_gemm_a8w4.py) | MoE A8W4 GEMM |
| [moe_op_mxfp4.py](moe/moe_op_mxfp4.py) | MoE MXFP4 GEMM |
| [moe_op_mxfp4_silu_fused.py](moe/moe_op_mxfp4_silu_fused.py) | MoE MXFP4 + SiLU Fusion |
| [moe_op_silu_fused.py](moe/moe_op_silu_fused.py) | MoE SiLU Fusion |
| [moe_op_gelu.py](moe/moe_op_gelu.py) | MoE GELU Activation |
| [moe_op_e2e.py](moe/moe_op_e2e.py) | E2E Fused MoE (gate+SiLU+down integrated, persistent variant) |
| [moe_align_block_size.py](moe/moe_align_block_size.py) | MoE Block Size Alignment |
| [moe_routing_sigmoid_top1_fused.py](moe/moe_routing_sigmoid_top1_fused.py) | Sigmoid Top-1 Fused Routing |
| [quant_moe.py](moe/quant_moe.py) | Quantized MoE |
| [moe_routing/routing.py](moe/moe_routing/routing.py) | MoE Routing Core |
| [moe_routing/topk.py](moe/moe_routing/topk.py) | MoE TopK Selection |
| [moe_routing/expt_data.py](moe/moe_routing/expt_data.py) | Expert Data Packing |
| [moe_routing/bitmatrix.py](moe/moe_routing/bitmatrix.py) | Bitmatrix Compressed Expert Assignment |

## Normalization

| File | Description |
|------|------|
| [rmsnorm.py](normalization/rmsnorm.py) | RMSNorm (including fused add, dynamic quantization, backward, 8 kernels) |
| [norm.py](normalization/norm.py) | General Normalization |
| [fused_add_rmsnorm_pad.py](normalization/fused_add_rmsnorm_pad.py) | Fused Add + RMSNorm + Padding |## Quantization

| File | Description |
|------|-------------|
| [fused_fp8_quant.py](quant/fused_fp8_quant.py) | Fused FP8 quantization (includes 7 kernels: RMSNorm+quant, SiLU+mul+quant, etc.) |
| [fused_mxfp4_quant.py](quant/fused_mxfp4_quant.py) | Fused MXFP4 quantization |
| [quant.py](quant/quant.py) | General-purpose quantization utilities |

## RoPE

| File | Description |
|------|-------------|
| [rope.py](rope/rope.py) | Rotary Position Embedding |
| [fused_qkv_split_qk_rope.py](rope/fused_qkv_split_qk_rope.py) | Fused QKV Split + QK RoPE |

## Fusions

| File | Description |
|------|-------------|
| [fused_bmm_rope_kv_cache.py](fusions/fused_bmm_rope_kv_cache.py) | BMM + RoPE + KV Cache (three-stage grid, FP4/FP8) |
| [fused_kv_cache.py](fusions/fused_kv_cache.py) | KV Cache write |
| [fused_mul_add.py](fusions/fused_mul_add.py) | Fused Mul + Add |
| [fused_qk_concat.py](fusions/fused_qk_concat.py) | Fused QK Concat |

## Others

| File | Description |
|------|-------------|
| [activation.py](activation.py) | Activation functions (SiLU, GELU, etc.) |
| [causal_conv1d.py](causal_conv1d.py) | Causal 1D Convolution |
| [gmm.py](gmm.py) | Grouped Matrix Multiply |
| [softmax.py](softmax.py) | Softmax |
| [topk.py](topk.py) | TopK |
