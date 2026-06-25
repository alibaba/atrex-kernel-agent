# MI308X / gfx942 FlyDSL Specific Articles

FlyDSL-specific optimization reference articles on MI308X (gfx942/CDNA3).

---

| File | Description |
|------|------|
| [cdna3-fused-moe-bf16-optimization.md](cdna3-fused-moe-bf16-optimization.md) | FlyDSL Fused MoE BF16 Optimization |
| [cdna3-sage-attention-flydsl-optimization.md](cdna3-sage-attention-flydsl-optimization.md) | FlyDSL SageAttention (FP8) Optimization |
| [cdna3-flash-attention-bf16-gqa-optimization.md](cdna3-flash-attention-bf16-gqa-optimization.md) | FlyDSL Flash Attention bf16 + GQA full v0→v15 optimization journey; v14 ds_swizzle V transpose; v15 ds_swizzle latency hiding; includes aiter fmha_v3_fwd comparison |
| [cdna3-flash-attention-bf16-nomask-isa-scheduling.md](cdna3-flash-attention-bf16-nomask-isa-scheduling.md) | FlyDSL Flash Attention bf16 **no-mask** on MI308X; V6 rocprofv3 ATT-driven manual ISA scheduling; `ds_bpermute_lgkm_sum` separates LDS reduction wait from VMEM drain; 60.19 TFLOPS, BF16 target met, historical CK behavior FP16 non-same-dtype comparison |
| [cdna3-flash-attention-bf16-mask-optimization.md](cdna3-flash-attention-bf16-mask-optimization.md) | FlyDSL Flash Attention bf16 + **bit-packed binary mask** (non-causal) on MI308X; V0→V7 journey; V7 bit-packed u32 bitmask (32× BW reduction) + sched_barrier(0) scheduling; **50.6 TFLOPS, 3.74× faster than CK/SDPA with mask** |
| [cdna3-flash-attention-bf16-mask-lse-optimization.md](cdna3-flash-attention-bf16-mask-lse-optimization.md) | FlyDSL Flash Attention bf16 + mask + **LSE output** on MI308X; V8→V10 journey (continuation of V7); SHARE_KV_LDS + v_pk_fma_f32 + waves_per_eu=4 + BHSD native layout; **71.8 TFLOPS, +42% vs V7**; includes LSE output for training backward |
| [cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md) | FlyDSL Attention Backward **dQ + dK+dV** (bf16, arbitrary mask) on MI308X; dQ 2.93ms (occupancy 7) + dK+dV 6.52ms = 9.45ms combined; 4.35× faster than PyTorch SDPA backward; V0→V15 journey with 16 successful optimizations + 8 failed experiments; key techniques: strided scalar LDS reads to eliminate transpose buffers, dO B-operand from LDS to eliminate redundant VMEM |
| [cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md](cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md) | FlyDSL Flash Attention Backward **API integration** (bf16, arbitrary mask); OOB guards eliminate F.pad, bit-packed u32 + precomputed loop bounds; end-to-end 11.7ms, **3.0× vs PyTorch, 2.26× vs aiter CK-tile**; includes aiter/PyTorch four-way comparison |
| [cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md](cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md) | FlyDSL Chunk-GDN wave-specialized megakernel on MI308X; ported FlashQLA warp-specialization approach, fused `recompute_w_u + fwd_h + fwd_o` back half; standalone 397B-TP2 back-half relative to Triton baseline `1.644-1.720x` |
