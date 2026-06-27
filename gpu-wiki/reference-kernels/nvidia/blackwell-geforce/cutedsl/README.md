# Blackwell GeForce CuTeDSL Kernels

Reference CuTeDSL kernels on the Blackwell GeForce architecture, organized by source repository.

---

| Directory | Description |
|------|------|
| [cutlass/](cutlass/) | CUTLASS framework reference kernels (Dense GEMM, SM120 NVFP4 inline PTX GEMM atom) |
| [flash-attention/](flash-attention/) | Flash Attention SM120 forward/backward reference implementations |
| [flashinfer/](flashinfer/) | FlashInfer reference kernels and diagnostics (SM103 block-scaled GEMM, SM120 b12x CuTe DSL task39 diagnostic fork) |
| [gdn_chunk_fwd/](gdn_chunk_fwd/) | Gated DeltaNet chunk-level forward for sm_120 (bf16 q/k/v, fp32 state accum). Production V113 no-cache 3-kernel pipeline: K0 preprocess + K_inv Neumann + K1 fused chunk_h+o+final_state. cp.async 128-bit staging, LDSM-fed MMA, R2S/STSM stores, direct output copy, scaled-vnew state update, reuse-B LDSM, and Bx2 non-tail launch. **P50=0.531-0.533ms at T=6144 directional `output_final_state=True`, 1.51× faster than same-process FLA (0.804ms).** |
| [gdn_decode/](gdn_decode/) | Gated DeltaNet decode kernel for sm_120 (fp32 state, bf16 q/k/v). V13 = cp.async + LoadCacheMode.GLOBAL + assumed_align=16, matches FLA Triton wall-clock at 1.04 TB/s = 100.8% memcpy ceiling. |
| [moe_data_prep/](moe_data_prep/) | INT32 MoE data-prep `fused_moe_data` for sm_120 (histogram + per-CTA prefix offsets + scatter). **CUDA C++ via `load_inline`, NOT CuTeDSL** (CuTeDSL cannot CG capture). V9 = V7 contention-free per-CTA offsets + V9-A 4-way bank-replicated histogram. **0.706× of vLLM CG at T=6144** (24.61 µs vs 34.85 µs). 11 iter, 3 documented null results. |
| [quack/](quack/) | QuACK kernel references |
| [fused_fa_epilogue_nvfp4/](fused_fa_epilogue_nvfp4/) | Fused (sigmoid·gate + NVFP4 quant) epilogue kernel for sm_120. Replaces the (gate-mul + sigmoid + standalone scaled_fp4_quant) two-kernel chain with a single CuTeDSL kernel: bf16 (attn_out, gate) → packed e2m1 + swizzled e4m3 SF, fully in registers. V_final = **88.05 us cuda.Event / 103.58 us ncu @ canonical 6144 = 91.9% memcpy ceiling**. Multi-shape 6.5-7.2× fused vs (sigmoid_mul + standalone fp4 quant). True FA-fwd fusion deferred (cluster cutlass-DSL 4.4.2 vs flash_attn-cute 4.5+ blocker). |
