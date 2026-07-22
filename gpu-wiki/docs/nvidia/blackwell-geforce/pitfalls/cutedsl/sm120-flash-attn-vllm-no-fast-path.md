# sm_120 trap: `vllm.vllm_flash_attn.flash_attn_varlen_func` has no fast path on Blackwell-Geforce

## trap

When porting an attention pipeline from datacenter Hopper/Blackwell (sm_90/sm_100) to Blackwell-Geforce sm_120 (RTX PRO 5000 / 4000 / 50xx), `vllm.vllm_flash_attn.flash_attn_varlen_func` runs **far slower than the memory-bandwidth ceiling would predict**, even though the function imports cleanly and produces numerically-correct output.

## symptom

Path-1 attention forward, M=6144 SEQ_LEN, Q (6144, 16, 256) bf16, K/V (6144, 2, 256) bf16 (GQA ratio 8), causal=True:

| Path | Wall-clock | Implied DRAM throughput |
|---|---|---|
| memcpy ceiling for ~100 MB Q+K+V+O traffic on sm_120 (1099 GB/s) | **~99 us** estimated lower bound | 1099 GB/s |
| `torch.nn.functional.scaled_dot_product_attention` (PyTorch SDPA, FP32 fallback) | 1736 us | (compute-bound, never reaches DRAM) |
| `vllm.vllm_flash_attn.flash_attn_varlen_func` | **1645 us** | ~60 GB/s effective |

vllm is **16× slower than the memory ceiling** — same order as PyTorch SDPA. This contradicts vllm's behavior on H100 (typically <5% above memcpy ceiling for memory-bound attention).

## reality

Speculated: vllm wraps Tri Dao's flash-attention which dispatches per-arch fast paths for sm_80 (Ampere), sm_89 (Ada), sm_90 (Hopper wgmma), and sm_100 (Blackwell tcgen05). **No published per-arch fast path exists for sm_120 (Blackwell-Geforce)** — it uses the same SM80-era `mma.sync.aligned.m16n8k16` instruction as sm_80 but without the per-arch tile/stage/pipeline tuning. The vllm dispatch likely falls through to a slow generic CUDA reference implementation rather than calling Tri Dao's hand-tuned sm_80 path.

This is consistent with the wiki's `flash_fwd_sm120.py` subclass (`reference-kernels/nvidia/blackwell-geforce/cutedsl/flash-attention/flash_fwd_sm120.py`) being a one-off subclass that overrides only `can_implement` for the 99 KB SMEM cap, suggesting upstream FA on sm_120 is a recent / less-mature path with limited tuning.

## why

sm_120 is **client Blackwell**, not datacenter:
- No `tcgen05` / TMEM / wgmma — must use SM80-era `mma.sync.aligned.m16n8k16`
- 99 KB SMEM cap (not 163 KB Ampere or 228 KB Blackwell-datacenter)
- Tile / stage tuning that works on H100 (228 KB SMEM) won't fit on sm_120

Upstream attention libraries (vllm, flashinfer, xformers) apparently haven't tuned their kernel selectors for this configuration yet. The fast path that exists for sm_80 / sm_89 / sm_90 / sm_100 either doesn't exist for sm_120 or is gated behind a feature flag we couldn't find.

## evidence

- `kernel_opt_attn_fp4_fusion/probe_v3_env.py` — Q3 confirms `vllm.vllm_flash_attn.flash_attn_varlen_func` imports + runs to completion on sm_120 (output shape correct, dtype bf16)
- `kernel_opt_attn_fp4_fusion` commit `0412f33` — V3 hybrid implementation; bench logs show vllm 1645.72 us vs memcpy estimate 99 us
- Stage 3 closeout `wiki_drafts/stage3-closeout.md` §7 — full math + speculated root cause

## lesson

1. **For sm_120 attention work, do not assume vllm is fast.** Bench it on actual sm_120 hardware before committing to a hybrid (vllm-FA + custom-epilogue) architecture. A Path-1 fused-attn-into-epilogue kernel is the only way to actually hit the memory ceiling on Blackwell-Geforce as of 2026-04.
2. **For correctness oracle role**, vllm's slow-but-correct path is fine — it doesn't matter how slow the oracle is, only that it computes the right answer.
3. **Recommended sm_120 attention paths** (in order of expected speed at memory-bound shapes):
   - Build a custom cute-DSL FA forward with sm_120-specific SMEM/stage tuning + epilogue fusion (requires cutlass 4.5+; see `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md` for the cutlass 4.4.2 vendor blocker)
   - Wait for vendor (vllm / flashinfer / Tri Dao) to ship sm_120-tuned wheel
   - Use Triton attention with sm_120-specific autotune sweep
4. **For pipeline architecture decisions**: when the attention forward dominates wall-clock but you can't speed it up, optimizing the consumer (fp4-quant epilogue, etc.) yields tiny end-to-end wins. Always profile the full pipeline to know where the real ceiling sits.

## related

- `wiki_drafts/v3-fa-fusion-deferred-plan.md` — the cute-DSL FA + fp4 fusion plan that would address this gap if cutlass upgrades
- `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md` — why the obvious shortcut (vendor flash_attn.cute) is blocked
- `docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md` — sm_120 memory-bound kernel that DOES hit the ceiling (proves the hardware can; it's a software gap)
