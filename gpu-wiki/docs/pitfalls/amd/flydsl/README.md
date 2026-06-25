# AMD FlyDSL Pitfalls

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| [flash-attn-pitfalls.md](flash-attn-pitfalls.md) | FlashAttention forward family: bf16 causal/GQA, bf16 bit-packed mask, fp16 no-mask CK-ISA tuning | MI308X (gfx942) | 40 |
| [attention-backward-dkdv-pitfalls.md](attention-backward-dkdv-pitfalls.md) | Attention backward dQ/dK/dV | MI308X (gfx942) | 12 |
| [flash-attn-bwd-mask-integration-pitfalls.md](flash-attn-bwd-mask-integration-pitfalls.md) | FlashAttention backward API integration with arbitrary mask | MI308X (gfx942) | 6 |
| [chunk-gdn-mi308x-wave-specialization-pitfalls.md](chunk-gdn-mi308x-wave-specialization-pitfalls.md) | Chunk-GDN wave-specialized megakernel | MI308X (gfx942) | 6 |
| [chunk-gdn-pitfalls.md](chunk-gdn-pitfalls.md) | Chunk-GDN FlyDSL / CDNA family pitfalls | CDNA | 11 |
| [flash-attn-d256-pitfalls.md](flash-attn-d256-pitfalls.md) | FlashAttention D256 path | MI355X / CDNA4-adjacent | 10 |

## Cross-kernel notes

- FlashAttention forward tuning is dtype- and feature-specific. The fp16
  no-mask CK95 work must use its own CK measurement and should not reuse bf16
  mask or bf16 causal/GQA numbers.
- For CK mimicry, real `rocprofv3 --att` disassembly is the source of truth:
  count `ds_read_b128`, `ds_read2_b32`, `s_waitcnt`, and `s_nop`, then verify
  wall time with kernel trace.
- LDS layout experiments need finite-output checks before profiling. Compiling
  or launching a K swizzle / dword-to-LDS path is not a correctness signal.

## Related optimization reports

- [cdna3-flash-attention-fp16-nomask-ck-isa-optimization.md](../../../ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-fp16-nomask-ck-isa-optimization.md)
- [cdna3-flash-attention-bf16-mask-optimization.md](../../../ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-mask-optimization.md)
- [cdna3-flash-attention-bf16-gqa-optimization.md](../../../ref-docs/amd/flydsl/gfx942/cdna3-flash-attention-bf16-gqa-optimization.md)
