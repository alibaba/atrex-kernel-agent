# AMD GPU Kernel Optimization in Practice

AMD architecture optimization patterns extracted from 42 kernel implementations in `reference-kernels/amd/`, covering CDNA3 (gfx942), CDNA4 (gfx950), and RDNA4 (gfx1250). Each file focuses on a specific optimization technique, including code examples and practical experience.

---

| File | Description |
|------|------|
| [MFMA Instruction Selection & Usage](mfma-instruction-selection.md) | CDNA3/CDNA4 MFMA instruction differences, mfma_scale for FP8 quantized GEMM |
| [LDS Bank Conflict Elimination](lds-bank-conflict-swizzle.md) | XOR16 Swizzle to eliminate LDS bank conflicts, 20-40% throughput improvement |
| [Preshuffle B Layout](preshuffle-b-layout.md) | Pre-arrange weight matrix to avoid runtime layout conversion, eliminating ds_bpermute overhead |
| [Software Pipelining](software-pipelining.md) | Three pipelining patterns: CDNA3 register-based, CDNA4 async DMA, RDNA4 TDM |
| [Instruction Scheduling Control](instruction-scheduling.md) | Manual instruction scheduling control via sched_barrier / sched_mfma for pipeline overlap optimization |
| [MoE 2-Stage Fusion](moe-2stage-fusion.md) | Expert GEMM + SiLU fusion, Mixed MoE mixed precision |
| [RDNA4-Specific Optimizations](rdna4-wmma-optimization.md) | WMMA (Wave32) matrix multiplication, ds_load_tr16_b128 transposed load |
| [Cross-Architecture Conditional Compilation](cross-architecture-conditional-compilation.md) | constexpr branching for CDNA3/CDNA4/RDNA4 conditional compilation |
| [Paged Attention Decode (FP8)](paged-attention-decode-fp8.md) | KV Cache paged management + FP8 quantization + online softmax |

---

## Related Documentation

- **Tuning Guide**: AMD GPU Kernel Tuning Guide — CDNA3 vs CDNA4 hardware spec comparison
- **General Triton Patterns**: [Triton Optimization Patterns in Practice](../../../generic/hands-on/README.md)
- **Reference Kernels**: `reference-kernels/amd/` — 42 AMD kernel source files
