# Blackwell (SM100) Kernel Optimization Hands-On

Blackwell architecture-specific optimization patterns extracted from `reference-kernels/nvidia/blackwell/` with 114 kernel implementations. Blackwell is currently the architecture with the most kernels, introducing entirely new hardware features such as tcgen05, TMEM, and CLC.

---

| File | Description |
|------|-------------|
| [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) | Tensor Memory accumulator and Double-Buffered TMEM, relieving VGPR pressure |
| [Three-Role Warp Specialization](three-role-warp-specialization.md) | TMA + MMA + Epilogue three-role division, fully overlapping load/compute/store |
| [CLC (Cluster Launch Control)](cluster-launch-control.md) | Hardware-level dynamic tile scheduling, replacing static pid mapping |
| [2CTA Cooperation](two-cta-cooperation.md) | Two CTAs cooperating on one tile, doubling TMA load bandwidth |
| [Block-Scaled MMA](block-scaled-mma.md) | MXF8/NVF4 block-scaled matrix multiplication, hardware quantization support |
| [Epilogue Fusion](epilogue-fusion.md) | GEMM + SwiGLU/RMSNorm/FP4Quantize fusion, reducing HBM reads/writes |
| [Distributed GEMM+AllReduce](distributed-gemm-allreduce.md) | Two-shot AllReduce, overlapping GEMM with communication |
| [Programmatic Dependent Launch](programmatic-dependent-launch.md) | No host synchronization between kernels, reducing launch latency |
| [MLA Decode](mla-decode.md) | Multi-Latent Attention inference optimization, low-rank KV cache compression |
| [Pipeline Mode Comparison](pipeline-comparison.md) | Blackwell vs Hopper pipeline type comparison |
| [TMA Gather Sparse Decode](tma-gather-sparse-decode.md) | SM100 TMA Gather for collecting non-contiguous tokens, MLA decode with TMEM/UMMA |

---

## Related Documents

- **Hopper Hands-On**: [Hopper Optimization Hands-On](../sm90/hands-on/README.md) — SM90 comparison
- **General Triton**: [Triton Optimization Patterns Hands-On](../../../generic/hands-on/README.md) — Cross-architecture general patterns
- **Reference Kernels**: `reference-kernels/nvidia/blackwell/` — 114 Blackwell kernel source code
