# Triton Kernel Optimization Patterns in Practice

Universal Triton optimization patterns extracted from 47 kernel implementations in `reference-kernels/generic/triton/`. These patterns apply to all hardware backends (NVIDIA/AMD) and do not depend on architecture-specific features.

---

| File | Title | Description |
|------|------|------|
| [autotune-config-pruning.md](autotune-config-pruning.md) | Autotune Configuration & Pruning | Multi-dimensional parameter search (tile/stages/warps) and heuristic pruning strategies |
| [persistent-kernel-tile-scheduling.md](persistent-kernel-tile-scheduling.md) | Persistent Kernel & Tile Scheduling | GROUP_SIZE_M swizzle to improve L2 hit rate, persistent matmul |
| [online-softmax-flash-attention.md](online-softmax-flash-attention.md) | Online Softmax & Flash Attention | Single-pass online softmax, exp2 optimization, causal mask block-level skipping |
| [fused-kernel-patterns.md](fused-kernel-patterns.md) | Fused Kernel Patterns | cross-entropy fusion, element-wise + reduction fusion decision criteria |
| [memory-access-optimization.md](memory-access-optimization.md) | Memory Access Optimization | max_contiguous/multiple_of hints, EVEN_M/EVEN_N boundary check elimination |
| [grouped-gemm-deepgemm.md](grouped-gemm-deepgemm.md) | Grouped GEMM (DeepGEMM Pattern) | MoE scenario multi-expert small GEMM batching |
| [cascade-state-merge.md](cascade-state-merge.md) | Cascade / State Merge (FlashInfer Pattern) | split-KV / distributed attention result merging |
| [mamba-ssm-state-management.md](mamba-ssm-state-management.md) | Mamba / SSM State Management | Mamba2 SSD chunk-wise state updates |

---

## Related Documents

- **Hopper Practice**: [Hopper Optimization Practice](../../nvidia/common/sm90/hands-on/README.md) — SM90-specific optimizations (TMA, WGMMA)
- **Blackwell Practice**: [Blackwell Optimization Practice](../../nvidia/common/hands-on/README.md) — SM100-specific optimizations
- **AMD Practice**: [AMD Optimization Practice](../../amd/common/hands-on/README.md) — MFMA, LDS swizzle
- **Reference Kernels**: `reference-kernels/generic/triton/` — 47 Triton kernel source files
