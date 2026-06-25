# Hopper (SM90) Kernel Optimization Guide

Hopper architecture-specific optimization patterns extracted from 21 kernel implementations in `reference-kernels/nvidia/hopper/`. Covers hands-on experience with two DSL frameworks: CuTeDSL and Gluon.

---

| # | File | Title | Description |
|---|------|------|------|
| 1 | [tma-tensor-memory-accelerator.md](tma-tensor-memory-accelerator.md) | TMA (Tensor Memory Accelerator) | Host-side TMA descriptor creation, asynchronous loading within kernel, TMA store epilogue writeback |
| 2 | [wgmma-warpgroup-mma.md](wgmma-warpgroup-mma.md) | WGMMA (Warpgroup Matrix Multiply-Accumulate) | 128-thread warpgroup cooperative MMA, SS/RS mode selection, differences from Ampere mma.sync |
| 3 | [mbarrier-software-pipeline.md](mbarrier-software-pipeline.md) | mbarrier Software Pipeline | PipelineTmaAsync multi-stage pipeline, producer/consumer pattern, equivalent implementation in Gluon |
| 4 | [warp-specialization.md](warp-specialization.md) | Warp Specialization | DMA/MMA warp role separation, setmaxregister dynamic register reallocation |
| 5 | [persistent-kernel-tile-scheduler.md](persistent-kernel-tile-scheduler.md) | Persistent Kernel and Tile Scheduler | Static persistent tile scheduler, swizzle for improved L2 locality |
| 6 | [flash-attention-hopper.md](flash-attention-hopper.md) | Flash Attention Hopper Specialization | TMA + WGMMA FMHA, online softmax, K/V double buffering |
| 7 | [gdn-decode-state-management.md](gdn-decode-state-management.md) | GDN Decode and Complex State Management | Multi-stage state update kernel, state merge optimization |
| 8 | [mamba2-ssd-warp-specialization.md](mamba2-ssd-warp-specialization.md) | Mamba2 SSD 7-Role Warp Specialization | Extreme warp specialization (7 roles), SSM kernel multi-stream parallelism |
| 9 | [vectorized-fp8-conversion.md](vectorized-fp8-conversion.md) | Vectorized FP8 Conversion via PTX | PTX vectorized FP8 type conversion, E4M3/E5M2 hardware acceleration |
| 10 | [cluster-level-reduction.md](cluster-level-reduction.md) | Cluster-Level Reduction | Thread Block Cluster cross-SM reduction, Split-K GEMM scenario |
| 11 | [seesaw-warpgroup-scheduling.md](seesaw-warpgroup-scheduling.md) | Seesaw Warpgroup Scheduling | FlashMLA core: two WGs alternating between CUDA Core/Tensor Core, 660 TFLOPS on H800 |
| 12 | [dsm-crossover-fp8-dequant.md](dsm-crossover-fp8-dequant.md) | DSM Crossover FP8 Dequantization | Cluster=2 cross dequantization, st.async write to peer SMEM, 410 TFLOPS on H800 |
| 13 | [split-kv-tile-scheduler.md](split-kv-tile-scheduler.md) | Split-KV Decode and Tile Scheduler | Long sequence KV splitting + pre-generated scheduling metadata + PDL kernel linking |

---

## Related Documents

- **General Triton Patterns**: [Triton Optimization Patterns](../../../../generic/hands-on/README.md)
- **Blackwell Guide**: [Blackwell Optimization Guide](../../hands-on/README.md)
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
