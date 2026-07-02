# Seesaw Warpgroup Scheduling

FlashMLA's core innovation: Two warp groups alternate execution between CUDA Cores (softmax/scaling) and Tensor Cores (WGMMA), forming a "seesaw" pattern that maximizes utilization of both compute units.


**Last updated**: 2026-07-01

**Source**: FlashMLA `csrc/sm90/decode/dense/splitkv_mla.cuh`

---

## Background

In MLA Decode, each token needs to execute:
1. **QK^T GEMM** (Tensor Core) → produces attention scores
2. **Softmax + Scaling** (CUDA Core) → normalization
3. **PV GEMM** (Tensor Core) → produces attention output

In the traditional serial execution approach, Tensor Cores and CUDA Cores are each idle approximately 50% of the time.

## Seesaw Pattern

```
English description
WG0: [QK GEMM₀] [softmax₀] [QK GEMM₁] [softmax₁] ...
WG1:            [PV GEMM₀]             [PV GEMM₁] ...
      ↑ Tensor Core  ↑ CUDA Core  ↑ Tensor Core
```

Two warp groups share the same output matrix (split into O_L and O_R by left/right halves) and alternate between using Tensor Cores and CUDA Cores:

```cpp
// WG0 execute QK GEMM(Tensor Core )
// WG1 execute softmax + scaling(CUDA Core )
__syncwarp();

// WG0 softmax(CUDA Core )
// WG1 PV GEMM(Tensor Core )
__syncwarp();
```

### Output Matrix Partitioning

```
Output [num_heads × head_dim_v]:
┌──────────────┬──────────────┐
│    O_L       │    O_R       │
│ (WG0 ) │ (WG1 ) │
└──────────────┴──────────────┘
     256 cols       256 cols
```

- WG0 handles O_L (left half accumulator)
- WG1 handles O_R (right half accumulator)
- Both share the same softmax row statistics (m_i, l_i)

### Three-Warpgroup Role Assignment

| Warpgroup | Role | Register Allocation |
|-----------|------|---------------------|
| WG0 | Compute (QK GEMM + softmax + PV GEMM left half) | 192 |
| WG1 | Compute (softmax + PV GEMM right half) | 160 |
| WG2 | Producer (TMA load KV) | Less |

## Practical Experience

- Seesaw improves Tensor Core utilization from ~50% to ~80%
- Key constraint: The CUDA Core and Tensor Core phases of the two WGs must have roughly equal duration, otherwise one side will stall
- Only applicable to the MLA Decode alternating pattern of QK GEMM → softmax → PV GEMM
- Dense decode achieves **660 TFLOPS** (compute-bound) and **3000 GB/s** (memory-bound) on H800

## Related

- **Warp Specialization**: [Warp Specialization](warp-specialization.md) — Foundation of DMA/MMA two-role task partitioning
- **WGMMA**: [WGMMA](wgmma-warpgroup-mma.md) — Foundation of 128-thread Warpgroup MMA
- **Flash Attention**: [Flash Attention Hopper](flash-attention-hopper.md) — General Hopper FMHA
- **DSM FP8 Dequant**: [DSM Crossover FP8 Dequant](dsm-crossover-fp8-dequant.md) — FlashMLA's FP8 version
