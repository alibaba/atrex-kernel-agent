# SM120 NVFP4 Split-K GEMV

> **Usability status:** `requires-external-checkout`
>
> Set `$CUTLASS_DIR` to a compatible CUTLASS checkout before building or running these files.

CUDA C++ NVFP4 decode GEMV Split-K reference for NVIDIA RTX PRO 5000
Blackwell-GeForce (`sm_120`).

---

| File | Description |
|------|-------------|
| [gemm_v3_splitk_sm120.cu](gemm_v3_splitk_sm120.cu) | Final CUDA kernel: phase-1 Split-K partial sums + phase-2 FP32 reduce to BF16. Tuned for M=1 decode GEMV C2-like shapes. |
| [cutlass_splitk_dispatch_sm120.py](cutlass_splitk_dispatch_sm120.py) | vLLM CUTLASS NVFP4 dispatch pattern: C2 decode shapes call Split-K; C1 / prefill remain on stock CUTLASS. |
| [gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu](gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu) | Scoped CTA-3D TMA source snapshot for TP=1 C2 decode `M=1,N=5120,K=17408`, `S=8`, `tile_n=8`. |
| [build_gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.sh](build_gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.sh) | Build helper for the archived CTA-3D TMA source. |
| [flashinfer_b12x_cta3d_dispatch_sm120.py](flashinfer_b12x_cta3d_dispatch_sm120.py) | vLLM FlashInfer b12x + CTA-3D hybrid dispatch wrapper for allowlisted small-M NVFP4 decode shapes. |
| [omoexplore-kernel-notes.md](omoexplore-kernel-notes.md) | Knowledge map for which archived kernel or dispatch pattern to study for each production lesson. |

## Hardware and scope

- Hardware: RTX PRO 5000 / 4000 Blackwell-GeForce, `sm_120`, 110 SMs.
- Kernel family: CUDA C++ + inline PTX `mma.sync.aligned.kind::mxf4nvf4`.
- Target shape: M=1 decode GEMV, small `N`, long `K`.
- Non-target shape: C1-like large `N`, prefill / batched GEMM, and generic GEMM.

## Related docs

- Optimization report:
  [docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md)
- Consolidated production lessons:
  [docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-decode-gemm-production-lessons.md](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-decode-gemm-production-lessons.md)
- Pitfalls:
  [docs/nvidia/blackwell-geforce/pitfalls/cuda/nvfp4-split-k-gemv-pitfalls.md](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cuda/nvfp4-split-k-gemv-pitfalls.md)
