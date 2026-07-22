# CUDA Tile vs Triton: Hopper/Blackwell GPU Benchmark Paper

Analysis of the paper "Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs," examining CuTile's positioning relative to cuBLAS, Triton, and WMMA for GPU kernel development.

---

## 1. Paper Background

Paper: "Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs" (arXiv, 2026-04-25, by Divakar Kumar Yadav, Tian Zhao, Deepak Kumar).

The core question is not merely a new CUDA abstraction, but a broader issue: **who should write custom kernels in AI systems?**

Traditional split:
- Standard GEMM → cuBLAS / torch.matmul
- Flexible fused operators → Triton
- Peak performance (Attention/MoE/quantization/communication fusion) → CUDA C++ / CUTLASS / FlashAttention handwritten

**CuTile = NVIDIA's attempt to fill this gap**: elevating GPU programming from the SIMT thread perspective to a **tile perspective** — describe "what math to perform on these tiles," and the compiler/runtime handles mapping to threads, Tensor Cores, and TMA. CUDA 13.1 simultaneously ships CUDA Tile IR + cuTile Python.

Test matrix: CuTile / cuBLAS / Triton / WMMA / raw SIMT × {H100 NVL, B200, RTX PRO 6000 Blackwell Server Edition} × {GEMM, fused MHA, end-to-end LLM inference}.

---

## 2. GEMM: CuTile is Good, but Does Not Replace cuBLAS

BF16 square GEMM **8192×8192 on B200**:

| Implementation | TFLOP/s |
| --- | --- |
| cuBLAS | **1671.8** |
| Triton | 1032.9 |
| CuTile | 875.8 |

CuTile on Blackwell = **52–79%** of cuBLAS, significantly faster than WMMA, but insufficient to replace cuBLAS.

The more interesting comparison is **CuTile vs WMMA**:
- CuTile outperforms WMMA by **1.5–5.0x** across most GEMM sizes
- Kernel code: CuTile **22 lines** vs WMMA **123 lines**

> Standard GEMM: do not switch to CuTile. Custom GEMM: CuTile is worth evaluating. Maintaining WMMA code: **strongly recommend** evaluating CuTile. This is not about performance championships — it is about engineering cost.

---

## 3. Attention: CuTile's Strongest Result on B200

BF16 causal, batch=8, 32 heads, d=128, **seq=4096 on B200**:

| Implementation | TFLOP/s | vs FA2 |
| --- | --- | --- |
| CuTile | **1007.4** | **2.51x** |
| Triton | 525.6 | 1.31x |
| FlashAttention-2 | 400.7 | 1.00x |

However, the same CuTile attention kernel on **RTX PRO 6000 (sm_120)**:

| Implementation | TFLOP/s |
| --- | --- |
| FlashAttention-2 | 335.4 |
| Triton | 287.7 |
| CuTile | **178.8** (-47%) |

**CuTile attention paradox**: 2.51x over FA2 on B200, but 47% slower on RTX PRO 6000 — a **5.6x cross-architecture gap**.

Paper assessment: sm_100 is the tileiras compiler's primary optimization target; sm_120 has maturity differences. TMA, Tensor Core microarchitecture, and shared memory configuration may all contribute. **"Blackwell" is not a sufficiently precise descriptor: B200 and RTX PRO 6000 may have entirely different kernel tuning outcomes.**

> Note: The paper compares against FlashAttention-**2** (no TMA), not Hopper FA3 or Blackwell FA4. CuTile's advantage primarily comes from leveraging TMA + new Tensor Core hardware capabilities.

---

## 4. Triton's Position Remains Unshaken

- Triton runs on all platforms (including H100, where CuTile was unavailable in the paper's test environment)
- Maintains **62–101%** of cuBLAS performance for GEMM
- Cross-architecture stability is Triton's strength; CuTile shows large variation across Blackwell variants

> CuTile provides an enticing upper bound on B200; Triton provides a stable engineering lower bound for cross-architecture fleets. These are complementary, not substitutes.

---

## 5. Code Volume: CuTile's Strongest Selling Point is Maintenance Cost

| Kernel | cuBLAS | Triton | CuTile | WMMA | FA2 / SDPA |
| --- | --- | --- | --- | --- | --- |
| GEMM | 1 line (API) | 53 | **22** | 123 | — |
| Attention | — | 62 | **60** | — | 1 line (library API) |

- **GEMM: CuTile significantly less than Triton and WMMA**: `ct.load/mma/store` abstracts away SMEM tiling, register allocation, and warp scheduling
- **Attention: CuTile and Triton comparable**: Attention complexity stems from online softmax, causal mask, running max/sum — not just tiling

> The most valuable abstraction is not making simple things simpler, but making complex-yet-repetitive hardware details unnecessary for every team to hand-write.

---

## 6. End-to-End Inference: Not Equivalent to LLM Serving Acceleration

The paper tested a LLaMA-7B-like **4-layer proxy** model:

- Fused attention backends (SDPA, FA2) deliver **1.7–2.4x prefill speedup** over naive
- Decode phase under fused backends is memory-bandwidth-bound: B200 batch=32, all backends produce **4594–4647 tok/s** (nearly identical)

> One cannot claim "CuTile attention is 2.5x faster, therefore LLM serving is 2.5x faster." This is kernel-level vs system-level. CuTile provides value for prefill-heavy, long-context scenarios; decode-heavy online serving is bottlenecked by KV cache, bandwidth, batching, scheduler, and speculative decoding.

---

## 7. Version Considerations

- Paper environment: CUDA Toolkit 12.8 + 13.1, CuTile 1.1.0; H100 unavailable
- Subsequent NVIDIA update: **CUDA 13.2** extends CUDA Tile support to CC **8.x, 10.x, 11.x, 12.x**; R580+ driver; Python 3.10-3.13; CUDA Toolkit 13.1+ or `cuda-tile[tileiras]`
- However, Hopper (H100/H200) is still not CuTile's primary battlefield

---

## 8. Selection Decision Table

| Scenario | Recommendation |
| --- | --- |
| Standard GEMM (torch.matmul / cuBLAS sufficient) | **Do not switch to CuTile** |
| Custom GEMM fusion on Blackwell | Worth trying CuTile |
| Maintaining WMMA Tensor Core kernels | **Strongly recommend** evaluating CuTile |
| Fused attention on B200/B100 (especially long-context prefill) | Prioritize CuTile |
| Attention on RTX PRO 6000 / sm_120 | Hold off — FA2 / Triton more stable |
| Mixed GPU fleet (H100 + Blackwell) | Triton better suited as unified solution |
| FP8 / INT8 production quantization kernels | Paper provides no evidence — cannot conclude |

---

## 9. The Bigger Signal

NVIDIA is pushing Tensor Core kernel programming toward higher-level, more Pythonic, more compiler-driven approaches. Triton already pushed the boundary one layer up; CuTile is NVIDIA pushing from the hardware side inward: **you write tile-level logic, we handle the underlying hardware mapping.**

Paper self-acknowledged limitations: No NCU roofline analysis, no FP8/INT8 coverage, single GPU instance, 4-layer proxy rather than full 32-layer LLaMA.

> The most reasonable stance is not "all-in on CuTile," but adding it to the toolbox:
> - cuBLAS handles standard operators
> - Triton handles cross-architecture customization
> - CuTile handles high-value kernels worth betting on for Blackwell
>
> New abstractions do not eliminate old ones — they redefine where each abstraction fits best.
