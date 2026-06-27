# Reference Kernels

Python GPU kernel implementations extracted from upstream reference repositories, organized by **hardware architecture → DSL language → source repository**.

## Usability Status

Reference kernels are indexed with one of these statuses:

| Status | Meaning |
|--------|---------|
| `runnable` | Expected to run with documented dependencies and environment variables. |
| `requires-external-checkout` | Requires an external repository root such as `$AITER_BASE`, `$CUTLASS_DIR`, or `$ATREX_OPEN_ROOT`. |
| `diagnostic-archive` | Preserved for a specific investigation, benchmark shape, or production diagnosis; read as evidence before adapting. |
| `historical-snapshot` | Preserved for source comparison or provenance; not expected to run unchanged. |

## Directory Structure

```
reference-kernels/
├── nvidia/
│   ├── ampere/                 # SM80 (A100)
│   │   ├── cutedsl/            # cutlass + flash-attention
│   │   ├── gluon/              # async-copy tutorial
│   │   └── triton/             # DeepGEMM legacy grouped GEMM
│   ├── hopper/                 # SM90 (H100/H20)
│   │   ├── cutedsl/            # cutlass + flash-attention + flashinfer (norm/GDN/Mamba) + quack (reduction/GEMM/MLP) + tilelang (inline-PTX utility library)
│   │   └── gluon/              # TMA, WGMMA, persistence, warp-specialization, conv
│   ├── blackwell/              # SM100 (B200)
│   │   ├── cutedsl/            # cutlass + cutex + cuLA + flash-attention + flashinfer (GEMM/MLA/MoE/quant) + quack (SM100 GEMM)
│   │   ├── gluon/              # tcgen05, CLC, multi-CTA, attention, convolution
│   │   └── triton/             # cuLA chunk_intra attention
│   └── blackwell-geforce/      # SM120
│       ├── cuda/               # CUDA C++ / inline PTX NVFP4 Split-K, prefill, RMSNorm-MLP PDL diagnostics
│       ├── cutedsl/            # cutlass + flash-attention + flashinfer + task39 b12x diagnostic fork + GDN chunk fwd + quack (SM120 GEMM)
│       └── triton/             # vLLM GDN post-processing fused norm+gate
├── amd/
│   ├── cdna/                   # CDNA3 (gfx942) + CDNA4 (gfx950) generic
│   │   ├── flydsl/             # FlyDSL framework kernel examples
│   │   └── triton/aiter/       # aiter inference operator library Triton kernels (Attention/GEMM/MoE/Norm/Quant 80+)
│   ├── cdna3/                  # CDNA3 (gfx942) specific
│   │   └── flydsl/             # MI308X Flash Attention / Attention backward / Chunk-GDN optimized kernels
│   ├── cdna4/                  # CDNA4 (gfx950) specific
│   │   ├── flydsl/             # FlyDSL chunk-GDN fwd_h (0.97x Triton)
│   │   └── gluon/              # Gluon matmul + aiter production-level GEMM/PA kernels
│   └── rdna4/                  # RDNA4 (gfx1250)
│       ├── flydsl/             # FlyDSL RDNA4 WMMA/FP8 GEMM
│       └── gluon/              # Triton gfx1250 GEMM/FA examples
├── generic/                    # Architecture-agnostic or multi-architecture
│   ├── triton/                 # Triton tutorials + triton-kernels multi-arch library + flash-attention + flashinfer + LeetCUDA
│   └── gluon/                  # Gluon basic tutorials (intro, layouts)
└── README.md
```

## Statistics

| Architecture | File Count | DSL |
|------|--------|-----|
| nvidia/ampere | 32 | CuTeDSL, Gluon, Triton |
| nvidia/hopper | 98 | CuTeDSL, Gluon |
| nvidia/blackwell | 135 | CuTeDSL, Gluon, Triton |
| nvidia/blackwell-geforce | 57 | CuTeDSL, Triton, CUDA |
| amd/cdna | 119 | FlyDSL, Triton (aiter) |
| amd/cdna3 | 15 | FlyDSL |
| amd/cdna4 | 26 | Gluon (triton + aiter) |
| amd/rdna4 | 21 | FlyDSL, Gluon |
| generic | 61 | Triton, Gluon |
| **Total** | **567** | |

## Source Repositories

| Repository | Description | Extracted DSL |
|------|------|-----------|
| `cutlass` | NVIDIA CUTLASS CuTeDSL official examples | CuTeDSL |
| `cutex` | Blackwell CuTeDSL GEMM/FA4 implementation | CuTeDSL |
| `cuLA` | Linear Attention / KDA kernel | CuTeDSL, Triton |
| `flash-attention` | Flash Attention v4 multi-architecture implementation | CuTeDSL, Triton |
| `FlyDSL` | AMD FlyDSL framework kernel examples | FlyDSL |
| `triton` | Triton/Gluon official tutorials + triton_kernels library | Triton, Gluon |
| `LeetCUDA` | Triton introductory kernels | Triton |
| `flashinfer` | FlashInfer inference acceleration library | CuTeDSL, Triton |
| `DeepGEMM` | DeepSeek GEMM library (legacy Triton kernels) | Triton |
| `aiter` | AMD official AI inference operator library (Attention/GEMM/MoE/Norm/Quant) | Triton, Gluon |
| `quack` | Dao-AILab QuACK high-performance kernel library (Reduction/GEMM/MLP/TopK) | CuTeDSL |
| `tilelang` | TileLang CuTeDSL backend contrib library (inline PTX utilities: atomic/ldsm/mma/ieee math/grid sync) | CuTeDSL |

### Repositories Not Extracted (No Python Kernel)

| Repository | Reason |
|------|------|
| `composable_kernel` | Pure C++ HIP template library |
| `cute-gemm` | Pure CUDA C++ |
| `hpc-ops` | Pure CUDA C++, Python is only a wrapper |
| `DeepGEMM` (sm90/sm100) | Hopper/Blackwell kernels are pure CUDA C++, only Ampere legacy uses Triton |

## Kernel Type Index

### GEMM / MatMul
- `nvidia/blackwell/cutedsl/cutlass/dense_gemm*.py` — Blackwell GEMM from basics to persistent
- `nvidia/blackwell/cutedsl/cutlass/tutorial_gemm/` — Blackwell GEMM tutorial series
- `nvidia/blackwell/cutedsl/flashinfer/dense_blockscaled_gemm_sm100.py` — Blackwell block-scaled GEMM
- `nvidia/blackwell/cutedsl/flashinfer/grouped_gemm_masked_blackwell.py` — Blackwell grouped GEMM
- `nvidia/hopper/cutedsl/cutlass/dense_gemm*.py` — Hopper GEMM
- `nvidia/ampere/cutedsl/cutlass/sgemm.py`, `tensorop_gemm.py` — Ampere GEMM
- `nvidia/ampere/triton/DeepGEMM/` — DeepGEMM grouped GEMM (Ampere legacy)
- `amd/cdna/flydsl/FlyDSL/preshuffle_gemm.py` — AMD CDNA preshuffle GEMM
- `amd/cdna4/gluon/matmul_gluon_gfx950_*.py` — CDNA4 Gluon matmul series
- `amd/cdna/triton/aiter/gemm/` — aiter quantized GEMM (A8W8/A4W4/FP4 + block-scale/preshuffle/split-K)
- `amd/cdna4/gluon/gemm_a8w8_blockscale.py` — aiter Gluon A8W8 block-scale GEMM
- `nvidia/hopper/cutedsl/quack/gemm*.py` — QuACK SM90 GEMM (WGMMA + pingpong + composable epilogue)
- `nvidia/blackwell/cutedsl/quack/gemm_sm100.py` — QuACK SM100 GEMM (UMMA/tcgen05 + CLC)
- `nvidia/blackwell-geforce/cutedsl/quack/gemm_sm120.py` — QuACK SM120 GEMM (warp MMA)
- `nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/` — CUDA NVFP4 decode GEMV Split-K (C2-like small-N / long-K shapes)
- `nvidia/blackwell-geforce/cuda/nvfp4_linear_qkvz_atrex/` — Diagnostic ATREX NVFP4 linear_qkvz Split-K source for structural DRAM-BW ceiling analysis
- `nvidia/blackwell-geforce/cuda/nvfp4_prefill_gemm/` — Experimental SM120 NVFP4 prefill GEMM router and CUDA candidates
- `nvidia/blackwell-geforce/cutedsl/flashinfer/dense_blockscaled_gemm_sm120_task39_diagnostic.py` — Diagnostic FlashInfer b12x CuTe DSL fork for SM120 prefill/gate-up SF-layout experiments
- `generic/triton/triton-tutorials/03-matrix-multiplication.py` — Triton matmul tutorial
- `generic/triton/triton-kernels/matmul_details/` — Multi-architecture matmul library

### Attention
- `nvidia/blackwell/cutedsl/cutlass/fmha*.py` — Blackwell Flash Attention
- `nvidia/blackwell/cutedsl/cutlass/mixed_input_fmha/` — Blackwell mixed-precision FMHA
- `nvidia/blackwell/cutedsl/cutlass/mla/` — Blackwell MLA decode (cutlass)
- `nvidia/blackwell/cutedsl/flashinfer/mla_decode_fp16.py`, `mla_decode_fp8.py` — Blackwell MLA decode (flashinfer)
- `nvidia/blackwell/cutedsl/cuLA/` — Linear attention / KDA
- `nvidia/hopper/cutedsl/cutlass/fmha.py` — Hopper FMHA
- `nvidia/hopper/cutedsl/flashinfer/gdn_decode_*.py` — Hopper GDN decode series
- `nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/` — SM120 GDN chunk forward V113 (CuTeDSL, no-cache directional final_state, 0.531-0.533ms at T=6144, 1.51× same-process FLA)
- `nvidia/ampere/cutedsl/cutlass/flash_attention_v2.py` — Ampere Flash Attention v2
- `amd/cdna/flydsl/FlyDSL/flash_attn_func.py` — AMD Flash Attention
- `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py` — MI308X BF16 Flash Attention causal+GQA
- `amd/cdna3/flydsl/FlyDSL/flash_attn_func_nomask_mi308x.py` — MI308X BF16 Flash Attention no-mask D64, ATT-guided `ds_bpermute_lgkm_sum`
- `amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py` — MI308X BF16 Flash Attention bit-packed arbitrary mask
- `amd/cdna/flydsl/FlyDSL/pa_decode_fp8.py` — AMD paged attention decode
- `amd/cdna4/flydsl/FlyDSL/chunk_gdn_fwd_h.py` — CDNA4 chunk-GDN fwd_h (0.97x Triton, barrier removal + double-buffered k-LDS)
- `generic/triton/triton-tutorials/06-fused-attention.py` — Triton fused attention tutorial
- `generic/triton/flash-attention/flash_attn_triton*.py` — Flash Attention Triton implementation
- `generic/triton/flashinfer/cascade.py` — Attention cascade/merge
- `amd/cdna/triton/aiter/attention/` — aiter PA decode/prefill, MLA decode, lean attention, unified attention (18 items)
- `amd/cdna/triton/aiter/flash_attn_amd/` — aiter AMD optimized Flash Attention (fwd/bwd)

### Softmax / LayerNorm / Reduction
- `nvidia/blackwell/cutedsl/cutlass/reduce.py`, `rmsnorm.py` — Blackwell reduce/RMSNorm
- `nvidia/hopper/cutedsl/flashinfer/rmsnorm.py`, `layernorm.py`, `fused_add_rmsnorm.py` — Hopper Norm series
- `nvidia/blackwell/cutedsl/flashinfer/rmsnorm_fp4quant.py`, `add_rmsnorm_fp4quant.py` — Blackwell fused Norm+FP4
- `nvidia/blackwell-geforce/cuda/rmsnorm_mlp_nvfp4_pdl/` — SM120 RMSNorm + MLP input NVFP4 quant, PDL handoff, and row-ready diagnostic sources
- `amd/cdna/flydsl/FlyDSL/softmax_kernel.py`, `layernorm_kernel.py`, `rmsnorm_kernel.py`
- `nvidia/hopper/cutedsl/quack/rmsnorm.py`, `softmax.py`, `cross_entropy.py`, `topk.py` — QuACK reduction kernels (~90% SOL)
- `nvidia/hopper/cutedsl/quack/reduction_base.py` — QuACK 4-level reduction framework (thread→warp→block→cluster)
- `generic/triton/triton-tutorials/02-fused-softmax.py`, `05-layer-norm.py`
- `amd/cdna/triton/aiter/normalization/rmsnorm.py` — aiter RMSNorm (with fused add/quant/bwd)

### Quantization
- `nvidia/blackwell/cutedsl/flashinfer/nvfp4_quantize.py`, `mxfp4_quantize.py`, `mxfp8_quantize.py` — Blackwell FP4/FP8 quantization
- `amd/cdna/triton/aiter/quant/` — aiter FP8/MXFP4 fused quantization (with RMSNorm+quant fusion)

### MoE
- `nvidia/blackwell/cutedsl/flashinfer/*gemm_swiglu_fusion.py`, `*gemm_finalize_fusion.py` — Blackwell fused MoE
- `amd/cdna/flydsl/FlyDSL/moe_gemm_2stage.py`, `moe_blockscale_2stage.py`, `mixed_moe_gemm_2stage.py`
- `amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/` — MI308X FP8 PTPC Fused MoE checkpoint (pause state, continuation map)
- `nvidia/blackwell/cutedsl/cutlass/blockwise_gemm/` — Blackwell grouped/masked GEMM
- `amd/cdna/triton/aiter/moe/` — aiter MoE GEMM (A8W8/A4W4/MXFP4 + E2E fused + routing, 17 items)

### SSM / Mamba
- `nvidia/blackwell/cutedsl/cutlass/mamba2_ssd/` — Blackwell Mamba2 SSD (cutlass)
- `nvidia/hopper/cutedsl/flashinfer/ssd_kernel.py` — Hopper Mamba SSD (flashinfer)
- `generic/triton/flashinfer/ssd_chunk_state.py` — Triton Mamba SSD

### CuTeDSL inline PTX (reference for `llvm.inline_asm` patterns)

CuTeDSL does not have `tl.inline_asm`; you need to embed PTX via `cutlass._mlir.dialects.llvm.inline_asm`. The following files are ready-to-use references for various inline PTX patterns. **Pattern overview**: [`docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md`](../docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md).

| What you want to do | Reference file |
|---------|---------|
| **Getting started: single PTX instruction** | [`nvidia/hopper/cutedsl/tilelang/utils.py`](nvidia/hopper/cutedsl/tilelang/utils.py) (`pack_half2`: `mov.b32 {r,r}`) |
| **Single math instruction (tanh.approx)** | [`nvidia/hopper/cutedsl/tilelang/math.py`](nvidia/hopper/cutedsl/tilelang/math.py) (`tanh.approx.f32`) |
| **f-string template generating multiple variants (rounding mode)** | [`nvidia/hopper/cutedsl/tilelang/ieee_math.py`](nvidia/hopper/cutedsl/tilelang/ieee_math.py) (`add/sub/mul/fma/rcp/sqrt/div.{rn,rz,rm,rp}.{f32,f64}`) |
| **fp16/bf16 via i16 bitcast** | [`nvidia/hopper/cutedsl/tilelang/atomic.py`](nvidia/hopper/cutedsl/tilelang/atomic.py) (`atom.add.noftz.f16/v2.f16/v2.bf16`) |
| **Multiple outputs (StructType + extractvalue)** | [`nvidia/hopper/cutedsl/tilelang/atomic.py`](nvidia/hopper/cutedsl/tilelang/atomic.py) (`atom.global.add.v4.f32` with 4 outputs) |
| **CAS spin loop (implementing float atomic max/min)** | [`nvidia/hopper/cutedsl/tilelang/atomic.py`](nvidia/hopper/cutedsl/tilelang/atomic.py) (`atom.cas.b32` + `setp.ne.b32` + `@p bra retry`) |
| **mbarrier `try_wait` spin** | [`nvidia/hopper/cutedsl/tilelang/cpasync.py`](nvidia/hopper/cutedsl/tilelang/cpasync.py) (`mbarrier.try_wait.parity.shared::cta.b64`) |
| **mma.sync factory (generating multiple dtype/shape/layout combos)** | [`nvidia/hopper/cutedsl/tilelang/ptx_mma.py`](nvidia/hopper/cutedsl/tilelang/ptx_mma.py) (dense + sparse, FP16/BF16/INT8/INT4/TF32/FP64/FP8) |
| **lop3 + sub.f16x2 quantized decode** | [`nvidia/hopper/cutedsl/tilelang/quantize.py`](nvidia/hopper/cutedsl/tilelang/quantize.py) (INT4→FP16) |
| **prmt + mul.bf16x2 quantized decode** | [`nvidia/hopper/cutedsl/tilelang/quantize.py`](nvidia/hopper/cutedsl/tilelang/quantize.py) (FP4→BF16 twiddling) |
| **bar.sync $0,$1 (multiple barrier ids)** | [`nvidia/hopper/cutedsl/tilelang/reduce.py`](nvidia/hopper/cutedsl/tilelang/reduce.py) (`bar_sync_ptx`) || **activemask** | [`nvidia/hopper/cutedsl/tilelang/warp.py`](nvidia/hopper/cutedsl/tilelang/warp.py)（`activemask.b32`） |
| **Multi-line PTX + .reg/.pred/labels + global variables** | [`nvidia/hopper/cutedsl/tilelang/grid_sync.py`](nvidia/hopper/cutedsl/tilelang/grid_sync.py)（grid soft sync: `atom.add.release.gpu.s32` + spinning + `st.release.gpu.global.s32`） |
| **fence.sc.gpu + ld.relaxed.gpu (seq_cst load/store)** | [`nvidia/hopper/cutedsl/tilelang/atomic.py`](nvidia/hopper/cutedsl/tilelang/atomic.py)（`AtomicLoad`/`AtomicStore`） |
| **SM120 NVFP4 mma.sync.aligned.kind::mxf4nvf4 end-to-end demo** | [`nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py`](nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py) + [test](nvidia/blackwell-geforce/cutedsl/cutlass/test_sm120_nvfp4_inline_ptx_gemm.py)（pitfall summary at [`docs/ref-docs/nvidia/cutedsl/sm120/sm120-nvfp4-inline-ptx-gemm.md`](../docs/ref-docs/nvidia/cutedsl/sm120/sm120-nvfp4-inline-ptx-gemm.md)） |

Recommended workflow for LLMs taking on inline PTX tasks:
1. First read [`docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md`](../docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md) for pattern overview (constraint strings, StructType, bitcast, `@dsl_user_op`, `has_side_effects` and 12 sections in total).
2. Then pick the closest reference file from the table above as a code style template.
3. For mma.sync-type instructions, refer directly to the `ptx_mma.py` factory; for other PTX, first check whether `cutlass._mlir.dialects.nvvm` already provides an op, and only write inline asm if none exists.

### FlyDSL inline_asm（`llvm.InlineAsmOp` / `llvm.inline_asm` usage reference）

Most FlyDSL instructions have already been wrapped by `flydsl._mlir.dialects.rocdl` and `flydsl.expr.buffer_ops`; only a few hardware control instructions require `llvm.InlineAsmOp` inline assembly. **Usage overview**: [`docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md`](../docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md).

| What you want to do | Reference file |
|---------|---------|
| **Getting started: single asm, no operands**（cache invalidation / writeback） | [`amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py`](amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py)（`buffer_inv sc1`、`buffer_wbl2 sc0 sc1`） |
| **Full: with operands + outputs + constraint strings**（global load/store with cache modifier） | [`amd/cdna/flydsl/FlyDSL/hgemm_splitk.py`](amd/cdna/flydsl/FlyDSL/hgemm_splitk.py)（`global_store_dword $0, $1, off sc0 sc1` constraints `v,v`；`global_load_dword $0, $1, off sc1` constraints `=v,v`） |
| **Multi-line asm + Python f-string compile-time concatenation** | [`amd/rdna4/flydsl/FlyDSL/gemm_fp8fp4_gfx1250.py`](amd/rdna4/flydsl/FlyDSL/gemm_fp8fp4_gfx1250.py)（`s_prefetch_inst_pc_rel` × 10 + `\n.join`） |
| **HW register control (`s_setreg_imm32_b32`)** | [`amd/rdna4/flydsl/FlyDSL/moe_gemm_2stage_wmma_gfx1250.py`](amd/rdna4/flydsl/FlyDSL/moe_gemm_2stage_wmma_gfx1250.py)（gfx1250 wave mode `hwreg(26, 4, 1)`） |
| **RDNA4 split barrier sequence** | [`amd/rdna4/flydsl/FlyDSL/rdna_f16_gemm.py`](amd/rdna4/flydsl/FlyDSL/rdna_f16_gemm.py)（`s_wait_dscnt + s_wait_storecnt + s_barrier_signal/wait`） |
| **Cross-GPU signal protocol（cache control + uncached load/store + L2 writeback full combination）** | [`amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py`](amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py)（multi-GPU AllReduce 1-stage / 2-stage） |

Recommended workflow for LLMs taking on FlyDSL inline_asm tasks:
1. First read [`docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md`](../docs/ref-docs/amd/flydsl/flydsl-inline-asm-patterns.md) for pattern overview (10 sections: API forms, AMDGPU constraint strings, eight usage categories, `has_side_effects` selection, comparison with CuTeDSL, pitfalls).
2. Before writing, verify whether `flydsl._mlir.dialects.rocdl`（such as `s_waitcnt` / `sched_barrier` / `wave_id`）and `flydsl.expr.buffer_ops`（buffer load/store with `cache_modifier`）already have existing ops, and use inline_asm only if none exist.
3. For AMDGPU constraint strings, only use `v`（VGPR）/ `s`（SGPR）；addresses should also go through `v`（do not write NVPTX-style `l`）.
4. `has_side_effects=True` is essentially the default; all inline_asm in FlyDSL projects use `True`.
