# tilelang/contrib/cutedsl

Source: `tilelang/tilelang/contrib/cutedsl/` (TileLang's CuTeDSL backend).

This is a utility library built on top of `cutlass.cute` and extensively using inline PTX via `cutlass._mlir.dialects.llvm.inline_asm`. It covers multiple generations of architectures — SM80 / SM90 / SM100 — and implements the CuTeDSL equivalent of the CUDA template (`tl_templates/cuda/*.h`).

For a complete summary of inline PTX patterns, see:
[`docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md`](../../../../../docs/ref-docs/nvidia/cutedsl/cutedsl-inline-ptx-patterns.md)

## File Inventory

| File | Main Content | Key Inline PTX Instructions |
|------|-------------|----------------------------|
| `__init__.py` | Exports + cutlass.cute re-exports | — |
| `utils.py` | bitcast, make_tensor helpers, `pack_half2` | `mov.b32 {r,r}` |
| `cpasync.py` | cp.async / TMA load/store/reduce / mbarrier | `mbarrier.try_wait.parity.shared::cta.b64` spin |
| `ldsm.py` | `ldmatrix` / `stmatrix` x1/x2/x4 + transpose variants | NVVM dialect (no PTX written) |
| `ptx_mma.py` | `mma.sync.aligned` factory for all dtypes (FP16/BF16/INT8/INT4/TF32/FP64/FP8) + sparse variants | `mma.sync.aligned.{shape}.{layout}.{dtype}` |
| `atomic.py` | AtomicAdd/Max/Min/Load/Store + AtomicAddx2/x4 | `atom.add.noftz.f16/v2.f16/v2.bf16`, `atom.global.add.v4.f32`, `atom.cas.b32` spin-based float max/min, `fence.sc.gpu` |
| `reduce.py` | min/max/SumOp/MaxOp/MinOp, CumSum1D/2D, AllReduce, NamedBarrier | `bar.sync $0, $1` (including multiple barrier IDs) |
| `warp.py` | `__activemask`, `__shfl_*_sync`, warp_reduce_{sum,max,min,bitand,bitor} | `activemask.b32` |
| `math.py` | exp/log/sin/cos/sqrt/rsqrt re-exports + `tanh` (fastmath) | `tanh.approx.f32` |
| `ieee_math.py` | Explicit IEEE-754 rounding modes: fadd/fsub/fmul/fma/fsqrt/fdiv | `add/sub/mul/fma/rcp/sqrt/div.{rn,rz,rm,rp}.{f32,f64}` |
| `quantize.py` | INT4→FP16 decoding, FP4→BF16 twiddling decoding | `lop3.b32 + sub.f16x2`, `prmt.b32 + mul.bf16x2` |
| `grid_sync.py` | Software grid barrier (requires cooperative launch) | Multi-line PTX: `ld.acquire.gpu.global.s32` spin + `st.release.gpu.global.s32` |
| `threadblock_swizzle.py` | `dim3` / `BlockIdx` / `GridDim` / 2D row-column rasterization | — |
| `gemm_v1.py` | SM80/SM90 generic GEMM dispatch (gemm_ss/rs/sr/rr) | Calls `ptx_mma`, `ldsm`, etc. |
| `gemm_v2.py` | SM90 WGMMA descriptor + warpgroup GEMM | NVVM dialect WGMMA |
| `gemm_tcgen05.py` | SM100 tcgen05 / TMEM GEMM | NVVM dialect tcgen05 |

## Learning Path

1. Start with `utils.py` and `math.py`: the shortest inline asm examples.
2. `ieee_math.py`: see how f-string templates generate multiple variants (rn/rz/rm/rp).
3. `atomic.py`: study fp16 bitcast, vectorized atomic, CAS spin patterns.
4. `ptx_mma.py`: see how the factory pattern handles dense + sparse + multi-dtype simultaneously.
5. `grid_sync.py`: study multi-line PTX, temporary `.reg`/`.pred`/labels, and `llvm.mlir.global` coordination.
6. `gemm_v1/v2/tcgen05.py`: see how a full GEMM assembles these primitives together.

## Relationship with the SM120 NVFP4 Demo

 uses the same `llvm.inline_asm` approach Xin for `mma.sync.aligned.kind::mxf4nvf4...`, with template strings, constraint rules, `StructType` output decomposition, and `u32` packed FP4 register — all consistent. You can use `ptx_mma.py` as a template Spring new block-scaled MMA factory.
