# QuACK — Blackwell (SM100) CuTeDSL Kernels

Dao-AILab QuACK SM100 (Blackwell) dedicated kernels.

> Source: `$REFERENCE_KERNEL_ROOT/quack/quack/`
> For common code, see: `nvidia/hopper/cutedsl/quack/`

## Files

| File | Description |
|------|-------------|
| `gemm_sm100.py` | SM100 GEMM: UMMA/tcgen05 + CLC dynamic persistent + setmaxregister + TMEM |
| `sm100_utils.py` | SM100 architecture-specific utility functions |
