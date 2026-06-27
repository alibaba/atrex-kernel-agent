# QuACK — Blackwell GeForce (SM120) CuTeDSL Kernels

Dao-AILab QuACK SM120 (Blackwell GeForce) dedicated kernels.

> Source: `$REFERENCE_KERNEL_ROOT/quack/quack/`
> See common code at: `nvidia/hopper/cutedsl/quack/`

## Files

| File | Description |
|------|-------------|
| `gemm_sm120.py` | SM120 GEMM: warp MMA + ldmatrix (no WGMMA) |
| `sm80_utils.py` | SM80 compatibility utility (SM120 path reuse) |
