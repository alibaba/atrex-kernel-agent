# QuACK — Blackwell GeForce (SM120) CuTeDSL Kernels

Dao-AILab QuACK SM120 (Blackwell GeForce) dedicated kernels.

> Source: `$REFERENCE_KERNEL_ROOT/quack/quack/`
> See common code at: `nvidia/hopper/cutedsl/quack/`

## Files

| File | Description |
|------|-------------|
| `gemm_sm120.py` | SM120 GEMM: warp MMA + ldmatrix (no WGMMA) |
| `sm80_utils.py` | Upstream-named fragment-partition helper imported by `gemm_sm120.py`; it is a support module, not a standalone SM80 or SM120 kernel. |

The helper stays beside the extracted SM120 consumer to preserve the upstream
`from quack import sm80_utils` dependency. Its filename is not an architecture
classification.
