pattern: matmul
type: config
applicable_scenarios:
  - 256×256×64 GEMM
  - MI308X architecture
  - fp16 data type
---

# Stopping Conditions

Use the general stopping conditions from optimization-guide.md §1.8, with the full checklist from §3.0-3.6 of `common_optimizations.md`.

# Final Best Configuration Template for 256×256×64 GEMM

```python
# 256×256×64 GEMM optimal config (fp16, MI308X)
blocked_a: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[8, 8],
    warps_per_cta=[8, 1], order=[1, 0]
)
blocked_b: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[4, 8], threads_per_warp=[2, 32],
    warps_per_cta=[8, 1], order=[1, 0]
)
mma: gl.constexpr = gl.amd.AMDMFMALayout(
    version=3, instr_shape=[16, 16, 16],       # NOT [32,32,8]
    warps_per_cta=[2, 4], transposed=True
)
shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
shared_b_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])  # NOT [0,1]
dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)
SUBK: tl.constexpr = 16

# No convert_layout / in_thread_transpose
# num_warps=8, num_stages=1
# WPS 4-subslice Round 3 pattern (ref: aiter/matmul_gluon_wps.py)

# Main loop
for k in range(0, K, BLOCK_K):
    # Use WPS (Warp Pipeline Stage) to implement 4-subslice pipeline
    # Ref: docs/amd/cdna3/pipeline.md
    pass
```

## Key Takeaways

1. `instr_shape=[16,16,16]` — Align K dimension to SUBK=16 to reduce the number of MFMA operations
2. `shared_b order=[1,0]` — Do not perform in_thread_transpose to avoid ds_bpermute
3. `blocked_b spt=[4,8]` — 8-element N dimension ensures dwordx4 coalesced memory access
4. 4 subslice WPS — Reduce the number of pipeline stages to lower VGPR pressure
5. 8 warps — A necessary condition for VGPR-limited tiles

## Measured Performance

**240.74 TFLOPS** (fp16, MI308X, 8192³)

**Comparison**: Triton 178.18 TFLOPS, **+35%**
