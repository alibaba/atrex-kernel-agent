# WGMMA (Warpgroup Matrix Multiply-Accumulate)


**Last updated**: 2026-06-30

## Pattern: 128-Thread Warpgroup Cooperative MMA

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm90.py`, `cutedsl/cutlass/fmha.py`

```python
# WGMMA: 4 warps (128 threads) cooperatively execute matrix multiplication
# The A operand is read from shared memory (not register), reducing register pressure
tiled_mma = cute.make_tiled_mma(
    cute.SM90_16x8x16_F32BF16BF16F32_SS,  # SS = both A and B read from shared memory
    # or cute.SM90_16x8x16_F32BF16BF16F32_RS  # RS = A from register, B from shared
)
```

**WGMMA in Gluon**:

```python
# Gluon automatically maps to wgmma via tl.dot
# but we need to ensure that BLOCK_M is a multiple of 128 (warpgroup requirement)
c = tl.dot(a, b)  # compiler automatically selects wgmma instruction
```

**Practical Experience**:
- WGMMA's A operand is read directly from shared memory, **no need to load it into registers first**
- This is fundamentally different from Ampere's `mma.sync` (which requires first `ldmatrix` into registers)
- BLOCK_M must be a multiple of 64 (128/256 recommended), because warpgroup = 4 warps = 128 threads
- SS mode (both A and B from shared memory) generally outperforms RS mode

---

## Related

- **PTX MMA Instruction Evolution**: [PTX MMA Instruction Evolution](../../common/ptx/nvidia-ptx-mma-instructions.md) — wmma → mma.sync → wgmma
- **CuTeDSL SM90**: [CuTeDSL SM90 Special Features](../cutedsl/hopper-cutedsl-sm90.md) — warpgroup MMA details
- **Hardware Specifications**: [Hopper Hardware Specs](../../common/hardware-specs/hopper.md) — H100/H20 peak TFLOPS
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
