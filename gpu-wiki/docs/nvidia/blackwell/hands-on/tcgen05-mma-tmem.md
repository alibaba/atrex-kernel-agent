# tcgen05 MMA and TMEM


**Last updated**: 2026-07-01

## Pattern: Tensor Memory (TMEM) Accumulator

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm100.py`, `cutedsl/cutlass/tutorial_gemm/`

```python
# Blackwell introduces Tensor Memory (TMEM): High-bandwidth memory dedicated for MMA accumulators
# No longer uses register file to store accumulators, freeing up a large number of VGPRs

tiled_mma = cute.make_tiled_mma(
    cute.SM100_MMA_F32BF16BF16F32_SS_TN,  # tcgen05 MMA
    # Accumulator is automatically allocated in TMEM
)

# Accumulator is in TMEM, does not occupy registers
acc = cute.make_fragment_like(tiled_mma.accumulator())
cute.clear(acc)  # TMEM clear

for k_tile in range(num_k_tiles):
    acc = cute.gemm(tiled_mma, smem_a[stage], smem_b[stage], acc)
    # acc remains in TMEM, register pressure is zero
```

**Practical Experience**:
- TMEM capacity is similar to shared memory (~256 KB/SM), but with higher bandwidth
- After migrating the accumulator from register → TMEM, VGPR usage can be reduced by 50%+
- TMEM can only be read/written by tcgen05 MMA instructions and cannot be directly loaded/stored
- The epilogue needs to first copy TMEM data to registers or shared memory

## Pattern: Double-Buffered TMEM

```python
# Double-buffered TMEM: One set computes, the other is being read by epilogue
tmem_acc_0 = cute.make_fragment_like(tiled_mma.accumulator())
tmem_acc_1 = cute.make_fragment_like(tiled_mma.accumulator())

# Tile 0: Compute to tmem_acc_0
for k in range(K_tiles):
    tmem_acc_0 = cute.gemm(mma, smem_a, smem_b, tmem_acc_0)

# Tile 1: Compute to tmem_acc_1, while epilogue processes tmem_acc_0
pipeline_tmem.producer_commit(tmem_acc_0)  # Mark acc_0 as readable
for k in range(K_tiles):
    tmem_acc_1 = cute.gemm(mma, smem_a, smem_b, tmem_acc_1)
    # Epilogue warp reads tmem_acc_0 in parallel
```

**Practical Experience**:
- Double-buffered TMEM allows GEMM computation and epilogue to fully overlap
- Requires enough TMEM capacity to hold 2 sets of accumulators
- Very effective for large tiles: GEMM takes a long time, allowing the epilogue to be completely hidden

---

## Related

- **Three-Role Warp Specialization**: [Three-Role Warp Specialization](three-role-warp-specialization.md) — TMEM enables the epilogue warp to independently read accumulators
- **Pipeline Pattern Comparison**: [Pipeline Pattern Comparison](pipeline-comparison.md) — TMEM-related pipeline types
- **CuTeDSL Basics**: [CuTeDSL Programming Model](../../common/cutedsl/cutedsl-programming-model.md) — Python DSL compilation pipeline
- **PTX MMA Evolution**: [PTX MMA Instruction Evolution](../../common/ptx/nvidia-ptx-mma-instructions.md) — wmma → mma.sync → wgmma → tcgen05
