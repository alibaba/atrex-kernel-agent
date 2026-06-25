# Warp Specialization

## Pattern: DMA Warp + MMA Warp Division of Labor

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm90.py`, `gluon/triton/07-persistence.py`

```python
# CuTeDSL: Explicit warp role assignment
if warp_group_id == 0:
    # DMA warp group: responsible for TMA loading
    for k_tile in range(num_k_tiles):
        pipeline.producer_acquire(stage)
        cute.copy(tma_a, src[k_tile], smem[stage])
        pipeline.producer_commit(stage)
elif warp_group_id == 1:
    # MMA warp group: responsible for matrix computation
    for k_tile in range(num_k_tiles):
        pipeline.consumer_wait(stage)
        acc = cute.gemm(tiled_mma, smem_a[stage], smem_b[stage], acc)
        pipeline.consumer_release(stage)
```

## Pattern: setmaxregister Register Reallocation

```python
# DMA warp doesn't need many registers → give them to MMA warp
if warp_group_id == 0:  # DMA warp
    cute.arch.setmaxregister(32)   # limit to 32 registers
else:  # MMA warp
    cute.arch.setmaxregister(256)  # use more registers
```

**Practical Experience**:
- Hopper allows dynamic adjustment of the register limit per warp at runtime
- DMA warps only perform address computation and TMA instructions; 32 registers are sufficient
- MMA warps need to store accumulators and intermediate results; 256 registers reduce spilling
- This is a Hopper-exclusive feature; Ampere does not support it

---

## Related Documents

- **CuTeDSL Architecture Primitives**: [CuTeDSL Architecture Primitives](../../../../../ref-docs/nvidia/cutedsl/nvidia-cutedsl-arch-primitives.md) — warp/lane intrinsics
- **CuTeDSL SM90**: [CuTeDSL SM90-Specific Features](../../../../../ref-docs/nvidia/cutedsl/sm90/hopper-cutedsl-sm90.md) — warpgroup collaboration
- **Profiling**: [Hopper Profiling Guide](../../../../../ref-docs/nvidia/gluon/sm90/profiling_guide.md) — ncu analysis of warp utilization
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
