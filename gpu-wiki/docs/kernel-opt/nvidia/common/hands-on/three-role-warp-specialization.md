# Three-Role Warp Specialization

## Pattern: TMA Warp + MMA Warp + Epilogue Warp

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm100.py`

```python
# Blackwell extends Hopper's 2-role to 3-role:
if warp_group_id == 0:
    # TMA Warp Group: data movement
    for tile in tiles:
        pipeline_load.producer_acquire(stage)
        cute.copy(tma_a, src_a[tile], smem_a[stage])
        cute.copy(tma_b, src_b[tile], smem_b[stage])
        pipeline_load.producer_commit(stage)

elif warp_group_id == 1:
    # MMA Warp Group: core computation
    for tile in tiles:
        pipeline_load.consumer_wait(stage)
        acc = cute.gemm(mma, smem_a[stage], smem_b[stage], acc)
        pipeline_load.consumer_release(stage)

        # Notify epilogue after computation is complete
        pipeline_store.producer_commit(acc)

elif warp_group_id == 2:
    # Epilogue Warp Group: result processing and write-back
    pipeline_store.consumer_wait()
    # Read accumulator from TMEM → type conversion → TMA store
    cute.copy(tma_c, acc_smem, dst_c)
```

**Practical Experience**:
- The three-role division enables complete overlap of the load, compute, and store phases
- Hopper has only 2 roles, with epilogue executed serially alongside MMA
- Blackwell's TMEM allows the epilogue warp to independently read the accumulator without blocking MMA
- `setmaxregister` allocation: TMA=32, MMA=256, Epilogue=128

---

## Related Documentation

- **tcgen05 MMA and TMEM**: [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) — The TMEM accumulator is the foundation of the three-role pattern
- **Epilogue Fusion**: [Epilogue Fusion](epilogue-fusion.md) — Fused computation in the epilogue warp
- **Hopper in Practice**: [Hopper Optimization in Practice](README.md) — Hopper 2-role comparison
- **CuTeDSL Pipeline**: [CuTeDSL Pipeline Patterns](../../../../ref-docs/nvidia/cutedsl/cutedsl-pipeline-patterns.md) — producer/consumer patterns
