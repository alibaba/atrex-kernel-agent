# mbarrier Software Pipeline


**Last updated**: 2026-06-30

## Pattern: PipelineTmaAsync Multi-Stage Pipeline

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm90.py`

```python
from cutlass.cute.nvgpu import PipelineTmaAsync

# createpipeline( 3-4 )
pipeline = PipelineTmaAsync(num_stages=4)

# Producer（TMA warp）：
for k_tile in range(num_k_tiles):
 pipeline.producer_acquire(stage) # wait buffer
    cute.copy(tma_a, src_a[k_tile], smem_a[stage])
    cute.copy(tma_b, src_b[k_tile], smem_b[stage])
 pipeline.producer_commit(stage) # data
    stage = (stage + 1) % num_stages

# Consumer（MMA warp）：
for k_tile in range(num_k_tiles):
 pipeline.consumer_wait(stage) # waitdata
    acc = cute.gemm(tiled_mma, smem_a[stage], smem_b[stage], acc)
 pipeline.consumer_release(stage) # buffer
    stage = (stage + 1) % num_stages
```

**Practical Experience**:
- The larger `num_stages` is, the deeper the pipeline, and the better the compute-memory access overlap
- However, each additional stage consumes an extra portion of shared memory (tile_size × 2 × num_stages)
- Hopper has 228 KB shared memory/SM, typically supporting 3-4 stages
- mbarrier is a hardware synchronization primitive, lighter than `__syncthreads()`

## Pipeline in Gluon

```python
# Gluon use fence + commit_group + wait_group
for k in range(0, K, BLOCK_K):
    # Prefetch next tile
    tl.async_copy_global_to_shared(next_a_ptr, smem_a_next)
    tl.async_copy_global_to_shared(next_b_ptr, smem_b_next)
    tl.commit_group()

    # Compute current tile
    acc = tl.dot(smem_a, smem_b, acc)

    # Wait for next tile
    tl.wait_group(0)
    smem_a, smem_a_next = smem_a_next, smem_a
```

---

## Related

- **CuTeDSL Pipeline**: [CuTeDSL Software Pipeline and Synchronization Patterns](../../common/cutedsl/cutedsl-pipeline-patterns.md) — producer/consumer state machine
- **PTX Synchronization**: [PTX Synchronization and Async Operations](../../common/ptx/nvidia-ptx-sync-and-async.md) — mbarrier details
- **Hardware Specifications**: [Hopper Hardware Specifications Table](../../common/hardware-specs/hopper.md) — shared memory capacity
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source codes
