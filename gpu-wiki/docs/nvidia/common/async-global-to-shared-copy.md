# Async Global-to-Shared Memory Copy (CC 8.0+)

`memcpy_async` enables data to go directly from global memory to shared memory, **bypassing the register file**.


**Last updated**: 2026-06-30

## Advantages

- Reduces register pressure (the traditional approach requires loading to registers first, then storing to shared memory)
- Copy and computation can be overlapped via pipelining
- Executed by the hardware DMA engine, freeing up compute units

## Usage Pattern

```c
// Pipeline double-buffering mode
__shared__ float smem[2][TILE_SIZE];
int stage = 0;

// Preload the first tile
__pipeline_memcpy_async(&smem[0][tid], &gmem[0 * TILE_SIZE + tid], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);
__syncthreads();

for (int i = 1; i < numTiles; i++) {
    // Asynchronously load the next tile
    __pipeline_memcpy_async(&smem[1-stage][tid], &gmem[i * TILE_SIZE + tid], sizeof(float));
    __pipeline_commit();

    // Compute on the current tile simultaneously
    compute(smem[stage]);

    // Wait for the load to complete
    __pipeline_wait_prior(0);
    __syncthreads();
    stage = 1 - stage;
}
```

## Related

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [PTX Synchronization and Async Operations](ptx/nvidia-ptx-sync-and-async.md) — Details on async copy instructions at the PTX level
- [CuTeDSL Software Pipelining and Synchronization Patterns](cutedsl/cutedsl-pipeline-patterns.md) — Pipeline abstractions in the upper-level DSL
- [GPU Memory Hierarchy and Optimization](../../generic/gpu-memory-hierarchy.md) — General principles of shared memory optimization
