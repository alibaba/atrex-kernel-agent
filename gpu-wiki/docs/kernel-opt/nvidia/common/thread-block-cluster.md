# Thread Block Cluster (CC 9.0+)

A new hierarchy introduced by Hopper: multiple blocks form a cluster, sharing an L2 cache region and allowing direct access to each other's shared memory.

## Distributed Shared Memory

```c
// Blocks within a cluster can directly access each other's shared memory
// Requires the cluster launch API
cudaLaunchConfig_t config;
config.gridDim = {numBlocks, 1, 1};
config.blockDim = {blockSize, 1, 1};

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {clusterSize, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, args...);
```

## TMA (Tensor Memory Accelerator)

A hardware unit that automatically handles multi-dimensional tensor transfers from global memory to shared memory:
- Automatically handles 2D/3D data layout transformations
- Supports multicast to multiple blocks within a cluster
- Offloads instruction overhead from address calculations

## Related Documentation

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [CuTeDSL SM90 (Hopper) Specialized Features](../../../ref-docs/nvidia/cutedsl/sm90/hopper-cutedsl-sm90.md) — WGMMA and TMA usage at the DSL layer
- [CuTeDSL Architecture Primitives](../../../ref-docs/nvidia/cutedsl/nvidia-cutedsl-arch-primitives.md) — cluster indexing and warp intrinsics
- [Hopper Kernel Optimization Hands-On](sm90/hands-on/README.md) — TMA + WGMMA hands-on practice
