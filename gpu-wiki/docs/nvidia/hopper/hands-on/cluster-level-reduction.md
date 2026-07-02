# Cluster-Level Reduction


**Last updated**: 2026-06-30

## Pattern: Intra-Cluster Reduction Across SMs

**Source**: `cutedsl/cutlass/`

```python
# Hopper introduces Thread Block Cluster: multiple CTAs can collaborate
# Supports cross-SM shared memory access and reduction

# Scenario: Split-K GEMM, multiple CTAs compute different K slices of the same output tile
# Finally, the partial results from each slice need to be reduced
cluster_shape = (2, 1, 1)  # 2 CTAs form a cluster

# CTA 0 and CTA 1 each compute half of K
# After synchronizing via cluster barrier, CTA 0 reads CTA 1's shared memoryyms for accumulation
```

**Practical Experience**:
- Cluster size is typically 2-4; larger sizes reduce SM utilization
- Primarily used in Split-K scenarios (where K is large but M/N is small)
- Cross-SM shared memory access is faster than global memory but slower than local shared memory

---

## Related

- **GPU Execution Model**: [GPU Execution Model and Thread Optimization](../../../generic/gpu-execution-model.md) — thread/warp/block/grid hierarchy
- **CuTeDSL SM90**: [CuTeDSL SM90 Specialized Features](../cutedsl/hopper-cutedsl-sm90.md) — cluster support
- **GEMM Optimization**: [Hopper GEMM Optimization](../gluon/matmul.md) — Split-K scenarios
- **Hardware Specifications**: [Hopper Hardware Specifications](../../common/hardware-specs/hopper.md) — SM count and shared memory
- **Reference Kernel**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
