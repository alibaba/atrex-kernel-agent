# Autotune Configuration and Pruning


**Last updated**: 2026-07-01

## Pattern: Multi-Dimensional Parameter Search + Heuristic Pruning

**Source**: `triton-tutorials/03-matrix-multiplication.py`, `triton-kernels/matmul_details/`

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
 # ... configuration
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

**Key Parameter Dimensions**:
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K`: tile sizes, directly affecting shared memory usage and compute density
- `num_stages`: software pipeline stages, affecting prefetch depth and register pressure
- `num_warps`: warp count, affecting occupancy and parallelism
- `GROUP_SIZE_M`: L2 cache locality grouping (see [Persistent Kernel and Tile Scheduling](persistent-kernel-tile-scheduling.md))

**Pruning Strategy**:

```python
def prune_configs(configs, named_args, **kwargs):
    M, N, K = named_args['M'], named_args['N'], named_args['K']
    pruned = []
    for cfg in configs:
        BLOCK_M, BLOCK_N = cfg.kwargs['BLOCK_M'], cfg.kwargs['BLOCK_N']
        # Prune configurations where tile is larger than problem size
        if BLOCK_M > M or BLOCK_N > N:
            continue
        # Small matrices don't need too many stages
        if M * N < 1024 * 1024 and cfg.num_stages > 3:
            continue
        pruned.append(cfg)
    return pruned
```

**Practical Experience**:
- The `key` parameter determines when to re-search: re-tune when matrix shapes change
- For inference scenarios (fixed shapes), autotune only incurs overhead on first execution
- Recommended configuration count is 8-20; too many will extend search time

---

## Related

- **Same Series**: [Persistent Kernel and Tile Scheduling](persistent-kernel-tile-scheduling.md) — GROUP_SIZE_M swizzle and persistent matmul
- **Same Series**: [Fused Kernel Patterns](fused-kernel-patterns.md) — autotune is often used in conjunction with fused kernels
- **Prerequisite Knowledge**: [GPU Execution Model](../gpu-execution-model.md) — warp and occupancy concepts
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — overview of all patterns
