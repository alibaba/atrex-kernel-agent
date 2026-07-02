# Persistent Kernel and Tile Scheduling


**Last updated**: 2026-07-01

## Pattern: GROUP_SIZE_M Swizzle to Improve L2 Hit Rate

**Source**: `triton-tutorials/03-matrix-multiplication.py`

```python
# Standard tile indexing (row-major) — low L2 hit rate
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Swizzle tile indexing — adjacent CTAs share input rows/columns
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

**Principle**: Group tiles of `GROUP_SIZE_M` rows together for execution, so that adjacent CTAs share the same row block of the A matrix, improving L2 cache reuse.

**Practical Experience**:
- `GROUP_SIZE_M=8` is a good starting point in most scenarios
- For tall-and-narrow matrices (M >> N), increasing GROUP_SIZE_M yields more noticeable improvements
- For square matrices, GROUP_SIZE_M=4-8 is usually sufficient

## Pattern: Persistent Matmul (Fixed CTA Count)

**Source**: `triton-kernels/matmul_details/persistent_matmul_kernel.py`

```python
# Launch a fixed number of CTAs, each CTA processes multiple tiles
num_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)
for tile_id in range(pid, num_tiles, NUM_SMS):
    # Compute (pid_m, pid_n) for the current tile
    pid_m, pid_n = swizzle_tile(tile_id, ...)
    # Execute GEMM computation
    ...
```

**Advantages**: Reduces CTA launch overhead and enables better cross-tile data reuse.

---

## Related

- **Same Series**: [Autotune Configuration and Pruning](autotune-config-pruning.md) — GROUP_SIZE_M as an autotune parameter
- **Same Series**: [Grouped GEMM](grouped-gemm-deepgemm.md) — Tile scheduling for batched GEMM
- **Prerequisites**: [GPU Memory Hierarchy](../gpu-memory-hierarchy.md) — L2 cache locality principle
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — Overview of all patterns
