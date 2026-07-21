# Grouped GEMM (DeepGEMM Mode)

## Pattern: Multi-Problem GEMM Batching

**Source**: `triton/DeepGEMM/`

```python
# Scenario: Small GEMMs for multiple experts in MoE need to be executed in batch
@triton.jit
def grouped_gemm_kernel(
    a_ptrs, b_ptrs, c_ptrs,    # pointer arrays for each problem
    group_sizes,                 # M size of each group
    NUM_GROUPS: tl.constexpr,
    ...
):
    pid = tl.program_id(0)

    # Find which group the current pid belongs to
    group_id = 0
    accumulated = 0
    for g in range(NUM_GROUPS):
        group_size = tl.load(group_sizes + g)
        tiles_in_group = cdiv(group_size, BLOCK_M) * cdiv(N, BLOCK_N)
        if pid < accumulated + tiles_in_group:
            group_id = g
            local_pid = pid - accumulated
            break
        accumulated += tiles_in_group

    # Use local_pid to compute tile position, execute standard GEMM
    ...
```

**Practical Experience**:
- Suitable for MoE scenarios with many experts and small per-expert matrices
- Compared to `torch.bmm`, reduces kernel launch overhead
- M sizes can vary across groups (supports irregular shapes)

---

## Related Documents

- **Same Series**: [Persistent Kernel and Tile Scheduling](persistent-kernel-tile-scheduling.md) — Tile scheduling strategies
- **Same Series**: [Autotune Configuration and Pruning](autotune-config-pruning.md) — Autotune configuration for grouped GEMM
- **Prerequisites**: [GPU Execution Model](../../ref-docs/gpu-execution-model.md) — CTA scheduling and kernel launch overhead
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — Overview of all patterns
