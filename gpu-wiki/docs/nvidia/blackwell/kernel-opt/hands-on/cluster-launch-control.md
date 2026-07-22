# CLC (Cluster Launch Control)

## Mode: Dynamic Tile Scheduling

**Source**: `gluon/triton/08b-persistent-matmul-clc.py`, `cutedsl/cutlass/`

```python
# CLC: Hardware-level dynamic work distribution
# Replaces the traditional "pid → tile" static mapping

# Gluon CLC usage:
from triton.experimental.gluon.nvidia.blackwell import clc

@gluon.jit
def persistent_matmul_clc(a, b, c, ...):
    # CLC dynamically retrieves the next pending tile
    tile_id = clc.get_work_item()

    while tile_id is not None:
        m_tile, n_tile = decode_tile_id(tile_id)
        # Execute GEMM
        compute_gemm(a, b, c, m_tile, n_tile)
        # Retrieve the next tile
        tile_id = clc.get_work_item()
```

## CuTeDSL CLC

```python
# CuTeDSL wraps CLC via PersistentTileSchedulerSm100
scheduler = PersistentTileSchedulerSm100(
    cluster_shape=cluster_shape,
    problem_shape=problem_shape,
)
work_tile = scheduler.get_current_work()
while work_tile.is_valid():
    # Process tile
    compute(work_tile)
    work_tile = scheduler.advance_to_next_work()
```

**Practical Experience**:
- CLC is better suited than static scheduling for irregular workloads (e.g., causal attention's triangular tile distribution)
- CLC implements work-stealing in hardware, which is faster than software atomic operations
- For square matrix GEMM (uniform workload), CLC performance is close to that of static scheduling

---

## Related Documents

- **2CTA Cooperation**: [2CTA Cooperation](two-cta-cooperation.md) — CLC can be used in conjunction with 2CTA
- **Generic Triton**: [Triton Optimization Patterns in Practice](../../../../generic/kernel-opt/hands-on/README.md) — persistent kernel pattern
- **CuTeDSL Basics**: [CuTeDSL Programming Model](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md) — Python DSL compilation pipeline
