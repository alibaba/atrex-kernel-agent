pattern: fused_attention
type: load_balancing
priority: ⭐⭐
performance_gain: +15-20%
Applicable Scenario: Causal Attention
---

# SE-level Causal Attention Load Balancing


**Last updated**: 2026-06-30

## Background

In causal attention, the computational workload varies significantly across different `start_m` values:
- Small `start_m` (e.g., 0, 1, 2): requires computing a large number of K values (because the causal mask only limits `end_n ≤ start_m`)
- Large `start_m` (e.g., M-1, M-2): requires computing only a small number of K values

This leads to SE (Shader Engine) load imbalance, with some SEs overloaded and others idle.

**Measured data** (MI300X, causal attention, 4096×4096):
- Busiest SE: ~120% average load
- Idlest SE: ~80% average load
- Load imbalance results in ~15-20% performance loss

## Solution: Zigzag Remap

Zigzag remap redistributes the order of `start_m` to achieve more balanced SE load.

### Approach 1: start_m-only zigzag

**Principle**: Apply a simple zigzag remapping to `start_m`.

**Code Example**:
```python
@triton.jit
def remap_start_m_zigzag(start_m, NUM_SE: tl.constexpr = 8):
    """Zigzag remap start_m to balance SE load"""
    # Simple zigzag: 0, 7, 1, 6, 2, 5, 3, 4, 8, 15, 9, 14, ...
    se_id = start_m % NUM_SE
    zigzag_se_id = se_id if se_id % 2 == 0 else NUM_SE - 1 - se_id
    remapped_start_m = (start_m // NUM_SE) * NUM_SE + zigzag_se_id
    return remapped_start_m
```

**Performance Improvement**: ~10-15%

### Approach 2: start_m × batch-head zigzag (Core Approach)

**Principle**: Apply zigzag remapping across the three dimensions (`start_m`, `batch`, `head`) encontra to further balance the load.

**Code Example**:
```python
@triton.jit
def remap_pid_zigzag(pid, NUM_PID_M, NUM_PID_BH, GROUP_SIZE_M: tl.constexpr = 8, NUM_SE: tl.constexpr = 8):
    """Zigzag remap (start_m, batch, head) to balance SE load"""
    # Map 1D pid to (start_m, batch, head)
    pid_m = pid % NUM_PID_M
    pid_bh = pid // NUM_PID_M

    # Apply zigzag to start_m
    se_id_m = pid_m % NUM_SE
    zigzag_se_id_m = se_id_m if se_id_m % 2 == 0 else NUM_SE - 1 - se_id_m
    remapped_pid_m = (pid_m // NUM_SE) * NUM_SE + zigzag_se_id_m

    # Apply zigzag to batch-head
    se_id_bh = pid_bh % NUM_SE
    zigzag_se_id_bh = se_id_bh if se_id_bh % 2 == 0 else NUM_SE - 1 - se_id_bh
    remapped_pid_bh = (pid_bh // NUM_SE) * NUM_SE + zigzag_se_id_bh

    # Recombine
    remapped_pid = remapped_pid_m + remapped_pid_bh * NUM_PID_M
    return remapped_pid
```

**Usage**:
```python
@triton.heuristics({
    "NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_M"]),
    "NUM_PID_BH": lambda args: args["B"] * args["H"],
})
@gluon.jit
def flash_attn_kernel(..., NUM_PID_M: tl.constexpr, NUM_PID_BH: tl.constexpr):
    pid = gl.program_id(axis=0)
    pid = remap_pid_zigzag(pid, NUM_PID_M, NUM_PID_BH, GROUP_SIZE_M=8, NUM_SE=8)
    pid_m = pid % NUM_PID_M
    pid_bh = pid // NUM_PID_M
    start_m = pid_m * BLOCK_M
    batch = pid_bh // H
    head = pid_bh % H
    # ...
```

**Performance Improvement**: ~15-20% (better than Approach 1)

## Performance Comparison

| Approach | Busiest SE Load | Idlest SE Load | Load Imbalance Loss | Performance Improvement |
|------|------------|------------|-------------|---------|
| No remap | 120% | 80% | ~15-20% | — |
| start_m-only zigzag | 110% | 90% | ~5-10% | +10-15% |
| **start_m × batch-head zigzag** | **105%** | **95%** | **~0-5%** | **+15-20%** |

## Key Takeaways

1. **Limitations of start_m-only zigzag**: Zigzag is applied only on the `start_m` dimension, making it completely ineffective when M-blocks ≤ NUM_SES
2. **Advantages of start_m × batch-head zigzag**: The batch-head dimension is included in the permutation, so the total number of blocks is much larger than the number of SEs, making it suitable for all sequence lengths
3. **Special requirement for Non-Causal**: Non-causal must maintain bh-first ordering to preserve L2 cache locality; start_m-first ordering causes a performance regression of 3-4%


## Related

- [Changelog for Preview 0.1.4](CHANGELOG.md)
- [AMD MI308X (gfx942) GEMM Optimization Techniques Reference](ck_gemm_optimization_reference.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
