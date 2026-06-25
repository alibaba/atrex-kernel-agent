# Mamba2 SSD 7-Role Warp Specialization

## Pattern: Extreme Warp Specialization

**Source**: `cutedsl/flashinfer/ssd_kernel.py`

```python
# Mamba2 SSD kernel divides warps into 7 roles:
# Role 0: TMA load A (SSM parameters)
# Role 1: TMA load B (input transform)
# Role 2: TMA load C (output transform)
# Role 3: TMA load X (input sequence)
# Role 4: GEMM compute (core matrix multiplication)
# Role 5: State update (SSM state recurrence)
# Role 6: TMA store (write back results)
```

**Practical Experience**:
- Only compute-intensive kernels are worth fine-grained warp specialization
- SSM kernels have a large number of independent data streams, making them suitable for multi-role specialization
- Excessive specialization reduces occupancy (requires more warps), and must be verified through benchmarks

---

## Related Documentation

- **Warp Specialization Basics**: [Warp Specialization](warp-specialization.md) — DMA/MMA dual-role specialization fundamentals
- **Linear Attention**: [Chunk Linear Attention Optimization](../../../../../ref-docs/nvidia/gluon/sm90/linear_attention.md) — Similar multi-stage computation patterns
- **Profiling**: [Hopper Profiling Guide](../../../../../ref-docs/nvidia/gluon/sm90/profiling_guide.md) — Analyzing warp occupancy
- **Reference Kernel**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
