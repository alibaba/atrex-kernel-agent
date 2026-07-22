# 2CTA Cooperation

## Pattern: Two CTAs Cooperating on One Tile

**Source**: `cutedsl/cutlass/dense_gemm_2cta_sm100.py`, `gluon/triton/10-multi-cta-matmul.py`

```python
# 2CTA: Two CTAs load A and B respectively, share results for computation
# CTA 0 is responsible for loading A tile
# CTA 1 is responsible for loading B tile
# Both perform GEMM computation

cluster_shape = (2, 1, 1)  # 2 CTAs form a cluster

if cta_id_in_cluster == 0:
    # Load A tile to shared memory
    cute.copy(tma_a, src_a, smem_a)
    # B tile is loaded by CTA 1, accessed via cluster shared memory
else:
    # Load B tile
    cute.copy(tma_b, src_b, smem_b)

# Both CTAs execute GEMM (using shared memory from the other SM)
acc = cute.gemm(mma, smem_a_remote, smem_b_local, acc)
```

**Practical Experience**:
- 2CTA doubles the TMA load bandwidth (TMA units from two SMs work in parallel)
- Suitable for memory-bound GEMM (small K, large M/N)
- Cross-SM shared memory access incurs additional latency; may not benefit compute-bound scenarios
- Requires cluster support (introduced in Hopper, enhanced in Blackwell)

---

## Related Documentation

- **CLC**: [CLC (Cluster Launch Control)](cluster-launch-control.md) — Dynamic tile scheduling can be combined with 2CTA
- **Hopper Practical**: [Hopper Optimization Practices](README.md) — Hopper cluster comparison
- **CuTeDSL Basics**: [CuTeDSL Programming Model](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md) — Python DSL compilation pipeline
