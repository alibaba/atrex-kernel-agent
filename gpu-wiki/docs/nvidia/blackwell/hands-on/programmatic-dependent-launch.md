# Programmatic Dependent Launch


**Last updated**: 2026-06-30

## Pattern: No Host Synchronization Between Kernels

**Source**: `cutedsl/cutlass/`

```python
# Blackwell supports kernel A directly launching kernel B without returning to host
# Scenario: GEMM → Softmax → GEMM Attention pipeline

# After Kernel A (QK GEMM) completes, directly launch Kernel B (Softmax)
cute.cluster_launch(softmax_kernel, grid, block,
                    args=(qk_result, ...),
                    dependency=current_kernel)  # Wait for A to complete before launching B
```

**Practical Experience**:
- Reduces host-device round-trip latency for kernel launches
- Particularly effective for small kernel sequences (where launch overhead is a large proportion)
- Blackwell only

---

## Related

- **CLC**: [CLC (Cluster Launch Control)](cluster-launch-control.md) — Another hardware-level scheduling mechanism
- **Hopper Practice**: [Hopper Optimization Practice](README.md) — Hopper does not have this feature
- **General Triton**: [Triton Optimization Patterns Practice](../../../generic/hands-on/README.md) — Kernel fusion as an alternative
