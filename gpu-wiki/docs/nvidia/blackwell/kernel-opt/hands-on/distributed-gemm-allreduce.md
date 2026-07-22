# Distributed GEMM+AllReduce (Two-Shot)

## Pattern: Overlapping GEMM with Communication

**Source**: `cutedsl/cutlass/`

```python
# Two-shot AllReduce: split GEMM output into chunks, compute and communicate concurrently
# Shot 1: each GPU computes its own partial GEMM the result is sent to the peer
# Shot 2: receive the peer result and perform reduction

# Flow:
# GPU 0: GEMM(A0, B) → partial_C0 ──nvshmem_put──→ GPU 1
# GPU 1: GEMM(A1, B) → partial_C1 ──nvshmem_put──→ GPU 0
# GPU 0: C = partial_C0 + partial_C1_from_gpu1
# GPU 1: C = partial_C1 + partial_C0_from_gpu0
```

**Practical Experience**:
- Two-shot has lower latency than NCCL AllReduce (eliminates one synchronization step)
- Uses nvshmemurn to implement point-to-point communication, bypassing NCCL protocol overhead
- Suitable for Tensor Parallel (TP) scenarios, where each GPU holds a portion of the weights
- Requires high-bandwidth interconnect between GPUs (NVLink/NVSwitch)

---

## Related Documentation

- **2CTA Cooperation**: [2CTA Cooperation](two-cta-cooperation.md) — Another form of inter-SM collaboration
- **Hopper Hands-on**: [Hopper Optimization Hands-on](README.md) — SM90 distributed comparison
- **CuTeDSL Basics**: [CuTeDSL Programming Model](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md) — Python DSL compilation pipeline
