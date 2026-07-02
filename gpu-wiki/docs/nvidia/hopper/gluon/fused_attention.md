# Fused Attention (Prefill / Paged Attention) Optimization Guide

> **Composite pattern**. For sub-pattern optimization, see `docs/ref-docs/nvidia/gluon/sm90/matmul.md` and `softmax_reduce.md`.

**Last updated**: 2026-06-30

> The following only describes content **specific to this pattern** and **interaction constraints** between sub-patterns.
> For the general ISA optimization checklist, see `common_optimizations.md`.

> **Status**: Skeleton document, to be supplemented with future optimization case studies.

---

## Pattern Characteristics

| Feature | Description |
|------|------|
| **Core Computation** | `Attention(Q, K, V) = softmax(Q × K^T / √d) × V` |
| **Sub-pattern Composition** | matmul(Q, K^T) → softmax → matmul(score, V) |
| **SASS Signature** | Contains `WGMMA` instructions (two groups); contains softmax-related `SHFL`/reduction operations |
| **Special Variants** | Causal mask, Paged KV cache, GQA/MQA |

**Identification Criteria**:
- Has Q×K^T → softmax → ×V three stages
- May have causal mask
- May have paged KV cache

---

## Bottleneck Characteristics

The bottleneck type of Fused Attention depends on sequence length and head dimension:
- **Long sequence + large head dim**: Tends toward Compute Bound (large computational cost of two matmuls)
- **Short sequence + small head dim**: Tends toward Memory Bound (KV cache memory access dominates)

---

## Sub-pattern Interaction Constraints (Specific to This Pattern)

> The following constraints are **not covered** in matmul.md and softmax_reduce.md, as they only appear when multiple sub-patterns are combined.

### Constraint 1: Two Matmuls Share Shared Memory

The Q×K^T and score×V matmul stages need to share limited shared memory:
- Q tiles may be reused across both matmuls (to avoid redundant loading)
- K and V shared memory may alternate in the same region
- Shared memory allocation strategy requires global planning; individual matmuls cannot be optimized independently

### Constraint 2: Register Pressure from Softmax Intermediate Results

The max and sum values from softmax must remain in registers between the two matmuls, increasing overall register pressure.

---

## Optimization Strategies

Source: `common_optimizations.md` Appendix A "Attention" row.

| Priority | Optimization | Description |
|--------|--------|------|
| ⭐⭐⭐ | §3.0 Coalesced memory access pre-check | Required |
| ⭐⭐⭐ | §3.1 Coalesced memory access + wide load | Required |
| ⭐⭐ | §3.3 Eliminate scratch/spill | High register pressure from two matmuls + softmax |
| ⭐⭐ | §3.4 async_copy pipeline | KV cache prefetch |
| ⭐⭐ | §3.5 wgmma correctness | Fence/wait ordering for two wgmma groups |
| ⭐ | §3.2 Bank conflict | Fine-tuning |

---

## Stop Conditions

Use the general stop conditions from optimization-guide.md §1.8.

---

## To Be Supplemented

- [ ] FlashAttention-style tiling strategy
- [ ] Tile skip optimization for causal mask
- [ ] Discontiguous memory access handling for paged KV cache
- [ ] K/V broadcast optimization for GQA/MQA
- [ ] Numerical stability vs. performance trade-offs of online softmax
- [ ] Measured case study data
- [ ] Lessons learned from pitfalls


## Related

- [Hopper (sm_90) General ISA Optimization Checklist](common_optimizations.md)
- [Hopper (sm_90) SASS Instruction Patterns and Optimization Reference](isa_patterns.md)
- [Chunk Linear Attention / Recurrent State Update Optimization Guide](linear_attention.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [Gluon Kernel Performance Optimization Guide (NVIDIA Hopper)](optimization-guide.md)
- [Fused Attention Optimization Guide](../../../amd/gluon/gfx950/fused_attention.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
