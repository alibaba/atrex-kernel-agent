# Fused Attention Optimization Guide

> This document focuses on specialized optimization for the **standard Flash Attention** (QK → softmax → PV) kernel.

**Last updated**: 2026-06-30

> For MLA Decode, refer to `docs/ref-docs/amd/gluon/gfx950/mla_decode.md` (with mxfp4 QK and online softmax special handling).
> For a general ISA optimization checklist, see `common_optimizations.md`.

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | Q×K^T → softmax → ×V |
| **Data Format** | Q/K/V/O: bf16/fp16 |
| **Main Loop Structure** | Iterate along KV seq len, loading K/V tiles each step, performing MFMA QK + softmax + MFMA PV |
| **Key Bottleneck** | V bandwidth + QK matmul bandwidth |
| **ISA Flags** | `v_mfma_bf16` (QK & PV), `buffer_load_dwordx4`, `v_cndmask_b32` (to be eliminated) |

**Identification Criteria**:
- Two mfma operations present (QK and PV)
- Online softmax: updates `e_max`, `e_sum`, rescale accumulator per step
- Causal mask (optional)

---

## Bottleneck Characteristics

### Roofline Analysis

Flash Attention is typically **Memory Bound** because:
- QK matmul output needs to be written back to HBM (or kept in registers)
- V load is the primary bandwidth consumer
- Softmax exp/log computation is relatively lightweight

```
Tile_AI ≈ (2×BM×BN×BK + 2×BM×BN×BK) / (BM×BK + BN×BK + BM×BN + BN×BK) × element_size
        ≈ 4×BM×BN×BK / ((BM+2×BN+2×BK) × element_size)
```

For a typical configuration (BM=64, BN=64, BK=64, bf16), Tile AI ≈ 20-30 FLOPs/Byte << Ridge Point 245.

---

## Optimization Strategy Priority

| Priority | Optimization | Description |
|--------|--------|------|
| ⭐⭐⭐ | Full/Tail split | Eliminate hot-path v_cndmask (same as MLA OPT-3) |
| ⭐⭐⭐ | V uses convert_layout instead of smem | Reduce LDS usage |
| ⭐⭐ | K uses immediate smem (value=) | Compiler auto-reuses LDS |
| ⭐⭐ | tl.assume(stride > 0) | Helps compiler optimize |
| ⭐ | 1/d_i reciprocal multiplication | Triggers Newton-Raphson optimization |
| ⭐ | iglp_opt(2) | Instruction-group-level parallelism hint |
| ⭐ | SE-level Zigzag Remap | Causal attention load balancing |
| — | warp_pipeline_stage | Not applicable (complex control flow) |

---

## Detailed Optimization Techniques

### Full/Masked Block Separation

Extract the inner loop into a sub-function, using the `gl.constexpr` parameter `DO_MASK` to control whether masking logic is executed:

```python
if n_full_blocks > 0:
    acc, d_i, m_i = _attn_inner(..., DO_MASK=False)
if masked_blocks > 0:
    acc, d_i, m_i = _attn_inner(..., DO_MASK=True)
```

**Effect**: Non-causal scenarios go from trailing Triton by 5% to leading by 8-13%; causal scenarios improve by +7-14%.

---

### V Uses convert_layout Instead of smem

V matrix (non-transposed) directly uses `buffer_load → gl.convert_layout(v, dot_op1)` into MFMA. Only K (transposed) uses smem.

**Effect**: Reduces LDS usage and VGPR pressure.

---

### SE-level Zigzag Remap (Causal Attention Load Balancing)

In causal attention, different M-blocks have drastically different workloads. The MI355X has 8 XCDs, and the hardware distributes consecutive PIDs to different XCDs in a round-robin manner. Without rearrangement, some XCDs may receive all heavy blocks while others receive all light blocks.

```python
NUM_XCDS = 8
total_blocks = num_m_blocks * num_bh
wave = flat_pid // NUM_XCDS
pos = flat_pid % NUM_XCDS
is_odd = wave % 2
if is_odd:
    logical_pid = wave * NUM_XCDS + (NUM_XCDS - 1 - pos)
else:
    logical_pid = flat_pid
logical_pid = tl.minimum(logical_pid, total_blocks - 1)
start_m = logical_pid // num_bh
off_bs_head = logical_pid % num_bh
```

**Effect** (bs=4, h=32, dim=64):
- S=2048 (16 M-blocks): +65%
- S=1024 (8 M-blocks): +52%

---

## References

- General Optimizations: `common_optimizations.md`
- MLA Decode: `docs/ref-docs/amd/gluon/gfx950/mla_decode.md`
- Lessons Learned: `pitfalls.md`


## Related

- [chunk-GDN (Gated Delta Net) Optimization Summary](chunk_gdn_lessons.md)
- [CDNA4 (gfx950) Generic ISA Optimization Checklist](common_optimizations.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [MLA Decode Attention Optimization Guide](mla_decode.md)
- [Gluon Kernel Performance Optimization Guide (AMD CDNA4)](optimization-guide.md)
- [Fused Attention (Prefill / Paged Attention) Optimization Guide](../../../nvidia/hopper/gluon/fused_attention.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
