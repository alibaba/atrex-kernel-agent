---
pattern: chunk_gdn_linear_attention
sub_patterns: [matmul]
pitfalls: [2, 7]
---

# chunk-GDN (Gated Delta Net) Optimization Summary

> Source: Qwen3.5-397B chunk-GDN kernel optimization practice on MI355X (2026-04-24)

**Last updated**: 2026-06-30

> Applicable: CDNA4 gfx950 + Gluon framework, similar to linear attention / SSM state accumulation kernels

---

## Key Findings (Sorted by Payoff)

### 1. ⭐⭐⭐ ds_write_b16 Narrow Writes Are the Hidden Killer of Narrow Tiles

**Symptom**: dot_op1 operands with BV ≤ 16, going through the `smem.store()` + `smem.load(dot_op1)` path, generate a large number of `ds_write_b16` / `ds_write_b16_d16_hi` (16-bit element-wise writes) instead of `ds_write_b128` (128-bit vector writes).

**Diagnosis**:
```bash
grep -o "ds_write_[a-z0-9_]*" $ASM | sort | uniq -c
# if ds_write_b16 -> write！
```

**Fix**: Replace `smem.store(data)` + `smem.load(dot_op1)` with `gl.convert_layout(data, dot_op1)`:
```python
# ❌ tile smem -> ds_write_b16
smem_b.index(0).store(b_h.to(gl.bfloat16))  # 44 × ds_write_b16!
h_dot = smem_b.index(0).load(dot_op1)

# ✅ convert_layout -> write, ds_read_b64_tr_b16
h_dot = gl.convert_layout(b_h.to(gl.bfloat16), dot_op1)
```

**Pattern**:
- BV ≤ 16: **convert_layout is better** (avoids ds_write_b16)
- BV ≥ 32: **smem is better** (ds_write_b128 is efficient)

**Measured Gain**: fwd_h kernel +6-13%

---

### 2. ⭐⭐⭐ k_width=8 (Not 4)

**Symptom**: Gluon kernel performance is ~10% worse than Triton, ISA instruction count looks reasonable.

**Diagnosis**: Extract dot_op kWidth from Triton TTGIR:
```bash
grep "kWidth" /tmp/triton_dump/*/*.ttgir
# : kWidth = 8 ( Gluon default 4)
```

**Fix**:
```python
# ❌ default k_width=4: ds_read
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)

# ✅ Triton TTGIR k_width=8: ds_read
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
```

**Measured Gain**: fwd_h +10%, fwd_o +0% (already using instrShape=[32,32,16] K=16, k_width=8 has minor impact)

---

### 3. ⭐⭐ Manual buffer_load Prefetching (Better Than async_copy)

**Background**: On CDNA4, mixing async_copy (buffer_load_to_shared) with buffer_load triggers in-order completion serialization, which actually slows things down.

**Correct Approach**: Pure buffer_load + early dispatch:
```python
# Prologue: prefetch iter 0
b_w_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs)

# Loop:
for i_t in range(NT):
 b_w = b_w_pre # useprefetchdata
    # ... compute ...

 # prefetch iter i+1 ( compute previous, hardware OOO latency)
    if i_t + 1 < NT:
        b_w_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=next_offs)
```

**Key**: Timing of k^T prefetch — issue buffer_load before gating (exp/exp2), so that load latency overlaps with exp computation.

**Measured Gain**: fwd_h +33% (w/v cross-iteration + k^T before gating)

---

### 4. ⭐⭐ VGPR Tradeoff Between DistributedLinearLayout and BlockedLayout

**Finding**: DLL produces better buffer_load patterns (more ds_read_b64_tr_b16) but increases VGPR. BlockedLayout has lower VGPR → better LLVM scheduling flexibility.

| Layout | Advantage | Disadvantage |
|--------|-----------|--------------|
| DLL (DistributedLinearLayout) | Better vectorization | VGPR +20 |
| BlockedLayout (tpw=[4,16]) | Lower VGPR | Needs correct tpw/wpc |

**Decision**: If VGPR > 450 (single warp kernel), prefer BlockedLayout. If VGPR has headroom, DLL may be better.

**Measured**: fwd_o DLL 484 VGPR → blocked 464 VGPR → +5% (due to improved LLVM scheduling)

---

### 5. ⭐⭐ Strictly Matching TTGIR Is Not Always Optimal

**Finding**: Fully restoring all layouts and operations from Triton TTGIR into Gluon results in ~4% worse performance than a cherry-pick approach.

**Reason**: Triton and Gluon's LLVM backends produce different instruction scheduling for the same IR. Certain Gluon-specific code structures (e.g., convert_layout + blocked_hv) enable Gluon's LLVM backend to generate better vmcnt interleaving.

**Recommendation**: Use TTGIR as a reference (extract kWidth, shared layout params), but verify the effect of each change through measurements — do not blindly match everything.

### 6. ⭐ async_copy Limitations on CDNA4

| Limitation | Description |
|------|------|
| order=[0,1] does not compile | SwizzledSharedLayout with order=[0,1] triggers `unrealized_conversion_cast` |
| Mixed usage slows down | async_copy + buffer_load in the same kernel causes serialization (-11%) |
| spt constraint | spt[contiguous_dim] × bits = 128 or 32 |
| [64,128] special case | Requires tpw=[4,16] to maintain spt=8 × 16bits = 128 |

**Conclusion**: For kernels with loops (e.g., fwd_h), pure buffer_load + manual prefetch is better than async_copy. async_copy is suitable for single-pass kernels without loops (e.g., K-reduction in GEMM).

---

### 7. ⭐ Memory-bound Kernel Tuning Essentials (l2norm Case)

For extremely memory-bound kernels with AI < 1:

| Parameter | Optimal Direction | Reason |
|------|---------|------|
| num_stages | **1** | Pipeline overhead > benefit |
| BT (tile rows) | **Small** (8-16) | More CTAs → better tail utilization |
| num_warps | **2** | Balances occupancy and utilization |
| Vectorization | Ensure dwordx4 | Critical bottleneck |

**BW Utilization Upper Bound**: Element-wise kernels with large data sizes (> L2 size) measured at ~60% BW, limited by L2 miss + TLB constraints.

---

## Diagnostic Process (Verified with chunk-GDN)

```
1. Roofline: compute-bound vs memory-bound
2. TRITON_KERNEL_DUMP -> Triton + Gluon ISA
3. ds_write type -> ds_write_b16 -> convert_layout
4. ds_read type -> kWidth -> k_width=8
5. buffer_load type -> ushort -> layout
6. check VGPR -> DLL vs BlockedLayout
7. verificationaccuracy + performance
```

---

## Applicability

This experience directly applies to:
- Gated Delta Net (chunk-GDN) fwd_h / fwd_o
- Any dot operation with narrow tiles (BV ≤ 16)
- Loop kernels with serial state dependencies (linear attention, SSM, Mamba)
- CDNA4 (gfx950) + Gluon framework

Indirectly applies to:
- Similar optimizations for CDNA3 (gfx942) (kWidth differences need verification)
- FlyDSL framework (convert_layout corresponds to layout conversion optimization)


## Related

- [CDNA4 (gfx950) Generic ISA Optimization Checklist](common_optimizations.md)
- [Fused Attention Optimization Guide](fused_attention.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [MLA Decode Attention Optimization Guide](mla_decode.md)
- [Gluon Kernel Performance Optimization Guide (AMD CDNA4)](optimization-guide.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120](../../../nvidia/blackwell-geforce/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
