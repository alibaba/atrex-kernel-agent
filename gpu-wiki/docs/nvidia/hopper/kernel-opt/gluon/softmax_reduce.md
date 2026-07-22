pattern: softmax_reduce
sub_patterns: []
pitfalls: [1, 5]
---

# Softmax / Reduction / Element-wise Optimization Guide

> This document is a **base pattern (leaf node)** with no sub-pattern dependencies.
> For the general ISA optimization checklist, see `common_optimizations.md`.
> For applicable pitfall experiences, see entries marked as `` in `pitfalls.md` (Pitfalls 1, 5).

> **Status**: Skeleton document, awaiting future optimization case studies.

---

## Pattern Characteristics

| Feature | Description |
|---------|-------------|
| **Core Computation** | Softmax: `exp(x - max(x)) / sum(exp(x - max(x)))`; Reduction: `sum`/`max`/`mean` along a dimension; Element-wise: per-element operations |
| **Main Loop Structure** | Iterate along the reduction dimension (if applicable), or no loop (pure element-wise) |
| **Inter-iteration Dependencies** | Softmax yes (max and sum require global reduction), Reduction yes, Element-wise no |
| **SASS Indicators** | **No** `WGMMA` instructions; `LDG`/`STG` present; may contain `SHFL` (warp-level reduction) |

**Identification Criteria**:
- No wgmma instructions
- Pure reduction or element-wise operations
- No matrix multiplication computation

---

## Bottleneck Characteristics: Memory Bound

Element-wise, reduction, and softmax operators are typically **Memory Bound on all GPUs**.

- Arithmetic Intensity is extremely low (typically < 10 FLOPs/Byte, far below any GPU's Ridge Point)
- Optimization focus: **reduce memory access volume, improve bandwidth utilization**

---

## Optimization Strategy Priorities

Source: `common_optimizations.md` Appendix A, "Element-wise" and "Reduction" rows.

### Element-wise

| Priority | Optimization Item | Description |
|----------|-------------------|-------------|
| ⭐⭐⭐ | §3.0 Coalesced Memory Access Pre-check | Mandatory |
| ⭐⭐⭐ | §3.1 Coalesced Access + Wide Loads | Primary gain for Memory Bound |
| — | §3.3, §3.4, §3.5 | Not applicable (no wgmma) |

### Reduction

| Priority | Optimization Item | Description |
|----------|-------------------|-------------|
| ⭐⭐⭐ | §3.0 Coalesced Memory Access Pre-check | Mandatory |
| ⭐⭐⭐ | §3.1 Coalesced Access + Wide Loads | Primary gain for Memory Bound |
| ⭐ | §3.2 Bank Conflicts | Possible bank conflicts when reductions exchange data via smem |
| — | §3.4, §3.5 | Not applicable (no wgmma) |

---

## Stopping Conditions

Use the general stopping conditions from optimization-guide.md §1.8. Stop when bandwidth utilization ≥ 90%.

---

## Applicable Pitfall Experiences

See `pitfalls.md` for details:

| # | Title | Key Points |
|---|-------|------------|
| 1 | Compiler is sensitive to code structure | Manual code rearrangement is usually a negative optimization |
| 5 | High benchmark variance | CUDA events + 200 samples + P50 |

---

## To Be Added

- [ ] Online normalization optimization pattern for Softmax
- [ ] SHFL instruction optimization for warp-level reduction
- [ ] Tile strategy for multi-dimensional reduction
- [ ] Measured case study data
