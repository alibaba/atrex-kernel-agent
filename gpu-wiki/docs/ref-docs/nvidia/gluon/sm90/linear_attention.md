---
pattern: linear_attention
sub_patterns: [matmul]
pitfalls: [6, 7, 8, 9, 10]
---

# Chunk Linear Attention / Recurrent State Update Optimization Guide

> **Composite pattern**. The internal wgmma is a matmul sub-pattern. See `matmul.md` for general matmul optimization.
> The content below only describes **pattern-specific content** and **applicability adjustments** Instruct sub-pattern optimization.
> See `common_optimizations.md` for the general ISA optimization checklist.

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | `h[i+1] = f(h[i], x[i])`, iterative state matrix update |
| **Main Loop Structure** | Iterates along the time dimension (chunk dimension), with wgmma + fence + wait in each iteration |
| **Inter-Iteration Dependency** | **Yes**. The output h[i] of iteration i is the input to iteration i+1 |
| **Typical Operators** | RNN/SSM state update, chunked linear attention recurrence, `h[i+1] = decay * h[i] + k[i] ⊗ v[i]` |
| **chunk_size Parameter** | Usually controlled by BT (chunk_size) to split the time dimension, affecting the iteration count NT = ceil(T/BT) |

**Identification Criteria**:
- The main loop has a serial dependency of the form `h[i+1] = f(h[i])`
- chunk_size splits the time dimension
- State matrices are passed between iterations

---

## Bottleneck Characteristic: Latency-Bound ⚠️


**Roofline analysis may yield misleading conclusions**. Kernels of this pattern are typically classified as Memory Bound by Roofline (Tile AI < Ridge Point), but the actual bottleneck is **neither bandwidth nor compute**, but rather **latency caused by the serial dependency chain between iterations**. Traditional ISA-level optimizations (memory coalescing, load width, bank conflicts, etc.) are nearly ineffective here, because the bottleneck is not in the efficiency of individual instructions but in the barrier/fence/wait synchronization overhead of each iteration.

### Characteristic Identification

| Characteristic | Description |
|------|------|
| **Inter-Iteration Data Dependency** | The output of iteration i is the input of iteration i+1 (e.g., RNN state update: `h[i+1] = f(h[i])`) |
| **Long Loop** | Main loop iteration count >> 1 (e.g., 100+ iterations), with wgmma + fence + wait in each iteration |
| **Very Low Bandwidth Utilization with Unknown Cause** | Roofline shows Memory Bound, but optimizing memory access yields no performance change |
| **Lower Bound Per-Iteration Latency** | Determined by the minimum cycles of wgmma issue + fence + wait, typically 3-10 µs/iter |

### Diagnosis Method

```
1. analysismainloopdata:
 - output[i] -> input[i+1] ？
 - if -> Latency-Bound

2. :
   theoretical_time = NT × per_iter_min_latency
   per_iter_min_latency ≈ num_wgmma_per_iter × wgmma_latency + sync_overhead
 (Hopper wgmma latency ≈ µs, fence + mma + wait)

3. :
 if elapsed time / theoretical_time < 1.3 ->
```

---

## ⚠️ Sub-Pattern Applicability Adjustment: matmul ISA Optimizations Yield Minimal Benefit


**Core lesson**: For a kernel with inter-iteration serial dependencies (e.g., chunked recurrent state update), after verifying memory coalescing ✅, load width ✅, async_copy ✅, pipeline reordering ❌ (-10%), and loop invariant hoisting ❌ (-32%), only tile dimension tuning yielded +7%. Extensive ISA-level optimization work (several hours) produced zero benefit.

**Adjustments to matmul.md optimization strategy**:
- ISA optimizations in `common_optimizations.md` §3.1-3.5 are **nearly ineffective** for this pattern
- §3.0 memory coalescing pre-check is still mandatory (to ensure no directional errors)
- §3.1 load width verification can be quickly confirmed, but do not spend time fixing
- §3.3/3.4/3.5 should be skipped (pipeline reordering may cause negative optimization; code structure changes carry high risk)

**Quick Diagnosis Flow**:
```
1. kernel mainloopwhether there is output[i] -> input[i+1] ？
 - -> Latency-Bound. "optimization"
 - -> workflow, reference matmul.md

2. per-iteration elapsed time:
 per_iter_min ≈ #wgmma × µs + fence/wait

3. :
   theoretical_time = NT × per_iter_min

4. if / < 1.3 -> ISA optimization
```

**Applicable Scenarios**: RNN/SSM state update, chunked linear attention recurrence, any loop of the form `h[i+1] = f(h[i], x[i])`.

---

## Optimization Strategies (Sorted by Priority)

### Strategy 1: Increase chunk_size (BT) — Most Effective ⭐⭐⭐


In the serial dependency chain `h[i+1] = f(h[i], x[i])`, each iteration has fixed overhead (fence/wait/barrier synchronization, smem allocation, prefetch issue). Doubling BT halves the iteration count, spreading the fixed overhead over more computation.

**Measured Data** (chunk_gated_delta_rule_fwd, H20, K=128, V=128, T=9418):

| BT | NT | Registers/Thread | smem Peak | P50 Latency | Speedup |
|----|-----|-------------|-----------|---------|------|
| 64 | 148 | 76 | 72 KB | 0.488 ms | baseline |
| 128 | 74 | 109 | 140 KB | 0.388 ms | **1.26x** |

**BT Increase Cascade Change Checklist** (all are required):
1. **smem allocation shape**: `[2, BT, 64]` and `[2, 64, BT]` change with BT
2. **BlockedLayout coverage constraints**: Layouts involving the BT dimension must satisfy `spt×tpw×wpc = BT`
3. **Layouts of different shapes cannot be shared**: At BT=64, a certain blocked layout covers both `[64, 16]` (h stores) and `[BT=64, 16]` (v loads). After BT=128, they must be split into two independent layouts
4. **swizzle_byte_width check**: Ensure it does not exceed `TILE_DIM × element_bytes`
5. **smem total check**: Must not exceed 228 KB (Hopper max)**Upper Limit of BT Increase**:
- BT=256 → smem 276 KB > 228 KB limit ❌
- Registers 109/thread already high, further increase may trigger spill
- Rule of thumb: Every time BT doubles, check both smem and register hard constraints

**Applicable Conditions**:
- Latency-Bound kernel (serial dependency across iterations)
- ISA is already clean under current BT (no spill, correct load width, async_copy already used)
- smem and registers have headroom
- Grid blocks count won't drop below ~30% of SM count

### Strategy 2: Tile Dimension Tuning (General Methodology)

When a kernel has multiple tunable tile dimensions (e.g., BM, BN, BV, BK), tuning follows these general principles:

#### Benefits and Costs of Increasing Tile Dimensions

| Factor | Effect of Increasing Tile Dimension |
|------|-------------------|
| ✅ Fewer grid blocks → reduced launch/scheduling overhead | Positive |
| ✅ Larger wgmma matrix → higher instruction-level efficiency | Positive |
| ✅ Fewer loop iterations → reduced barrier/fence overhead | Positive (**Core benefit for Latency-Bound**) |
| ❌ Increased register pressure → potential spill to local memory | Negative |
| ❌ Reduced SM utilization → more SMs idle | Negative |
| ❌ Increased smem usage → lower occupancy | Negative |

#### Decision Process

```
1. dimensioncurrent
2. dimension, compute 2× :
 a. grid_blocks -> SM utilization 20%？
 b. accumulator register -> spill？
 c. smem -> 228 KB？
3. SM utilization ≥ 20% register spill tile
4. verification(do notanalysis)
```

#### Dimension Selection Preferences for Latency-Bound

- **Prioritize adjusting dimensions that reduce iteration count** (e.g., BT), rather than dimensions that increase wgmma computation volume (e.g., BV)
- Doubling BT → halving iteration count → amortizing fixed overhead → effective speedup
- Doubling BV → increasing wgmma N dimension → but iteration count unchanged → ineffective or even counter-optimizing for Latency-Bound

#### BlockedLayout Recalculation Rules (Required After Modifying Tile Dimensions)

After modifying any tile dimension, all `BlockedLayout` that reference that dimension must be recalculated. Constraints:

```
: spt[d] × tpw[d] × wpc[d] = TILE_DIM[d] (dimension d)
Warp : tpw[0] × tpw[1] = 32 (Hopper warp size)
Warp total count: wpc[0] × wpc[1] = num_warps
128-bit : spt[contiguousdimension] × element_bytes ≥ 16
```

#### NVMMASharedLayout swizzle_byte_width Constraint

**Hard constraint**: `swizzle_byte_width ≤ tile_byte_width_in_swizzle_dim`

```
tile_byte_width = TILE_DIM[swizzle_dim] × element_size_bytes
swizzle_byte_width must ≤ tile_byte_width

example (bf16, 2B/element):
  TILE_DIM=16 → 32 bytes → swizzle_byte_width ≤ 32
  TILE_DIM=32 → 64 bytes → swizzle_byte_width ≤ 64
  TILE_DIM=64 → 128 bytes → swizzle_byte_width ≤ 128
  TILE_DIM=128 → 256 bytes → swizzle_byte_width = 128 (max)
```

Violating this constraint results in **LLVM compilation error** (`Block shape is too small for the swizzle byte size`), not a runtime error.

---

## Anti-Patterns (Things Not to Do)

### Anti-Pattern 1: Do Not Increase BV

Increasing BV from 16 to 32, expecting each wgmma to perform more useful computation (64×32×BT vs 64×16×BT). Actual performance **degraded by 30%**.

**Root Cause**: Increasing BV changes the N dimension of wgmma (output width), which is fundamentally different from the effect of BT (K dimension/reduction):
- Doubling BV → accumulator registers double (`[64, BV]` fp32 → per b_h usage doubles)
- Doubling BV → h_smem/vn_smem in smem doubles, potentially changing swizzle pattern
- Doubling BV → grid V dimension block count halves, but per-block workload doubles (net effect not positive)
- For Latency-Bound kernels, wgmma throughput is not the bottleneck; increasing wgmma operand size does not reduce fixed overhead

**Lesson**: Tile tuning for Latency-Bound kernels should prioritize **dimensions that reduce iteration count** (e.g., BT), not dimensions that increase wgmma computation volume (e.g., BV).

### Anti-Pattern 2: Do Not Increase num_warps

Attempting to increase num_warps from 4 to 8, expecting to hide latency through higher occupancy, but theoretical analysis shows it is ineffective.

**Root Cause**: In a serial dependency chain, all warps within the same block must wait for the wgmma in each iteration to complete before entering the next iteration (synchronized via fence + wait). More warps cannot reduce the latency of a single iteration — they are all waiting for the same wgmma to complete. Instead, they increase register pressure (more warps × more accumulator registers), potentially causing spill.

**Rules**:
- For Latency-Bound kernels, do not increase num_warps as an optimization technique
- Increasing num_warps is only effective in scenarios: insufficient SM utilization + kernels without serial dependencies (e.g., standard GEMM)
- Modifying num_warps requires recalculating `warps_per_cta` for all layouts (constraint (3)), which involves significant effort and is not worth attempting in Latency-Bound scenarios

### Anti-Pattern 3: Cannot Skip smem allocate

Multiple `warpgroup_mma(k_smem, vn_smem, ...)` calls within the loop use the same data. Attempting to do only one `allocate_shared_memory`, with subsequent wgmma directly referencing the same smem variable, resulted in severe precision regression (max_diff=0.16).**Root Cause**: The Gluon compiler determines the lifetime of smem addresses based on SSA liveness analysis. When an smem variable has no new `allocate_shared_memory` calls after `warpgroup_mma_wait`, the compiler considers that smem region's liveness to have ended mad will assign the same address to subsequent `allocate_shared_memory`. As a result, the original smem data gets overwritten by later writes.

**Key Insight**: `allocate_shared_memory` in Gluon is not just about writing data — it also **declares the liveness starting point of the smem address**. Removing this call is equivalent to telling the compiler "this smem region is no longer needed."

**Rules**:
- ❌ Do not remove `allocate_shared_memory` calls to "save one smem write," even if the data being written is exactly the same
- ❌ Do not assume that smem operands remain valid after `warpgroup_mma_wait` — the compiler may have already reused the address
- ✅ Every time wgmma needs an smem operand, there must be a corresponding `allocate_shared_memory` to ensure liveness
- ✅ If smem data genuinely needs to be reused, keep `allocate_shared_memory(value=same_data)` + `fence_async_shared()`, even if it appears redundant

**Diagnosis**: If accuracy degrades after removing `allocate_shared_memory` but there is no compilation error, first suspect that smem addresses are being overwritten due to reuse. Compare the smem addresses (`[R_base+offset]`) in the SASS before and after optimization to confirm.

---

## Stopping Conditions

- **Measured time / theoretical lower bound < 1.3×** → Approaching architectural limit, stop ISA-level optimization
- **BT has reached the smem or register limit** → chunk_size cannot be further increased
- Consider **algorithm-level changes** (e.g., kernel fusion to eliminate intermediate state, reordering computations to reduce serial dependency chain length)

> **Key**: The optimization headroom for a Latency-Bound kernel is fundamentally limited by the length of the serial dependency chain. When the kernel is within 1.3× of the theoretical lower bound, ISA-level optimization should be stopped.

---

## Optimization Strategy Priority Quick Reference

Source: `common_optimizations.md` Appendix A, "Recurrent state update" row.

| Priority | Optimization Item | Description |
|----------|-------------------|-------------|
| ⭐⭐⭐ | Increase chunk_size (BT) | Reduces iteration count, most effective |
| ⭐⭐ | §1.7 Tile dimension tuning | Choose dimensions that reduce iteration count |
| ⭐ | §3.0 Coalesced memory access pre-check | Mandatory, rule out directional errors |
| ⭐ | §3.1 Load width verification | Quick verification only |
| — | §3.5 wgmma correctness | Optional confirmation |
| ❌ | §3.3 Scratch elimination | Pipeline/code refactoring may cause negative optimization |
| ❌ | §3.4 async_copy reordering | May cause negative optimization |
| ❌ | smem allocate omission | Pitfall 8: causes data to be overwritten |
| ❌ | BV increase | Pitfall 10: measured -30% |
| ❌ | num_warps increase | Pitfall 7: ineffective under serial dependencies |

---

## Applicable Pitfall Experience

See `pitfalls.md` for details. Summary below:

| # | Title | Key Point |
|---|-------|-----------|
| 6 | ISA optimization yields extremely low gains | Determine Latency-Bound first before acting, avoid wasting hours |
| 7 | Increasing num_warps is ineffective | More warps in a serial dependency chain do not reduce per-iteration latency |
| 8 | smem allocate cannot be omitted | The compiler will reuse smem addresses; removing allocate causes data overwrite |
| 9 | Increasing chunk_size is most effective | BT 64→128 achieved +20% speedup (measured data) |
| 10 | BV increase is a negative optimization | Does not reduce iteration count, increases register pressure, measured -30% |
