# Hopper (sm_90) Practical Pitfall Experience Index

**Last Updated**: 2026-03-20

> The following experiences come from actual optimization projects, each verified through real measurements. Listed by number, with each entry annotated with applicable mode tags for mode-based filtering.
> Pitfalls 8-11 are from the chunk_gated_delta_rule optimization case studylu, including measured data.

---

## Quick Reference Index

| # | Title | Applicable Mode | Topic Inline |
|---|------|---------|---------|
| 1 | Compiler is sensitive to code structure | [General] | `matmul.md` |
| 2 | Tile dimension tuning has a sweet spot | [General] | `matmul.md` |
| 3 | swizzle_byte_width constraint | [General] | `matmul.md` |
| 4 | BlockedLayout must be recalculated holistically | [General] | `matmul.md` |
| 5 | High benchmark variance | [General] | `matmul.md` |
| 6 | ISA optimization yields minimal gains | [Recurrent] | `linear_attention.md` |
| 7 | Increasing num_warps is ineffective | [Recurrent] | `linear_attention.md` |
| 8 | smem allocate cannot be omitted | [Recurrent] | `linear_attention.md` |
| 9 | Increasing chunk_size is most effective | [Recurrent] | `linear_attention.md` |
| 10 | Increasing BV is a negative optimization | [Recurrent] | `linear_attention.md` |
| 11 | Using ncu profile to identify launch sequence numbers | [General] | — |

---

## Pitfall 1: Gluon/Triton Compiler Is Highly Sensitive to Code Structure — Manual Refactoring Is Usually a Negative Optimization [General]

**Symptom**: Manually rearranging code structure (e.g., hoisting loop invariants, adjusting prefetch positions) results in 10%-30% performance regression, even though the logic is completely equivalent at the instruction level.

**Root Cause**: The Gluon/Triton compiler's CSE (Common Subexpression Elimination), register allocation, and instruction scheduling are **globally coupled**. Changing code structure alters variable liveness intervals, causing the compiler to make different register allocation decisions, potentially triggering spills or suboptimal scheduling.

**Rules**:
- ❌ Do not manually hoist loop invariants outside the loop (the compiler's CSE already handles this)
- ❌ Do not manually adjust prefetch/async_copy positions to "improve overlap" (the compiler's scheduling for the current position may already be locally optimal)
- ❌ Do not merge smem allocations within a loop to "save smem" (the compiler's smem reuse strategy may depend on the current structure)
- ✅ Only make mathematically equivalent modifications that do not change code structure (e.g., modifying layout parameters, tile sizes, constexpr values)

**Verification**: Any code structure changes must be performance-tested. If regression occurs, abandon the change immediately — do not attempt to "fix" the regression. The original structure is usually optimal.

---

## Pitfall 2: Tile Dimension Tuning Has a Sweet Spot — Bigger Is Not Always Better and May Be Counterproductive [General]

**Symptom**: The result of increasing any tile dimension (e.g., from 16 → 32 or higher) is **highly dependent on the specific kernel**. In practice, what seems like a reasonable 2× increase can result in a 10% to 40% performance regression.

**Root Cause**: Increasing tile dimensions has dual effects: reducing iteration count (a benefit) while increasing register pressure alsod smem usage (a cost). The sweet spot varies by kernel structure, matrix size, and hardware model. The current tile size may already be at the sweet spot, and increasing it pushes past the optimal point into regression territory.

**⚠️ Key Lesson**: **Do not assume that increasing tile dimensions will always yield benefits**. Even a 2× increase may be a negative optimization. Always measure.

**Rules**:
- **Always benchmark the current tile size first, then benchmark the modified tile size** — do not rely on intuition
- Change only one dimension at a time MAD only by a factor of 2×; do not change multiple dimensions or use larger jumps simultaneously
- Register pressure is a hidden cost: doubling a tile dimension typically doubles accumulator register usage, potentially causing spills
- **Increasing tile dimensions also changes smem layout and blocked layout**, and these cascading changes may introduce additional performance penalties
- **Rule of thumb**: grid_blocks ≥ num_SMs × 0.3 is a safe lower bound. Values below this require strong justification to consider

---

## Pitfall 3: NVMMASharedLayout swizzle_byte_width Must Not Exceed Tile Byte Width [General]

**Symptom**: After modifying the `shared_v` of `swizzle_byte_width` from 64 to 128, the kernel fails to compile (LLVM ERROR: Block shape is too small for the swizzle byte size).

**Root Cause**: The swizzle of `NVMMASharedLayout` performs an XOR transformation on one dimension of the tile. If the byte width of that dimension is smaller than `swizzle_byte_width`, the swizzle cannot cover a complete row, and the hardware cannot execute.

**Hard Constraint**:
```
swizzle_byte_width ≤ TILE_DIM[swizzle_dim] × element_size_bytes

bf16 example:
  TILE_DIM=16 → max swizzle = 32
  TILE_DIM=32 → max swizzle = 64
  TILE_DIM=64 → max swizzle = 128
```

**Rules**:
- After modifying tile dimensions, you **must** check and adjust `swizzle_byte_width`
- This is a compile-time error and will not produce silent correctness issues, but it will waste debugging time
- Safe strategy: `swizzle_byte_width = min(128, TILE_DIM × element_bytes)`

---

## Pitfall 4: BlockedLayout Modifications Must Be Recalculated Holistically — Isolated Parameter Adjustments Will Break Coverage Constraints [General]

**Symptom**: After modifying BV, only `spt` was changed without adjusting `tpw`/`wpc`, causing the layout to be unable to cover the target tile (compile error or silent data error).

**Constraint System** (all conditions must be satisfied simultaneously):
```
(1) spt[d] × tpw[d] × wpc[d] = TILE_DIM[d] (, dimension d)
(2) tpw[0] × tpw[1] = 32                         (Hopper warp size)
(3) wpc[0] × wpc[1] = num_warps (warp total count)
(4) spt[contiguousdimension] × element_bytes ≥ 16 (128-bit load )
```**Method**: List target tile shapes, enumerate all combinations satisfying (1)(2)(3)(4), and select the most reasonable one. For 2D tile `[M, N]`:
```python
for spt0 in divisors(M):
    for spt1 in [s for s in divisors(N) if s * elem_bytes >= 16]:
        for tpw0 in divisors(32):
            tpw1 = 32 // tpw0
            for wpc0 in divisors(num_warps):
                wpc1 = num_warps // wpc0
                if spt0*tpw0*wpc0 == M and spt1*tpw1*wpc1 == N:
                    # valid combination
```

---

## Pitfall 5: Large Benchmark Variance — Must Use CUDA Events + Large Sample Size + P50 [General]

**Symptom**: The same kernel shows 15%-20% runtime variance across different runs (e.g., P10=468µs, P90=558µs), making single measurements unreliable.

**Root Cause**: Dynamic GPU frequency scaling (thermal throttling), L2 cache state, memory controller scheduling, OS interrupts, etc. This is especially pronounced on H20 (low TDP leads to more frequent clock throttling).

**Rules**:
- ✅ Use `torch.cuda.Event(enable_timing=True)` to measure single kernel runtime (excluding CPU overhead)
- ✅ At least 200 samples, sort and use P50 (median) as the representative value
- ✅ A/B comparisons must be interleaved within the same process and same run (to eliminate system state differences)
- ❌ Do not use `time.time()` or `torch.cuda.synchronize()` + wall clock (includes CPU overhead and scheduling jitter)
- ❌ Do not use single or few measurements for performance comparisons
- **Significance threshold**: Differences < 3% are unreliable on H20 and may be noise. Only differences ≥ 5% that are consistent across runs should be considered valid improvements.

**Standard benchmark template**:
```python
def benchmark(fn, warmup=20, repeat=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    results = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        results.append(s.elapsed_time(e))
    results.sort()
    return results[len(results)//2]  # P50
```

---

## Pitfall 6: ISA-Level Optimization Yields Minimal Gains for Latency-Bound Kernels — Identify First, Then Act [Recurrent]

**Symptom**: For a kernel with serial dependencies between iterations (e.g., chunked recurrent state update), we verified coalesced memory access ✅, verified load width ✅, verified async_copy ✅, but got: pipeline reordering ❌(-10%), loop-invariant code motion ❌(-32%). Ultimately, only a +7% gain was achieved through tile dimension tuning. Hours of ISA-level optimization yielded zero benefit.

**Lesson**: If a kernel has a **serial dependency chain between iterations**, this should be identified in Step 1 (§1.6) and you should directly jump to tile dimension tuning (§1.7), skipping ISA-level analysis and optimization in Steps 2-3.

**Quick Identification Flow**:
```
1. kernel mainloopwhether there is output[i] -> input[i+1] ？
 - -> Latency-Bound. §1.7
 - -> workflow Step 2-3

2. per-iteration elapsed time:
 per_iter_min ≈ #wgmma × µs + fence/wait

3. :
   theoretical_time = NT × per_iter_min

4. if / < 1.3 -> ISA optimization
```

**Applicable Scenarios**: RNN/SSM state update, recurrence in chunked linear attention, any loop of the form `h[i+1] = f(h[i], x[i])`.

---

## Pitfall 7: Increasing num_warps Is Ineffective for Serial-Dependency Kernels [Recurrent]

**Symptom**: Attempting to increase num_warps from 4 to 8, expecting higher occupancy to hide latency, but theoretical analysis shows this is ineffective.

**Root Cause**: In a serial dependency chain, all warps within the same block must wait for the wgmma in each iteration to complete before entering the next iteration (synchronized via fence + wait). More warps cannot shorten the latency of a single iteration — they are all waiting for the same wgmma to complete. Instead, this increases register pressure (more warps × more accumulator registers), potentially causing spilling.

**Rules**:
- For latency-bound kernels, do not increase num_warps as an optimization strategy
- Increasing num_warps is only effective when: SM utilization is insufficient + the kernel has no serial dependencies (e.g., standard GEMM)
- Changing num_warps requires recomputing `warps_per_cta` for all layouts (constraint (3)), which involves significant work and is not worth attempting in latency-bound scenarios

---

## Pitfall 8: Gluon Compiler Reuses smem Addresses — Removing allocate_shared_memory Causes Data Overwrite [Recurrent]

**Symptom**: Within a loop, multiple `warpgroup_mma(k_smem, vn_smem, ...)` calls use the same `b_v_new_bf16` data. Attempting to call `allocate_shared_memory` only onceatore and having subsequent wgmma calls reference the same `vn_smem` resulted in severe precision degradation (max_diff=0.16).

**Root Cause**: The Gluon compiler determines smem address lifetimes based on SSA liveness analysis. When `vn_smem` has no subsequent `allocate_shared_memory` callsito after `warpgroup_mma_wait`, the compiler considers that smem region's liveness to have ended and may assign the same address to later `allocate_shared_memory` operations (e.g., `k3_smem`). As a result, `vn_smem` data gets overwritten by writes to `k3_smem`.**Key Insight**: `allocate_shared_memory` in Gluon is not just about writing data — it simultaneously **declares the starting point of smem address liveness**. Removing this call tells the compiler "this piece of smem is no longer needed."

**Rules**:
- ❌ Do not remove a `allocate_shared_memory` call just to "save one smem write," even if the data being written is identical
- ❌ Do not assume that smem operands remain valid after `warpgroup_mma_wait` — the compiler may have already reused the address
- ✅ Every time a wgmma needs smem operands, there must be a corresponding `allocate_shared_memory` to guarantee liveness
- ✅ If you genuinely need to reuse smem data, keep `allocate_shared_memory(value=same_data)` + `fence_async_shared()`, even if it appears redundant

**Diagnosis**: If accuracy degrades after removing `allocate_shared_memory` without any compilation errors, first suspect smem address reuse/overwrite. Compare smem addresses (`[R_base+offset]`) in the SASS before and after optimization to confirm.

---

## Pitfall 9: The Most Effective Optimization for Latency-Bound Kernels Is Increasing chunk_size — Measured +20% [Recurrent]

**Phenomenon**: For a chunked recurrent state update kernel (chunk_gated_delta_rule_fwd), ISA-level optimizations were attempted (coalesced memory access ✅, load width ✅, async_copy ✅), all to no avail. Increasing `chunk_size` (BT) from 64 to 128 yielded a stable **+20% speedup** (alternating A/B measurements, 500 sample pairs, consistent P10/P50).

**Root Cause**: In the serial dependency chain `h[i+1] = f(h[i], x[i])`, each iteration incurs fixed overhead (fence/wait/barrier synchronization, smem allocation, prefetch dispatch). Doubling BT halves the number of iterations (148→74), amortizing the fixed overhead across more computation. The number of wgmma operations per iteration remains unchanged (determined by the K-dimension split, independent of BT), and although doubling the wgmma reduction dimension K_wgmma=BT makes each wgmma larger, the total computation remains the same.

**Measured Data** (chunk_gated_delta_rule_fwd, H20, K=128, V=128, T=9418):

| BT | NT | Registers/thread | smem peak | P50 Latency | Speedup |
|----|-----|-----------------|-----------|---------|------|
| 64 | 148 | 76 | 72 KB | 0.488 ms | baseline |
| 128 | 74 | 109 | 140 KB | 0.388 ms | **1.26x** |

**Cascading Changes Checklist When Increasing BT** (all are mandatory):
1. **smem allocation shape**: `[2, BT, 64]` and `[2, 64, BT]` change with BT
2. **BlockedLayout coverage constraint**: layouts involving the BT dimension must satisfy `spt×tpw×wpc = BT`
3. **Layouts with different shapes cannot be shared**: When BT=64, `blocked2` covers both `[64, 16]` (h stores) and `[BT=64, 16]` (v loads); when BT=128, they must be split into `blocked2` ([64,16]) and `blocked4` ([128,16])
4. **swizzle_byte_width check**: ensure it does not exceed `TILE_DIM × element_bytes`
5. **Total smem check**: ensure it does not exceed 228 KB (Hopper maximum)

**Upper Limits for BT Increase**:
- BT=256 → smem 276 KB > 228 KB limit ❌
- Registers at 109/thread is already high; further increases may trigger spills
- Rule of thumb: check both smem and register hard constraints every time BT is doubled

**Applicable Conditions**:
- Latency-Bound kernel (serial dependency between iterations)
- ISA is already clean at current BT (no spills, correct load widths, async_copy already in use)
- smem and registers have headroom
- Number of grid blocks does not drop below ~30% of SM count

---

## Pitfall 10: Increasing BV Can Be a Negative Optimization for Latency-Bound Kernels — Measured -30% [Recurrent]

**Phenomenon**: Increasing BV from 16 to 32, expecting each wgmma to perform more useful computation (64×32×BT vs 64×16×BT). Measured performance degraded by 30%.

**Root Cause**: Increasing BV changes the N-dimension (output width) of wgmma, which has a completely different effect from BT (K-dimension / reduction):
- Doubling BV → accumulator registers double (`[64, BV]` fp32 → per b_h usage doubles)
- Doubling BV → h_smem/vn_smem in smem double, potentially changing swizzle patterns
- Doubling BV → number of grid blocks along V dimension halves, but per-block workload doubles (net effect is not positive)
- For Latency-Bound kernels, wgmma throughput is not the bottleneck; increasing wgmma operand size does not reduce fixed overhead

**Lesson**: Tile tuning for Latency-Bound kernels should prioritize dimensions that **reduce the number of iterations** (such as BT), rather than dimensions that increase wgmma computation (such as BV).

---

## Pitfall 11: ncu Profiling Must First Identify the Correct Kernel Launch Index [General]

**Phenomenon**: Using a fixed value like `ncu --launch-skip 10` to profile a Gluon kernel, but what actually gets profiled is a PyTorch internal `torch.randn` or `torch.cat` kernel, not the target Gluon kernel.

**Root Cause**: The Python test script for Gluon kernels triggers a large number of PyTorch internal kernel launches during the setup phase (random number generation, tensor concatenation, cumsum, etc.), and the count is not fixed. For example, a typical test script has 23 PyTorch kernel launches before the target kernel.

**Rules**:
```bash
# Step 1: column kernel launch
ncu --print-summary per-kernel python kernel.py

# Step 2: kernel ( "chunk_gated_delta_rule_fwd_ke..." - 23)

# Step 3: profile
ncu --set full --launch-skip 23 --launch-count 1 -o profile python kernel.py
```**Complete flow for checking spill**:
```bash
# check STL/LDL(register)
ncu --import profile.ncu-rep --page source --print-source sass | grep -cE 'STL|LDL'

# register
ncu --import profile.ncu-rep --metrics launch__registers_per_thread
```

## Related Documents

- **Cross-architecture comparison**: [CDNA4 Pitfalls](../../../../amd/cdna4/ref-docs/gluon/pitfalls.md) — 10 CDNA4-specific pitfalls
- **🔴 #1 Conflicts with AMD**: This document considers manual code refactoring to be almost always a negative optimization, but manual optimizations in [CDNA3 ISA Optimization](../../../../amd/cdna3/ref-docs/gluon/common_optimizations.md) yield +1-4% gains
- **Referenced by**: [matmul](matmul.md) references #1-5 | [linear_attention](linear_attention.md) references #6-10
