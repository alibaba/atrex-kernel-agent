# Pitfalls: FlyDSL Attention Backward dQ + dK+dV on MI308X (gfx942)

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

Traps encountered while optimizing the FlyDSL attention backward (dK+dV)
kernel on MI308X. Companion report:
[cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](../../ref-docs/flydsl/cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md).

Forward kernel pitfalls (shared traps like K_PAD, rocdl.exp2) are in
[flash-attn-pitfalls.md](flash-attn-pitfalls.md).

---

## 1. Two-Pass (split dV/dK loops) regresses 46% despite improving occupancy

**Trap**: Accum VGPR = 196 limits occupancy to 2 waves/SIMD. The natural fix is to split the single loop into two passes — Pass 1 computes dV only, Pass 2 computes dK only — cutting Accum VGPR roughly in half and potentially doubling occupancy.

**Result**: Correctness passed, but 16.23 ms vs 11.08 ms baseline (+46.6% regression).

**Why**: Each pass must recompute S=K@Q^T and P=softmax(S) from scratch (Q/K are too large to keep in registers between passes). This doubles GEMM1 + softmax work. The occupancy improvement (2→3-4 waves) cannot compensate for 2× compute increase when the kernel is already compute-bound.

**Lesson**: For compute-bound backward kernels, trading compute for occupancy is a losing proposition. Only split passes if the recomputed work is cheap relative to the saved accumulator cost.

---

## 2. block_n=64 silently drops half the K/V rows (correctness failure)

**Trap**: Increasing block_n from 32 to 64 doubles the K/V tile size per workgroup, reducing total workgroups and loop iterations. Benchmark shows 22% TFLOPS improvement, suggesting a clear win.

**Result**: Rows 32-63 of dK and dV output are all zeros. 50% of KV rows are never computed.

**Why**: MFMA32x32x8 C output maps `lane_mod_32` (lane_id % 32) to the row index. With 64-wide waves, only 32 unique row positions exist. The store logic uses `k_row = n_start + lane_mod_32`, which only covers the first 32 rows of the 64-row tile. Supporting block_n=64 would require an inner n_chunk loop that processes two 32-row sub-tiles sequentially.

**Lesson**: Always run correctness checks before trusting benchmark numbers. A kernel that silently computes half the output will appear faster while being wrong. The MFMA32x32x8 output layout fundamentally limits the N dimension to 32 rows per tile without inner tiling.

---

## 3. Cooperative LDS transpose for dO/Q cannot be eliminated

**Trap**: The dV MFMA uses dO^T as A operand (lane_mod_32 = d_col), requiring 4 consecutive q values per lane. Since dO is stored as (..., S, D), loading 4 different q values at the same d_col requires 4 strided global loads (stride = D = 64 elements apart). The obvious thought: "skip LDS transpose, load directly from global like GEMM1 loads Q."

**Result**: Not possible.

**Why**: In GEMM1 (S=K@Q^T), Q is the B operand where lane_mod_32 selects the q_row, and each lane loads 4 consecutive D-dim values — which ARE contiguous in memory. But for dV MFMA, dO^T is the A operand where lane_mod_32 selects d_col, and each lane needs 4 consecutive q_row values at the same d_col — which are NOT contiguous (stride = D between q_rows). LDS transpose converts row-major dO(q,d) into column-major dO^T(d,q) to make the q dimension contiguous.
**Lesson**: Whether a GEMM operand can be loaded directly from global or needs LDS transpose depends on which dimension is contiguous in memory vs. which dimension MFMA maps to the lane_mod_32 (row) vs. the K-pack (4 consecutive). If the K-pack dimension is strided in memory, LDS transpose is mandatory.

---

## 4. Multi-wave (4 waves) is 2.9× slower for backward despite working for forward

**Trap**: The forward kernel uses BLOCK_SIZE=256 (4 waves) successfully. For backward, the natural assumption is that more waves = better latency hiding = better performance.

**Result**: block_n=32, 4 waves: 32.0 ms vs 11.1 ms single-wave (2.9× slower).

**Why**: The backward kernel has more inter-wave synchronization points (3 barriers per Q-tile vs 1-2 in forward). Each barrier stalls all waves. Additionally, the backward kernel's VGPR pressure is already at 196 Accum VGPR for 1 wave; 4 waves would require each wave to fit in 512/4 = 128 VGPR total, causing massive spilling.

**Lesson**: Multi-wave is not universally better. For kernels with high VGPR pressure and many barriers, single-wave avoids synchronization overhead and register spilling. Always benchmark both configurations.

---

## 6. Not hoisting loop-invariant global loads wastes 33% performance

**Trap**: The inner loop of the dQ kernel iterates over KV-tiles, reloading Q, dO, LSE, and Delta on every iteration. These data are invariant for a fixed Q-tile, but the compiler will not hoist them automatically because they have global memory side effects.

**Result**: After manually hoisting outside the loop, dQ improved from 52.43 TFLOPS to 78.23 TFLOPS (+49.2%, latency -33.0%).

**Why**: Each iteration re-issues 21 global load instructions (Q: 8, dO: 8, LSE: 1, Delta: 1; mask is variable and not counted). The latency of these loads cannot be hidden by a single wave. After hoisting outside the loop, the values are kept in VGPRs (~40 VGPR), significantly reducing HBM bandwidth pressure.

**Lesson**: The FlyDSL/MLIR compiler does not automatically perform loop-invariant code motion (LICM) on global memory accesses. Any loop-invariant global load must be manually hoisted. This is the single largest optimization opportunity in the backward kernel.

---

## 7. dS register bypass requires precise matching of MFMA register layout

**Trap**: The dS computation produces an f32 accumulator, which needs to be converted to bf16 and then used as the B operand of the dQ GEMM. The intuitive approach is to write to LDS and then read back (LDS repack). However, if the register layout of dS_acc happens to match the layout of the GEMM B operand, the LDS roundtrip can be skipped.

**Result**: After analyzing the MFMA32x32x8 layout and confirming a match, we directly perform f32→bf16 truncation + pack in registers, saving 4 LDS writes + 4 LDS reads per iteration, yielding +0.76%.

**Why**: MFMA32x32x8 C output layout: `lane_mod_32 = B operand's lane_mod_32 = C's i dimension`. dS comes from `mfma_acc(K_a, Q_b, s_acc)` where B=Q, so `lane_mod_32 = q_row`. The dQ GEMM is `mfma_acc(KT_a, dS_b, dq_acc)` where B=dS, requiring `lane_mod_32 = q_row` — a perfect match! r=0..3 maps to consecutive k_col values tolerant of direct packing into v4bf16.**Lesson**: Before attempting register bypass, you must fully derive the lane-to-element mapping of the MFMA C output, and verify that (lane_mod_32, r) is perfectly aligned with the (row, k-pack) of the target GEMM operand. P^T and dS^T in the dK+dV kernel cannot use bypass (lane_mod_32 = q_row, but the B operand requires lane_mod_32 = k_row).

---

## 8. All Barriers in a 1-wave Kernel Are Redundant — Removing Them Gains +5-7%

**Trap**: The backward kernel has multiple `__syncthreads()` barriers protecting LDS write-after-read (WAR) and read-after-write (RAW) dependencies (P^T write → P^T read, cooperative load → LDS read, dS write → dS read). These barriers are required under multi-wave execution.

**Result**: Removing all barriers under 1-wave (BLOCK_SIZE=64): dQ +3.0%, dK+dV +5.4%.

**Why**: BLOCK_SIZE=64 = 1 wavefront = 64 lanes. Within the same wavefront, LDS writes are automatically visible to subsequent LDS reads (no out-of-order execution within a wavefront; VALU/LDS instructions complete in program order). Barriers only matter for synchronization between multiple wavefronts. Under 1-wave, `__syncthreads()` becomes pure overhead (~20 cycles stall per barrier).

**Lesson**: For a 1-wave kernel (BLOCK_SIZE ≤ 64), all LDS barriers are redundant. You can safely remove them, but always verify correctness first. The forward kernel uses BLOCK_SIZE=256 (4 waves), so barriers must be retained there.

---

## 9. Dead LDS Buffer Causes -15% Performance Loss (No Error, No Crash)

**Trap**: When copying code from the dK+dV kernel to the dQ kernel, the allocation code for the ``lds_dot`` buffer (used for the dO^T LDS transpose) was copied along with it, but no code in the dQ kernel uses this buffer. It compiles fine, runs fine, and produces correct results.

**Result**: After removing ``lds_dot``, dQ improved from 44.51 TFLOPS to 52.43 TFLOPS (+17.8%, execution time -15.1%).

**Why**: ``lds_dot`` occupies ~4.3 KB of LDS. The FlyDSL allocator allocates LDS at the highest watermark; the excess LDS usage reduces the kernel's occupancy (the number of wavefronts that can run concurrently), but produces no compilation warnings or runtime errors.

**Lesson**: When copying code from another kernel, always verify that every LDS buffer allocation is actually used. FlyDSL will not warn about unused LDS allocations. A simple ``grep -n 'lds_dot' kernel.py`` can reveal the problem.

---

## 10. Strided Scalar LDS Reads Can Eliminate Transpose Buffers (dQ lds_kt: -37%)

**Trap**: The K^T transpose buffer (lds_kt, 4,352 bytes) seems necessary to provide the transposed K data for the dQ MFMA B operand. K is stored row-major in lds_k, but the MFMA requires K^T with the k_row dimension packed into 4 consecutive bf16 elements per lane.

**Result**: Eliminating lds_kt and reading K^T via 4 strided scalar loads from lds_k: LDS 13,056 → 8,704 bytes, occupancy 5 → 7, dQ 4.3 ms → 2.93 ms (-37%).

**Why**: lds_k stores K[k_row, d_col] at lds_k[k_row * K_STRIDE + d_col]. K^T[d_col, k_row] = K[k_row, d_col] can be read by iterating over k_row at a fixed d_col — that is simply stride-K_STRIDE scalar reads from lds_k. Four ``vector.load_op(v1_type, lds_k, [k_row * K_STRIDE + d_pos])`` replace one ``vector.load_op(v4_type, lds_kt, [vector_idx])``. The 3 extra LDS instructions per MFMA are trivially hidden by the occupancy jump from 5 to 7 waves per SIMD. The same technique does NOT work for P^T/dS^T in dK+dV because P resides in registers (MFMA accumulator), not in LDS — there is no buffer to stride-read from.

**Lesson**: Before allocating a separate transpose buffer, check whether the source data is already present in another LDS buffer in a different layout. Strided scalar reads trade 3 extra LDS ops per MFMA for the entire transpose buffer's LDS footprint. The break-even point is when the saved LDS crosses an occupancy threshold.

---

## 11. Pre-reading LDS to Registers Destroys the MFMA–LDS Overlap

**Trap**: To eliminate lds_pt (2,176 bytes, occupancy 4 → 5), pre-read all P^T/dS^T data from lds_dot into registers before overwriting lds_dot with new data. This avoids needing a separate lds_pt buffer.**Result**: Full pre-read: 7.66ms→7.91ms (+3.3% regression). Partial pre-read (only d_chunk=0 since P^T only overwrites 32 rows): same 7.91ms regression.

**Why**: The current code reads P^T/dS^T from lds_pt inline between MFMA instructions. The hardware overlaps these LDS reads with MFMA execution (LDS and MFMA use different pipelines). Pre-reading all packs BEFORE the MFMA chain creates a serial LDS read phase that stalls the MFMA pipeline at the start. The occupancy gain 4→5 provides ~25% more latency hiding, but the added serial LDS phase costs more than this gain. The math: 8 LDS vector reads at ~4 cycles each = 32 cycles of serial LDS, but when spread across MFMA gaps (64 cycles per MFMA), they cost 0 cycles (fully hidden).

**Lesson**: When LDS reads are interleaved between MFMAs, the hardware pipeline overlap hides LDS latency for free. Pre-reading to consolidate LDS access into a single phase breaks this overlap. Only pre-read when the occupancy gain crosses a major threshold (e.g., 2→4, not 4→5).

---

## 12. Redundant global loads hiding in cooperative load + MFMA pack pattern

**Trap**: dK+dV kernel loads dO data from global memory TWICE per iteration: once via cooperative load to lds_dot (for dO^T transpose), and once as 8 direct VMEM loads for dP GEMM B-operand (do_b_packs). This double-load is non-obvious because the two loads serve different MFMA operand layouts.

**Result**: Eliminating the 8 direct loads by reading dO B-operand from lds_dot: 7.66ms→6.52ms (-14.9%). Required reordering dP GEMM before Q^T→lds_dot write (so lds_dot still holds dO^T).

**Why**: PMC profiling (rocprofv3) revealed 172 VMEM instructions per wave, with 60% of wave time spent waiting (latency-bound). 8 VMEM/iter × 5.5 avg iterations = 44 redundant VMEM loads per wave. The dP GEMM B-operand needs lane_mod_32=q_col with 4 packed d-elements at stride DOT_STRIDE — exactly what strided scalar reads from lds_dot provide. By reordering the dP GEMM before Q^T overwrites lds_dot, the dO^T data is still valid in LDS, and the strided reads (4× v1 loads per MFMA) cost ~16 LDS cycles vs ~4,000 cycles of VMEM latency saved.

**Lesson**: When the same data is needed in two different MFMA operand layouts, check if one layout is already in LDS from a cooperative load. Strided scalar LDS reads can often substitute for direct global loads when the data is accessible at a different stride. Profile with PMC counters (SQ_INSTS_VMEM, SQ_WAIT_ANY) to identify redundant global loads — they won't be obvious from code review alone.

---

## Quick reference: do vs don't

| Do | Don't |
|----|-------|
| block_n=32, BLOCK_SIZE=64 (1 wave) | block_n=64 (drops rows) |
| Single-pass dK+dV in one loop | Two-pass split (recompute penalty) |
| Separate dQ kernel (host-side) | dQ in inner loop (+8 MFMA + 32 atomic/iter) |
| LDS transpose for dO^T and Q^T | Direct global load for A operand (strided) |
| 1 wave per workgroup | 4 waves (3× slower due to barriers + spill) |
| Correctness check before trusting benchmarks | Trust faster time without checking outputs |
| Manual LICM all loop-invariant global loads | Rely on MLIR/LLVM to auto-hoist global loads |
| Remove all LDS barriers under 1 wave | Remove barriers under multiple waves (data race) |
| Analyze MFMA register layout before deciding to bypass LDS | Blindly attempt register bypass (layout mismatch) |
| Check if all LDS buffers are actually used | Copy code from other kernels without checking |
| Do not blindly port scheduling hints from forward kernel | Blindly port scheduling hints from forward kernel |
| Strided scalar reads replace separate transpose buffers (lds_kt, dO packs) | Allocate separate LDS buffers for every MFMA operand layout |
| Inline LDS reads between MFMAs (leverage pipeline overlap) | Pre-read all LDS data into registers before executing MFMA chain |
| PMC profiling to find redundant VMEM (SQ_INSTS_VMEM, SQ_WAIT_ANY) | Rely solely on code review Quotes to find performance issues |
