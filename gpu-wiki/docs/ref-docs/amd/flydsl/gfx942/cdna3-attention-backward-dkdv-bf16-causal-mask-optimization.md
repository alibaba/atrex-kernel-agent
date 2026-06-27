# FlyDSL Attention Backward dQ + dK+dV (bf16, Causal Mask) on MI308X (gfx942)

Applicability: backend: flydsl; hardware: amd; topic: reference

## Target hardware

- Chip: AMD MI308X (CDNA3, gfx942)
- Peak bf16 MFMA: 206 TFLOPS
- HBM BW: 5.3 TB/s
- CUs: 80, LDS: 64 KB/CU, VGPR: 512/SIMD (4 SIMD/CU)

## Algorithm baseline

Attention backward pass computing dQ, dK and dV gradients with arbitrary additive mask.
dK+dV kernel: Inner loop per Q-tile: GEMM1(S=K@Q^T) → softmax → P LDS transpose → dV MFMA → dP GEMM(V@dO^T) → dS → dS LDS transpose → dK MFMA.
dQ kernel: Grid per Q-tile, inner loop over KV-tiles: GEMM1(S=K@Q^T) → softmax → dS GEMM(dO@V^T) → dQ MFMA.

Test shape: B=1024, H=8, S=326 (padded to 352), D=64, causal mask.

## Kernel resource footprint (final)

### dK+dV kernel
| Resource | Value | Notes |
|----------|-------|-------|
| VGPR | 68 | No spill |
| Accum VGPR | 196 | 4× v16f32 (dv_accs + dk_accs) |
| LDS | ~20 KB | lds_k + lds_v + lds_dot + lds_qt + lds_pt |
| Scratch | 0 | No stack spill |
| Workgroup | 64 threads (1 wave) | BLOCK_SIZE=64 |
| Occupancy | 2 waves/SIMD | Limited by 196 Accum VGPR |

### dQ kernel
| Resource | Value | Notes |
|----------|-------|-------|
| VGPR | ~52 | No spill |
| Accum VGPR | 32 | 1× v16f32 (dq_accs) |
| LDS | ~15 KB | lds_k + lds_v + lds_kt + lds_pt |
| Scratch | 0 | No stack spill |
| Workgroup | 64 threads (1 wave) | BLOCK_SIZE=64 |
| Occupancy | 4 waves/SIMD | Limited by VGPR |

## Optimization journey

### V0 — Initial baseline (11.08 ms, 46.92 TFLOPS, 22.8% peak)
Starting point: single-pass dK+dV kernel with block_n=32, BLOCK_SIZE=64.
Key design decisions inherited from forward kernel: K_PAD=4, rocdl.exp2, MFMA32x32x8 bf16.
Dual LDS transpose for P^T and dS^T to remap MFMA C output layout to B-operand layout.
3 barriers per Q-tile iteration (P write, cooperative load, dS write).

### V1 — Two-Pass split dV/dK (16.23 ms, 32.01 TFLOPS — reverted)
Split single loop into two independent loops: Pass 1 (dV only), Pass 2 (dK only).
Goal: reduce Accum VGPR from 196 to ~80-100, improve occupancy from 2 to 4+ waves/SIMD.
Result: correctness passed but +46% regression. Recomputing S=K@Q^T and P=softmax(S) twice per tile doubled MFMA work, which outweighed occupancy gains.
Lesson: cannot trade compute for occupancy when MFMA is already the bottleneck.

### V2 — Larger block_n=64 (10.78 ms, 57.39 TFLOPS — reverted, correctness failure)
Increased KV-tile from 32 to 64 rows. Benchmark showed 22% TFLOPS improvement.
But correctness check revealed kernel only processes first 32 rows (lane_mod_32 limitation in MFMA32x32x8). Rows 32-63 output zeros.
Lesson: MFMA32x32x8 C output maps lane_mod_32 to row index; block_n > 32 requires inner n_chunk loop.

### V3 — Micro-optimization analysis (no change)
Analyzed: (a) remove wave_id==0 guards (compiler already optimizes), (b) Q prefetch via iter_args (+16 VGPR risk), (c) eliminate cooperative load (impossible — LDS transpose required for non-contiguous A operand), (d) P/dS LDS write batching (j_formula non-contiguous).
Conclusion: performance ceiling reached at ~23% peak for 2 waves/SIMD.
### V4 — Barrier removal for dK+dV (10.509 ms, 49.45 TFLOPS, +5.4%)
Removed all 4 `s_barrier` in the dK+dV inner loop. Under 1-wave (BLOCK_SIZE=64), LDS writes within the same wavefront are automatically visible to subsequent reads, making the barriers completely redundant.

### V5 — dQ baseline (11.052 ms, 35.27 TFLOPS)
Standalone dQ kernel baseline implementation. The grid is partitioned by Q-tiles, with the inner loop iterating over KV-tiles. dQ MFMA: A=K^T (LDS), B=dS (LDS repack). LDS uses 4 buffers: lds_k, lds_v, lds_kt (K^T), lds_pt (dS repack).

### V6 — K^T transpose merged (9.048 ms, 43.08 TFLOPS, -18.1%)
Merged the K^T LDS transpose into the cooperative K load, eliminating the independent transpose pass. Each thread writes to both lds_k[row-major] and lds_kt[col-major] when loading K.

### V7 — Vectorized dS LDS write (8.757 ms, 44.51 TFLOPS, -3.2%)
Vectorized the v4bf16 dS LDS write (4 v4 stores instead of 16 v1 stores), and also removed the wave0-only IfOp guard.

### V8 — Dead LDS buffer removal (7.433 ms, 52.43 TFLOPS, -15.1% 🔥)
Discovered that the lds_dot buffer (copied over from the dK+dV kernel) was completely unused in the dQ kernel. Removing it reduced LDS usage from ~20 KB to ~15 KB, improving occupancy.

### V9 — Loop-invariant hoisting (4.982 ms, 78.23 TFLOPS, -33.0% 🔥🔥)
Hoisted the global loads of Q, dO, LSE, and Delta out of the inner loop. These data remain constant across the entire KV-tile iteration; reloading them each loop iteration wasted significant HBM bandwidth. A single optimization delivered a massive 33% improvement.

### V10 — Softmax FMA fusion (4.899 ms, 79.56 TFLOPS, -1.7%)
Fused the softmax ``s * scale`` + ``- lse`` + ``* log2e`` into ``fma(s, scale_log2e, neg_lse_log2e)``, reducing from 5 VALU/r to 3 VALU/r tallies, saving 32 VALU instructions per iteration.

### V11 — Scale deferral (4.834 ms, 80.63 TFLOPS, -1.3%)
Deferred the dS scale factor from the element-wise multiply in the inner loop to the post-loop dQ store. Saves 16 MulF instructions per iteration.

### V12 — Full barrier removal (4.690 ms, 83.11 TFLOPS, -1.6%)
Removed all barriers in the dQ kernel inner loop (barrier after K+V coop_load + barrier after dS LDS write). Under 1-wave, LDS writes within the same wavefront are automatically visible to subsequent reads.

### V12c — dS register bypass (4.655 ms, 83.74 TFLOPS, +0.76%)
After analyzing the MFMA32x32x8 register layout, discovered that dS_acc's lane_mod_32=q_row perfectly matches the dQ GEMM B operand layout. Performed f32→bf16 truncation and packing directly in registers (bitcast+ShRUI+trunci), bypassing the dS→LDS roundtrip. Eliminated 4 LDS writes + 4 LDS reads per iteration.

---

*The following optimizations were completed in a subsequent session (test shape changed to B=1024, H=8, S=316, D=64, S_PAD=320).*

### V13 — dK+dV: lds_qt → lds_dot merge (7.66 ms dK+dV, -27.1%)
Merged lds_qt and lds_dot into the same buffer (time-division multiplexed): dO^T is written to lds_dot first; after the dV MFMA reads it, Q^T overwrites the same location in lds_dot for use by the dK MFMA. The lds_qt buffer is completely eliminated. LDS reduced from ~20 KB to 15,232 bytes, occupancy improved from 2→4 waves/SIMD (now LDS-limited rather than VGPR-limited).

### V14 — dQ: lds_kt elimination via strided scalar reads (2.93 ms dQ, -37.0% 🔥🔥)
Eliminated the independent K^T LDS buffer (lds_kt, 4,352 bytes) by reading K^T data from the existing lds_k buffer using strided scalar reads. Each MFMA B-operand read uses 4 ``vector.load_op(v1_type, lds_k, ...)`` instead of 1 ``vector.load_op(v4_type, lds_kt, ...)``. LDS reduced from 13,056→8,704 bytes, occupancy 5→7. 4.3ms→2.93ms — the largest single optimization in dQ history.

Key insight: K is stored in lds_k in row-major order (``K[k_row, d_col]``), while K^T needs to be read in ``K^T[d_col, k_row]`` order. The original approach maintained a separate lds_kt transpose bufferarts, writing to both buffers during the cooperative load of K. The new approach reads directly from lds_k with stride=K_STRIDE: ``lds_k[k_row * K_STRIDE + d_pos]``, iterating k_row with stride. Four scalar reads replace one vector read, adding 3 extra LDS instructions/MFMA, but the latency hiding provided by occupancy 7 far outweighs the extra LDS overhead.

### V15 — dK+dV: dO B-operand from LDS (6.52 ms dK+dV, -14.9% 🔥)
Eliminated the 8 global VMEM loads (do_b_packs) for the dP GEMM, instead using strided scalar reads from lds_dotega to load dO^T as the B operand. The key was to reorder the dP GEMM to precede the Q^T→lds_dot write, so that lds_dot still contains the dO^T data at that point.

Before loop body: ... → dV MFMAs → **Q^T→lds_dot** → dP GEMM(V @ dO^T, dO from global) → ...
After loop body: ... → dV MFMAs → **dP GEMM(V @ dO^T, dO from lds_dot)** → Q^T→lds_dot → ...rocprofv3 PMC analysis confirms: among 172 VMEM per wave, 8×5.5=44 are redundant global loads of dO MFMA packs (dO is already stored in lds_dot via cooperative load), and 60% of wave time is VMEM latency waiting. After elimination, VMEM reduces by ~25%, 7.66ms→6.52ms.

### Failed experiments (reverted)
- **sched_dsrd + sched_mfma hints** (V12): -8.2% regression. FlyDSL soft scheduling hints interfere with the compiler's default scheduling.
- **SSA pre-load + sched_mfma(4)** (V12a): -1.4% regression. VGPR pressure increases.
- **Direct V global load** (V8): +14.8% regression. LDS coalesced access > scattered global load.
- **lds_pt reuse lds_v offset** (V5): +1.1% regression. Bank conflict increases.
- **dK+dV ds_swizzle paired store** (iter4): -26% regression. With 1-wave, ds_swizzle latency cannot be hidden.
- **dK+dV wave0 IfOp removal** (V12): 0% change. Compiler already optimizes away.
- **dK+dV lds_pt elimination (full pre-read)** (V15): 7.66→7.91ms (+3.3%) regression. Pre-reading all dO^T and Q^T packs into registers before overwriting lds_dot with P^T adds serial LDS read phases that destroy the natural MFMA-LDS pipeline overlap.
- **dK+dV lds_pt elimination (partial pre-read)** (V15): Same regression. Even pre-reading only d_chunk=0 packs (since P^T only overwrites rows 0-31 of lds_dot) still 7.91ms. The occupancy gain 4→5 is insufficient to offset the added LDS instruction overhead.

## Final perf vs baseline

Test shape: B=1024, H=8, S=316, D=64, S_PAD=320, causal mask.

| Implementation | Time (ms) | Relative |
|---|---|---|
| FlyDSL dQ (V14, lds_kt eliminated) | 2.93 | - |
| FlyDSL dK+dV (V15, dO from LDS) | 6.52 | - |
| FlyDSL dQ+dK+dV kernel-only | 9.45 | 1.00× |
| FlyDSL dQ+dK+dV fast-path (incl. API overhead) | 9.75 | - |
| PyTorch SDPA backward (dQ+dK+dV) | 42.30 | 0.22× |

**4.35× faster than PyTorch SDPA backward**

Previous best (V12c+V4, S=326): dQ 4.655ms + dK+dV 10.509ms = 15.164ms.
Current best (V14+V15, S=316): dQ 2.93ms + dK+dV 6.52ms = 9.45ms. **37.7% faster.**

## Kernel resource footprint (updated)

### dK+dV kernel (V15)
| Resource | Value | Notes |
|----------|-------|-------|
| Arch VGPR | 20 | No spill |
| Accum VGPR | 244 | 4× v16f32 accumulators + intermediate |
| LDS | 15,232 bytes | lds_k(4352) + lds_v(4352) + lds_dot(4352) + lds_pt(2176) |
| Occupancy | 4 waves/SIMD | Limited by LDS (floor(65536/15232)=4) |
| Workgroup | 64 threads (1 wave) | BLOCK_SIZE=64 |

### dQ kernel (V14)
| Resource | Value | Notes |
|----------|-------|-------|
| Arch VGPR | ~24 | No spill |
| Accum VGPR | ~32 | 2× v16f32 (dq_accs) |
| LDS | 8,704 bytes | lds_k(4352) + lds_v(4352), lds_kt eliminated |
| Occupancy | 7 waves/SIMD | floor(65536/8704)=7 |
| Workgroup | 64 threads (1 wave) | BLOCK_SIZE=64 |

## Remaining bottlenecks

1. **lds_pt in dK+dV cannot be eliminated**: P^T and dS^T transpose requires LDS because MFMA C output lane_mod_32=q_row but B operand needs lane_mod_32=k_row. Two pre-read approaches tried and both regressed (serial LDS reads destroy MFMA pipeline overlap).
2. **Latency-bound dK+dV**: PMC analysis shows 60% of wave time is wait cycles. Per-wave MFMA pipeline = 11,264 cycles but total wave lifetime = ~33,000 cycles. Occupancy 4 is insufficient to fully hide VMEM latency (172 VMEM/wave × ~500 cycles latency vs 4 × 11,264 compute slots).
3. **Instruction scheduling gap**: FlyDSL soft scheduling hints (sched_dsrd + sched_mfma) cause regression. True ISA-level scheduling needs CK V3 CoreLoopScheduler or equivalent cycle-level manual scheduling.
4. **Cooperative dO^T transpose overhead**: 16 element-wise LDS stores per iteration for row→column transpose, unavoidable since dO is row-major but MFMA A operand needs column-major q-dimension contiguous.
## What would close the remaining gap

1. **Software prefetching**: Issue Q MFMA pack loads at end of previous iteration (overlap with dK MFMAs). Would require 8 additional v4bf16 iter_args but could hide ~2,000 cycles of VMEM latency.
2. **Fused dQ+dK+dV kernel**: Amortize global loads across all three gradients, but complex due to different loop structures (dQ loops over KV-tiles, dK+dV loops over Q-tiles).
3. **CK V3–style ISA scheduling**: `CoreLoopScheduler` for cycle-level MFMA/VALU/LDS interleaving to reduce dependency stalls.

## Sustained recipe

### dK+dV kernel
1. Use block_n=32, BLOCK_SIZE=64 (1 wave) — do NOT try multi-wave (4 waves → 2.9× slower)
2. K_PAD=4 for LDS bank-conflict avoidance
3. Merge lds_qt into lds_dot (time-share: dO^T first, then Q^T overwrites) — occupancy 2→4
4. **dO B-operand from lds_dot**: Move dP GEMM before Q^T write, read dO^T from LDS with strided scalar reads instead of redundant global loads — saves 8 VMEM/iter, 15% speedup
5. rocdl.exp2 for single-instruction exp2 in softmax
6. Dual LDS transpose for P^T and dS^T (MFMA B-operand remapping) — cannot be eliminated
7. Remove all barriers in 1-wave configuration
8. Pre-issue Q^T global loads (overlap with P^T write + dV MFMAs)

### dQ kernel
1. **Loop-invariant hoisting**: Move Q, dO, LSE, Delta global loads outside KV-tile loop — 33% impact
2. **lds_kt elimination**: Replace K^T LDS buffer with strided scalar reads from lds_k — 37% impact, occupancy 5→7
3. **dS register bypass**: Bitcast+ShRUI+trunci to bypass dS→LDS roundtrip — 0.76% impact
4. **Barrier removal**: All barriers removed in 1-wave configuration
5. **Softmax FMA fusion**: Reduce VALU from 5/r to 3/r
6. **Scale deferral**: Defer dS scale to post-loop dQ store

## Related docs

- Forward kernel optimization: [cdna3-flash-attention-bf16-mask-optimization.md](cdna3-flash-attention-bf16-mask-optimization.md)
- Forward kernel pitfalls: [flash-attn-pitfalls.md](../../../../pitfalls/amd/flydsl/flash-attn-pitfalls.md)
- Backward kernel pitfalls: [attention-backward-dkdv-pitfalls.md](../../../../pitfalls/amd/flydsl/attention-backward-dkdv-pitfalls.md)
- Reference kernel (dK+dV): [attn_bwd_dkdv_mi308x.py](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/attn_bwd_dkdv_mi308x.py)
- Reference kernel (dQ): [attn_bwd_dq_mi308x.py](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/attn_bwd_dq_mi308x.py)
- API integration + arbitrary mask + end-to-end perf: [cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md](cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md)
- API integration pitfalls: [flash-attn-bwd-mask-integration-pitfalls.md](../../../../pitfalls/amd/flydsl/flash-attn-bwd-mask-integration-pitfalls.md)
