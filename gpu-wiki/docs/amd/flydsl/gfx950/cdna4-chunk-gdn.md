# FlyDSL Chunk-GDN Optimization (MI355X / gfx950)

Applicability: backend: flydsl; hardware: amd; topic: reference


**Last updated**: 2026-06-30

From 2.64x to **0.78x** versus the Triton complete-pipeline comparison
baseline. Chunk-GDN (Gated Delta Network) forward pass consists of 5 kernels,
with the FlyDSL implementation surpassing that baseline in full pipeline
performance on MI355X (CDNA4).

Pitfall notes: `pitfalls/amd/flydsl/chunk-gdn-pitfalls.md`
Supporting code: `reference-kernels/amd/cdna4/flydsl/FlyDSL/chunk_gdn_*.py` (5 kernels + pipeline)

---

## 1. Pipeline Overview

5 kernels execute in order:

```
g_cumsum = cumsum(g) # block
A = kkt_solve(k, g_cumsum, beta) # k@k^T + gating +
w, u = recompute_wu(k, v, beta, A, g) # w u
h, v_new = fwd_h(k, w, u, g_cumsum) # h + v fix
o = fwd_o(q, k, v_new, h, g_cumsum) # outputcompute
```

### Parameters

| Config | H | Hg | K | V | BT | Description |
|--------|---|---|---|---|---|-------------|
| TP2 | 32 | 8 | 128 | 128 | 64 | Current default |
| TP1 | 64 | 16 | 128 | 128 | 64 | H/Hg doubled |

Note: H_SIZE/Hg_SIZE are **compile-time constants** for each kernel. Switching TP requires patching the source code and recompiling.

---

## 2. Performance Summary

### TP2 (H=32, Hg=8) — benchmarked with `triton.do_bench` min (us)

| Kernel | T=4K | Ratio | T=16K | Ratio | T=65K | Ratio | T=262K | Ratio |
|---|---|---|---|---|---|---|---|---|
| cumsum | 55 | 8.60x | 55 | 4.11x | 53 | 1.39x | 52 | **0.30x** |
| kkt_solve | 49 | **0.82x** | 125 | **0.95x** | 432 | **0.92x** | 1641 | **0.90x** |
| recompute_wu | 40 | **0.87x** | 131 | **0.85x** | 460 | **0.76x** | 1841 | **0.79x** |
| fwd_h | 132 | **0.78x** | 526 | **0.79x** | 1976 | **0.74x** | 8367 | 1.00x |
| fwd_o | 113 | 2.36x | 182 | 1.14x | 638 | **0.94x** | 2484 | **0.81x** |
| **TOTAL** | 390 | 1.18x | 1019 | **0.90x** | 3560 | **0.80x** | 14385 | **0.91x** |

### TP1 (H=64, Hg=16) — benchmarked with `triton.do_bench` min (us)

| Kernel | T=4K | Ratio | T=16K | Ratio | T=65K | Ratio |
|---|---|---|---|---|---|---|
| cumsum | 57 | 6.26x | 51 | 2.39x | 54 | **0.76x** |
| kkt_solve | 49 | **0.65x** | 134 | **0.56x** | 472 | **0.52x** |
| recompute_wu | 37 | **0.41x** | 138 | **0.40x** | 471 | **0.39x** |
| fwd_h | 243 | 1.04x | 925 | 1.03x | 4108 | 1.04x |
| fwd_o | 113 | 1.20x | 204 | **0.65x** | 713 | **0.53x** |
| **TOTAL** | 500 | 0.99x | 1452 | **0.80x** | 5818 | **0.78x** |

### Key Observations

1. **kkt_solve and recompute_wu have a massive advantage at TP1** (0.39x-0.52x), because doubling H doubles the CU utilization of these two kernels, and FlyDSL's MFMA tiling is more efficient than Triton's at high H.
2. **fwd_h slightly loses at TP1** (1.04x), the only kernel-TP combination that consistently loses.
3. **cumsum has a large gap at small T but does not affect the overall performance**, and surpasses Triton at large T.
4. **fwd_o has a large gap at small T** (2.36x), but significantly outperforms at large T.

### Optimization Journey Summary

| Stage | Description | Total (us) | vs Triton comparison |
|-------|-------------|-----------|-----------|
| V0 | All kernels naive implementation | 11840 | 2.64x |
| V1 | fwd_o: BV=128, 1wf | 5744 | 1.38x |
| V2 | recompute_wu: A→LDS, BV=BK=128, vec8 | 5314 | 1.19x |
| V3 | kkt_solve: vec8 k staging, MFMA mat_mul | 4928 | 1.11x |
| V4 | fwd_o: load-ahead + grid reorder | 4201 | 1.04x |
| V5 | recompute_wu: row-major vec8 + ds_read_tr | 3952 | 1.02x |
| V6 | output_final_state=True (fair comparison) | 3794 | **0.84x** |
| V7 | fwd_h batched IfOps + cumsum wave scan | 3779 | **0.84x** |
| **V8** | **TP1/TP2 generalization sweep** | **3560 (TP2@65K)** | **0.80x** |

---

## 3. Optimization Techniques per Kernel

### 3.1 fwd_h — 4-LDS Double Buffering + O=3 + Pre-Load

fwd_h is the hottest kernel in the pipeline (~55% of total time), implementing the recurrence update for chunked linear attention.

| Parameter | Value |
|------|---|
| Block | 256 threads (4 wavefronts) |
| BV | 16 |
| K | 128 |
| Grid | (V/BV=8, N*H) |
| MFMA/iter | 8 (w·h: 4, k^T·v_new: 4) |
| Barriers/iter | 2 (barrier 3 removed) |
| LDS | 43 KB (4× k-LDS + h-LDS + v_new-LDS) |

#### Algorithm

Per chunk (BT=64 timesteps):
1. **Store h**: Write current h state to global and LDS
2. **w·h MFMA**: Compute matrix multiplication of w and h to get the correction term
3. **v_new = v - w·h**: Compute the corrected value, apply gating
4. **Stage v_new to LDS**: Write gated v_new to LDS
5. **k^T · v_new MFMA**: Compute matrix multiplication of k transpose and gated v_new, update h

#### Performance Evolution

| Version | Description | Time (us) | vs Triton comparison |
|------|------|-----------|-----------|
| Triton comparison ref | chunk_gated_delta_rule_fwd_kernel_h | 2241 | 1.00x |
| V1 O=2 | 1 wavefront | 8289 | 3.69x |
| V2 O=2 | 4 wavefronts | 5665 | 2.53x |
| V3 O=2 | Row-major k-LDS | 5394 | 2.39x |
| V5 O=2 | Col-major k-LDS + 3 barriers | 5489 | 2.45x |
| **V5 O=3** | **Col-major k-LDS + LLVM O=3** | **3136** | **1.40x** |
| V7 O=3 | Pre-load w/v/g before barrier | 3158 | 1.41x |
| V8 O=3 | SW-pipeline k via iter_args | 3888 | 1.74x ↑ |
| **V11 O=3** | **4 separate k-LDS + no barrier 3 + pre-load** | **2183** | **0.97x** |

#### Core Technique 1: LLVM O=3 Monkey-Patch (42% Speedup)

FlyDSL's `RocmBackend.pipeline_fragments()` hardcodes the optimization level for `rocdl-attach-target` to O=2. O=3 enables LLVM's partial wait optimization (`vmcnt(N)`, `lgkmcnt(N)`), allowing global loads and LDS operations to partially overlap.

```python
from flydsl.compiler.backends.rocm import RocmBackend

_orig_pipeline = RocmBackend.pipeline_fragments
def _patched_pipeline(self, *, compile_hints=None, **kw):
    if compile_hints is None: compile_hints = {}
    frags = _orig_pipeline(self, compile_hints=compile_hints, **kw)
    return [f.replace('O=2', 'O=3') if 'rocdl-attach-target' in f else f for f in frags]
RocmBackend.pipeline_fragments = _patched_pipeline
```

**ISA Difference (O=2 vs O=3)**:
- O=2: `s_waitcnt vmcnt(0) lgkmcnt(0)` — full wait, all loads must complete
- O=3: `s_waitcnt vmcnt(3) lgkmcnt(1)` — partial wait, allowing instruction-level parallelism

**Lesson**: `waves_per_eu`, `maxnreg`, `fast_fp_math` in `compile_hints` have no additional effect on top of O=3. O=3 is the only critical compilation option.

#### Core Technique 2: Column-Major k-LDS Layout

Storing k in LDS in column-major order (transposed) allows Phase 5's k^T · v_new MFMA to directly `ds_read_b128` read contiguous data.

**Row-major (V3)**:
- Global load: `buffer_load_b128` (contiguous reads)
- LDS write: `ds_write_b128` (write by row)
- LDS read (Phase 5): requires 32 `ds_read_u16` scattered reads

**Column-major (V5)**:
- Global load: `buffer_load_b128` (contiguous rows)
- LDS write: 32 `ds_write_b16` scattered writes (write to transposed positions)
- LDS read (Phase 5): `ds_read_b128` (contiguous column reads)

```python
K_LDS_STRIDE = 72 # BT(64) + 8 padding, bank conflict

# columnmainwrite: k[dk][t] layout
for j in range_constexpr(8):
    k_elem = vector.extract(k_vec, static_position=[j], dynamic_position=[])
    lds_k_off = (dk_start + j) * K_LDS_STRIDE + t_local
    k_elem_vec = vector.from_elements(v1bf16_type, [k_elem])
    vector.store(k_elem_vec, lds_k[kc], [to_idx(lds_k_off)])
```

**Key Insight**: Under O=3, scatter writes (`ds_write_b16`) are preferable to scatter reads (`ds_read_u16`), because writes are issued before the barrier with no data dependencies passive, allowing LLVM to schedule them freely; whereas scatter reads occur after the barrier, with each read result serving as input to subsequent MFMA operations, forming a dependency chain.

#### Core Technique 3: 4-LDS Double Buffering + Barrier 3 Removal (30% Speedup)

This is the largest single optimization. V5 has 3 barriers, where barrier 3 protects k-LDS read/write conflicts.

**Solution**: 4 independent SmemPtr memrefs implement k-LDS double buffering:

```python
# 4 independent SmemPtr(2 kc × 2 ping-pong)
lds_k_kc0_buf0 = SmemPtr(lds_base_val, offset_kc0_buf0, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc1_buf0 = SmemPtr(lds_base_val, offset_kc1_buf0, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc0_buf1 = SmemPtr(lds_base_val, offset_kc0_buf1, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc1_buf1 = SmemPtr(lds_base_val, offset_kc1_buf1, T.bf16, shape=(LDS_K_ELEMS,)).get()

# row buffer (scf.IfOp, uniform branch, none divergence)
is_buf0 = arith.cmpi(arith.CmpIPredicate.eq,
                     arith.remui(i_t_i32, arith.constant(2, type=T.i32)),
                     arith.constant(0, type=T.i32))
```

**Why it works**: After removing barrier 3, LLVM sees a large scheduling region from barrier 2 to the next iteration's barrier 1. Phase 5's MFMA can overlap with Phase 1's k global loads.

**Key Pitfall**: Must use 4 **independent** SmemPtr memrefs, not a single large memref + dynamic offset. See pitfalls documentation for details.

#### Core Technique 4: Pre-load w/v/g Before Barrier

In Phase 1, before k's global load + LDS scatter write, simultaneously issue w/v/g global load requests. These loads naturally complete during the barrier wait.

Pre-loading alone is nearly ineffective when barrier 3 is present (V7: 3158 us ≈ V5: 3136 us). But combined with barrier 3 removal, the synergy is significant (V11: 2183 us).

#### ISA Comparison (V11 vs Triton)

| Metric | FlyDSL V11 | Triton |
|--------|-----------|--------|
| Time | 2183 us | 2241 us |
| MFMA/iter | 8 | 16 (software-pipelined) |
| Barriers/iter | 2 | ~9 |
| VGPR | ~70 | ~100 + 16 accvgpr |
| Key Technique | O=3 partial waits + barrier removal | num_stages=2 SW pipeline |

#### Failed Optimization Attempts

**SW-Pipeline k via iter_args (V8: Rollback)**: Moved k's global load to the previous iteration's Phase 5, passing via `scf.for_`'s `iter_args`. Added 4 `v8bf16` iter_args, increasing VGPR pressure. Performance regressed from 3136 us to 3888 us.

**Lesson**: FlyDSL's `scf.for_` iter_args do not pass pointers for free like Triton's `tl.advance`. Each iter_arg is a full register value copy.

**Single Large SmemPtr Memref (V9/V10: Correctness Failure)**: Attempted to use a single `SmemPtr(shape=(LDS_K_ELEMS * 4,))` covering all 4 k-LDS buffers. Even with compile-time constant offsets, it produced incorrect results. This is a known bug in FlyDSL SmemPtr.

**BV=32 (V15: Rollback)**: Increased BV from 16 to 32 to halve the grid. However, MI355X has ~304 CUs, and reducing the grid from 256 to 128 caused CU underutilization, resulting in performance regression.

#### LDS Layout

| Buffer | Elements | Bytes |
|--------|----------|-------|
| lds_bh0 | 1024 | 2048 |
| lds_bh1 | 1024 | 2048 |
| lds_bv | 1024 | 2048 |
| lds_k × 4 | 4608 × 4 | 36864 |
| **Total** | | **43008** |

MI355X LDS per CU: 64 KB. 43 KB occupancy allows 1 workgroup/CU.

### 3.2 fwd_o — BV=128 Single Wavefront + Load-Ahead + Grid Reordering

| Parameter | Value |
|-----------|-------|
| Block | 64 threads (1 wavefront) |
| BV | 128 (vs Triton BV=128) |
| Grid | (B*H, NT, 1) |
| MFMA/iter | 160 (q@h: 64, q@k^T: 32, b_A@v: 64) |

**V0→V1 (7.1x)**: Changed from BV=16/4wf to BV=128/1wf, eliminating 8x redundant q@k^T computation.

**V4 (1.27x)**: 4wf + load-ahead pipelining + grid reordering (B*H on grid_x), improving L2 cache locality. Eventually switched back to 1wf because MI355X performs better with 1wf.

**Key**: ds_read_tr_b64 (gfx950 hardware transposed LDS read) for B operand, avoiding manual transposition.

### 3.3 recompute_wu — Row-Major Vec8 + ds_read_tr + Grid Reordering

| Parameter | Value |
|-----------|-------|
| Block | 256 threads (4 wavefronts) |
| BV, BK | 128, 128 |
| Grid | (B*H, NT, 1) |

**V2 (2.3x)**: Changed A matrix from global reads to row-major Vec8, improving memory access efficiency. BV and BK increased from 64 to 128.**V5 (1.4x)**: Switched to row-major vec8 staging (buffer_load vec_width=8) + ds_read_tr for reading B operand + grid reordering (B*H on grid_x).

**Key insight**: Row-major storage enables naturally aligned global loads (128-byte line), and ds_read_tr performs hardware transpose during MFMA reads, avoiding scatter write overhead.

### 3.4 kkt_solve — Vec8 k Staging + MFMA mat_mul

| Parameter | Value |
|------|---|
| Block | 64 threads (1 wavefront) |
| BC | 16 (sub-chunk) |
| Grid | (NT, B*H, 1) |

4 BC=16 sub-chunks compute k@k^T (10 MFMA blocks), then gating + beta + 16×16 triangular solve (Hillis-Steele style scf.for_ row-by-row unrolling).

**V3**: Changed from scalar k staging to vec8 global load + LDS staging, with mat_mul helper function encapsulating MFMA 16×16 matrix multiplication.

### 3.5 cumsum — ds_bpermute Wave Scan

| Parameter | Value |
|------|---|
| Block | 256 threads (4 wavefronts, 4 heads/block) |
| Grid | (NT, H/4, 1) — dynamic NT |

Hillis-Steele prefix sum via `rocdl.ds_bpermute` (zero-barrier communication within wavefront). Each wavefront processes 64 timesteps for one head.

**Note**: The NT dimension of the grid uses a dynamic value (previously hardcoded MAX_NT=1024, which limited T<=65K).

---

## 4. General Optimization Takeaways

### 4.1 O=3 Monkey-Patch Is Universally Applicable

All 5 kernels should use the O=3 monkey-patch. The effect varies by kernel:
- fwd_o: **42% speedup** (largest beneficiary)
- fwd_h: ~0% (ISA already optimally scheduled)
- Other kernels: 5-15% varying

### 4.2 Grid Dimension Has Significant Performance Impact

- fwd_o and recompute_wu use `(B*H, NT, 1)` instead of `(NT, B*H, 1)`, because adjacent CTAs share q/k data, resulting in higher L2 cache hit rate
- fwd_h uses `(V/BV, N*H)` because splitting across the V dimension is the primary source of parallelism
- At small T, CUs are underutilized (e.g., T=4K, NT=64, total grid blocks may be < 304 CUs), and performance is affected by launch overhead

### 4.3 Applicable Scenarios for ds_read_tr16_b64

The hardware-transpose LDS read `ds_read_tr16_b64` on gfx950 is suitable for MFMA B operand reads (requiring column-contiguous layout), **but not for all scenarios**:

- **Applicable**: B operand in fwd_o and recompute_wu (row-major storage, transposed during MFMA read)
- **Not applicable**: k reads in fwd_h (scatter write is better in fwd_h because it occurs before the barrier Myth 3 is latency-hidden)

Key criteria: whether the scatter operations are before or after a barrier. Scatter before a barrier is free (LLVM freely schedules it), while scatter after a barrier is on the critical path.

### 4.4 TP Scaling Limitations

The H_SIZE/Hg_SIZE for all kernels are **module-level compile-time constants**, captured in the `build_*()` closure. Switching TP requires:
1. sed replacement of constants across all kernel source files
2. Deleting `__pycache__`
3. Restarting the process (`_fn = None` cache prevents recompilation)

In the future, H/Hg should be changed to launch parameters or `@flyc.kernel` compile-time parameters.

### 4.5 Benchmark Methodology

- `triton.testing.do_bench(fn, return_mode="min")` is the standard approach, including warmup + multiple measurements
- At small T, kernel launch overhead accounts for a high proportion; both FlyDSL and Triton are affected
- Some kernels OOM at TP1 (T=262K, H=64); memory limits must be noted

### 4.6 MFMA Usage Pattern (fwd_h)

8 MFMAs per iteration (`mfma_f32_16x16x32_bf16`):

| Phase | Purpose | MFMA Count | A Source | B Source |
|-------|------|-----------|--------|--------|
| Phase 2 | w·h | 4 (2 kc × 2 halves) | w (global pre-load) | h (LDS) |
| Phase 5 | k^T·v_new | 4 (2 kc × 2 halves) | k (LDS) | v_new (LDS) |

Each MFMA processes a 16×16 output tile, K_reduce=32 (bf16). K=128 is split into 2 kc blocks (64 each), each block further divided into 2 32-wide halves → 4 MFMAs per phase.

---
- [FlyDSL Programming Guide](../flydsl-programming-guide.md)
- [FlyDSL Kernel Authoring](../flydsl-ref-kernel-authoring.md)
- [MFMA Instruction Selection](../../common/hands-on/mfma-instruction-selection.md)
- [LDS Bank Conflict Optimization](../../common/lds-bank-conflict-optimization.md)
- [Software Pipelining](../../common/hands-on/software-pipelining.md)
- [SmemPtr Pitfalls](../pitfalls/chunk-gdn-pitfalls.md)


## Related

- [Flash Attention GQA D=256 Optimization on MI355X (CDNA4 / gfx950)](cdna4-flash-attention-gqa-d256.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Software Pipeline Depth Optimization](../../../nvidia/common/software-pipeline-depth-optimization.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
