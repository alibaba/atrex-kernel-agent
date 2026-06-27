# Triton Fused RMSNormGated (bf16, GDN post) on RTX PRO 5000 (sm_120)

> **Path-of-record for vLLM `_deltanet_post`-style epilogues on Blackwell-GeForce.**
> Cross-arch baseline: see CuTeDSL real-fusion attempt
> [`v3-fa-fusion-deferred-plan.md`](../../cutedsl/sm120/v3-fa-fusion-deferred-plan.md)
> (blocked on cluster cutlass-DSL ≥ 4.5; tracked separately).

---

## §0 Punchline

| Path | norm+gate kernel | norm+gate + fp4_quant | end-to-end deltanet_forward | vs eager |
|---|---|---|---|---|
| Eager (vLLM stock `_deltanet_post`) | 1404.66 us | 1411.62 us | 2402.17 us | 1.00× |
| **V3 (this kernel)** | **107.96 us** | **129.02 us** | **1112.36 us** | **2.16× e2e** |
| Speedup vs eager | **13.01×** | **10.94×** | **2.16×** | |

**Stop condition met** — V3 reaches **122.6 % of the sm_120 D2D memcpy ceiling**
(108 us measured vs 133 us theoretical 1:1 R/W ceiling). The kernel's actual
read:write ratio is 2:1 (100 MB read + 48 MB write = 148 MB), which beats the
balanced-traffic ceiling — there is no headroom left in this standalone shape.

The remaining **88 %** of end-to-end time is the upstream `chunk_gated_delta_rule`
(Triton from FLA), which cannot be improved without monkey-patching the FLA
kernel itself (deferred to Phase 2 — see §7).

---

## §1 Target hardware & shape

- **GPU**: NVIDIA RTX PRO 5000 / 4000 Blackwell-GeForce (`sm_120`, 110 SMs, 72 GB)
- **Stack**: PyTorch 2.11 + CUDA 13.0, Triton 3.x bundled with torch
- **Workload**: `attn.py::_deltanet_post`'s RMSNormGated + SiLU(z) gating block
  (consumed by `scaled_fp4_quant` + `cutlass_scaled_fp4_mm` downstream — those
  remain unchanged).
- **Canonical shape** (only sweep target):
  - `core_out`: bf16 `[N=6144, H_V=32, D=128]`
  - `z`:        bf16 `[N=6144, H_V=32, D=128]`
  - `norm_w`:   bf16 `[D=128]`
  - `out`:      bf16 `[N=6144, H_V*D=4096]`

---

## §2 Algorithm baseline (matches `attn.py::_deltanet_post` lines 75-92)

```python
x        = core_out.float()
var      = x.pow(2).mean(dim=-1, keepdim=True)        # per-head RMSNorm
x_normed = x * rsqrt(var + eps) * norm_w.float()
silu_z   = z.float() * sigmoid(z.float())
out      = (x_normed * silu_z).to(bf16).reshape(N, H_V*D)
```

**Eager profile (reference)**: 1404.66 us = ~8 separate ops on the bf16 tensors
(rsqrt + mean + 4× elementwise mul + sigmoid + cast). Each op reads/writes the
full 50 MB tensor → ~7× redundant DRAM round-trips.

**Memcpy ceiling at this size**: 1110.71 GB/s @ 148 MB working set
→ theoretical lower bound for a single-pass fused kernel ≈ **133 us** (R+W balanced).

---

## §3 Kernel resource footprint (V3 final)

| Resource | V3 value | Notes |
|---|---|---|
| BLOCK_M | **2** | Sweep critical knob; see §4 |
| num_warps | 4 | Sweep insensitive (2 also tied) |
| num_stages | 3 | Marginal (108.6 → 108.5 us vs ns=2) |
| shared mem / program | 512 B | Below sm_120's 99 KB cap; not a constraint |
| `st.local` / `ld.local` | **0** | Verified via PTX grep across BM ∈ {1,2,4,8,16}; **no register spill** |
| Grid (canonical) | 6144 / 2 = 3072 programs | Waves ≈ 3072 / 110 ≈ 27.9 |
| LDG per program | ~17 LDG.E.128.cg | (2×128 / 8 elems × 2 inputs) — manageable |

---

## §4 Optimization journey

### V1 — naive baseline
- BLOCK_M=8, num_warps=4, num_stages=2. Default cache mode, no vec hints.
- Output: **420.1 us** kernel-only. e2e 1457 us = **1.66× over eager**.
- Already a 3.35× speedup over the 8-kernel eager chain (one pass vs many).

### V2 — cache hint + vectorization hint (Agent B's P0 recipe)
- Added `cache_modifier=".cg"` on `core_out` / `z` loads (bypass L1) and
  `cache_modifier=".ca"` on the small `norm_w` load.
- Added `tl.multiple_of(d_off, 8)` + `tl.max_contiguous(d_off, 128)` to force
  Triton's codegen toward LDG.E.128.
- **PTX verified the hints landed**: `ld.global.cg.v4.b32` emitted as expected.
- **Performance**: **422.1 us** — i.e. **0 % gain**.
- Lesson: hints can be necessary-but-not-sufficient. See pitfalls
  [#1](../../../../pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md#1-cachemodifiercg-ldg128-hints-can-land-in-ptx-with-zero-performance-gain).

### V3 probe — falsify the "register spill" hypothesis
- Hypothesis (Agent B): BM=8 packs `8 × 32 × 128 × 2 inputs / 128 threads = 256`
  fp32 values per thread, exceeding sm_120's 255 register cap → spill.
- **PTX grep across BLOCK_M ∈ {1, 2, 4, 8, 16}**:
  - `st.local`: 0 in every config
  - `ld.local`: 0 in every config
- **Hypothesis falsified**. No register spill. The expected per-thread register
  pressure was over-counted because Triton folds reuse.

### V3 sweep — `BLOCK_M / num_warps / num_stages` joint search
Direct measured `kernel_time` decided the next move (since theory had failed twice).

| BLOCK_M | time | vs ceiling | vs V2 |
|---|---|---|---|
| 1 | 108.23 us | 122.9 % | 3.88× |
| **2** | **107.88 us** | **123.3 %** | **3.89×** |
| 4 | 108.57 us | 122.5 % | 3.87× |
| 8 (V2) | 434.11 us | 30.6 % | 0.97× |
| 16 | 2193.83 us | 6.1 % | 0.19× |
| 32 | 2908.91 us | 4.6 % | 0.14× |

Spread = **27× max/min** — tile size is by far the dominant knob.
Chosen `BLOCK_M=2`. `num_warps=4` and `num_stages=3` were minor (≤ 0.5 %).

### V3 final config
```python
BLOCK_M=2, num_warps=4, num_stages=3
```

---

## §5 Final perf vs baseline

| Stage | eager | V3 | speedup |
|---|---|---|---|
| norm+gate (this kernel only) | 1404.66 us | 107.96 us | **13.01×** |
| norm+gate + scaled_fp4_quant | 1411.62 us | 129.02 us | **10.94×** |
| **end-to-end `deltanet_forward`** | **2402.17 us** | **1112.36 us** | **2.16×** |

Bandwidth utilisation: 148 MB / 108 us = **1370 GB/s** ≈ **122 % of the
balanced-traffic memcpy ceiling** (1110 GB/s). The "over-ceiling" effect is
because the kernel is asymmetric: read 100 MB / write 48 MB ≈ 2:1, while
the memcpy ceiling assumes 1:1 — see §6.

Correctness: `x_fp4` and `x_bs` are **100.0000 % bit-exact** vs eager + vLLM
`scaled_fp4_quant`. End-to-end GEMM Frobenius rel-err = **0.0** (identical
output bytes). bf16 intermediate matches eager to max-rel `7.7e-3` (within bf16
rounding noise), 99.9993 % of bytes bit-exact.

---

## §6 Why V1 / V2 were 4× slower — the actual bottleneck

`BLOCK_M=8` had **two independent things going wrong simultaneously**, masking
each other:

1. **Wave tail**: grid = 6144 / 8 = 768 programs / 110 SMs → waves ≈ 7.03,
   leaving a fractional final wave (3 × 110 = 330 progs unused → 30 % of SMs
   idle on tail).
2. **In-flight programs per SM collapse**: each program has ~65 LDG.E.128
   instructions; with only ~7 programs in flight per SM, the scheduler runs
   out of independent work to overlap with DRAM latency. Bandwidth utilisation
   stalls at ~30 % memcpy ceiling.

Reducing `BLOCK_M=2` makes grid = 3072, waves ≈ 27.9 (3 % tail loss), AND
each program now has only ~17 LDG → SM stays saturated with concurrent work.

V2's cache+vec hints **were correct** for the eventual config but had nothing
to attack as long as the SM was idle. Once V3 unblocked the SM, the hints
silently kicked in (the hints are still in V3's code).

---

## §7 Remaining bottleneck & what would close the gap

End-to-end breakdown of V3:

| Component | Time | % of e2e |
|---|---|---|
| `chunk_gated_delta_rule` (FLA Triton) | ~983 us | 88.4 % |
| **fused rmsnorm+gate (this kernel)** | **108 us** | **9.7 %** |
| `scaled_fp4_quant` + `cutlass_scaled_fp4_mm` | ~21 us | 1.9 % |

Phase 2 (deferred): monkey-patch FLA's `chunk_fwd_kernel_o` to write the
post-processed bf16 output **directly inside its epilogue**, eliminating the
`core_out` HBM round-trip (~50 MB) entirely. Projected end-to-end → ~600 us
(another ~1.85× over V3, ~4× over eager). Implementation cost: editing
upstream FLA Triton source.

For pure-Triton standalone post kernels at this shape on sm_120: **V3 is at
the wall**.

---

## §8 Sustained recipe

When implementing similar fused-norm-gating-elementwise kernels on sm_120:

1. **Tune `BLOCK_M` first; theoretical reasoning about register pressure on
   Triton is unreliable** — spills do not happen at the per-thread register
   counts theory predicts (Triton folds reuse aggressively). Always grep PTX.
2. **Default to `BLOCK_M = 1, 2, or 4`** for shapes where each row already
   has thousands of elements. Larger BM only helps when row work is tiny.
3. **Compute waves = grid / SM count**. Keep waves close to integer multiples
   of SM count, OR keep waves ≥ 20 so tail loss < 5 %.
4. **Apply `cache_modifier=".cg"` + `multiple_of` + `max_contiguous` hints
   even when they show 0 % gain at first** — they often unlock the actual
   speedup once another knob (BLOCK_M, num_warps) is tuned. Removing them
   is a regression trap.
5. **Verify "over memcpy ceiling" results by checking the read:write ratio
   of your kernel** vs the ceiling assumption (1:1 D2D memcpy). 122 % ceiling
   is real if your R:W ≠ 1:1.
6. **`num_stages` for memory-bound elementwise is mostly a no-op** at the
   sm_120 SMEM budget; sweeping 1/2/3/4 changes time by < 0.5 % at BM=2.

---

## §9 Related docs

- **Code**: [`reference-kernels/nvidia/blackwell-geforce/triton/gdn_post/fused_rmsnorm_gated_pro5000.py`](../../../../../reference-kernels/nvidia/blackwell-geforce/triton/gdn_post/fused_rmsnorm_gated_pro5000.py)
- **Pitfalls**: [`docs/pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md`](../../../../pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md)
- **CuTeDSL real-fusion (different approach, deferred)**: [`v3-fa-fusion-deferred-plan.md`](../../cutedsl/sm120/v3-fa-fusion-deferred-plan.md)
- **CuTeDSL standalone fused-quant epilogue (3-arch memcpy wall study)**:
  [`sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md`](../../cutedsl/sm120/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md)
- **sm_120 cp.async / cache mode (CuTeDSL specific)**:
  [`sm120-gdn-decode-cpasync-cache-mode.md`](../../../../kernel-opt/nvidia/cutedsl/sm120/sm120-gdn-decode-cpasync-cache-mode.md)
- **NVIDIA `ncu` profiling**: [`ncu-profiling-guide.md`](../../common/ncu-profiling-guide.md)

