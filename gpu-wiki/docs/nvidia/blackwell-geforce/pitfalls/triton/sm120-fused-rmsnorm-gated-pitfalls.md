# sm_120 Triton fused-RMSNormGated pitfalls

Hardware: NVIDIA RTX PRO 5000 / 4000 Blackwell-GeForce (`sm_120`)
Stack: PyTorch 2.11 + CUDA 13.0, Triton bundled with torch
Discovered while writing the V1 → V3 fused-RMSNormGated kernel for vLLM
`_deltanet_post`. Full report:
[`docs/nvidia/blackwell-geforce/ref-docs/triton/sm120-fused-rmsnorm-gated-bf16-optimization.md`](../../ref-docs/triton/sm120-fused-rmsnorm-gated-bf16-optimization.md)

---

## 1. `cache_modifier=".cg"` + LDG.128 hints can land in PTX with **zero** performance gain

**Trap**: After establishing a memory-bound baseline, the textbook P0 move is
"add `cache_modifier=".cg"` on streaming loads + `tl.multiple_of(8)` +
`tl.max_contiguous(128)` on the contiguous dim". You run it, grep PTX,
confirm `ld.global.cg.v4.b32` is emitted, expect 25-35 % bandwidth gain.

**Result**: Measured kernel time **422.1 us → 420.1 us** (V1 → V2 on the BM=8
config). **0 % gain**, despite PTX clearly showing the new instructions.

**Why**: The kernel was not actually L1-saturated and was not actually LDG-width
limited. The real bottleneck was **insufficient in-flight programs per SM**
(BLOCK_M=8 packed too much work into each program → only ~7 programs per SM,
DRAM latency unhideable). Cache hints + vec hints can only optimise the inside
of a program; if the SM has nothing else to overlap with the long-scoreboard
stall, faster individual loads don't help.

**Lesson**: When a P0 textbook hint shows 0 % gain and PTX confirms it landed,
**stop adding more hints**. The bottleneck is structural (tile size, occupancy,
wave count), not microarchitectural. Pivot to `BLOCK_M` / `num_warps` /
grid-shape sweep before more low-level tuning. **Do not remove the hints either**
— they are necessary-but-insufficient and may unlock the gain once the
structural fix lands.

---

## 2. "Per-thread register count > 255 ⇒ register spill" reasoning is **unreliable** on Triton

**Trap**: Compute per-thread register pressure from naive theory:
`(BLOCK_M × H_V × D × 2 inputs) / (num_warps × 32) = bytes/thread`. If this
exceeds `255 × 4 = 1020 bytes`, conclude "register spill, must shrink BLOCK_M".

For our V2 (BM=8, H=32, D=128, num_warps=4): predicted **256 fp32 / thread =
1024 B/thread > 1020 B**. Reasoning concluded "spill very likely".

**Result**: PTX grep across `BLOCK_M ∈ {1, 2, 4, 8, 16}` shows
**`st.local: 0`** and **`ld.local: 0`** in every config. **No spill at any BM**.

**Why**: Triton aggressively folds register reuse — the same physical register
is reused across multiple Python-level intermediates because Triton sees the
data flow. A "per-thread fp32 count" that assumes each Python intermediate
materialises into a separate register over-counts by 2-4×.

**Lesson**: **Always grep PTX for `st.local` / `ld.local` before believing a
register-spill hypothesis.** Even if theory says you should be over-budget,
Triton may have folded enough state to fit. Three commands:

```bash
TRITON_CACHE_DIR=$PWD/.triton_cache python your_script.py
find .triton_cache -name '*.ptx' | xargs grep -c 'st.local\|ld.local'
```

If both counts are 0, the spill hypothesis is dead — investigate occupancy /
wave count / SMEM instead.

---

## 3. `BLOCK_M` sweep can show **27× max/min spread** at the same shape

**Trap**: Once you have a working Triton kernel, it's tempting to keep
`BLOCK_M` at whatever value worked first and tune other knobs.

**Result** (canonical shape [N=6144, H_V=32, D=128] bf16, num_warps=4,
num_stages=2, on sm_120):

| BLOCK_M | time | vs best |
|---|---|---|
| 1 | 108.23 us | 1.00× |
| 2 | 107.88 us | **best** |
| 4 | 108.57 us | 1.01× |
| 8 | 434.11 us | 4.02× slower |
| 16 | 2193.83 us | 20.3× slower |
| 32 | 2908.91 us | **27.0× slower** |

**Why**: Per-program work × in-flight programs per SM is a U-shaped curve.
`BM=1/2/4` saturates the SM with concurrent programs (3072 / 110 ≈ 28 waves);
`BM=8` already collapses concurrency (768 / 110 ≈ 7 waves, partial wave tail);
`BM ≥ 16` further halves concurrency AND likely starts triggering SMEM /
register pressure issues that don't show as spill but stall the scheduler.

**Lesson**: For elementwise/normalisation kernels at "many small rows" shapes
on sm_120, **always sweep `BLOCK_M ∈ {1, 2, 4, 8, 16}` early**. The cost
is one minute, the payoff can be 4-27×. Default to **`BLOCK_M = 1, 2, or 4`**
when each row already has > 1k elements; only go larger if rows are tiny.

---

## 4. "% of memcpy ceiling > 100 %" is **real**, not a measurement artifact, when R:W ≠ 1:1

**Trap**: Standard practice is to declare a memory-bound kernel "done" at
~90 % of D2D memcpy ceiling. When measurements show **122.6 %**, instinct says
"my measurement is wrong" or "the ceiling number is stale".

**Result** (V3 final on sm_120 RTX PRO 5000):
- Measured kernel time: 108 us
- Working set: 100 MB read + 48 MB write = **148 MB** total
- Achieved bandwidth: 148 / (108 × 1e-6) ≈ **1370 GB/s**
- D2D memcpy ceiling at 148 MB: **1110.71 GB/s** (measured separately)
- Apparent utilisation: **123 %**

**Why**: The standard memcpy ceiling assumes balanced 1:1 read:write
(D2D: read N bytes, write N bytes). Our kernel is **2:1 (read 100 MB, write
48 MB)**. HBM controllers achieve higher effective bandwidth for asymmetric
R:W traffic than for the worst-case D2D copy. The 122 % is a real measurement
of higher hardware efficiency — the ceiling number is just inappropriate as a
yardstick for this kernel's traffic profile.

**Lesson**: When checking "% of ceiling", **compute the ceiling for your
kernel's actual R:W ratio**, not just D2D-copy ceiling. Either:
1. Run a synthetic kernel with the same R:W ratio as ceiling, OR
2. Treat any value within 80-130 % of D2D ceiling as "at the wall" — both
   under (different access pattern) and over (better R:W) are explained by
   the asymmetry.

If you go optimising past a 100 %+ result, you'll waste days chasing a
non-existent gap.

---

## 5. Triton `num_stages` is a near-no-op for memory-bound elementwise on sm_120

**Trap**: `num_stages` is the standard knob for hiding DRAM latency via
software pipelining. Default Triton autotune treats it as a primary tuning
parameter (`num_stages ∈ {1, 2, 3, 4, 5}` is common).

**Result** at `BLOCK_M=2, num_warps=4` on sm_120:

| num_stages | time |
|---|---|
| 1 | 113.91 us |
| 2 | 108.63 us |
| 3 | 108.49 us |
| 4 | 108.54 us |

Going from 1 → 2 saves 5 us; 2 → 3 → 4 are within noise (< 0.5 %).

**Why**: At `BLOCK_M=2`, each program is so small (~17 LDG.E.128 total) that
the compiler can already overlap loads with the small amount of compute even
at single-stage. The cp.async pipeline depth doesn't help because there's
nothing to pipeline against — the kernel finishes before deeper stages would
matter.

**Lesson**: For sm_120 memory-bound elementwise/normalisation kernels with
small per-program work, **don't waste autotune budget on `num_stages > 2`**.
Keep `num_stages = 2` as default and spend the budget on `BLOCK_M` (huge
spread, see #3) and `num_warps` (small but real spread, e.g. `num_warps=1`
gave 7× regression in our sweep).

---

## "Use what / don't use what" cheatsheet

| Use | Don't use |
|---|---|
| Sweep `BLOCK_M ∈ {1,2,4,8,16}` first thing on every new kernel | Trust theoretical "register spill" reasoning without PTX grep |
| Grep PTX for `st.local` / `ld.local` to confirm spill | Believe `cache_modifier=".cg"` solves bottlenecks on its own |
| Compute waves = grid / SM count, target ≥ 20 OR integer multiple | Add hints "in case they help" without measuring before/after |
| Compute kernel R:W ratio when checking memcpy ceiling | Treat 100 % memcpy ceiling as a hard cap |
| Keep `num_stages ∈ {2, 3}` and not sweep further | Sweep `num_stages` as if it's a primary knob |

---

## Cross-references

- Optimisation report (full V1 → V3 journey):
  [`sm120-fused-rmsnorm-gated-bf16-optimization.md`](../../ref-docs/triton/sm120-fused-rmsnorm-gated-bf16-optimization.md)
- Final kernel source:
  [`fused_rmsnorm_gated_pro5000.py`](../../../../../reference-kernels/nvidia/blackwell-geforce/triton/gdn_post/fused_rmsnorm_gated_pro5000.py)
- Adjacent CuTeDSL pitfalls on the same sm_120 hardware (different DSL,
  partially overlapping concerns):
  [`docs/nvidia/blackwell-geforce/pitfalls/cutedsl/`](../cutedsl/README.md) — especially
  `sm120-ncu-l1-hit-rate-shared-pollution.md` (another sm_120 measurement
  trap).
