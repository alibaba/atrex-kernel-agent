# Gluon Pitfalls

Traps from writing Blackwell kernels in **Gluon** (Triton's explicit-layout,
Blackwell-primitive dialect — `gl.*` / `bw.*`).

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| [sm100-blackwell-primitives-pitfalls.md](sm100-blackwell-primitives-pitfalls.md) | DSA sparse attention (`H=16` MLA) + FP8 top-k indexer, ported Triton→Gluon | sm_100 (B200 Blackwell datacenter) | 6 |

## Cross-Pitfall Summary (sm_100 Gluon)

- **`gl.dot_fma` is ~60-80× slower than `tl.dot`** — it is software FMA with no
  tensor cores. Tensor-core throughput requires `bw.tcgen05_mma` + `bw.alloc_tmem`
  + SMEM staging. `dot_fma` is a correctness scaffold only.
- **`gl.dot_fma` has unforgiving layout/dtype rules** — `k_width=0` (required
  positional) for blocked parents; operands must match the accumulator dtype (no
  auto-upcast).
- **`bw.tcgen05_mma` is structurally uncompetitive at small `H`** — `blockM=64`
  minimum forces ≥75% phantom-row padding at `H=16`; Triton's `wgmma` scheduler
  hides the same padding with ~5× less framework overhead. Gluon pays off only
  when the problem fills the MMA tile.
- **`gl.barrier` is listed but not callable** from `gluon.jit` — use atomic-spin
  or `bw.mbarrier` for sync.
- **Layouts are explicit** — `warps_per_cta` must sum to `num_warps`; 1-D `arange`
  needs a plain `BlockedLayout`; reduction results need `convert_layout` before
  combining.
- **`translator_helpers` is a bring-up tool, not a perf port** — auto-translation
  emits generic layouts + extra `convert_layout`, never reaches `tcgen05_mma`, and
  is net-slower on single-dot FP8. A real Gluon win is hand-authored.

## Cross-Platform Correlation

- Triton-side traps for the same DSA kernels:
  [`../triton/sm100-sparse-decode-split-k-pitfalls.md`](../triton/sm100-sparse-decode-split-k-pitfalls.md)
- Blackwell hardware primitives referenced here (tcgen05/tmem/mbarrier):
  [`../../../kernel-opt/nvidia/common/blackwell/hardware/`](../../../kernel-opt/nvidia/common/blackwell/hardware/)
