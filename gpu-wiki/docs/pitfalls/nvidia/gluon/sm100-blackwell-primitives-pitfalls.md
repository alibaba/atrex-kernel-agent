# sm_100 Gluon Blackwell-primitive pitfalls

Hardware: NVIDIA **B200 / Blackwell datacenter (`sm_100a`)**
Stack: CUDA 13.x, **Triton 3.7 Gluon** (the explicit-layout, Blackwell-primitive
dialect — `gl.*`, `bw.*`)
Distilled from MLSys26 FlashInfer-contest sessions porting two Triton DSA
kernels (MLA-style sparse attention `H=16`; FP8 top-k indexer) to Gluon. The
recurring theme: **Gluon is not a "sprinkle helpers on Triton" swap** — it is a
hand-authored tensor-core pipeline, and the easy paths are correctness scaffolds,
not performance paths.

---

## 1. `gl.dot_fma` is ~60-80× slower than `tl.dot` — it has no tensor cores

**Trap**: Port a `tl.dot` to Gluon's `gl.dot_fma` as the "matmul op." It compiles
and is bit-correct.

**Result**: Large-shape median **1.30 ms vs 0.016 ms** for the Triton `tl.dot`
original — ~80× slower. `gl.dot_fma` is pure **software FMA**; it never touches
the tensor cores.

**Why**: Gluon exposes the MMA hardware only through the Blackwell primitives.
`dot_fma` is a portable scalar fallback.

**Lesson**: To get tensor-core throughput in Gluon you **must** use
`bw.tcgen05_mma` with tensor-memory descriptors (`bw.alloc_tmem`) + shared-memory
staging. Treat `gl.dot_fma` as a correctness scaffold only — never benchmark
against it as if it were the perf path.

---

## 2. `gl.dot_fma` layout/dtype rules differ from `tl.dot`

**Trap**: Reuse `tl.dot` operand handling for `gl.dot_fma`.

**Result**: Two hard failures —
- `DotOperandLayout(op, parent, k_width=0)` with a `BlockedLayout` parent: any
  `k_width` other than `0` errors ("kWidth not supported when parent is a blocked
  layout"); `k_width` is a **required positional** arg.
- Operand dtypes must all match the accumulator's dtype — `gl.dot_fma` does **not**
  auto-upcast (unlike `tl.dot`). Cast every operand to a common dtype
  (`.to(gl.float32)`) *before* `gl.convert_layout`, or hit
  `FMA.cpp aElem.getType() == tgtTy`.

**Lesson**: Gluon makes layout and dtype explicit and unforgiving. Set
`k_width=0` for blocked parents and pre-cast operands to the accumulator dtype.

---

## 3. `bw.tcgen05_mma` is structurally uncompetitive at small `H`

**Trap**: Having learned #1, port the kernel to the "real" `bw.tcgen05_mma` path
expecting it to beat Triton.

**Result**: Correct, but **6.1× slower** than the Triton `tl.dot` version at
`H=16` (0.095 ms vs 0.016 ms). Even with all follow-on stages (mbarrier launch-tax
removal, clustering) the best case stayed ~4.4× slower.

**Why**: `tcgen05` MMA's minimum `blockM=64` forces **≥75% phantom-row padding**
for `H=16`; the larger `HM=128` tile needs D-chunking to fit SMEM, compounding to
~160 MMA fences/call. Triton's `tl.dot` hides the *same* padding inside its
`wgmma` scheduler with no surfaced fence — a ~5× framework-overhead advantage on
this shape.

**Lesson**: Gluon/`tcgen05` only pays off when the problem fills the MMA tile.
At small head counts (`H=16` MLA), Triton's scheduler wins. Consider Gluon only
after restructuring to a larger effective `blockM` (e.g. fusing N tokens across
batch) — otherwise it's a net regression.

---

## 4. `gl.barrier` is listed but not callable from a `gluon.jit` kernel

**Trap**: Use `gl.barrier` for in-kernel sync — it's in `language/__init__.py`.

**Result**: `AttributeError` at JIT parse time.

**Lesson**: For cross-CTA sync use `gl.atomic_add(ptr, 0, sem="acquire")` in a
spin loop. For true within-CTA barriers, only **`bw.mbarrier`** (Blackwell) is
available — and it carries its own tmem/shmem allocation setup. Don't assume a
listed symbol is callable in Gluon.

> **Why this differs from the Triton side.** The Triton trap doc (Trap #6)
> deliberately *avoids* `atomic_add(0, sem="acquire")` for the spin and uses
> `tl.load(volatile=True)` instead — a contention-free L2 read, whereas
> `atomic_add(0)` is a read-modify-write that hammers the same L2 line under
> contention. Gluon's `gl.*` surface exposes **no volatile-load equivalent**, so
> the acquire-atomic spin is the only portable cross-CTA primitive here, and you
> accept the L2-line contention the Triton path designs around. Prefer
> `bw.mbarrier` where the sync is within a cluster. See
> [`../triton/sm100-sparse-decode-split-k-pitfalls.md`](../triton/sm100-sparse-decode-split-k-pitfalls.md)
> Trap #6.

---

## 5. Gluon layout bookkeeping is explicit — `warps_per_cta`, 1-D layouts, `convert_layout`

**Trap**: Let layouts default the way Triton does.

**Result / rules**:
- `warps_per_cta` **must sum to the kernel's `num_warps`** — you can't
  under-allocate to a small tensor's row count; the same warp over-covers extra
  rows (e.g. `H=16` with 8 warps → `warps_per_cta=[8,1]`).
- 1-D `gl.arange` must be a plain `BlockedLayout([x],[y],[z],[0])` — **not** a
  `SliceLayout`-wrapped 2-D. Use `SliceLayout(axis, parent)` only to build an
  offset vector you then broadcast into a 2-D tile.
- A reduction result (`gl.max(logits, axis=1)`) inherits `SliceLayout(1, parent)`;
  to combine it with another layout you must `gl.convert_layout(...)` first —
  scalar ops across mismatched 1-D layouts error (the conversion is shmem-free
  when distributions coincide).

**Lesson**: In Gluon, layouts are part of the program. Budget time for explicit
`warps_per_cta` / `BlockedLayout` / `convert_layout` plumbing that Triton's
autoscheduler did invisibly.

---

## 6. `translator_helpers` (Triton→Gluon) is not a parity swap — it's net slower on single-dot FP8

**Trap**: Auto-port `@triton.jit` → `@gluon.jit` with
`triton.tools.triton_to_gluon_translater.translator_helpers` (`tl_dot`,
`tl_trans`, `default_blocked_layout`, `reset_to_default_layout`, …). Correctness
is perfect.

**Result**: **+8.5% mean** (all slow workloads +4.8 to +11.1%) vs the Triton
original on the same image. `default_blocked_layout` emits generic distributed
layouts and the helpers insert explicit `convert_layout` ops that Triton would
have folded into the dot's SMEM epilogue. The Blackwell `tcgen05_mma` path is
**not** reached this way — the port lowers to `mma_v2`-style ops with extra SMEM
transposes.

**Lesson**: The translator helpers are a *correctness bring-up* tool, not a perf
port. A real Gluon win requires hand-authored TMA loads + explicit SMEM staging +
`tcgen05_mma` descriptors + a custom load/MMA/epilogue pipeline. If you're not
writing that, stay on Triton.

> **Harness caveat:** Gluon needs a recent image (Triton 3.7); an older paired-A/B
> image will `COMPILE_ERROR` on the Gluon side. Run Gluon variants on the
> Triton-3.7 image with a back-to-back kernel swap for the paired comparison.

---

## "Use what / don't use what" cheatsheet

| Use | Don't use |
|---|---|
| `bw.tcgen05_mma` + `bw.alloc_tmem` for tensor-core throughput | `gl.dot_fma` as a perf path (it's software FMA) |
| `k_width=0` + pre-cast operands to accumulator dtype | `tl.dot` operand habits on `gl.dot_fma` |
| Gluon only when the problem fills `blockM≥64` | `tcgen05` at small `H` (phantom-row padding) |
| `gl.atomic_add(...,0,sem="acquire")` / `bw.mbarrier` for sync | `gl.barrier` (not callable in `gluon.jit`) |
| Explicit `warps_per_cta` summing to `num_warps`, `convert_layout` | Assuming Triton-style default layouts |
| Hand-authored TMA + SMEM + `tcgen05` pipeline | `translator_helpers` as a parity perf swap |

---

## Cross-references

- Triton-side counterpart traps for the same kernels:
  [`../triton/sm100-sparse-decode-split-k-pitfalls.md`](../triton/sm100-sparse-decode-split-k-pitfalls.md)
- Blackwell hardware primitives (tcgen05, tmem, mbarrier):
  [`../../../kernel-opt/nvidia/common/blackwell/hardware/`](../../../kernel-opt/nvidia/common/blackwell/hardware/)
- Measurement trust (paired A/B, image-pin caveats, noise floor):
  [`../../../ref-docs/nvidia/common/ncu-measurement-discipline.md`](../../../ref-docs/nvidia/common/ncu-measurement-discipline.md)
