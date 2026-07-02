# NCU Measurement Discipline — Trusting Your Profile and Timing Numbers

How to *trust* the numbers a profile-driven loop produces: what NCU's

**Last updated**: 2026-06-30

`Duration` means (and doesn't), how much a latency delta has to move before it
is signal rather than noise, and the ways CUDA graph capture silently corrupts
both profiling and correctness. This is the "measurement-trust" layer that sits
underneath the rest of the NCU docs — it does not tell you *what* to optimize,
it tells you *whether to believe* the evidence you are optimizing against.

> Empirical numbers below were observed on **B200 / CUDA 13.2 / Triton 3.6 /
> cutlass-dsl 4.3.4** profile-driven sessions. The *mechanisms* are general
> (they follow from how NCU and CUDA-graph capture work); the exact
> multipliers are illustrative, not constants — re-measure on your stack.

## TL;DR

1. **NCU `Duration` is not a wall-time number.** Profiling overhead inflates
   per-kernel duration ~5–20×. Use NCU for **ratios** (occupancy %, IPC, stall
   reasons, throughput %, cache hit %, block limits) — never to compare kernel
   *speed*. Get speed from a separate timing harness (`triton.testing.do_bench`
   / `torch.cuda.Event`).
2. **A single latency run is not a measurement.** A standalone bump of
   +0.5–1× on a short kernel is usually session noise. Confirm with a
   drift-cancelled A/B comparison or a repeated variance check before you
   believe a code change helped.
3. **CUDA graph capture breaks profiling and can break correctness silently.**
   "No kernels were profiled" has two unrelated causes; graph-captured kernels
   can be skipped on replay while still passing correctness on stale bytes.

---

## 1. NCU `Duration` ≠ timing-harness latency

NCU works by injecting a measurement library, serializing kernel launches, and
**replaying the kernel across multiple passes** to collect every metric. That
machinery inflates the reported `Duration` far above the real execution time.

> Observed on B200: a Triton forward kernel reports **11.58 µs in NCU vs ~1.5 µs
> in the timing harness**; a CuTe reduce reports **13.15 µs vs ~0.9 µs**. The
> inflation is not a constant factor — it depends on pass count and kernel
> shape — so NCU durations are not even comparable *to each other* across
> kernels.

**Rules:**

- Treat NCU `Duration` as a **"kernel launched and was profiled" sanity
  signal**, not a performance number.
- Read NCU only for **dimensionless ratios and percentages**: SOL %, achieved
  occupancy, warp-stall breakdown, IPC, memory throughput %, cache hit rates,
  `launch__occupancy_limit_*`. These are pass-stable and trustworthy.
- Get **absolute latency** from the timing harness, on a separate run, with NCU
  *not* attached. Never justify a code change with an NCU duration delta — that
  is exactly the case the repo's profile-driven rule (`ncu` for bottleneck,
  `do_bench`/`Event` for latency only) is guarding against.
- For the related trap — believing NCU's per-rule **`Est. Speedup %`** as a
  wall-time predictor — see
  [`ncu-rule-est-speedup-meta-rules.md`](ncu-rule-est-speedup-meta-rules.md).
  Two specific blind spots of that estimate are worth restating here because
  they are *measurement* errors, not optimization errors:
  - **SMEM-staging fixes:** "raise occupancy by staging registers through
    shared memory" — the estimate ignores the new SMEM round-trip +
    `__syncthreads` it introduces. On register-resident kernels with small
    per-thread state, staging often *regresses* despite a double-digit
    predicted local speedup. Treat the estimate as an upper bound; measure the
    staged variant before committing.
  - **CTA-coarsening / packing fixes:** the estimate only models *raising*
    occupancy (more warps/CTA, fewer regs, smaller tile). It does **not** model
    the inverse — packing several logical units per CTA to amortize launch
    overhead. Packing reduces CTA count; on an already-under-loaded kernel that
    can drop below **1 wave** on the target SM count and remove across-CTA
    latency hiding entirely. Before trusting an occupancy "Est. Speedup",
    sanity-check `CTA_count / num_SMs ≥ 1` on the *smallest* shape after the
    proposed change.

---

## 2. Noise discipline — when a latency delta is signal

Short kernels measured across separate process / container launches drift from
cold cache, JIT warm-up, cuBLAS/CUTLASS lazy init, GPU thermal state, and (on
shared infra) neighbour load. The drift is commonly a few-percent CV but can
spike higher, which is enough to fake a "win" on a sub-1× change.

**The failure mode to avoid:** a single labeled run shows +1× for a new config
that looks like a breakthrough; an apples-to-apples re-comparison then shows
Δ ≈ 0. The +1× was pure drift. Committing on the standalone bump "merges" a
no-op — or a regression — on noise.

**Discipline:**

- **Never decide signal-vs-regression from ≤3 workloads / shapes.** A smoke
  test on a handful of shapes typically shows **2–3× variance** vs the
  full-suite mean. Use small subsets for *compile / correctness* checks only.
- **For any sub-1× delta, use a drift-cancelled comparison.** Run the new and
  old variants **back-to-back in the same process / container**, so both share
  one cold-start, one thermal point, one load sample — the cross-session drift
  nearly cancels and the *delta* is tight even when *absolute* scores wander.
  This is the first-line tool for iterating under noise (and for confirming a
  suspected revert).
- **To quote a noise floor, repeat the unchanged variant N times** and report
  mean / std / CV (overall and per-shape). A change must clear the noise floor
  to count. Note that per-shape numbers from a repeat-the-same-code variance
  pass are **not** apples-to-apples with a labeled run's per-shape numbers
  (autotune-config drift across the two paths) — use the same-process A/B for
  per-shape comparison, and the variance pass only for the overall CV.
- **Warm up before measuring.** The very first run after a cold start (empty
  JIT cache, GPU not at steady temperature) scores low; throw one run away so
  the loop starts in steady state.
- **On shared/cloud GPUs the drift can be >10×, not a few percent.** The same
  unchanged code has been observed at 0.33 ms on one VM and 4.0 ms on another.
  Where cross-host placement varies run-to-run, *absolute* numbers from
  different sessions are meaningless below ~15%; only same-host paired deltas
  carry signal.
- **Stratify the A/B by the workloads you actually changed.** A per-host bias
  can flip the *unaffected* shapes and either fake or mask the real signal in
  the aggregate. Split results into "touched by this change" vs "untouched"
  before judging — if the untouched shapes also moved, that movement is host
  bias, and the real verdict is in the touched subset.
- **Require a mechanism before keeping a marginal win.** A sub-1%
  direction-positive result *without a mechanism story* is noise, not signal
  (e.g. `input_precision="ieee"` on a bf16 dot lowers to the same PTX — any
  "win" is drift). Keep only on *direction-positive + ~zero-cost + a concrete
  reason it should help*.

This complements the repo rule that latency comes from `do_bench`/`Event` only:
those tools give you a *number*, this section tells you *how many of them you
need and how to compare them* before the number means anything.

---

## 3. Profiling under CUDA graph capture

When a kernel is launched from a captured CUDA graph (`torch.cuda.CUDAGraph`,
`torch.cuda.graph(g)`, `g.replay()`), launches go through `cuGraphLaunch`, not
`cuLaunchKernel`. NCU's kernel-symbol machinery does not see them the same way.

### "No kernels were profiled" — two unrelated causes

Diagnose before reaching for any fix:

1. **Kernel-filter / NVTX-scope miss (most common, no graph involved).** Your
   `--kernel-name` regex or an NVTX-range include filter didn't match any
   launched kernel. NCU prints "No kernels were profiled" and lists what it saw
   under **"Available Kernels"**. *Diagnose:* re-run **without**
   `--kernel-name`. If your kernel now appears in "Available Kernels" but no
   profile data is collected, it is a filter/scope miss — graph-capture
   workarounds do nothing here. Fix the regex (`ncu` `--kernel-name` is
   full-match; use `regex:` / `.*name.*` for substring) or the NVTX include.
2. **CUDA graph capture** — *only* if `kernel.py` actually captures a graph.
   Graph-captured launches don't surface the same symbol info, so the regex can
   miss them; worse, NCU has been observed profiling **only the first kernel of
   a replay** even with no name filter, so an "Available Kernels" listing under
   capture is not trustworthy as complete.

### Workaround for capture: an env gate

Add a module-level env gate that bypasses the graph branches, profile with it
set so every kernel launches eagerly and NCU sees them all, then unset it for
labeled timing (the graph capture is itself a real wall-time win worth keeping
in the production path):

```python
import os
_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))      # module scope
...
if not _NO_GRAPH:
    with torch.cuda.graph(g):      # capture branch
        ...
    g.replay()
else:
    run_eager()                    # same kernels, no capture
```

Then profile with `NO_GRAPH=1` exported (or passed into the container/env), and
run the labeled bench without it. This is cheaper than hand-rolling a
`torch.cuda.Event` harness around each kernel. (Also available:
`ncu --graph-profiling node|graph` to profile per-node or the whole graph —
see [`ncu-profiling-guide.md`](ncu-profiling-guide.md) — but the eager-gate
gives the cleanest per-kernel attribution.)

---

## 4. Silent kernel skipping under graph capture (a correctness trap)

If a kernel in a captured pipeline fails to enter the graph — it runs once
during capture, then is skipped on every replay — its destination tensor keeps
**coincidentally-correct stale bytes** from the capture-time run. Correctness
checks pass, and the headline latency is **artificially low** because the real
work isn't being done on replay. This is a *correctness* bug that hides as a
*performance* win.

**Known causes:**

- A DSL launch path whose stream binding doesn't pick up capture mode (e.g.
  CuTe DSL's `@cute.kernel.launch()` TVM-FFI binding).
- A bare CUDA chevron `<<<grid, block>>>` with no explicit stream — the legacy
  null stream is never in capture mode.

**Detection (any one triggering ⇒ bug confirmed):**

- **Zero-output replay** — `output.zero_()` after warmup, run, synchronize,
  assert `output.abs().sum() > 0`.
- **Poison-cell** — write a sentinel (`output[0,0,0] = -1e9`), replay, assert
  the cell was overwritten.
- **Varying inputs** — keep the same tensor addresses but mutate contents
  between replays (`q.normal_()`); assert the output changes.

Run one of these whenever a graph-capture change produces a "too good"
speedup with no algorithmic reason.

---

## 5. Per-call overhead floor

There is a fixed cost between Python entering the launch function and the first
GPU instruction executing (observed floor ~80 µs in one harness). When a kernel
is fast enough that this floor dominates, no amount of in-kernel optimization
moves the headline. The two levers that actually help are CUDA-runtime, not
kernel-internal:

- **Audit pre-kernel GPU↔CPU syncs** — a stray `.item()`, `.cpu()`, a
  data-dependent shape op (`.nonzero()`, `.unique()`, boolean-mask indexing), or
  `torch.cuda.synchronize()` on the launch path serializes the whole pipeline.
  (Plain `tensor.shape` is *not* a sync — it's free metadata; only ops whose
  output shape depends on device values force the device-to-host wait.)
- **Capture stable shapes into a CUDA graph** — amortizes per-launch overhead
  across replays (then apply §3/§4 so you can still profile and trust it).

If your latency is at the per-call floor, stop tuning the kernel body and go
after these two first.

> **Measurement blind spot — PDL / cross-kernel overlap is invisible to
> per-kernel timing.** Programmatic Dependent Launch (`launch_pdl` +
> `griddepcontrol.launch_dependents`) lets the next kernel's prologue overlap
> the current kernel's tail. Its win is *wall-clock between launches*, not any
> single kernel's duration — so a CUPTI/`torch.cuda.Event` harness that times
> each kernel's individual start/stop **cannot see it** and reports PDL as
> neutral even when it is correctly emitted and running. (Contrast `.item()` /
> host syncs, which the same harness *does* see, because they serialize the
> stream and gate the next launch.) If PDL matters for your pipeline, measure
> end-to-end wall time across the launch sequence, not per-kernel duration.

---

## Practical checklist

Before acting on any profile-driven measurement:

1. Did the speed comparison come from the **timing harness**, not NCU
   `Duration`? (If NCU, discard it as a speed number.)
2. Is the delta **bigger than the noise floor**, confirmed by a same-process
   A/B or a repeat-variance pass? (If from one standalone run on few shapes,
   it is not yet signal.)
3. Does the kernel **capture a CUDA graph**? If so, did you profile with the
   eager gate, and run a silent-skip detection check on any surprising win?
4. Is the kernel **at the per-call overhead floor**? If so, audit syncs +
   consider graph capture before touching the body.

## Related docs

- NCU CLI / metrics / sections reference: [`ncu-profiling-guide.md`](ncu-profiling-guide.md)
- Profile-driven optimization loop: [`ncu-profile-driven-optimization-workflow.md`](ncu-profile-driven-optimization-workflow.md)
- `Est. Speedup %` is a ceiling, not a wall-time delta: [`ncu-rule-est-speedup-meta-rules.md`](ncu-rule-est-speedup-meta-rules.md)
- Community Nsight practice: [`nsight-profiling-practice.md`](nsight-profiling-practice.md)
