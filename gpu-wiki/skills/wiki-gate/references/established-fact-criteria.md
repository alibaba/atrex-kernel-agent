# anti-strategy established-fact adjudication guide

The adjudication rules used when auditing the 404 historical anti-strategy
records. They were generalized from a close reading of the first 38, one by one;
writing them down keeps later bulk adjudication to **the same standard** as those
first 38 and leaves an auditable basis.

## The three hard criteria (all must hold to stay in the store)

| # | Criterion | Where it lands in the schema |
|---|---|---|
| 1 | A checkable condition C: arch, shape regime, dtype or toolchain | at least one non-empty field in `established_fact.condition` |
| 2 | A causal mechanism: why it **necessarily** fails under that condition | `established_fact.mechanism` ≥ 40 characters |
| 3 | A verdict that concludes | may not be `unknown` / `unstable` (both removed from the schema enum) |

The key constraint on criterion 1: **"this operator" is not a condition**. A
technique failing on one operator is an observation, not a law.

## Statements that count as a mechanism (→ backfill)

Every record backfilled among the first 38 falls into one of these five kinds:

**① Quantified trade-off** — states outright that "what it saves < what it costs",
with numbers on both sides
> "padding M 901→960 requires copying the A matrix, and that copy costs 8.22us >
> the 2.1us the kernel saves"
> "torch.bmm's Python dispatch overhead exceeds the transpose it removes
> (+0.5us@N=256, +5us@N=1024)"

**② A hard resource or capacity limit** — cites a concrete capacity or spec and
derives the necessary outcome
> "at N=8192 a 128MB buffer > the B200's 50MB L2, so the transpose has to read
> HBM: 23us against 8us warm"
> "a 2-CTA cluster's tile needs m≥256, so at M=128/256 some clusters sit idle"

**③ An API or semantic limitation** — the tool or library does not support it
semantically; this is not under-tuning
> "cuDNN-FE only accepts an NHWC/RowMajor epilogue on SM100, so the
> channels_last conversion cannot be skipped"
> "batched-strided GEMM cannot express a reduction across the full 2048
> dimension, so batching computes the wrong result (error=288)"
> "torch.compile bakes the weight data pointer into the graph, so a
> pointer-based cache cannot possibly hold"

**④ A microarchitectural causal chain** — reasons from hardware behaviour through
to the performance outcome
> "PADDED_M=8 → fewer active warps → lower ILP → load-latency stall exceeds the
> cp.async pipeline depth"

**⑤ A measurement-model limitation** — the metric itself makes this class of
optimization invisible (the tool has to be written into `condition.toolchain`)
> "the GPU pipeline already overlaps CPU launch overhead with kernel execution,
> so a CUPTI kernel-span cannot show a host-side optimization"
> "the kernel is dispatch-bound and host ctypes overhead ~38us ≈ the GPU time, so
> a GEMM-level 2-7% improvement is masked completely"

## Statements that do not count as a mechanism (→ demote)

**Ⓐ Exhaustion bookkeeping** — "we looked, there is nothing new"
> "179th consecutive dead-end", "all queries New?=No", "STALL_COUNT=49",
> "ORCHESTRATOR QUIESCE RECOMMENDED"

**Ⓑ Bare comparison numbers** — conditioned and reproducible, but **it never says
why**
> "Triton GEMM is 48-122% slower than cuBLASLt", "cuDNN FP16 is much slower than
> CUTLASS"
> Demoted across the board under standard A. Facts like these have value but lack
> a mechanism, so they return to staging until that is supplied.

**Ⓒ Inside the noise band** — the conclusion lands within measurement noise, which
is no conclusion
> "1.2% improvement but the multi-seed interval overlaps HEAD",
> "flat-within-noise", "+0.6% within noise"

**Ⓓ A methodology note** — a way of working, unrelated to any particular operator
> "a sub-5% geomean delta needs ≥2 runs in the same session", "the profile can be
> reused while the md5 is unchanged"
> Exception: if the same record also holds a real mechanism (for example "the
> bottleneck is inside the closed-source nvjet binary and unreachable from the
> host layer"), backfill from that mechanism.

**Ⓔ An asserted ceiling** — declares the ceiling reached without explaining it
> "confirmed mma_v2 Ampere ceiling", "kernel at hardware floor", "at strong local
> optimum"

**Ⓕ Speculative wording** — the mechanism carries may / might / possibly
> "the 3-element inner loop may be less efficient than the grid-stride loop"

## Borderline cases (the ones actually met in the first 38)

| Situation | Adjudication | Basis |
|---|---|---|
| A record tried 8 approaches and only 1 has a mechanism | **backfill**, with `established_fact` describing only that one | The gate asks "what did this record establish", not "everything it tried" |
| The lesson is methodology, but `attempted` holds a real mechanism | **backfill**, filling in the mechanism from `attempted` | It is enough that the mechanism exists; where it sits does not change that it is a fact |
| The mechanism is in `attempted` rather than `root_cause` | **backfill** | Historical records commonly have an empty `root_cause`, with the mechanism scattered through `attempted` |
| The condition is "it failed on every shape" | **backfill**, writing the measured range into `shape_regime` | Checkable — but the mechanism still has to hold on its own |
| A measurement-tool property used as the condition | **backfill**, written into `condition.toolchain` | Metric semantics are versioned and checkable |
| `rediscovered` is high but there is no mechanism | **demote** | A reproduction count is no substitute for a mechanism (standard A) |

## Backfill discipline

- The mechanism **may only be taken from that record's own prose**; never borrow a
  number or a conclusion from another record
- Never supply a causal explanation the record does not contain
- Backfilled content must pass `validate_fact()`: at least one non-empty condition
  field under a legal key, and a mechanism of ≥40 characters that does not match
  the hollow-wording blacklist
- When `validate_fact` rejects a backfill it automatically becomes a demotion
  instead (see the `rejected_backfill` branch in `audit_anti_strategy.py`)

## Demotion procedure

A record that fails is **not deleted**. It moves back to the mining skill's staging
area, with `status: active` and the added fields `demoted_from_main` /
`demoted_reason` / `demoted_at`. Once a mechanism is supplied later, or the record
has been independently reproduced several times, it can be admitted again through
wiki-gate.
