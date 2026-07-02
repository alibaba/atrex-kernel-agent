# FlashAttention-4 Kernel Architecture on Blackwell (roles, barriers, TMEM)

> **Source / attribution**: Distilled in our own words from MLC-AI's *Modern GPU

**Last updated**: 2026-06-30

> Programming for MLSys*, Part IV — Flash Attention 4
> (`chapter_flash_attention`), available as a 3rd-party submodule at
> `gpu-wiki/3rdparty/modern-gpu-programming-for-mlsys/` and online at
> <https://mlc.ai/modern-gpu-programming-for-mlsys/>. That walkthrough uses the
> TIRx DSL; the structure below describes FA4 on Blackwell (sm100, tcgen05 /
> TMEM / TMA) and is framework-agnostic. Identifier names (`s_ready`,
> `p_o_rescale`, …) are kept so the prose maps onto a real kernel. Companion
> docs: [`../kernels/flash-attention-4.md`](../kernels/flash-attention-4.md),
> [`flash-attention-4-warp-specialization-2cta.md`](flash-attention-4-warp-specialization-2cta.md),
> [`flash-attention-4-source-code-analysis-part1.md`](flash-attention-4-source-code-analysis-part1.md).

## Why attention is harder than GEMM

GEMM is one MMA chain repeated. FA4 is **two MMA phases with softmax wedged in
the middle**:

```
Q,K --score MMA--> S --softmax--> P --value MMA--> O
```

The middle stage is the whole difficulty. Softmax (row-max, `exp`, row-sum) runs
on CUDA cores **on the critical path between two Tensor-Core MMAs**, and online
softmax must *revisit and rescale* the output `O` accumulated so far whenever the
running row-max grows. So most of "attention optimization" is really softmax
optimization: cheaper exponentials and overlapping softmax with the MMAs instead
of stalling on them.

Two consequences drive the whole design:
- `S` is never materialized in full (≈64 MB/head at seq=4096); K/V stream in
  blocks and three per-row running states (`row_max`, `row_sum`, and the `O`
  accumulator) summarize everything seen so far.
- Normalization by `row_sum` is **deferred to the epilogue**; per block the kernel
  only produces the softmax numerator `P`, not the normalized probabilities.

`exp` is evaluated as hardware `exp2` on raw scores by folding `1/sqrt(d)` and
`log2(e)` into one constant `scale_log2 = log2(e)/sqrt(d)` (so
`exp(x/sqrt(d)) = exp2(x·scale_log2)`) — `exp2` is faster than natural `exp`.

## Tile-primitive graph (everything lives in TMEM)

| Tile | Produced by | Where |
|------|-------------|-------|
| `S` (scores) | score MMA `Q·Kᵀ` | TMEM |
| `P` (softmax numerator) | softmax: TMEM→registers→TMEM (fp16) | TMEM (fp16 view) |
| `O` (output accumulator) | value MMA `P·V` | TMEM (fp32) |

The rescale-`O`-on-max-change step is itself a tile op (TMEM→registers→TMEM),
not scalar bookkeeping. Every stage has the same shape: a tile placement, a
hardware dispatch (`tcgen05.mma` / `tcgen05.ld` / TMEM store / TMA), and a
barrier that proves the next consumer may run.

## Warp roles (warp specialization)

One CTA runs **two Q pipeline stages** concurrently; four warpgroups split the work:

| Role | Warpgroup | Job |
|------|-----------|-----|
| Softmax (Q stage 0) | WG0 | read `S` from TMEM, compute `P`, write `P` to TMEM |
| Softmax (Q stage 1) | WG1 | same, for the second Q slot |
| Correction + epilogue | WG2 | rescale `O` in TMEM; final normalize + stage output |
| MMA + data movement | WG3 | warp 0 issues **all** MMAs (score *and* value); warp 1 issues TMA loads; warp 2 issues the TMA store |

Key asymmetry: **every MMA issues from WG3 warp 0 alone** — WG0/WG1 never issue an
MMA, they only consume `S`, run softmax, and write `P`. The "two Q stages" are
*not* two heads; they are two slots in the Q pipeline so two Q tiles are in
flight at once (hence softmax appears twice, on WG0 and WG1).

## The two MMA phases

- **Score MMA** (WG3 w0): `S = Q·Kᵀ`, both operands from SMEM → `S` in TMEM;
  one elected lane arrives `s_ready` to release softmax.
- **Softmax** (WG0/WG1): waits `s_ready`; reads `S` TMEM→registers in chunks;
  computes row-max/row-sum and `P`; writes `P` back as **fp16 in TMEM** (an MMA
  needs `P` as a tile operand, not scattered registers). Arrives `p_o_rescale`
  (first 96 cols) and `p_ready_2` (last 32).
- **Value MMA** (WG3 w0): `O += P·V`, with **`P` from TMEM and `V` from SMEM**,
  accumulating into `O` in TMEM. `accum` is false on the first K/V tile
  (initialize) and true afterward (add).

**96 + 32 split**: softmax writes `P` in four 32-col chunks; the value MMA fires
on the first three chunks (96 cols) immediately and a second sub-MMA takes the
last 32 after `p_ready_2`. This overlaps the last chunk's `exp`/TMEM-write with a
96-wide MMA already in flight, keeping the Tensor Core busy.

## TMEM layout and reuse (why barriers and layout are inseparable)

`S`, `P`, and `O` **share one 128×512 TMEM allocation**. With Q-pipeline depth 2,
the two `S` slots (2×128 cols) plus the two `O` slots (2×128 cols) already use all
512 fp32 columns — there is no room left for `P`. So `P` **aliases the same
bytes through a narrower fp16 view**: take one fp32 view for `S`/`O`, rewind the
pool base, take a second fp16 view over the same physical bytes (twice as many
indexable columns). The aliasing only shows up as a `×2` stride on the `P`
region. This is legal **only because each region is reused strictly after its
previous consumer finishes** — and that timing is exactly what the barriers
guarantee. In FA4 the barriers don't just schedule; they make the layout legal.

## Barrier graph

Data-carrying handoffs (everything else is buffer-recycle bookkeeping):

| Barrier | Producer → consumer | What becomes safe |
|---------|---------------------|-------------------|
| `q_load` / `kv_load` (`.full`/`.empty`) | TMA ↔ MMA | Q/K/V SMEM tile ready / stage reusable |
| `s_ready` | score MMA → softmax | `S` readable in TMEM |
| `p_o_rescale` | softmax + WG2 → value MMA | first 96 cols of `P` ready **and** `O` slot safe to accumulate |
| `p_ready_2` | softmax → value MMA | last 32 cols of `P` ready |
| `o_ready` | value MMA → epilogue | final `O` ready |
| `softmax_corr` (`.full`/`.empty`) | softmax ↔ WG2 | scalar `acc_scale` / final `row_sum` passed via a 1-slot SMEM mailbox |
| `corr_epi` (`.full`/`.empty`) | epilogue ↔ TMA store | `O_smem` ready / reusable |

Notes that prevent real bugs:
- `softmax_corr.empty` means only that WG2 consumed the scale slot — it is **not**
  the gate that lets the value MMA start. That gate is `p_o_rescale`.
- The softmax→correction edge passes a **scalar through an SMEM mailbox**, so it
  needs a `full`/`empty` pair, unlike the tile-ready gates.
- Barrier type follows the producer: TMA loads use a TMA barrier (byte-count
  completion), MMA completion a tcgen05 barrier (`tcgen05.commit`), pure
  thread→thread handoffs an mbarrier (explicit arrive).
- All the *new* barriers (`s_ready`, `p_o_rescale`, `p_ready_2`, `softmax_corr`)
  cluster around softmax — they exist precisely because the score and value MMAs
  are no longer adjacent.

## Pipelining

Different tile streams advance at different rates, so each gets its own ring:
**Q depth 2**, **K/V depth 3**, **TMEM depth 2**. Crucially the MMA warp does
*not* run all score MMAs then all value MMAs; once both Q stages are primed it
**interleaves** them (value for current V, score for next K, …), which is what
lets the score / softmax / correction / value rows overlap in time.

## Rescaling, writeback, LSE

The rescale is **mandatory**: when `row_max` grows, the `O` from earlier blocks
was scaled by the old max and is too large by `exp(m_new−m_old)`; skip it and the
result is simply wrong:

```
O_old ← O_old · exp((m_old − m_new)/sqrt(d))
```

Work is split: softmax computes the per-row scale and drops it in the mailbox;
WG2 reads `O` from TMEM, multiplies, writes back. A **conservative skip** avoids
wasted work: if the log2-scaled max delta hasn't moved past `-rescale_threshold`,
keep the old max and set `acc_scale = 1.0`; WG2 reduces `should_rescale` with
`any_sync` and leaves `O` untouched when no row needs it (rescaling `O` is a full
TMEM→RF→TMEM pass over the whole accumulator). The epilogue (WG2) does the
deferred normalize `O · 1/row_sum`, casts to fp16, stages `O_smem`; WG3's store
warp TMAs it to GMEM.

**LSE caveat**: this is forward-output only. To support a training backward pass,
store `LSE_i = log(row_sum_i) + row_max_i/sqrt(d)` (the `1/sqrt(d)` must be
re-applied because `row_max` is kept on the *raw* unscaled scores).

## Causal masking & GQA

- **Causal**: two complementary mechanisms — (1) skip whole K/V blocks above the
  diagonal (`get_n_block_max` trims the trip count); (2) for diagonal-straddling
  blocks, mask invalid columns to `-inf` **in registers before `exp2`** (applied
  as a bitmask over the 32-wide chunk, not element-by-element). The data path is
  unchanged; only trip count + an in-softmax mask step are added.
- **GQA**: a group of query heads shares one K/V head. The 128 Q-tile rows are
  repacked as `seq_pos × query_head` (e.g. `GQA_RATIO=4` → 32 seq positions × 4
  heads) so all heads in a group ride the same K/V tile (`seq_pos = row //
  GQA_RATIO`), saving K/V bandwidth.

## Key takeaways

1. FA4 = score MMA → softmax → value MMA, with **deferred normalization** and
   **online-softmax rescaling of `O` on the critical path**.
2. **Warp specialization**: one warp issues all MMAs; separate warpgroups own the
   two Q stages' softmax and the correction/epilogue — softmax overlaps MMAs.
3. **TMEM is the shared currency**: `S`/`P`/`O` alias one 128×512 region (`P` via
   an fp16 view), and **barriers are what make that reuse legal**.
4. The `96+32` value-MMA split and the **conditional rescale skip** are the two
   cheap-but-high-leverage tricks that keep the Tensor Core busy and cut wasted
   TMEM traffic.
</content>


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [FlashAttention 1–4: GPU Generational Evolution](../../common/flash-attention-1-to-4-gpu-evolution.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
