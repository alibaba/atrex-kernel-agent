# Triton on Blackwell


**Last updated**: 2026-06-30

## Overview

Triton is used for many attention and linear-attention kernels (NSA, GatedDeltaNet, FLA). Starting with Triton 3.6 (released `2026-01-21`), Triton ships native Blackwell (SM100) lowering through `tcgen05.mma` + Tensor Memory (TMEM). The earlier framing -- "Triton compiler generates wgmma, not tcgen05" -- was correct for Triton 3.5 and earlier but is no longer correct on 3.6+. See the "Pre-3.6 historical context" subsection below for the historical framing; the rest of this page describes the current 3.6+ behavior.

When to use Triton on Blackwell:
- Rapid prototyping (Triton's fast turnaround still beats CuTe-DSL for exploration)
- Memory-bound kernels (bandwidth is bottleneck, not compute)
- Linear / sparse attention (GatedDeltaNet, FLA, NSA) where Triton's grid scheduling is convenient
- Block-scaled matmul on Blackwell via `tl.dot_scaled` (NVFP4 / MXFP families) -- first-class hardware-accelerated path
- Warp-specialized descriptor/TMA matmul kernels following the Triton persistent matmul tutorial pattern

Use CuTe-DSL / CUTLASS / FA-4 / TRT-LLM instead when:
- Peak-performance compute-bound matmul or attention on Blackwell. SGLang `pr-sglang-5390` measured a CUTLASS `tcgen05_mla` backend ~27% faster than the Triton MLA decode baseline on Blackwell.
- Production routing where vendor kernels are mature: SGLang `pr-sglang-21595` changes Blackwell datacenter multimodal attention default from `triton_attn` to FA-4.

## Triton 3.6+ Blackwell path

The 3.6 release adds Blackwell-native infrastructure for `tcgen05.mma`, TMEM allocation/copy/load/store, and warp-specialization plumbing. Source-of-record: `doc-triton-3.6-blackwell`. Per-pathway breakdown with verified-vs-needs-verification status: `triton-3.6-evidence`.

Verified lowering surfaces (with caveats):

1. **Descriptor/TMA + `tl.range(warp_specialize=True)` + `tl.dot`** -- strongest checked `tl.*`-surface evidence. The official Triton persistent matmul tutorial states the warp-specialized mode "only works on Blackwell right now"; backed by warp-specialization aref plumbing (PRs `#8262`, `#7826`, `#8009`, `#8123`, `#8534`, `#8451`, `#8651`) and Blackwell TMEM / `tcgen05` backend work (PRs `#8136`, `#8148`, `#8202`, `#8386`, `#8421`, `#8495`, `#8102`, `#8338`, `#8225`).

2. **`tl.dot_scaled` block-scaled matmul** -- hardware-accelerated by fifth-generation Tensor Cores on compute capability 10. The 3.6 dialect doc exposes `ttng.tc_gen5_mma_scaled` with TMEM-token semantics plus `ttng.tmem_copy`. Source PRs: Gluon NVIDIA `tcgen05 mma scaled` (`#8393`); frontend fixes (`#8564`, `#8658`); shared TMEM/tcgen05 backend (`#8136`, `#8148`, `#8202`). Format coverage centered on NVFP4 / MXFP per the official block-scaled matmul tutorial.

3. **Gluon front-end + `gl.warp_specialize` + `num_ctas`** -- the most explicit Blackwell-native surface. Initial 2-CTA cluster support landed in `#8644`, `#8653`; `num_ctas` plumbing in `#8645`; Gluon-side `tcgen05 mma scaled` in `#8393`. The release notes describe 2-CTA / cluster as initial support, so cluster-scope kernels via Gluon should be treated as early-stage.

Caveats -- what the verified evidence does and does not establish:
- **What is not yet verified by in-corpus evidence**: that arbitrary plain `tl.dot` kernels AUTOMATICALLY emit `ttng.tc_gen5_mma` / `tcgen05.*` PTX on every shape and configuration. The 3.6 infrastructure exists (`#8136`, `#8148`, `#8202`, `#8225`, `#8338`, `#8386`, `#8421`, `#8495`, `#8102` in the release notes) and downstream tl.dot kernels for SM100 are landing, but the question of "is the new path the default for every `tl.dot` shape on Blackwell" has not been confirmed by inspectable downstream PTX in any tracked-repo PR locally. Treat plain `tl.dot` on Blackwell as "uses the 3.6+ infrastructure when applicable" rather than as a blanket equivalence-to-CuTe-DSL claim.
- **What is also not yet verified**: that fused attention forward kernels with `warp_specialize=True`, `tl.dot_scaled` block-scaled matmul, and Gluon multi-CTA paths produce the same lowering as the upstream Triton tutorials predict -- the upstream tutorials demonstrate these paths; tracked-downstream repos haven't yet landed PRs with inspectable PTX confirming each one.

### What changed vs the pre-3.6 framing

Before Triton 3.6, the Blackwell story was: the compiler generated `wgmma.mma_async`, accumulators stayed in registers, and direct `tcgen05` / TMEM access was unavailable from `tl.*`. Pages and inclusion-policy text written under that premise are obsolete. The current premise is: **`tcgen05` + TMEM lowering paths exist on SM100, but coverage and performance leadership are workload-dependent**.

## Downstream Triton-on-Blackwell adoption (post-3.6 evidence)

The `evidence_basis` is anchored on:

- **`doc-triton-3.6-blackwell`** (`source_category: official-doc`) -- verifies that Triton 3.6 ships native Blackwell lowering infrastructure (TMEM, tcgen05, warp_specialize plumbing, Gluon multi-CTA / 2CTA, tl.dot_scaled). This is the "infrastructure exists" half of AC-1.2.

Together the two anchors establish that **the 3.6+ Blackwell lowering infrastructure is real AND tracked downstream repos are landing real Triton matmul kernels for Blackwell production decode paths today** -- which is the substance of the rewrite: the pre-3.6 "lacked-tcgen05 / lacked-TMEM" framing is no longer correct.

Two clarifications about what these anchors do NOT prove (since the verified-surface section above flags some shapes as needs-verification):

- The anchors do not prove that EVERY plain `tl.dot` kernel on SM100 emits `tcgen05.mma` PTX. They prove that real Triton kernels with `tl.dot` matmul are landing on Blackwell-only paths in tracked downstream repos (per `pr-vllm-34597`'s `triton_decode_attention.py` kernel) and that the 3.6 release added the infrastructure those kernels lower through. Whether the lowering automatically targets `tcgen05.mma` for every shape and dtype on every Blackwell SKU is a separate question.
- The anchors do not include explicit inspectable `tcgen05.mma` PTX dumps from a tracked-downstream merged PR. Such proof would be an even stronger anchor than what we have today; until one is found, the verified claim should be read as "Triton 3.6+ Blackwell is real and downstream-adopted", not "every Triton matmul on Blackwell is now optimal".

Supplementary post-refresh anchor: `pr-vllm-29339` -- vLLM bugfix that scopes the upstream `triton_kernels` library (the `triton-lang/triton/python/triton_kernels` collection shipped with Triton 3.6) to `[sm100, sm90]` for the MXFP4 quantization path. Modifies only a dispatch gate (`vllm/model_executor/layers/quantization/mxfp4.py`), which is why `pr-vllm-34597` is the primary anchor and `pr-vllm-29339` is supplementary.

Pre-refresh historical anchors (retained as supplementary context, not as AC-1.1 "new tracked-repo PR page" evidence on their own):

- `pr-sglang-22079` -- Gemma4 NVFP4 SGLang `extend_attention` Triton kernel doing actual `tl.dot(q,k)` / `tl.dot(p,v)` matmul on `[sm100, sm90]`, merged 2026-04-03. Strongest in-corpus example of a real `tl.dot` Triton matmul landing for SM100 post-3.6, but pre-refresh per `previous_pages_manifest`.
- `pr-sglang-21019` -- Qwen3.5 GDN projection fused split/reshape/cat kernel merged 2026-03-20. `tl.load`/`tl.store` only (memory rearrangement, no `tl.dot`); demonstrates "Triton on SM100 post-3.6" but not the matmul lowering path.

Caveat anchors `pr-sglang-5390` and `pr-sglang-21595` provide ecosystem context (CUTLASS still leads on peak; Blackwell defaults moved away from triton_attn for some workloads).

## FlashInfer-Bench: AI-Generated Triton Performance

The following table reflects benchmark snapshots from when the original page was written (Triton 3.5 era) and is preserved for historical reference. It does NOT reflect the 3.6+ tcgen05 path.

| Model | Avg Speedup vs FlashInfer | Resolved % |
|---|---|---|
| Gemini 2.5 Pro | 0.628x | 73.1% |
| GPT-5 | 0.467x | 92.3% |
| Claude Opus 4.1 | 0.456x | 73.1% |

A re-run on Triton 3.6 with FlashInfer-Bench is not yet available locally; future refresh rounds should update this table or remove it in favor of a pointer to the live leaderboard.

## GatedDeltaNet Decode in Triton

```python
@triton.jit
def gated_delta_net_decode(
    Q, K, V, Gate, State, Output,
    qk_dim: tl.constexpr, v_dim: tl.constexpr, d: tl.constexpr,
):
    """Single-token decode: O(d^2) per token."""
    head_id = tl.program_id(0)
    # Load recurrent state S: [qk_dim*d, v_dim*d]
    s = tl.load(State + head_id * qk_dim * d * v_dim * d + offsets)
    q = tl.load(Q + offsets)
    k = tl.load(K + offsets)
    v = tl.load(V + offsets)
    g = tl.load(Gate + head_id)

    # Delta rule: S = g*S + k @ (v - S^T @ k)^T
    sk = tl.dot(tl.trans(s), k)
    delta_v = v - sk
    s = g * s + tl.dot(k[:, None], delta_v[None, :])
    o = tl.dot(tl.trans(s), q)

    tl.store(State + offsets, s)
    tl.store(Output + offsets, o)
```

## NSA Sparse Attention in Triton

```python
@triton.jit
def sparse_attention_fwd(Q, K, V, Output, TopK_Indices,
                         block_size: tl.constexpr, topk: tl.constexpr):
    """Attend to top-k sparse token blocks only."""
    qid = tl.program_id(0)
    q = tl.load(Q + qid * d + tl.arange(0, d))
    acc = tl.zeros([d], dtype=tl.float32)
    for i in range(topk):
        bidx = tl.load(TopK_Indices + qid * topk + i)
        k_block = tl.load(K + bidx * block_size * d + offsets)
        v_block = tl.load(V + bidx * block_size * d + offsets)
        scores = tl.dot(q[None, :], tl.trans(k_block))
        # softmax + accumulate...
```

## Pre-3.6 historical context

> The text in this subsection describes Triton 3.5 and earlier. It is preserved for historical accuracy and is NOT a current statement about Triton 3.6+ behavior. Do not cite it as current limitations.

Triton on Blackwell -- Triton 3.5 and earlier:

1. **No direct tcgen05 access**: Triton compiler generates wgmma, not tcgen05.
2. **No TMEM**: accumulators stay in registers.
3. **CPU launch overhead**: impacts small-batch decode latency.
4. **Workaround**: CUDA graphs (vLLM default for GatedDeltaNet).

These four bullets describe the world before Triton 3.6.0 landed (`2026-01-21`). The first two are no longer correct on 3.6+; see the "Triton 3.6+ Blackwell path" subsection above. CPU launch overhead and CUDA-graph workarounds remain workload-relevant on small-batch decode paths regardless of compiler era.

## Blackwell Triton Examples (verbatim upstream code shipped locally)


| File | Purpose | PR |
|---|---|---|

The current AC-1.1 **post-refresh primary upstream-code anchor** is `pr-vllm-34597` (above), with the actual Triton decode-attention kernel shipped verbatim. Supplementary post-refresh anchor: `pr-vllm-29339`, a vLLM bugfix scoping the Triton 3.6 `triton_kernels` library to `[sm100, sm90]` for the MXFP4 quantization path (no evidence bundle because the change is dispatch-gate-only).

The original source tracked a 42-PR universe and marked entries outside the captured sub-scopes as not shipped locally; this wiki page keeps only the optimization-relevant conclusions.


## Related

- [CUDA C++ for Blackwell Kernels](cuda-cpp.md)
- [CuTe DSL on Blackwell (SM100)](cute-dsl.md)
- [PTX Instructions for SM100](ptx-sm100.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../common/triton/triton-tile-ir-beyond-simt.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
