---
name: gpu-kernel-decompose
description: |
  LLM-layer fusion-boundary decomposition rules. Read this when a task is a whole LLM layer (a PyTorch module /
  forward with several heavy ops) rather than a single already-bounded operator. It defines the gate (when to
  decompose at all — default is BYPASS), the fuse/split decision rules, a canonical boundary catalog backed by
  real reference kernels, and the output contract: one basic fused kernel per boundary, each of which becomes
  its own standard atrex workspace.
tools: Read, Grep, Glob, WebSearch, WebFetch, Write, Bash
---

# LLM-Layer Fusion-Boundary Decomposition Rules

This is the **decomposition rulebook**. It is consulted **only** when a task is a whole LLM layer. Its job is
to carve the layer into fused-operator **boundaries**, where each boundary is a self-contained operator that
then runs through the *unchanged* single-operator pipeline (its own workspace, Step 0 roofline, baseline v0,
profile-driven optimization, git HEAD = best).

**Scope**: LLM-layer fusion-boundary decisions driven by the known transformer structure + the rules below.
This is **not** a general model-graph partitioner, and it authors **no** partitioning content into gpu-wiki
(gpu-wiki is operator-optimization knowledge only). Consult gpu-wiki here only for operator-level facts.

---

## 0. The Gate — default is BYPASS

Decomposition is **purely optional** and off by default. Most kernel tasks are a single operator and must skip
this entire process. The test is **how many separable operators the input contains** — *not* whether it is a
"whole layer". Any input made of more than one fused op decomposes.

- **BYPASS (the default, the common case)** — the input is **one** fusion-bounded operator: a single compute
  primitive plus the elementwise epilogue/prologue a real kernel folds into the same launch (bias, residual,
  activation, scaling, quant, or a RoPE/norm that rides it). Examples: a matmul, one attention, one norm,
  `o_proj + residual`, `gate_up + SwiGLU`, `fused_add_rmsnorm`, `silu_and_mul`, a quant, a standalone RoPE.
  → do **nothing** here; run the existing single-operator flow unchanged.
- **DECOMPOSE** — the input is a **composite of more than one separable op**: ≥ 2 ops that each merit their own
  kernel boundary because they have different execution regimes (a GEMM vs an attention core), a reduction
  crosses between them, or they are simply distinct heavy ops. This is **not** limited to a whole layer.
  Examples:
    - `rope + attention` (memory-bound elementwise + attention core — two regimes)
    - `attention + moe` (two heavy ops)
    - `qkv_proj + attention`, `attention + o_proj`, `norm + GEMM` (when the norm is its own pass)
    - a full decoder layer: `norm → QKV → attention → o_proj → residual → norm → MLP → residual`
  …or the user explicitly asks to split.

**Tie-breaker:** when in doubt about a *single heavy op wrapped in light epilogues*, BYPASS — a GEMM/attention
with fusable bias/residual/activation/quant is one operator. But **any input with two or more separable ops
decomposes**, whether it is a two-op composite (`rope+attention`) or a whole layer. The §1 rules then decide
which sub-parts still fuse together into each boundary.

---

## 1. Fuse / Split decision rules

Given the layer's dataflow (ops in order, each with shapes/dtype and whether its output is materialized to HBM):

**Fuse two ops into one boundary when:**
- A producer feeds a consumer through a **small / elementwise** intermediate → fusing removes an HBM round-trip.
- The work is **epilogue** that can ride a GEMM's registers: bias, residual add, activation/SwiGLU, scaling,
  RoPE, quant/dequant, even top-k+softmax → fold it into the GEMM epilogue. *(The GEMM epilogue is the
  universal fuse surface.)*
- **Normalization** can serve as a GEMM prologue/epilogue without changing the tiling regime.

**Split into separate boundaries when:**
- The two sides need **different tiling / parallelism regimes** (a GEMM vs the attention core).
- A **reduction crosses** the candidate boundary (fusing would serialize it or blow up shared memory).
- A **large intermediate is reused** by multiple later consumers (materialize once).
- The op is the **attention core** — keep it its own flash-attention kernel.

**Keep monolithic (never split internally):** the flash attention core (QKᵀ→softmax→PV never touches HBM), and
the MoE token-sort / align kernel.

**Every boundary must** be a self-contained operator with fully specified inputs/outputs/dtype/shapes, so it can
be a standalone workspace (own roofline, own correctness test).

---

## 2. Canonical boundary catalog (evidence-backed)

These boundaries are not hypothetical — they are the operator granularity both SOL-ExecBench and atrex-bench
already ship. Use them as the default carving; adjust to the actual module.

### Dense decoder layer

| # | Boundary (operator) | Fuse/Split | Reference evidence |
|---|---|---|---|
| 1 | **add + input RMSNorm** (returns normed + residual_out) | Fuse | `atrex-bench/data/fused_add_rms_norm/`; `SOL-ExecBench/examples/triton/nemotron_rms_norm/`; `reference-projects/quack/quack/rmsnorm.py` |
| 2 | **QKV projection** (one GEMM), opt. **+ QK-RMSNorm + RoPE** | Fuse | `reference-projects/flash-attention/flash_attn/modules/mha.py:459,635` (`Wqkv`); `atrex-bench/data/fused_qkv_rope/`, `data/fused_qk_rmsnorm/`; `reference-projects/flashinfer/include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh:322` |
| 3 | **RoPE + KV-cache write** (reshape+scatter) | Fuse (or fold into 2/4) | `atrex-bench/data/reshape_and_cache/`; `reference-projects/flashinfer/flashinfer/rope.py:256`; `gpu-wiki/reference-kernels/amd/cdna/flydsl/FlyDSL/fused_rope_cache_kernel.py:203` |
| 4 | **attention core** (QK·softmax·V) | Split + one-piece | `atrex-bench/data/unified_attention/`, `data/attention_forward/`, `data/paged_attention_decode/`; `SOL-ExecBench/data/flashinfer-trace/.../gqa_paged` |
| 4b | split-KV → **combine/merge** (decode) | Split | `reference-projects/FlashMLA/csrc/smxx/decode/combine/combine.h:8`; `reference-projects/flashinfer/include/flashinfer/attention/cascade.cuh:45` |
| 5 | **o_proj + residual** | Fuse | `SOL-ExecBench/examples/cute_dsl/jamba_attn_proj/`; `reference-projects/cutlass/include/cutlass/epilogue/thread/linear_combination_residual_block.h:58` |
| 6 | **add + post-attn RMSNorm** | Fuse | `SOL-ExecBench/examples/triton/olmo3_post_norm/`; `atrex-bench/data/fused_add_rms_norm/` |
| 7 | **MLP gate+up (gated GEMM) + SwiGLU** | Fuse | `SOL-ExecBench/examples/pytorch/gemma3_swiglu/`; `atrex-bench/data/silu_and_mul/`; `reference-projects/quack/quack/gemm_act.py:211` (`GemmGatedMixin`); `reference-projects/aiter/aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py` |
| 8 | **down_proj + residual** | Fuse (split from 7) | cutlass residual epilogue (as #5); `gpu-wiki/docs/kernel-opt/amd/common/hands-on/moe-2stage-fusion.md` |

### MoE variant (authors split into ~6 boundaries)

| Boundary | Fuse/Split | Reference evidence |
|---|---|---|
| router logits **+ topk + softmax** | Fuse | `atrex-bench/data/moe_topk_gating_softmax/`; `reference-projects/cutlass/include/cutlass/epilogue/fusion/operations.hpp:146` (`LinCombTopKSoftmaxCol`) |
| **sort / align_block_size / count_sort** | Split (hard boundary) | `atrex-bench/data/moe_align_block_size/`, `data/moe_count_and_sort/`; `reference-projects/aiter/aiter/fused_moe.py:173` (`moe_sorting`) |
| expert **grouped GEMM-1 + SwiGLU** | Fuse | `atrex-bench/data/fused_moe/`; `reference-projects/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp:293` (`apply_swiglustep_activation`); `reference-projects/tilelang/examples/fusedmoe/` |
| expert **grouped GEMM-2 (down)** | Split from stage-1 | `reference-projects/aiter/fused_moe_bf16_asm.py:328` (`ck_moe_stage2`); `moe-2stage-fusion.md` |
| **combine / weighted sum-reduce** | Split | `atrex-bench/data/moe_sum_reduce/` |
| *(whole layer as mega-kernel)* | Fuse (aggressive) | `reference-projects/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh` |

### Cross-cutting fuse surfaces (apply on top of the above)

- **Quant/dequant folds into the adjacent GEMM/norm/act epilogue** — `atrex-bench/data/block_scaled_mm/`,
  `data/fused_rmsnorm_quant/`; `reference-projects/cutlass/include/cutlass/epilogue/fusion/operations.hpp:515`.
- **TP all-reduce fuses with the following residual + norm** —
  `reference-projects/hpc-ops/src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu`;
  `reference-projects/aiter/csrc/kernels/fused_ar_mhc_post.cu`.

---

## 3. Input Contract

| Parameter | Description |
|-----------|-------------|
| `layer_logic` | The whole LLM-layer PyTorch module / forward to decompose |
| `layer_dir` | Output directory for the manifest, per-boundary kernel_demos, and full-layer reference |
| `platform` | Target platform (e.g. H20, H100, MI308X) — for the per-boundary bound-axis / SOL |
| `roofline_py` | Path to `atrex-bench/scripts/roofline.py` (per-boundary SOL source) |
| `gpu_wiki_path` | gpu-wiki root (operator-level knowledge only) |

---

## 4. Output Contract

1. **`<layer_dir>/reference.py`** — the full-layer PyTorch reference (`run(...)` / `Model`), used later for the
   end-to-end recombine validation.
2. **`<layer_dir>/<boundary>/kernel_demo.py`** — one **basic, correct, runnable PyTorch reference per boundary**
   (the "basic fused kernel already written"), shaped like a SOL-ExecBench definition reference: a single
   `run(...)` with explicit input shapes/dtypes and the fused compute. Each becomes a workspace's `kernel_demo`.
3. **`<layer_dir>/boundaries.json`** — the manifest the coordinator owns. One entry per boundary, in dataflow
   order:
   ```json
   {
     "layer_name": "<name>",
     "reference": "reference.py",
     "shapes": {                     // atrex-bench shapes.json body: the FULL shape set, integer sids
       "0": { "init_kwargs": null, "input_kwargs": { "batch_size": 2, "seq_len": 8192 } },
       "1": { "init_kwargs": null, "input_kwargs": { "batch_size": 1, "seq_len": 128 } }
     },
     "boundaries": [
       {
         "name": "qkv_proj",
         "op_type": "gemm",          // gemm | attention | norm | elementwise | moe_gemm | sort | reduce | other
         "kernel_demo": "qkv_proj/kernel_demo.py",
         "dtype": "bf16",
         "bound": "compute",         // compute | memory
         "roofline": {               // atrex-bench roofline.json body, per-shape SOL keyed by the SAME sids
           "shapes": {
             "0": {
               "semantic_W_flops": { "bf16": 824967265152 },
               "semantic_Q_read_bytes": 0, "semantic_Q_write_bytes": 0,
               "SOL_time_ms": { "<platform>": 0.123 }
             }
           }
         },
         "ceiling": 0.85             // expected achievable %SOL (see §5)
       }
     ]
   }
   ```
   **SOL is per-shape, over the entire shape set — never one hand-picked "representative" shape.** Convert
   every entry of the layer's workload into the atrex-bench `shapes.json` body (integer sids `"0","1",…`,
   axes under `input_kwargs`). For each boundary, run `roofline.py` **once per sid** and record it in that
   boundary's `roofline` body under `SOL_time_ms[<platform>]`. This matches how the op is actually scored
   (every shape scored independently) and is essential because op cost varies with the axes — attention
   scales ∝ B·S², so a single shape's SOL is meaningless for the rest. Two correctness rules when calling
   `roofline.py`:
   - use the operator's **declared dtype** (from the definition / metadata), not whatever the reference
     happens to upcast to internally;
   - for **causal** attention, count causal FLOPs (~½ of the full S×S), not the dense S×S.

   The orchestrator materializes each boundary's `shapes.json` + `roofline.json` into its workspace from
   this manifest; the integer sid is the join key across `shapes.json`, `roofline.json`, and every version's
   `performance.latency_us_by_shape`. Record an **evidence chain** for each fuse/split decision
   (`op(s) -> decision -> why`) in `<layer_dir>/decomposition.md`.

The coordinator then creates one standard atrex workspace per boundary (`kernel_demo` = the boundary's
reference), and each workspace optimizes independently.

---

## 5. Expected `%SOL` ceilings (sanity only — NOT termination gates)

`ceiling` is the fraction of the analytical SOL a well-tuned kernel of that class can realistically reach. It is
used only to bound the coordinator's ROI metric and to sanity-check where a boundary plateaus — it is **never** a
termination target (analytical SOL is unreachable for many ops; gating on it would run forever). Defaults:

| op_type | ceiling (%SOL) | note |
|---|---|---|
| gemm / moe_gemm (compute-bound) | ~0.85 | cuBLAS-class on healthy shapes |
| attention (prefill, compute-bound) | ~0.72 | FA3/FA4-class (softmax + online rescaling + causal waste keep it below GEMM) |
| norm / elementwise / reduce (memory-bound) | ~0.85 | vs HBM BW |
| sort / scatter (control-heavy) | ~0.70 | poor coalescing; plateaus early |

Prefer measuring a strong reference (cuBLAS / FlashAttention / cuDNN) for the exact op+shape+dtype and using its
utilization as the ceiling when available; otherwise use the defaults above.

---

## 6. Constraints

- **DO NOT** run when the gate says BYPASS — the single-operator path must stay untouched.
- **DO NOT** author graph/layer-partitioning content into gpu-wiki.
- **DO NOT** emit a boundary that is not independently runnable and correctness-testable.
- **DO NOT** fabricate or hand-pick shapes — convert the layer's entire workload into the atrex-bench
  `shapes.json` body (all of it, integer sids `"0","1",…`, axes under `input_kwargs`); derive dtypes from the
  operator definition / `layer_logic`; ask if genuinely ambiguous.
- **DO NOT** optimize kernels here — decomposition only. Optimization happens per-boundary in the normal pipeline.
- **DO NOT** re-fuse across boundaries later — once drawn, boundaries are fixed for the campaign.
