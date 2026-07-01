# CLAUDE.md — atrex-kernel-agent (AKA)

Guidance for Claude when using the `gpu-kernel-optimizer` skill to implement/optimize GPU kernels
(including SOL-ExecBench cases). This file is the anti-cheat **policy** (guidance, not a code gate).

> Note on scope of effect: a repo-level `CLAUDE.md` is auto-loaded only when the working directory
> is inside this repo. When you optimize a SOL case with CWD elsewhere (e.g. `/home/admin/SOL-ExecBench`),
> this file may not be auto-loaded — the skill's `SKILL.md` (always loaded on skill invocation) carries
> a short pointer back here. Follow these rules whenever you produce a kernel/`solution.json`.

## Anti-cheat policy (default: self-written kernel)

When the task requires a self-written kernel (the default for SOL-ExecBench's "no调库 / no library
delegation" intent), the operator's **core compute must be a kernel YOU write and launch from
`run()`** using Triton (`@triton.jit`), CuteDSL (`@cute.jit`/`@cute.kernel`), cuTile, FlyDSL, or
inline CUDA. Do NOT do any of the following:

- **C1 — Library delegation.** Do not import/call `flashinfer`, `flash_attn`, `xformers`, `vllm`, or
  `aiter`; do not use `torch.nn.functional.scaled_dot_product_attention` as the compute path; do not
  wrap the benchmark's target library op. `torch` is for setup / allocation / reshape / indexing /
  launch glue only (a lone `F.linear` inside an otherwise real fused kernel needs justification).
- **C2 — Language-tag camouflage.** Do not declare a framework in `solution.json` that you do not
  actually launch from `run()`, and do not paste dead `@cute.kernel` / `@triton.jit` decorators "for
  language classification". Label `spec.languages` by the framework actually on the data path.
- **C3 — Input/shape-keyed memoization.** No process-global caches or `lru_cache` keyed on input-shape
  metadata that move per-call work (host sync, python loops, H2D copies) out of the timed region.
  SOL's allocator only varies `data_ptr`; keying on shape to defeat it is a cheat.
- **C4 — Timing-methodology gaming.** Do not rely on the allocator's pre-zeroing or CUPTI GPU-span
  quirks; your kernel must write **all** output bytes.
- **C5 — Masked-error PASS.** Use SOL's exact tolerance (`max_atol` / `max_rtol` /
  `required_matched_ratio` / `max_error_cap` / NaN-Inf / `allow_negative_inf`), never a looser global
  `rel_err`.
- **C6 — Fabricated target.** Never invent a performance target or leaderboard number. `T_b` must be a
  **measured** baseline: a library impl (FlashInfer / DeepGEMM / cuDNN / torch) measured through the
  SOL harness, cross-checked against the leaderboard "Scoring Baseline" row. `SOL Score` needs the
  hidden per-workload `T_SOL` — report a clearly-labelled roofline **estimate** or `N/A`, never a
  fabricated official number.

If the user **explicitly allows libraries** (`allow_libs` intent), C1 is relaxed; **C2–C6 still hold.**

## SOL-ExecBench: always report the four leaderboard metrics

For any SOL-ExecBench result, report all four (see `tools/sol_metrics.py`, `tools/fetch_leaderboard.py`):
- **Latency** = median over workloads of per-workload median `T_k` (EXACT)
- **Fast** = `count(T_k < T_b)/N` vs the measured baseline (EXACT)
- **Avg Speedup** = `mean(T_b / T_k)` vs the **Scoring Baseline** `T_b`, not the naive reference (EXACT)
- **SOL Score** = `mean 1/(1+(T_k−T_SOL)/(T_b−T_SOL))` — labelled roofline-`T_SOL` estimate, or `N/A`

Performance **target = match/beat the leaderboard top-3** (fetch with `tools/fetch_leaderboard.py`),
not merely the roofline peak.
