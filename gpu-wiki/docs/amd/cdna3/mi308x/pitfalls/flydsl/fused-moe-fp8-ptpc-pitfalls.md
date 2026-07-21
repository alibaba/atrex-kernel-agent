# Pitfalls: FlyDSL FP8 PTPC Fused MoE on MI308X

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

This document records the non-obvious failure modes from the omoExplore `proj007
task66` pause checkpoint for FlyDSL FP8 PTPC Fused MoE on AMD MI308X (CDNA3 /
gfx942).

Companion report:

- [`cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md`](../../ref-docs/flydsl/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md)
- [`cdna3-fused-moe-fp8-ptpc-atrex-v2.md`](../../ref-docs/flydsl/cdna3-fused-moe-fp8-ptpc-atrex-v2.md)

Reference source:

- [`reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/`](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/)
- [`reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/`](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/)

## 1. Treating The Pause Checkpoint As A Completed Optimization

**Trap**: Because task66 has a clean table and a promoted source state, assume the FP8 PTPC Fused MoE kernel is complete.

**Result**: The full-scope gate is only `1/14`: stage1 token512 passes, while stage1 has `1/7` passes and stage2 has `0/7` passes.

**Why**: task66 intentionally freezes a pause checkpoint. It preserves current source, byte contract, negative boundaries, and remaining TODOs; it does not finish the optimization.

**Lesson**: Describe this archive as a continuation checkpoint. Future work starts at task67 and must revalidate `task66_current_gap_summary.csv` before claiming user-visible benefit.

## 2. Reusing BF16/F16 References For The FP8 PTPC Gate

**Trap**: Use BF16 or F16-derived reference rows when evaluating FP8 PTPC progress.

**Result**: The apparent target can become easier or inconsistent with the byte model, causing false pass/fail decisions.

**Why**: The task66 contract explicitly models FP8 GEMM input reads, BF16 stage/output activations, F32 scales, I32 metadata, and BF16 output atomic read-modify-write traffic.

**Lesson**: Use the corrected FP8 measured-byte CM2 load-only proxy for this gate. BF16/F16-derived rows are invalid for FP8 PTPC acceptance.

## 3. Recomputing `target_us` With The Wrong Timing Source

**Trap**: Recreate `target_us` with torch events, torch copy timing, theoretical peak bandwidth, default Python/runtime drift, or `cache_modifier=0` rows.

**Result**: The target can move by enough to create false pass/fail decisions, especially on small-byte rows where profiler variance is already visible.

**Why**: Task66 defines `target_5pct_us = 1.05 * fp8_loadonly_ref_us_cm2`. The reference is a FlyDSL load-only kernel over measured HBM bytes, timed through the `profile_cuda_kernels` device-time path by extracting `bw_kernel` device time. `cache_modifier=0` rows and copy/event timings are different measurements.

**Lesson**: Reproduce target rows with `bandwidth_reference.py` or the equivalent `measure_flydsl_load(..., cache_modifier=2, mode="loadonly", tiles_per_thread=4, warmup=10, iters=50, cold=True)` call. Treat other timing paths as diagnostics only.

## 4. Counting Skip-Atomic Or No-Output Stage2 As Positive Evidence

**Trap**: Treat a fast stage2 skip-atomic or no-output diagnostic as evidence that stage2 is solved.

**Result**: The GEMM body may approach large-token targets, but the candidate does not produce the required output.

**Why**: The real stage2 contract includes BF16 output atomic/read-modify-write traffic. Removing the final output path removes the main unresolved cost.

**Lesson**: Skip-atomic and no-output timings are diagnostic only. A promotable stage2 path must preserve BF16 output and correctness.

## 5. Trusting `block_m=8` Fast Timings

**Trap**: Promote a fast `block_m=8` result.

**Result**: The run can look fast but is not computing valid MFMA rows.

**Why**: The kernel computes `m_repeat = tile_m // 16`. With `tile_m=8`, the loop structure is invalid for the intended row computation.

**Lesson**: Treat existing `block_m=8` timings as invalid. Do not use them in performance tables or tuning decisions.

## 6. Promoting The Full Intermediate Reduce Path Because It Is Correct

**Trap**: Replace atomic output with a full `[tokens, topk, model_dim]` intermediate reduce path because it avoids the stage2 atomic issue.

**Result**: The path is correct but slower in its tested form.

**Why**: The intermediate materialization and reduction cost exceed the benefit of avoiding the direct atomic output in the current implementation.

**Lesson**: Keep this path as a negative result unless a future design changes the data movement enough to beat the direct BF16 atomic contract.

## 7. Reusing Task65 Rowctx / Rowinfo-LDS As Promoted Source

**Trap**: Reapply task65 row-context or rowinfo-LDS changes because some rows show tiny local improvement.

**Result**: No task65 probe reaches the 5 percent gate; some changes regress important rows.

**Why**: The current rowctx/rowinfo-LDS reimplementation does not reproduce the older positive path. The best token512 row remains far above target, and token128 regresses.

**Lesson**: Task65 is documentation-only negative evidence. Do not retain its source changes without a new correctness and gate-passing proof.

## 8. Reintroducing Host-Sync `valid_blocks` On The v2 Profile Path

**Trap**: Use `num_valid_ids[0].item()` in the hot profile path to compute the exact stage launch bound every iteration.

**Result**: The profiler can show host synchronization or transfer-side artifacts, and full-pipeline parity no longer matches the atrex-open v2 trace guard.

**Why**: The atrex-open v2 profile path reached parity by avoiding per-iteration host reads. The standalone archive uses cached or upper-bound launch policy so the visible kernel trace stays in the same routing/quant/stage/overhead buckets as atrex-open.

**Lesson**: Keep exact host-derived `valid_blocks` out of the profile path. If exactness is needed for a diagnostic, run it outside the parity measurement and label it as diagnostic.

## 9. Removing The M=1 Stream Packet Boundary

**Trap**: Delete the M=1 stream marker because it launches no CUDA kernel and appears to be overhead-free dead code.

**Result**: M=1 stage1 and e2e timing become bimodal. The small-row profile can fail parity even when the classified kernels are otherwise aligned.

**Why**: The stream packet boundary preserves the timing boundary that the old `.item()` host sync accidentally provided, without adding host sync or profiler-visible memcpy.

**Lesson**: Preserve the M=1 stream packet boundary unless a replacement is validated by trace guard, stage1/stage2 sums, and repeated e2e checks.

## 10. Re-Sorting Stage2 When v2 Can Reuse Routing Output

**Trap**: Treat the archived stage1 and stage2 kernels as independent isolated kernels and add a separate stage2 sort before the second GEMM.

**Result**: Routing cost increases and the full pipeline no longer matches the atrex-open v2 trace, even if isolated GEMM timings look reasonable.

**Why**: The integrated v2 path is a full fused_moe pipeline. It relies on the AITER opus sorting result and reuses the routing metadata across the two FlyDSL stages.

**Lesson**: Preserve the integrated routing topology when migrating the full pipeline. Isolated stage harnesses are useful for task66 but are not proof of atrex-open v2 full-pipeline parity.

## 11. Treating M=1 Event Average As The Only Parity Signal

**Trap**: Fail or accept M=1 based only on a single e2e event average.

**Result**: A clean kernel trace can be misclassified as a regression, or a real trace regression can be missed if one event average happens to pass.

**Why**: M=1 has tiny stage kernels and visible event-average jitter. In task11, one full run saw `622.0us` M=1 e2e average while the kernel trace and kernel sum were aligned; the immediate M=1 rerun was `526.2us`, and the final full run passed at `530.4us`.

**Lesson**: Diagnose M=1 with trace guard first, then stage1/stage2 sums and e2e min, then e2e average. Rerun event timing only after the trace is clean.

## Use This / Do Not Use This

| Use | Do not use |
|---|---|
| task66 gap table as the pause checkpoint | task66 as proof of full optimization completion |
| FP8 measured-byte CM2 load-only target | BF16/F16-derived reference targets |
| `profile_cuda_kernels` `bw_kernel` device time | torch event/copy timing, theoretical peak, or cm0 rows as target_us |
| Stage2 BF16 atomic/output cost as the next bottleneck | Skip-atomic/no-output timings as acceptance evidence |
| `tile_m >= 16` valid MFMA row configurations | `block_m=8` fast timings |
| task65 as negative probe documentation | task65 rowctx/rowinfo-LDS source as promoted code |
| atrex-open v2 trace guard for full-pipeline migration | isolated-kernel-only migration that changes routing topology |
| M=1 stream event marker without host sync | deleting the boundary or reverting to `.item()` in the profile path |
| cached or upper-bound valid-block policy | per-iteration host reads for launch bounds |
| atrex-open full-pipeline profile parity | task66 bandwidth `target_us` as a full-pipeline latency target |
