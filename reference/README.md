# Kernel Opt Session Configuration

## Target
- platform: <H20 / MI308X / MI355X / ...> (parsed from user input)
- arch: <Hopper / CDNA3 / CDNA4 / ...> (derived from platform)
- framework: <CuteDSL / Gluon / FlyDSL / Triton / ...> (parsed from user input)
- dtype: <bf16 / fp16 / fp8 / ...>
- kernel type: <GEMM / Attention / Norm / MoE / ...>

## Execution
- execution_mode: local
- local_gpu: <local GPU model>

## Inputs
- kernel_demo: <path to the initial kernel implementation file> (parsed from user input and copied into the workspace as kernel.py)
- shapes: <...>
- reference: <path to reference.py>
- correctness threshold: rel_err < 0.01 (bf16 default) / 0.05 (fp8/fp4)

## Hard Constraints
- profiling: trust only ncu (tools/profile_nvidia.sh) / tools/profile_kernel.sh;
- performance targets: sourced from gpu-wiki; see Stop Conditions
- workspace: the run root (your current working directory); commit every accepted iteration with git
- knowledge base: `/tmp/aka-opt/gpu-wiki/`; the entire repository may be searched, not only docs/ or reference-kernels/;
- reference_project: `/tmp/aka-opt/reference-projects/` (optional source of similar optimized kernels)

## Additional Notes (parsed from user input)
- <extra information such as constraints, known bottlenecks, preferred optimization directions, edge cases; use none if empty>

## Tools (top-level paths)
- compute_utilization.py / bench_bandwidth.py / measure_bandwidth_ceiling.py
- measure_kernel_time.py / extract_asm.py / profile_kernel.sh
- profile_nvidia.sh / classify_ncu.py / extract_nvidia_asm.py (NVIDIA; helpers in ncu_helpers/)

## Stop Conditions
The following targets are filled by Step 0 after calculation as `hardware peak * 90%`. Prefer measured maxima from gpu-wiki when available; otherwise use hardware spec values.

- compute-bound target: <dtype> TFLOPS >= <peak * 90%> T, for example MI308X bf16 peak 206T -> target >= 185.4T
- memory-bound target: bandwidth >= <peak * 90%> TB/s, for example MI308X HBM peak 4.3TB/s -> target >= 3.87TB/s
- end-to-end latency: measured via `triton.testing.do_bench(kernel_fn, warmup=N, rep=N)` (p50 median, in ms); use this as the performance data source for TFLOPS and bandwidth calculation

## Task Context
- platform:
- arch:
- framework:
- dtype:
- shapes:
- correctness_threshold:
- stop_condition:
- reference_project:

## ISA Optimization Targets

### AMD
- Global memory: increase the share of `buffer_load_dwordx4`; avoid heavy use of `buffer_load_dword` and `buffer_load_dwordx2`.
- LDS memory: increase the share of `ds_read_b128` and `ds_write_b128`.
- Registers: keep `vgpr_spill_count == 0`; keep `scratch_load` and `scratch_store` counts at 0.
- LDS conflicts: keep `SQ_LDS_BANK_CONFLICT` below the target threshold.
- Compute utilization: push `mfma_busy` / `valu_busy` toward the target threshold.
- Pipeline: keep the `memory dependency` share in warp stalls below the target threshold.
- Occupancy: keep active wavefronts per CU at the target threshold.
- Stall cycles: keep average `s_waitcnt` stall cycles below the target threshold.

### NVIDIA
- Memory throughput / SOL reaches the target threshold.
- L2 hit rate reaches the target threshold.
- Tensor Core utilization reaches the target threshold.
- Warp stall reason distribution remains below target limits.
- TMA / `cp.async.bulk` usage reaches the target share.
- Shared-memory bank conflict rate stays below the target threshold.
