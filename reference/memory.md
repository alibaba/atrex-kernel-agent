# Kernel Opt Session Memory

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

## Pitfalls

## Iteration Log
| Version | Evidence | Action | TFLOPS | Bandwidth(GB/s) | Gain |
|---------|----------|--------|--------|-----------------|------|
