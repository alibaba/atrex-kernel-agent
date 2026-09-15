# Metric scope, names, and units

Metric names depend on the GPU, Nsight Compute version, and collected sections. The families below
come from the upstream NCU material, including its B200/sm_100 examples; they do not assert that the
current target is B200 or that every counter was collected. Prefer actual names in the saved result
or `action.metric_names()` from an available raw report. Never infer a GPU chip identifier from an
unrelated example.

| Question | Example metric families |
| --- | --- |
| Launch/resources | `launch__grid_size`, `launch__block_size`, `launch__waves_per_multiprocessor`, `launch__registers_per_thread`, `launch__shared_mem_per_block`, `launch__occupancy_limit_*` |
| Occupancy | `sm__maximum_warps_per_active_cycle_pct`, `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Duration/throughput | `gpu__time_duration.sum`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes_read.sum.per_second` |
| Memory volume | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Cache/transactions | `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| Instruction mix | `smsp__sass_inst_executed_op_global_ld.sum`, `smsp__sass_inst_executed_op_local_ld.sum`, `smsp__sass_inst_executed_op_local_st.sum` |
| Tensor/compute pipelines | `sm__pipe_tensor_cycles_active.*`, `sm__inst_executed_pipe_fma.*`, `sm__pipe_fp64_cycles_active.*` |
| Aggregate stalls | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |
| Source-correlated stalls | `smsp__pcsamp_warps_issue_stalled_<reason>`, `smsp__pcsamp_sample_count` |
| Temporal samples | `pmsampling:smsp__warps_issue_stalled_<reason>.avg` and available SM/DRAM series |

## Preserve the denominator

- `.sum`, `.avg`, and `.max` are different aggregates, not interchangeable aliases.
- `pct_of_peak_sustained_elapsed` uses elapsed time; `pct_of_peak_sustained_active` uses active
  cycles. Compare like denominators and distinguish active-SM efficiency from whole-launch use.
- Check `metric.unit()` before converting values. `gpu__time_duration.sum` is commonly in ns,
  but a guessed unit can create a thousand-fold error.
- Compute sectors/request only with matching operation, scope, and nonzero request count. Expected
  values depend on active lanes, instruction width, alignment, and access layout.
- Add read and write bytes only when both components exist with compatible scope. A missing
  `dram__bytes.sum` is not evidence of zero traffic.
- A profiler kernel duration excludes parts of the evaluator's callable path; do not substitute it
  for full-evaluation latency or average it together with that latency.

## Missing is not zero

An absent metric may be uncollected, renamed, unsupported, or unavailable for the selected action.
A numeric zero can be a real counter or a synthetic value with missing dependencies. Inspect its
availability and collection context before drawing conclusions.

For raw reports, preserve `None`/missing values. Check `num_instances()` before interpreting a
time series and `has_correlation_ids()` before attributing samples to PCs. Do not divide by empty
sample counts, replace missing metrics with zero, or print every counter merely to find one fact.
If the required evidence cannot be obtained through permitted requests, report that limitation.
