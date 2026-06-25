# NCU Profile-Driven Optimization Workflow

This document outlines the Nsight Compute-driven GPU kernel optimization workflow: first collect reproducible profiles, then construct a reasoning chain using counter / source / PM sampling / PTX-SASS evidence, and finally select only one verifiable kernel modification.

Core principle:

```text
profile first, diagnose second, optimize third
```

It is not simply about determining "memory-bound / compute-bound", but rather connecting measured metrics into:

```text
counter signal → possible mechanism → exclude other hypotheses → one concrete next edit
```

## Applicable Scenarios

Prioritize using NCU-driven optimization in the following scenarios:

- The baseline benchmark is already correct, but there is no baseline NCU report or comparable profile evidence yet.
- The candidate is correct, performance is close to the baseline / current best version, and the next direction is unclear.
- The candidate regresses on one or more shapes, and the mechanism needs to be explained.
- The candidate is noticeably faster, and profile evidence is needed to prove the source of the optimization.
- Blackwell / Hopper kernels may involve pipeline bubbles, tail effects, TMA / mbarrier waits, tensor pipe underuse, inline PTX hotspots, or source-correlated stalls.
- Review requires profile evidence to support optimization conclusions.

Do not perform performance profiling when correctness fails, unless the issue itself is NCU collection failure.

## Recommended Artifact Structure

Keep an independent directory for each optimization attempt for easy version comparison:

```text
profile-artifacts/${version}/
  report.ncu-rep
  raw.csv
  details.txt
  source.csv                 # Exported when NCU supports source page
  sampling.csv               # PM sampling or related sampling metrics export
  kernel.ptx                 # Exported for instruction-level analysis
  kernel.sass                # Exported for instruction-level analysis
```

## Standard Workflow

1. First select a representative shape. Prioritize the smallest shape that still exposes regression, plateau, tail effects, or suspected bottlenecks.
2. Build a stable harness: fix warmup, dtype, shape, seed, and kernel-name pattern, and ensure baseline and candidate use the same command.
3. Collect a complete NCU report: include `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `Occupancy`, `LaunchStats`, `MemoryWorkloadAnalysis`, `SourceCounters`.
4. If `ncu --list-sections` shows `PmSampling` or `PmSampling_WarpStates`, add the PM sampling section tile to observe timeline stalls and long-tail evidence.
5. Export `raw.csv` and `details.txt`; if the NCU version supports it, additionally export `source.csv` and sampling information.
6. When encountering source-correlated hotspots, inline PTX, codegen-sensitive, or instruction-selection-sensitive issues, export PTX / SASS and align hotspot source rows to the generated instructions.
7. The candidate must be compared against baseline / parent, not just evaluated by absolute counters.
8. Diagnose bottlenecks by metric groups, and clearly state counter signals, possible mechanisms, weaker assumptions, and risks.
9. Finally, select only one next concrete edit and specify the verification method and expected metric changes.

## Common Collection Commands

### Full Baseline Capture

```bash
version=v000_baseline
out=profile-artifacts/$version
kernel_name_regex="gemm|attention|decode"
profile_shape="m=4096,n=4096,k=4096"
profile_dtype="bf16"
mkdir -p "$out"

ncu --target-processes all \
    --kernel-name regex:"$kernel_name_regex" \
    --launch-skip 5 --launch-count 1 \
    --set full --import-source on \
    --section SpeedOfLight \
    --section SchedulerStats \
    --section WarpStateStats \
    --section Occupancy \
    --section LaunchStats \
    --section MemoryWorkloadAnalysis \
    --section SourceCounters \
    -o "$out/report" \
    python benchmarks/bench.py --impl baseline --shape "$profile_shape" --dtype "$profile_dtype"

ncu --import "$out/report.ncu-rep" --page raw --csv > "$out/raw.csv"
ncu --import "$out/report.ncu-rep" --page details > "$out/details.txt"
```

### Candidate Using the Same Harness

```bash
version=v001_candidate
out=profile-artifacts/$version
kernel_name_regex="gemm|attention|decode"
profile_shape="m=4096,n=4096,k=4096"
profile_dtype="bf16"
mkdir -p "$out"

ncu --target-processes all \
    --kernel-name regex:"$kernel_name_regex" \
    --launch-skip 5 --launch-count 1 \
    --set full --import-source on \
    --section SpeedOfLight \
    --section SchedulerStats \
    --section WarpStateStats \
    --section Occupancy \
    --section LaunchStats \
    --section MemoryWorkloadAnalysis \
    --section SourceCounters \
    -o "$out/report" \
    python benchmarks/bench.py --impl candidate --shape "$profile_shape" --dtype "$profile_dtype"

ncu --import "$out/report.ncu-rep" --page raw --csv > "$out/raw.csv"
ncu --import "$out/report.ncu-rep" --page details > "$out/details.txt"
```### Quick Retest of Core Metrics

Once you have a complete baseline, you can collect only key metrics for candidate versions:

```bash
kernel_name_regex="gemm|attention|decode"
profile_shape="m=4096,n=4096,k=4096"
profile_dtype="bf16"

ncu --target-processes all \
    --kernel-name regex:"$kernel_name_regex" \
    --launch-skip 10 --launch-count 1 \
    --import-source on \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_bytes.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__warp_issue_stalled_long_scoreboard.sum,\
smsp__warp_issue_stalled_short_scoreboard.sum,\
smsp__warp_issue_stalled_barrier.sum,\
smsp__warp_issue_stalled_membar.sum,\
smsp__warp_issue_stalled_mio_throttle.sum,\
smsp__warp_issue_stalled_no_instruction.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    -o profile-artifacts/v002_fast/report \
    python benchmarks/bench.py --impl candidate --shape "$profile_shape" --dtype "$profile_dtype"
```

If a metric does not exist, first check what is supported on the current machine:

```bash
ncu --query-metrics | grep -E 'long_scoreboard|bank_conflict|pipe_tensor|throughput'
```

## PM Sampling and Source Hotspots

Aggregate metrics can only indicate "where there might be a problem." Source / PM sampling is used to pinpoint "when, which line of code, which instruction window."

```bash
ncu --list-sections | grep -E '^PmSampling'

# English note
#   --section PmSampling --section PmSampling_WarpStates

ncu --import profile-artifacts/v001_candidate/report.ncu-rep \
    --page source --csv > profile-artifacts/v001_candidate/source.csv || true
ncu --import profile-artifacts/v001_candidate/report.ncu-rep \
    --page raw --csv | grep -Ei 'sampling|sample|stall|source' \
    > profile-artifacts/v001_candidate/sampling.csv || true
```

If `--page source` is unavailable, note the absence in the optimization record Tudor and use the closest section in `details.txt` to support your analysis.

## PTX / SASS Alignment Analysis

PTX / SASS inspection is needed in the following situations:

- SourceCounters point to inline PTX or generated code hotspots.
- `no_instruction` / `imc_miss` suggest front-end or code size pressure.
- High-level code appears correct, but you suspect scalarized loads, extra conversions, spills, barriers, or incorrect cache modifiers.

```bash
NVCC_FLAGS="-lineinfo -Xptxas=-v" python setup.py build_ext --inplace

version=v001_candidate
out=profile-artifacts/$version
mkdir -p "$out"

binary_path="build/libkernel_extension.so"
cubin_path="build/kernel.cubin"
kernel_name_regex="gemm|attention|decode"
hot_source_regex="load_qkv|mma_loop|store_output"
inline_ptx_mnemonic_regex="tcgen05|wgmma|ldmatrix|cp.async|mbarrier|bar.sync|ld.global|st.global|cvt|selp|pred"

cuobjdump --dump-ptx "$binary_path" > "$out/kernel.ptx" || true
cuobjdump --dump-sass "$binary_path" > "$out/kernel.sass" || true
nvdisasm -g "$cubin_path" > "$out/kernel.nvdisasm.sass" || true

grep -E -n "$kernel_name_regex|$inline_ptx_mnemonic_regex" "$out/kernel.ptx" "$out/kernel.sass"
grep -E -n "$hot_source_regex|$inline_ptx_mnemonic_regex" "$out/source.csv" "$out/details.txt" "$out/kernel.sass"
```

Common findings and edit families:

| PTX / SASS Signal | Possible Mechanism | Priority Fix Direction |
|---|---|---|
| Expected vector load but generated scalar `ld.global` | Alignment / aliasing not proven | Add vector path, alignment guard, `__restrict__` |
| Low bytes/sector and load scalarized | Poor coalescing or load scalarized | Vectorize, adjust layout / stride |
| Local memory load/store appears | Register spill | Reduce tile / unroll / live range |
| Repeated `cvt` / pack / unpack | Dtype or epilogue path mismatch | Fix dispatch, accumulator, or quant layout |
| Excess `bar.sync` / mbarrier | Excessive synchronization or phase lifetime error | Simplify barrier lifetime, adjust pipeline stages |
| `tcgen05` / `wgmma` sparse or missing in tensor kernel | Lowering took the wrong path | Force tensor path or adjust kernel stack config |
| Large instruction window with `imc_miss` | I-cache / front-end pressure | Split / specialize kernel, reduce unroll |
| Cache modifier not as expected | Incorrect cache policy | Use DSL cache-policy primitive or targeted inline PTX |## Metric Grouping and Interpretation

### Execution and Launch Identity

| Metric | Purpose |
|---|---|
| Kernel name / demangled name | Verify that the captured target is the intended kernel |
| `gpu__time_duration.sum` | kernel duration, should align with benchmark timing |
| `launch__grid_size` | Determine whether the grid is too small or underfilled |
| `launch__block_size` | Confirm the CTA shape |
| `launch__registers_per_thread` | register pressure and occupancy limiter |
| `launch__shared_mem_per_block_static` / `dynamic` | shared memory budget |
| `launch__occupancy_limit_registers` / `shared_mem` / `blocks` | reasons for occupancy limitations |

### Speed-of-Light

| Metric | Explanation |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | overall SM pressure |
| `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed` | tensor pipe utilization |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` | ALU / SFU / epilogue pressure |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM pressure |
| `lts__t_bytes.avg.pct_of_peak_sustained_elapsed` | L2 traffic pressure |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | coarse-grained compute / memory roofline hints |

High tensor, low memory usually indicates compute/tensor-bound; high memory, low SM usually indicates bytes/layout/coalescing issues; both low usually requires investigating latency, scheduler, occupancy, barriers, launch overhead, or tail effects.

### Scheduler and Warp State

| Metric | Explanation |
|---|---|
| `smsp__warps_eligible.avg.per_cycle_active` | low values indicate the scheduler frequently has no ready warp |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved active warp occupancy |
| `smsp__inst_executed.avg.per_cycle_active` | issue rate |
| `smsp__warp_issue_stalled_long_scoreboard.sum` | waiting on global / L2 / texture memory dependency |
| `smsp__warp_issue_stalled_short_scoreboard.sum` | waiting on shared-memory or MIO dependency |
| `smsp__warp_issue_stalled_barrier.sum` | CTA barrier / named barrier pressure |
| `smsp__warp_issue_stalled_membar.sum` | memory barrier / ordering pressure |
| `smsp__warp_issue_stalled_mio_throttle.sum` | MIO pipe saturated, often adjacent to shared / TMA / ldmatrix |
| `smsp__warp_issue_stalled_no_instruction.sum` | front-end starvation, divergence, code size, or I-cache |
| `smsp__warp_issue_stalled_imc_miss.sum` | instruction-cache miss |
| `smsp__warp_issue_stalled_not_selected.sum` | warp ready but not selected; may be a healthy signal if many warps are ready |

`smsp__warp_issue_stalled_*` should be normalized to a percentage. The largest stall is not necessarily the root cause, but it narrows the hypothesis space for further investigation.

### Memory Path

| Metric | Explanation |
|---|---|
| `dram__bytes_read.sum` / `dram__bytes_write.sum` | actual HBM bytes |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM utilization |
| `lts__t_bytes.sum` | L2 bytes, can be compared against DRAM bytes to infer L2 reuse |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1/TEX global-load sectors |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | global-load request count |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | global load sector utilization |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `st.sum` | shared-memory bank conflict |

If the bank-conflict metric name changes across NCU versions, consult `bank_conflict` and select the closest shared-memory load / store conflict counter.

### Hopper / Blackwell Specific Signals

```bash
ncu --query-metrics | grep -E 'wgmma|tcgen05|tensor|tma|mbarrier|tmem|cluster'
```| Signal | Explanation |
|---|---|
| tensor pipe active below expectations | Wrong dtype path, tile too small, heavy epilogue, or pipeline bubble |
| TMA / mbarrier wait hotspot | Stage count, phase tracking, arrive/wait lifecycle, or producer/consumer split issues |
| TMEM load / store hotspot | Epilogue drain or accumulator movement becoming a bottleneck |
| 2-SM cooperative imbalance | CTA pair scheduling or cluster shape mismatch |
| CLC / tail-wave evidence | Static scheduling leaving low-utilization tail waves |
| WGMMA wait-group hotspot | Pipeline depth, wait group placement, or smem visibility issues |

## Mapping Diagnosis to Modifications

| Dominant Signal | Priority Edit Family |
|---|---|
| long scoreboard | prefetch, coalesce, vectorize, reduce footprint, shared staging |
| short scoreboard + bank conflict | shared layout padding / swizzle, reduce smem round trip |
| barrier / membar / mbarrier | Fix phase tracking, stage count, arrive/wait lifecycle, warp specialization |
| tensor pipe low | Enlarge tile, force tensor path, lighten epilogue, deepen pipeline |
| DRAM high, SM low | Reduce bytes, fuse, vectorize, improve layout |
| L2 high, DRAM low | Increase reuse, persistent tiles, adjust cache policy |
| Both SM and memory low | Check occupancy, divergence, launch overhead, tail waves, PM timeline |
| I-cache / no-instruction | Reduce unroll / code size, split or specialize kernel |
| tail waves | Persistent scheduling, CLC, tile splitting, shape-aware routing |

## Output Rules

- End with exactly one next edit; do not write "try multiple approaches."
- Must reference the metric values that support the diagnosis.
- When a comparable report exists, must compare against baseline / parent.
- When key metrics are missing, state the omission and use the closest available section; do not fabricate values.
- If the kernel already exceeds approximately 85% of the relevant tensor / SM / DRAM peak and there is no low-cost edit, consider shape routing, fusion, launch overhead, or stop conditions.
- `.ncu-rep`, CSV, source / sampling exports, and PTX / SASS paths must be written to the optimization ledger.

## Related Documentation

- [NVIDIA Nsight Compute (NCU) Profiling Guide](ncu-profiling-guide.md) — Complete manual for NCU commands, sections, and metric system.
- [Nsight Profiling in Practice](nsight-profiling-practice.md) — Practical cases for Nsight Systems / Compute.
- [NCU rule est. speedup meta rules](ncu-rule-est-speedup-meta-rules.md) — Understanding the boundaries of NCU rule estimated speedup.
- [PTX/SASS Programming](ptx-sass-programming.md) — Instruction-level hotspot analysis and PTX/SASS background.
