# Diagnose the missing fact

Choose the relevant rows below; this is not a checklist of mandatory experiments. Compare the same
specialization, launch, workload, device, and collection conditions. Use opaque Shape IDs as given;
do not reconstruct private evaluation inputs or change the operator contract to fit a measurement.

| Observation | Competing explanations and useful evidence | Possible investigation |
| --- | --- | --- |
| Small grid or few waves per SM | Insufficient parallel work, large resource footprint, or an inherently small workload | Inspect grid, block size, SM count, and occupancy limits before splitting work |
| Achieved occupancy below theoretical occupancy | Short launches, partial waves, work imbalance, or resource/runtime effects | Correlate SM activity and duration; occupancy is not the optimization objective by itself |
| Long utilization tail | Unequal per-CTA work or a partial last wave | Compare per-SM activity and available temporal samples; consider work partitioning within the contract |
| High long-scoreboard stalls | Dependent memory access, low memory-level parallelism, or inefficient accesses | Examine hotspot instructions, DRAM throughput, cache behavior, and sectors/request together |
| High DRAM throughput | Potential bandwidth pressure, possibly alongside other limits | Estimate actual bytes moved and reuse; investigate avoidable transfers before launch-geometry tweaks |
| Low DRAM throughput despite slow execution | Latency, dependency chains, insufficient parallelism, or compute pressure | Do not call this bandwidth-bound merely because the code loads much data |
| Excess sectors or low useful bytes per sector | Strided/misaligned access, sparse active lanes, gather/scatter | Inspect actual lane addresses and instruction widths; a universal sectors/request threshold is misleading |
| Local-memory traffic | Register spills or deliberately addressable per-thread arrays | Check compiler spill diagnostics and source; reducing register count can introduce spills rather than help |
| Short-scoreboard stalls around shared accesses | Shared-memory dependencies or bank conflicts | Correlate shared transaction/wavefront counters with source access patterns before changing layout |
| Barrier/membar stalls | Required synchronization, arrival imbalance, or unnecessary synchronization | Establish dependency scope before removing or weakening barriers/fences |
| Math/LSU pipeline throttling | Pipeline saturation or avoidable instruction volume | Examine instruction mix; a busy pipeline is not automatically a defect |
| Low tensor activity in a matrix workload | Unsupported precision/shape, launch overhead, poor feed rate, or scalar implementation | Check generated instructions and data readiness; stay within the assigned DSL and numerical contract |
| Alternating compute/memory activity | Possible pipeline bubbles | Correlate time series with dependencies; investigate overlap only if correctness permits it |
| Atomic hotspots | Contention or required reduction work | Compare target distribution and operation count; evaluate hierarchical aggregation if valid |
| Low active lanes | Divergence, predication, or boundary work | Locate affected source regions and estimate how much total work they represent |
| Unexpected FP64 work | Promotions, constants, or intentionally wider computation | Check actual instructions and accuracy requirements before changing precision |

## Interpret stall evidence carefully

`long_scoreboard`, `short_scoreboard`, `wait`, `barrier`, and throttle counters describe different
waiting mechanisms. A per-issued-warp ratio is not a percentage of total wall time. For sampled
stall percentages, use the corresponding sample-count denominator, not a different aggregate.
`not_selected` can mean useful ready-warps are available; it is not inherently wasted work.

Source/PC correlation can localize where stalls were sampled, but the causal dependency may originate
earlier. Missing line information prevents source attribution; it does not mean there were no stalls.
Only request richer source evidence if that ambiguity matters to the next decision.

## Interpret timelines and estimates carefully

PM sampling may be sparse or absent for short kernels. Flat, tailed, or alternating curves suggest
hypotheses; they do not prove a mechanism. Different launches cannot be combined into one timeline.
If a specific intra-kernel ordering question remains, the optional Timeline Skill provides
instrumentation; its probe overhead must not be mistaken for an optimization gain.

NCU rule estimates are leads, not additive speedups or acceptance thresholds. Report the underlying
metric values, units, scope, and uncertainty in Journal analysis. Do not manufacture a complete
diagnosis from missing counters or force every Kernel through all of these investigations.
