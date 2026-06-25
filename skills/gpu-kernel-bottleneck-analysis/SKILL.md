---
name: gpu-kernel-bottleneck-analysis
description: Helper skill for GPU kernel bottleneck analysis. It provides Roofline analysis, same-size bandwidth baseline measurement, TFLOPS and bandwidth utilization calculation, and profiling evidence extraction. This skill is not a standalone stage; its capabilities are reused by Step 0 and by the first Stage 2 iteration.
---

# GPU Kernel Bottleneck Analysis

## Positioning

This skill is a helper toolkit, not an independent router stage. In the full optimization workflow:

- **Step 0** reuses Roofline analysis to write hardware specs, Roofline results, and Stop Conditions into workspace `README.md`.
- **The first Stage 2 iteration** reuses bottleneck analysis to produce ISA-level optimization targets.

It may also be triggered directly when the user asks to:

- Analyze why a kernel is slow.
- Run Roofline analysis.
- Calculate bandwidth or compute utilization.
- Locate bottlenecks from profile results.

## Workflow

1. Identify the compute pattern and estimate whether it is more likely compute-bound or memory-bound.
2. Run tile-level Roofline analysis.
3. Calculate current-kernel TFLOPS and bandwidth in GB/s, compare them with target-platform theoretical peaks, and report utilization percentages.
4. If memory-bound, measure same-size memcpy bandwidth ceiling.
5. Run official profiling tools and collect evidence.
6. Extract at least one concrete bottleneck evidence item.
7. Format the evidence as optimization input for the next stage.
8. Write the analysis to an archive file: workspace `README.md` in the full flow, or `bottleneck_report.md` for standalone use.

## Mandatory Constraints

- Optimization direction must be guided by official profiling tools.
- Do not replace profile evidence with ad-hoc timing.
- Do not output vague claims such as "there may be a bottleneck"; include concrete values, metrics, or instructions.
- Performance evaluation must include both TFLOPS and bandwidth in GB/s. Do not rely on latency alone.

## Common Tools

```bash
python tools/compute_utilization.py   --gpu h20 --dtype bf16   --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2"   --time-ms 0.5 --grid-blocks 64
```

```bash
python tools/bench_bandwidth.py --size-bytes <kernel_data_bytes>
python tools/measure_bandwidth_ceiling.py --size-bytes <kernel_data_bytes>
```

NVIDIA:

```bash
ncu --set full --launch-skip <N> --launch-count 1 -o ./profile python kernel.py
```

AMD:

```bash
bash tools/profile_kernel.sh python kernel.py
python tools/extract_asm.py kernel.py --output kernel.s
```

## Bottleneck Evidence Examples

- `Memory throughput` reaches a high ratio but `SOL%` is low.
- `L2 hit rate` is low, suggesting non-coalesced global loads.
- `ds_read` or `ds_write` shows bank-conflict stalls.
- High `VGPR` usage lowers occupancy.
- `buffer_load_dword` appears instead of `dwordx4`, indicating insufficient vectorization.
- `MMA/MFMA` utilization is low while address calculation or type conversion dominates.

## Output Requirements

The output must include:

- Bound classification
- Roofline results
- TFLOPS and bandwidth in GB/s, plus peak utilization percentages
- Same-size memcpy bandwidth baseline
- One or more citeable bottleneck evidence items
- Suggested optimization search keywords

For the full optimization flow, write the result to workspace `README.md`; for standalone use, write `bottleneck_report.md`:

```markdown
# Bottleneck Analysis Report

## Bound Type
<compute bound / memory bound>

## Roofline Analysis
<tile-level Roofline analysis>

## Performance Evaluation
| Metric | Current | Theoretical Peak | Utilization(%) |
|--------|---------|------------------|----------------|
| TFLOPS | | | |
| Bandwidth(GB/s) | | | |
| memcpy baseline bandwidth(GB/s) | | | |

## Bottleneck Evidence
<at least one concrete evidence item with values, metrics, or instructions>

## Optimization Direction
<search keywords and proposed directions>
```
