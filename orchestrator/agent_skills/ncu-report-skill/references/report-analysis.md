# Analyze an existing raw NCU report

The normalized Profile result is the first choice. Use raw reports only when the needed counter,
source correlation, or time series is not present there. GPU request options and output sync are
defined by `skills/gpu-measurement/SKILL.md`; record/source lookup is defined by
`skills/runtime-records/SKILL.md`. Do not remeasure a Kernel just to obtain a different prose summary.

## Included helpers

| File | Purpose |
| --- | --- |
| [analyze_reports.py](../helpers/analyze_reports.py) | Extract curated metrics and compare supplied reports |
| [extract_stall_hotspots.py](../helpers/extract_stall_hotspots.py) | Attribute sampled stalls to available source locations |
| [plot_timeline.py](../helpers/plot_timeline.py) | Render available per-instance temporal metrics |
| [ncu_utils.py](../helpers/ncu_utils.py) | Shared report loading, safe access, and metric extraction; imported by the other helpers |

These helpers parse report files; they do not submit jobs or establish acceptance. They use the
`ncu_report` Python module from an existing Nsight Compute installation. When it is available only
on the GPU worker, run analysis there through Dev. If the module is unavailable, preserve the error;
do not install a system profiler or modify service configuration.

For example, after obtaining an actual report at `scratch/profile/target.ncu-rep`:

```bash
python3 tools/sandbox.py --kind dev --input skills/ncu-report-skill/helpers --input scratch/profile/target.ncu-rep --sync scratch/ncu-analysis -- python3 skills/ncu-report-skill/helpers/analyze_reports.py --run-dir scratch/ncu-analysis --report scratch/profile/target.ncu-rep --tag candidate
```

Replace the report path with the actual downloaded file; it is not a standardized Gateway filename.
The helper writes under `scratch/ncu-analysis/analysis/`. The other two entrypoints accept the same
`--run-dir`, repeatable `--report`, and matching `--tag` options. `extract_stall_hotspots.py` also
accepts `--top`; `plot_timeline.py` accepts `--metric`, `--rows`, and `--cols`. Upload all files read
by the command and sync only the needed output directory. No fixed run-directory hierarchy outside
`scratch/` or standalone Markdown report is required.

## Confirm which action you are reading

The supplied helpers select the first range/action from each report. Check that it is the intended
Kernel; this can be wrong for reports with multiple launches. Their multi-report comparison is not
a Shape aggregate or ABBA measurement. For other actions, write a small analysis script using the
available API and submit it through the same Dev mechanism:

```python
import ncu_report

report = ncu_report.load_report("scratch/profile/target.ncu-rep")
action = report.range_by_idx(0).action_by_idx(0)  # Select the intended range/launch.
print(action.name())
name = "gpu__time_duration.sum"
if name in action.metric_names():
    metric = action[name]
    print(name, metric.value(), metric.unit())
```

The helpers' curated counter list includes B200 names. Missing counters stay unknown on other
devices. Use [metric interpretation](metrics.md), then inspect only the relevant available names.
Raw source information requires matching debug/line data. Samples lacking PC correlations cannot
support source-line claims; unavailable PM samples cannot support a temporal conclusion.

Keep extracted values tied to the original Profile record and exact Kernel. Dev analysis success
means the script completed, not that it evaluated the Kernel. Journal the measured evidence and
interpretation separately using the Runtime records Skill.
