# ACU collection reference

## Collection

Precompile the workload and capture exactly one target launch. Prefer explicit
PM metrics for FP8 Tensor, CU/IPC, DRAM, KSD/KVD request activity/hit rate, and
L2. The validated ACU 2.2 stock section omitted FP8 Tensor utilization, and its
`--section-folder` did not discover a local custom section.

Recommended metric names:

```text
ce__total_cta_num.sum
cu__cycles_active.avg
cu__inst_executed.avg.per_cycle_active
cu__inst_executed.avg.pct_of_peak_sustained_elapsed
cu__inst_executed_pipe_tensor_fp8.avg.pct_of_peak_sustained_active
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
ksd__requests_hit_rate.pct
ksd__requests_load_pipe_ws.sum
ksd__requests_store_pipe_ws.sum
kvd__requests_hit_rate.pct
kvd__requests_load_pipe_lsu.sum
kvd__requests_store_pipe_lsu.sum
l2__requests_hit_rate.pct
```

Pass them with the `pmsampling:` prefix, kernel replay, disabled PM warp
sampling, and explicit physical device. The requested interval is only a hint.

```bash
acu --devices <PHYSICAL_DEVICE> \
  --pm-sampling-interval 5000 \
  --disable-pm-warp-sampling \
  --replay-mode kernel \
  --cache-control all \
  --kernel-name <EXACT_KERNEL_NAME> \
  --launch-count 1 \
  --metrics <COMMA_SEPARATED_PMSAMPLING_METRICS_ABOVE> \
  --export profile.acurep --force-overwrite \
  <TARGET_COMMAND>
```

If ACU reports a GPM permission or monitoring conflict, disable GPM only on the
target device and retry once:

```bash
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi gpm -i <PHYSICAL_DEVICE> -s 0
```

## Report extraction

Use ACU raw CSV for kernel identity, duration, launch dimensions, registers,
shared memory, CU count, and occupancy.

ACU 2.2 stores PM windows in `.acurep` Perfetto protobuf under top-level Trace
field 1, TracePacket field 88, and nested message field 15. Each sample contains
start ns, a fixed64 double value, and end ns. Decode only this verified subtree,
keep the report immutable, and preserve unknown headers, metrics, and packet
membership.

KSD/KVD hit rate is valid only when matching load/store requests are positive.
L2 hit remains unknown without an activity denominator.

Run `python "$PPU_PROFILE_SKILL/scripts/acu_report.py"` after exporting the ACU
raw page. The exporter produces a long-form CSV plus extraction metadata and
leaves `.acurep` unchanged.
