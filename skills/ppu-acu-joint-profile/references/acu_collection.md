# ACU collection reference

## Collection

Precompile the workload and capture exactly one target launch. Choose the smallest PM metric set
that can answer the current question. There is no default dtype: inspect the workload, source, or
compiled instructions before selecting a Tensor metric, and do not collect a Tensor metric for a
non-Tensor kernel.

The following names are metrics whose extraction semantics were verified with ACU 2.2. They are a
menu, not one default collection. The FP8 metric is relevant only to an FP8 question:

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

Do not request an isolated hit-rate metric for decision evidence. Pair KSD/KVD hit rates with their
listed load/store activity metrics in the same replay packet and exact windows. No verified L2
activity denominator is listed here, so `l2__requests_hit_rate.pct` remains raw exploratory data.
Preserve any other ACU metric in the exported samples, but do not infer its unit, scope, or validity
until its semantics have been verified on the target ACU/PPU version.

Pass selected PM metrics with the `pmsampling:` prefix, kernel replay, and an explicit physical
device. The requested interval is only a hint. Use `--disable-pm-warp-sampling` only for a lightweight
PM-only capture; do not use that flag when the current question requires ACU warp-sampling evidence.
Keep a separate attempt when changing the sampling mode.

```bash
acu --devices <PHYSICAL_DEVICE> \
  --pm-sampling-interval 5000 \
  <OPTIONAL_SAMPLING_MODE_FLAGS> \
  --replay-mode kernel \
  --cache-control <POLICY_MATCHING_THE_PERFORMANCE_QUESTION> \
  --kernel-name <EXACT_KERNEL_NAME> \
  --launch-count 1 \
  --metrics <AGENT_SELECTED_METRICS_FOR_THIS_DTYPE_AND_QUESTION> \
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

Before extraction, write a collection descriptor beside the immutable report. Paths are resolved
relative to this descriptor. `decision` evidence requires at least one clean source, compiled
binary, and workload input; use `diagnostic` only when an unavailable binding makes the result
unsuitable for joint analysis or terminal reuse.

Capture the producer identity in the same allocation before collection:

```bash
acu --version > acu-version.txt
```

```json
{
  "schema": "ppu-acu-collection/v1",
  "producer": {"name": "acu", "version": "2.2"},
  "producer_artifact": {
    "path": "acu-version.txt",
    "identity": "stdout from the ACU binary used for this collection"
  },
  "evidence_grade": "decision",
  "report": "profile.acurep",
  "kernel_name": "exact filtered kernel name",
  "kernel_specialization": "compile-time specialization and architecture flags",
  "workload_identity": "shape, dtype, layout, and input case",
  "device_identity": {"physical_device": 0, "serial": "..."},
  "runtime_identity": {"compiler": "hggc ...", "runtime": "PPU SDK ..."},
  "cache_policy": "exact ACU cache-control and harness cache state",
  "clock_configuration": "locked/default clocks and observed power state",
  "requested_metrics": [
    "exact metric names expected in the exported PM payload"
  ],
  "source_artifacts": [
    {"path": "clean-source/kernel.cu", "identity": "probe-free target source"}
  ],
  "binary_artifacts": [
    {"path": "build/kernel.so", "identity": "loaded probe-free binary"}
  ],
  "workload_inputs": [
    {"path": "workload.json", "identity": "representative input descriptor"}
  ]
}
```

`requested_metrics` is optional, but include it when the exact requested list is known so a missing
metric is visible. KSD/KVD hit rate is valid only when matching load/store requests from the same
replay packet and exact window are positive. Duplicate sample identities are rejected. L2 hit
remains unknown without an activity denominator. Non-finite PM values reject extraction rather than
being serialized as JSON `NaN` or treated as numerical evidence.

Run the exporter after exporting the ACU raw page:

```bash
python "$PPU_PROFILE_SKILL/scripts/acu_report.py" profile.acurep \
  --raw-csv profile.raw.csv \
  --collection profile.collection.json \
  --csv profile.samples.csv \
  --metadata profile.extract.json
```

The exporter leaves `.acurep`, `profile.raw.csv`, and the decoded PM values available to the agent.
It also validates the exact raw kernel row, physical device, launch dimensions and finite launch
resources; checks PM window continuity, duration coverage, interval agreement, dropped samples, and
requested metric presence; and emits per-packet/per-metric valid counts and time-weighted summaries.
Unknown metrics remain in `profile.samples.csv` with `scope: unknown` and
`validity: unknown_semantics`.

`profile.extract.json` uses `ppu-acu-extraction/v3`. Its validation status has narrow data-quality
semantics:

- `accepted`: no detected integrity or sampling-quality issue;
- `warning`: raw data remains available, but short coverage, few samples, dropped samples, a missing
  requested metric, or another stated limitation prevents terminal reuse;
- `rejected`: a kernel/device mismatch, malformed launch fact, contradictory PM windows, duration
  overrun, or another hard integrity failure was detected.

A warning never triggers collection, timeline, or retry automatically. The agent reads the compact
metadata first, inspects only the relevant raw rows, and chooses whether the evidence is already
enough. The exporter hashes the report, raw page, collection descriptor, bound
source/binary/workload files, and resulting PM CSV. `merge.py` recomputes the raw/PM hashes before
consuming only accepted decision-grade metadata.
