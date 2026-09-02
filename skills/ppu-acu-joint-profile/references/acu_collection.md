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
  "device_identity": {"physical_device": 6, "serial": "..."},
  "runtime_identity": {"compiler": "hggc ...", "runtime": "PPU SDK ..."},
  "cache_policy": "acu --cache-control all; harness cache state ...",
  "clock_configuration": "locked/default clocks and observed power state",
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

KSD/KVD hit rate is valid only when matching load/store requests from the same replay packet and
exact window are positive. Duplicate sample identities are rejected. L2 hit remains unknown without
an activity denominator. Non-finite PM values reject extraction rather than being serialized as
JSON `NaN` or treated as numerical evidence.

Run the exporter after exporting the ACU raw page:

```bash
python "$PPU_PROFILE_SKILL/scripts/acu_report.py" profile.acurep \
  --raw-csv profile.raw.csv \
  --collection profile.collection.json \
  --csv profile.samples.csv \
  --metadata profile.extract.json
```

The exporter verifies the captured producer output identifies ACU 2.2, hashes the report, raw page, collection
descriptor, bound source/binary/workload files, and resulting PM CSV, and leaves `.acurep`
unchanged. `merge.py` recomputes the raw/PM hashes before consuming the extraction metadata.
