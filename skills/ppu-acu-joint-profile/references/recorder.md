# PPU fixed-slot recorder

## ABI and host setup

`ppu_timeline.cuh` defines a 64-byte header, 16-byte records, and one 64-bit writer claim per
declared owner. Allocation size is exact:

```text
64 + owner_count * records_per_owner * 16 + owner_count * 8
```

An owner is an agent-selected writer, not a synonym for block. `owner_count` is the number of
explicit `(block, thread)` entries in the manifest. Entries use dense ids `0..owner_count-1`; several
owners may come from one block, and a capture may sample only part of the grid. Size
`records_per_owner` for the busiest declared owner. Unused trailing slots remain zero.

Host pseudocode:

```cpp
using namespace ppu_acu_profile::timeline;
const u32 owners = selected_writers.size();
const u64 bytes = allocation_bytes(owners, records_per_owner);
std::vector<unsigned char> host(bytes);
if (!initialize_host_buffer(host.data(), bytes, owners, records_per_owner,
                            grid.x, grid.y, grid.z,
                            block.x, block.y, block.z, launch_id)) {
  throw std::runtime_error("invalid PPU timeline allocation");
}
// Copy host -> device, launch once, synchronize, copy every byte device -> host.
```

Python/Torch launchers can use `backends/ppu_backend/adapter.py` instead:

```python
device_index = ...  # selected by the launch harness
timeline_buffer = allocate_torch_buffer(
    owner_count=len(selected_writers),
    records_per_owner=records_per_owner,
    grid=grid,
    block=block,
    launch_id=launch_id,
    device=f"cuda:{device_index}",
)
# Pass timeline_buffer through the existing JIT launch interface.
save_torch_buffer(timeline_buffer, "timeline.bin")
```

For an HGGC JIT string, `instrument_source(source, source_name=...)` adds the CUDA-compatible runtime
declarations, recorder switch, and an absolute include of the adjacent `ppu_timeline.cuh`, then
restores compiler source coordinates with `#line`. HGGC must parse the complete recorder through its
native include path; do not copy the CUDA backend's text-embedding or device-only-header assumption.
Pass `enabled=False` only for an explicit no-op/control build. A project that owns its include path
may include `ppu_timeline.cuh` directly instead.

The host helpers are available unless `PPU_TIMELINE_DEVICE_ONLY` is defined. A device-only RTC
source must initialize the identical ABI in its launcher.

`Recorder(storage, owner, selected)` does nothing when `selected` is false. A selected recorder
claims its declared slot with the actual linear block and linear thread. A second selected writer for
the same owner sets `duplicate_owner`; an undeclared owner sets `bad_owner`; capacity exhaustion sets
`overflow`. Each record is written to `owner * records_per_owner + sequence`, with the committed bit
written last. The decoder rejects all of those failures.

Immediate `range_begin`, `range_end`, `mark`, and `count` read the timer and write one record at the
call site. For a short sensitive region, `timestamp()` reads only `%globaltimer`; `range_at`,
`mark_at`, and `count_at` write previously captured timestamps later. Flush deferred records after
the sensitive region without adding synchronization or changing useful-work control flow, and emit
them in chronological order for that owner. `range_at` emits its begin and end consecutively and is
the simplest safe form:

```cpp
const auto begin = trace.timestamp();
// Existing work.
const auto end = trace.timestamp();
trace.range_at(20, begin, end);
```

This moves fixed-slot writes out of the measured range but does not remove the timer-read cost.
Compare immediate and deferred variants only when that distinction matters to the hypothesis.

## Choosing owners and sites

Derive the topology from the hypothesis rather than from a fixed convention:

1. Identify the operation or role whose local ordering matters.
2. Choose the block(s) needed to distinguish representative, boundary, persistent-worker, or tail
   behavior.
3. Choose the thread or warp role that actually executes the boundary. Thread 0 is valid only when
   the source and hypothesis make it useful.
4. Assign one dense owner id to each selected writer and construct exactly one recorder for it.
5. Start coarse. Add fine sites only inside the region left ambiguous by coarse evidence.

Ranges use the same stable `site_id` for begin and end. Sites describe semantic boundaries, not the
conclusion the agent hopes to prove. Required semantic fields are:

- `boundary_semantics`: what is known complete before and after the timestamp;
- `async_domain`: such as `global_to_shared`, `tensor`, `epilogue`, or `control`;
- `source_anchor`: file plus a stable symbol or surrounding operation.

An asynchronous issue range measures issue work. Completion is observable only at the original
wait, barrier, or first dependent consume. Never add synchronization or control flow merely to make
the trace easier to read.

## Timer contract and correctness evidence

The recorder reads `%globaltimer`. The TIX contract for `ppu001` and `ppu0015` defines it as a
64-bit nanosecond timer, so manifest v4 declares the source and unit and the decoder applies an
identity conversion. Do not fit a `timer_tick_ns` scale from device-event timings: that would mix
event overhead and measurement error into every decoded phase. If a synchronized sanity experiment
materially contradicts the documented unit, reject the capture and investigate the runtime.

`%clock64` is a different per-CU cycle counter. Its delta includes scheduling, memory, and resource
waits and must not be substituted for `%globaltimer` or used as the timeline conversion.

The capture harness also writes numerical correctness evidence from the representative output:

```json
{
  "schema": "ppu-timeline-correctness/v1",
  "validation": "accepted",
  "kernel_name": "target_kernel",
  "workload_identity": "m=...;n=...;k=...;dtype=...",
  "device_identity": {"physical_device": 7, "serial": "..."},
  "checks": [
    {"name": "output relative L2", "status": "passed", "value": 0.00012},
    {"name": "state max absolute error", "status": "passed", "value": 0.00098}
  ]
}
```

The harness chooses metrics and tolerances appropriate to the operator contract. The decoder checks
that every declared check passed and that kernel, workload, and device identities match; it does not
invent numerical tolerances.

## Manifest v4

Write launch facts next to the raw buffer. This fine example intentionally uses a nonzero writer
thread and partial grid coverage:

```json
{
  "schema": "ppu-fixed-slot-timeline-manifest/v4",
  "backend": "ppu_fixed_slot",
  "capture_mode": "fine",
  "sampling_rationale": "coarse capture localized the unresolved load/MMA overlap to the steady-state loader role",
  "kernel_name": "target_kernel",
  "kernel_duration_ns": 323640,
  "grid": [148, 1, 1],
  "block": [128, 1, 1],
  "launch_id": 17,
  "records_per_owner": 32,
  "timer": {"source": "globaltimer", "unit": "ns"},
  "correctness_artifact": "correctness.json",
  "runtime_identity": {"compiler": "hggc ...", "runtime": "PPU SDK ..."},
  "provenance": {
    "evidence_grade": "decision",
    "kernel_specialization": "compile-time shape, dtype, layout, and architecture flags",
    "cache_policy": "harness cache preparation and reuse policy",
    "clock_configuration": "locked/default clocks and observed power state",
    "instrumented_sources": [
      {"path": "instrumented-source/kernel.cu", "identity": "exact instrumented target source"}
    ],
    "compiled_binaries": [
      {"path": "build/kernel.so", "identity": "loaded instrumented binary"}
    ],
    "workload_inputs": [
      {"path": "workload.json", "identity": "representative input descriptor"}
    ]
  },
  "clock_scope": "owner_local",
  "owner_layout": {
    "kind": "explicit_writers",
    "owners": [
      {
        "owner": 0,
        "block": 17,
        "thread": 64,
        "label": "steady-loader",
        "purpose": "observe issue, wait, and dependent consume in the localized mainloop"
      }
    ]
  },
  "analysis": {
    "owner": 0,
    "window_site_id": 10,
    "site_ids": [20, 21, 22],
    "tile": 4,
    "k_stage": 2
  },
  "coverage": {"all_blocks": false},
  "workload_identity": "m=...;n=...;k=...;dtype=...",
  "device_identity": {"physical_device": 0, "serial": "..."}
}
```

For a coarse capture, use `capture_mode: "coarse"`; `analysis` may be absent. If the question truly
requires a comparable range from every block, enumerate owners whose block fields cover the grid and
declare:

```json
"coverage": {
  "all_blocks": true,
  "range_site_id": 1,
  "instrumented_launch": {
    "cu_count": 64,
    "occupancy_blocks_per_cu": 2,
    "evidence_artifact": {
      "path": "instrumented-launch-resources.json",
      "identity": "compiler or profiler occupancy evidence for this instrumented binary"
    }
  }
}
```

The decoder then requires exactly one such range per linear block. The occupancy evidence must come
from the instrumented binary; clean ACU occupancy cannot prove the instrumented launch is one wave.
This is an opt-in topology, not a baseline requirement.

`kernel_duration_ns` comes from synchronized HGGC events around the single launch. The correctness
artifact path is resolved relative to the manifest unless absolute. Keep it inside the attempt
directory for remote capture and handoff.

The decoder hashes every declared artifact. `diagnostic` evidence may omit compiled binaries or
workload files when they are genuinely unavailable, but it cannot enter joint analysis or terminal
memory. `decision` evidence requires all three binding classes. Do not copy a hash from an earlier
attempt: the decoder computes hashes from the files present when it accepts the capture.

## Event dictionary v2

Omit `owners` when any declared owner may emit a site, or list the exact permitted owner ids. Roles
are descriptive strings; the analysis and coverage site ids in the manifest carry merge semantics.

```json
{
  "schema": "ppu-fixed-slot-events/v2",
  "sites": [
    {
      "site_id": 10,
      "name": "localized_mainloop_window",
      "kind": "range",
      "role": "analysis_window",
      "owners": [0],
      "boundary_semantics": "entry to exit of the coarse-localized steady-state region",
      "async_domain": "control",
      "source_anchor": "target kernel mainloop"
    },
    {
      "site_id": 20,
      "name": "next_async_load_issue",
      "kind": "range",
      "role": "fine_observation",
      "owners": [0],
      "boundary_semantics": "the original asynchronous load issue call only",
      "async_domain": "global_to_shared",
      "source_anchor": "mainloop load call"
    },
    {
      "site_id": 21,
      "name": "dependency_wait",
      "kind": "range",
      "role": "fine_observation",
      "owners": [0],
      "boundary_semantics": "the original wait until the loaded operand is consumable",
      "async_domain": "global_to_shared",
      "source_anchor": "mainloop wait call"
    },
    {
      "site_id": 22,
      "name": "tensor_issue_group",
      "kind": "range",
      "role": "fine_observation",
      "owners": [0],
      "boundary_semantics": "issue of the original tensor operation group",
      "async_domain": "tensor",
      "source_anchor": "mainloop tensor issue"
    }
  ]
}
```

Allowed kinds are `range`, `instant`, `counter`, and `any`. Range begin/end pairs must balance within
the same owner and site. Move a boundary when an original early exit can break the pair; do not let
the decoder infer missing control flow.
