# Decoded PPU timeline and joint-merge contract

Raw ABI integration and manifest examples live in [recorder.md](recorder.md). The `decode` command in
`scripts/timeline.py` is the supported path from a capture to merge input.

## Decoder contract

For `--output-prefix fine.timeline`, the decoder writes:

- `fine.timeline.canonical.json`: typed owner-local events with raw timestamps, relative nanoseconds,
  writer and launch identity, pairing, payloads, timer/correctness validation, and semantic
  site fields;
- `fine.timeline.perfetto.json`: one lane per explicitly declared owner and PPU timeline schema 4;
- `fine.timeline.summary.json`: capture mode, topology, range distributions, and interpretation limits;
- `fine.timeline.receipt.json`: accepted validation invariants and launch identity.

Manifest v4 declares `%globaltimer` with its documented nanosecond unit and binds the capture to an
accepted numerical correctness artifact. A legacy conversion field, invalid timer source/unit, or
correctness evidence for another kernel/workload/device is rejected before canonical output is
written.

The first committed record of each owner defines only that owner's local zero. Lanes are displayed
together for inspection, but their raw starts are neither sorted nor aligned. The decoder accepts a
sparse subset of blocks and arbitrary declared writer threads. It rejects missing or duplicate writer
claims, claim/manifest mismatch, holes, overflow, unknown sites, owner exclusions, kind mismatch,
decreasing timestamps, unbalanced ranges, invalid declared analysis windows, and false all-block
coverage.

The Perfetto root carries the agent's choices rather than inferring a topology:

```json
{
  "ppuTimeline": {
    "schemaVersion": 4,
    "captureMode": "fine",
    "samplingRationale": "...",
    "kernelName": "target_kernel",
    "kernelDurationNs": 323640,
    "grid": [148, 1, 1],
    "blockDims": [128, 1, 1],
    "owners": [
      {"owner": 0, "block": 17, "thread": 64, "label": "steady-loader", "purpose": "..."}
    ],
    "analysisOwner": 0,
    "analysisBlock": 17,
    "analysisThread": 64,
    "analysisWindowSiteId": 10,
    "analysisSiteIds": [20, 21, 22],
    "analysisWindowEventName": "localized_mainloop_window",
    "coverage": {"allBlocks": false, "rangeSiteId": null, "rangeEventName": null},
    "clockScope": "owner_local",
    "timerSource": "globaltimer",
    "timerUnit": "ns",
    "timerContractValidation": "accepted",
    "correctnessValidation": "accepted",
    "captureValidation": "accepted"
  },
  "traceEvents": []
}
```

Chrome/Perfetto timestamps and durations are microseconds. Manifest, canonical, ACU, and joint
durations are nanoseconds.

## Coarse and fine evidence

A coarse capture is valid without an `analysis` object and is used to decide where more resolution is
worth its probe cost. It is not mergeable merely because it contains ranges. A fine capture becomes
mergeable only when it declares one analysis owner, one enclosing window site, and a non-empty list
of range site ids inside that window.

This separation keeps topology choice with the agent:

- coarse captures may use one representative writer, several semantic roles, sampled blocks, or an
  explicitly complete all-block range;
- fine captures may retain only the owner and boundaries needed for the localized question;
- another owner-local clock requires another analysis, not a cross-owner timestamp comparison.

## Critical-path closure across captures

Use `scripts/critical_path.py` only after decoding accepted captures. It consumes a plan rather than
assuming that every kernel has the same phases:

```json
{
  "schema": "ppu-critical-path-plan/v1",
  "parent": {"site_id": 10, "name": "chunk"},
  "components": [
    {"site_id": 20, "name": "wait_and_publish"},
    {"site_id": 21, "name": "issue_and_projection"},
    {"site_id": 22, "name": "state_and_output"}
  ],
  "clean_reference": {
    "duration_ns_samples": [2713.3, 2708.1, 2718.6],
    "source": "probe-free synchronized harness on the same workload"
  },
  "stability": {"material_relative_spread": 0.03}
}
```

`components`, `clean_reference`, and `stability` are optional. Omit components for parent-duration
and owner/topology aggregation without closure. Their values come from the experiment and the
optimization decision; the analyzer supplies no architecture-wide threshold. Pass one
`--canonical` flag per distinct launch. The analyzer requires equal kernel, workload, device, and
runtime identity while preserving each launch id.

When components are declared, every parent occurrence reports component sums, interval-union
coverage, explicit overlap, uncovered time, and per-component occurrence counts. Across owners and captures it emits
duration distributions, the slowest observed owner by parent median, and owner-median spread. It
never compares owner-local start times. Uncovered time remains unattributed evidence: add a new
semantic boundary only when a specific unresolved dependency requires it.

## Joint merge semantics

The merger accepts only accepted capture/timer contracts, `clockScope=owner_local`, and schema 4. ACU PM
windows are kernel-start-relative. Fine ranges are analysis-owner-origin-relative. If the declared
analysis window ends at local time `window_end_ns`, the unknown owner-origin offset is bounded by:

```text
[0, kernelDurationNs - window_end_ns]
```

For every exact ACU PM window, the merger preserves two distinct statements:

- possible overlap: at least one allowed owner-origin offset makes the local range overlap;
- guaranteed overlap: every allowed offset makes the local range overlap.

It never aligns raw clocks or rescales either run. ACU values retain device-global scope and replay
packet membership. Optional A/B perturbation and B/C density-sensitivity files must match timeline
workload/device identity and remain separate evidence.

Normalized all-block duration survival is optional. It is produced only when:

1. the manifest declares `coverage.all_blocks: true` and a comparable range site;
2. decode proves exactly one such range for every linear block;
3. ACU launch data proves the grid fits the one-wave capacity used by that statistic.

Otherwise partial sampling is accepted and the distribution is omitted. Even when emitted, it is a
normalized duration distribution, not proof that independent owner clocks were synchronized.
