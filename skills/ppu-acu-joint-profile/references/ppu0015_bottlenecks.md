# PPU0015 bottleneck interpretation

Read this only when a `ppu0015` result needs architecture-specific interpretation. It supplies
decision-relevant facts for selecting probes and testing bottleneck hypotheses; it is not a product
peak-performance sheet and does not make timeline collection mandatory.

Authoritative sources:

- [T-Head SAIL TIX programming guide](https://developer.t-head.cn/docs_center/doc_detail/index.html?projectId=39&chapterId=197)
- [T-Head SAIL HGGC programming guide](https://developer.t-head.cn/docs_center/doc_detail/index.html?projectId=39&chapterId=196)

## Timer and topology facts

- `%globaltimer` is the 64-bit nanosecond timer used by the fixed-slot recorder.
- `%clock64` is a per-CU cycle counter. Its delta includes scheduling, memory, and other resource
  waits, so it is not pure instruction latency and must not supply the timeline conversion.
- `%cuid` identifies the executing CU. `%ncuid` is the upper bound of the CU identifier space and may
  exceed the physical CU count because CU identifiers need not be contiguous.
- Warp size is 32; a CU supports up to 2048 threads and a block up to 1024 threads.
- A CU exposes 256 KiB shared memory and 64K vector registers. Use launch resources and ACU
  occupancy evidence to determine the active-block or active-warp constraint; do not infer it from
  one resource in isolation.

Keep decoded starts owner-local unless another accepted contract proves cross-owner synchronization.
The nanosecond unit does not by itself establish cross-CU ordering.

## Select semantic boundaries from the hypothesis

For an asynchronous tiled mainloop, useful boundaries may include AIU issue, commit, the original
wait, first dependent shared-memory consume, Tensor Cell work, and epilogue or store. These are
choices, not a required event vocabulary. Instrument only boundaries that can separate the current
alternatives.

Do not add a wait or barrier to observe completion. An issue range measures issue work. Unhidden
latency becomes visible around the original wait or first dependent consume. Compare neighboring
tiles only when the same owner performs semantically comparable work.

## Compute interpretation

`ppu0015` uses `tc02` Tensor Cell instructions and adds FP8 E4M3/E5M2 and MXFP4 E2M1. The guides
state relative theoretical throughput of up to 2x FP16 for FP8 and 4x FP16 for FP4. They do not
publish the absolute FP16 peak, Tensor Cell count, or MMA issue rate needed for an absolute TFLOPS
roofline.

For a declared MMA shape, mathematical work is:

```text
FLOPs per MMA = 2 * M * N * K
```

For example, `m16n16k64` represents 32768 FLOPs. Combine source or disassembly instruction counts
with the owner-local compute range to estimate observed phase throughput. Treat it as a phase model,
not an absolute hardware-efficiency percentage unless an authoritative or measured ceiling is also
available.

## Memory and overlap interpretation

- Shared memory has 32 banks. `ppu0015` divides it into four groups with a 1024-byte continuous
  address-mapping span per group. Requests mapping to one group can serialize.
- The 1024-byte value describes address organization, not bytes per cycle; never use it as shared
  memory bandwidth.
- Global access optimization should reduce the number of touched 64-byte blocks.
- Load/AIU prefetch hints support 128-byte and 256-byte granularity. A hint may be ignored by
  hardware and is not completion evidence.
- AIU provides asynchronous global-to-shared tensor movement, including 2D tile and 5D im2col
  modes. Swizzle modes are intended to reduce later shared-memory conflicts.
- PPU exposes L1, L2, LLC, and HBM levels. Do not attribute a long wait to HBM from timeline alone;
  use ACU memory/cache evidence or a controlled layout experiment.

## Evidence patterns

| Owner-local timeline | Device-global ACU or launch evidence | Hypothesis to test next |
| --- | --- | --- |
| Tensor range dominates | Tensor utilization is sustained and high | Compute throughput or required work dominates. |
| Tensor range dominates | Tensor utilization is low | Issue dependency, insufficient eligible warps, or non-Tensor work may dominate. |
| Original wait is exposed on the critical path | DRAM activity is sustained and high | Global-to-shared movement throughput may dominate. |
| Original wait is exposed on the critical path | DRAM activity is low | Access dispersion, latency, dependency, or insufficient in-flight work may dominate. |
| Shared consume or `ldmatrix` region expands | Cache/DRAM evidence does not explain it | Test bank/group conflict, swizzle, alignment, or shared layout. |
| Only pipeline prologue or drain expands | Steady-state utilization remains stable | Fill/drain or tail overhead may dominate. |
| Comparable owner durations have material spread | Occupancy or active-CU evidence is uneven | Test load imbalance, dispatch waves, or resource-limited residency. |

These patterns select the next experiment; they do not prove causality. Let the agent choose which
owners, sites, ACU metrics, or controlled code variant can falsify the leading alternative.

## Missing absolute ceilings

The public programming guides do not provide absolute dtype peak FLOPS, HBM bandwidth, Tensor Cell
issue rate, or cache/shared bandwidth. Use ACU's vendor-normalized `pct_of_peak_sustained` metrics
for device-level saturation, or a same-device measured ceiling for roofline calculations. Label the
latter as measured rather than theoretical.
