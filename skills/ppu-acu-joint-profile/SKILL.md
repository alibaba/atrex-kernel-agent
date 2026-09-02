---
name: ppu-acu-joint-profile
description: Choose and run ACU-only, adaptive PPU in-kernel timeline, or optional bounded joint analysis for a PPU kernel. Use for device-level bottleneck diagnosis, kernel-internal critical-path questions, or evidence that genuinely needs both; do not require all three modes.
---

# PPU Profile Routing: ACU, Timeline, or Joint

Use this only after a correct runnable kernel and representative workload exist. Follow the NVIDIA
profiling pattern: start from the missing fact, choose the least intrusive evidence route that can
answer it, and escalate only when the current route leaves a concrete ambiguity. ACU, timeline, and
joint analysis are three selectable modes, not mandatory stages of one pipeline.

## Decide whether this optimization iteration needs profiling

Start and finish every optimization iteration with the probe-free kernel. Before invoking any
profiler, name the unresolved performance question and how its answer could change the next edit.
Skip profiling when source inspection, compiler output, the clean benchmark, or still-valid evidence
from an earlier iteration already separates the plausible bottlenecks.

Do not repeat ACU or timeline merely because an earlier iteration used it. Reuse accepted evidence
while the relevant kernel specialization, launch topology, workload, device, and control or pipeline
structure remain comparable. Collect new evidence only when a change invalidates the evidence needed
for the current decision, or when clean results expose a new ambiguity. Timeline is an escalation for
a kernel-internal timing question, never a required per-round validation step.

Across long-horizon episodes, read reusable evidence from
`memory/vN.json.profile_evidence.accepted_ppu_diagnostics`; the raw episode archive is not a
prerequisite. Compare every recorded identity and `invalidation_conditions` entry with the current clean
kernel before reuse. A canonical evidence reference identifies the prior conclusion, but does not
make it valid after specialization, workload, device, topology, or pipeline changes.

## Choose the evidence route

| Route | Choose it when | What it can establish |
| --- | --- | --- |
| ACU only | The bottleneck or expensive kernel is not yet localized at device level. This is the default first profiler for an unknown bottleneck. | Kernel duration, launch resources, occupancy, and device-wide compute, memory, cache, and tail behavior. |
| Timeline only | A specific kernel-internal ordering or dependency question already exists, whether or not ACU was run first. | Ordering and intervals within explicitly selected writers, such as issue, wait, consume, MMA, and epilogue boundaries. |
| Optional joint | Both accepted evidence sets exist and the remaining question depends on their relationship. | Possible and guaranteed overlap under a bounded owner-origin uncertainty; never direct ownership of a device-global metric. |

If the probe-free benchmark already answers the question, do not profile. After each selected route,
stop when its evidence answers the question. Do not collect timeline merely because ACU ran, collect
ACU merely because timeline ran, or invoke `merge.py` merely because both artifacts exist.

Keep mode-specific attempts separate, for example under `<PROFILE_DIR>/acu/attempt-N`,
`<PROFILE_DIR>/timeline/attempt-N`, and `<PROFILE_DIR>/joint/attempt-N`. Never combine events from
different launches into one apparent execution.

## Persist only terminal-reusable evidence

In a long-horizon episode, add `accepted_ppu_diagnostics` to the terminal journal outcome only for
ACU, timeline, or joint conclusions that still apply to the terminal probe-free kernel. Omit an
invalidated intermediate capture. Each row records the question and finding, the exact comparison
identity, how it affected the optimization decision, and the conditions that require collection of
new evidence:

```json
{
  "accepted_ppu_diagnostics": [
    {
      "route": "timeline",
      "question": "Does the tensor wait serialize the steady-state load pipeline?",
      "kernel_specialization": "target kernel specialization and compile-time parameters",
      "workload_identity": "representative shape, dtype, layout, and cache policy",
      "device_identity": "physical device and PPU runtime architecture",
      "launch_topology": "grid, block, selected writer roles, and relevant occupancy facts",
      "control_pipeline_identity": "mainloop stages, waits, barriers, and epilogue structure",
      "finding": "owner-local ranges show the wait on the measured critical path",
      "decision_impact": "next edit targets the load/tensor handoff instead of the epilogue",
      "evidence": {
        "artifact": "profiles/episode_N/timeline/attempt-N/fine.timeline.receipt.json",
        "sha256": "lowercase SHA-256 of that exact JSON artifact",
        "schema": "ppu-fixed-slot-receipt/v4",
        "evidence_id": "evidence_id read from the artifact"
      },
      "invalidation_conditions": [
        "a change to the measured specialization or workload",
        "a change to launch topology or mainloop synchronization"
      ]
    }
  ]
}
```

The supervisor resolves each workspace-relative artifact, recomputes its hash, and accepts only an
`accepted`, decision-grade artifact whose schema and evidence id match the row. It writes stable
`source_memory_version`, `source_episode`, `memory_ref`, and hash-bound `evidence_ref` fields into
canonical memory. Diagnostic-grade or warning artifacts may guide the current investigation but
must not enter terminal-reusable memory. An empty or omitted list is valid when profiling was
skipped or all collected evidence was invalidated.

## Shared evidence boundaries

- Preserve the exact kernel specialization, workload, cache policy, clocks, and physical device
  identity needed for the claim.
- Collect ACU from a probe-free launch. Never run ACU collection and device timeline recording in
  the same launch or process.
- Treat instrumented sources as temporary evidence snapshots. Restore a probe-free kernel before
  correctness, benchmark, commit, promotion, or handoff.
- Report ACU claims as device-global and timeline claims as owner-local. Agreement strengthens a
  hypothesis; temporal overlap does not assign an ACU counter to one owner or range.

For any route, set `PPU_PROFILE_SKILL` to the directory containing this `SKILL.md`.

When a `ppu0015` question depends on Tensor Cell, AIU, shared-memory, occupancy, or timer semantics,
read [references/ppu0015_bottlenecks.md](references/ppu0015_bottlenecks.md). It is an interpretation
aid, not a reason to profile an otherwise understood kernel.

## Route A: ACU-only analysis

Read [references/acu_collection.md](references/acu_collection.md) and collect the smallest useful
metric set on one exact probe-free target launch. Export the raw page and PM windows without creating
any timeline manifest or instrumented source:

```bash
acu -i profile.acurep --page raw --csv --csv-file profile.raw.csv
python "$PPU_PROFILE_SKILL/scripts/acu_report.py" profile.acurep \
  --raw-csv profile.raw.csv --collection profile.collection.json \
  --csv profile.samples.csv --metadata profile.extract.json
```

Use the ACU report independently to classify device-level compute, memory, cache, occupancy, or tail
evidence. Stop here when that evidence selects or rejects the optimization hypothesis. Do not infer
which source interval owns a device-global metric.

If ACU reports a GPM permission or monitor conflict, disable GPM only on the target device and retry
once:

```bash
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi gpm -i <DEVICE> -s 0
```

## Route B: Timeline-only analysis

Choose this route only for a falsifiable kernel-internal timing question. ACU is not a prerequisite
when the question is already precise. The agent owns the hypothesis, coarse/fine transition,
selected blocks, writer threads or roles, owner count, sites, density, and stop condition.

One owner is one declared writer in one launch. Construct one recorder for that owner and reuse it;
do not let several threads race for the same owner id. Every owner has an independent local origin,
so compare events within an owner and never order starts from different owners.

### Start coarse

Read [references/recorder.md](references/recorder.md), then choose the smallest topology that can
separate the current alternatives. Examples are choices, not defaults:

- one representative writer around major mainloop and epilogue phases;
- a few owners for blocks or worker roles expected to behave differently;
- one comparable range from every block only for a dispatch, imbalance, or tail question;
- a writer other than thread 0 when that thread or warp owns the operation being studied.

Set `capture_mode: "coarse"`, record the selection in `sampling_rationale`, and enumerate the exact
`(owner, block, thread)` writers. Compile the temporary snapshot with `PPU_TIMELINE_ENABLED`:

```cpp
#include "ppu_timeline.cuh"
namespace ptl = ppu_acu_profile::timeline;

const bool selected_writer = /* exact block/thread/role predicate */;
const unsigned owner = /* dense id for that selected writer */;
ptl::Recorder trace(params.timeline_buffer, owner, selected_writer);

trace.range_begin(10);  // one semantic coarse phase
// Existing kernel work; control flow and synchronization stay unchanged.
trace.range_end(10);
```

When an immediate record write would perturb the region being measured, capture only the timer at
the real boundaries and flush the pair later, in chronological order, after the sensitive work:

```cpp
const auto phase_begin = trace.timestamp();
// Existing sensitive work.
const auto phase_end = trace.timestamp();
// Flush outside the sensitive region; do not insert synchronization to move this flush.
trace.range_at(10, phase_begin, phase_end);
```

Use immediate writes when their cost does not change the conclusion. Use deferred writes when a
density check shows local interval distortion or when the hypothesis concerns a short critical
region. A deferred timestamp reduces the boundary operation to the timer read; it does not make the
capture probe-free, and records must still be flushed in owner-local timestamp order.

Do not add barriers, waits, atomics, predicates, or control-flow changes to simplify the trace. Map
each timestamp to a real semantic boundary. An asynchronous issue marker is not completion; observe
the original wait, barrier, or first dependent consume when completion matters.

Use the target project's real PPU compiler, runtime, architecture flags, and launch path; do not
replace them with a standalone CUDA-SDK executable or a synthetic launch route.

For a remote attempt, read [references/remote-capture.md](references/remote-capture.md). Upload this
skill as an explicit sandbox input, keep clean/instrumented snapshots under one attempt directory,
decode before the remote job exits, and synchronize only that attempt's evidence.

The recorder reads `%globaltimer`, whose `ppu001`/`ppu0015` TIX contract is a 64-bit nanosecond
timer. Declare `"timer": {"source": "globaltimer", "unit": "ns"}` and do not fit or copy a tick
conversion. If a synchronized sanity experiment materially contradicts that contract, reject the
capture and investigate the runtime instead of rescaling the timeline. `%clock64` is a CU cycle
counter that includes scheduling and resource waits; it is not this timeline's time source.

Read [references/recorder.md](references/recorder.md) for the timer and correctness artifact
contracts. The manifest binds correctness to the exact device, kernel, and workload. The decoder
rejects legacy conversion fields, an invalid timer contract, or correctness evidence whose checks
did not pass.

Initialize the ABI buffer on the host, copy it to the device, launch once, synchronize, and copy the
entire allocation back. Emit manifest v4 and the event dictionary from actual launch and source
facts, then decode:

```bash
python "$PPU_PROFILE_SKILL/scripts/timeline.py" decode \
  --raw coarse.timeline.bin \
  --manifest coarse.timeline.manifest.json \
  --event-dictionary coarse.timeline.events.json \
  --output-prefix coarse.timeline
```

Use only an accepted receipt. A diagnostic receipt is intentionally incomplete and may answer a
local exploratory question; a joint merge or terminal-reusable conclusion requires a decision-grade
receipt with source, compiled-binary, and workload-input bindings. If the coarse trace answers the
question, stop without fine probes, ACU, or merge. Read
[references/timeline_contract.md](references/timeline_contract.md) when interpreting decoder outputs
or preparing a fine capture for optional joint analysis.

### Refine only the unresolved region

Create a new reversible fine snapshot only when coarse evidence leaves a narrower question. Retain
the minimum context and add only the issue/wait/consume or subphase boundaries needed inside that
region. Set `capture_mode: "fine"`. A fine timeline remains valid standalone evidence and does not
require ACU or `merge.py`.

Declare `analysis.owner`, `analysis.window_site_id`, and `analysis.site_ids` only when preparing a
fine capture for optional joint analysis or when those labels help the standalone interpretation.
The decoder rejects declared analysis ranges outside the window. Analyze another owner in another
attempt instead of pretending unsynchronized owner clocks share an axis.

### Bound probe effects when the claim needs it

Preserve A (clean), B (minimal useful probes), and, only when density sensitivity matters, C (denser
nearby probes). Each timing command must warm up, iterate, synchronize, validate the representative
output, and emit one `__PPU_TIMELINE_SAMPLE__=...` JSON line:

```bash
python "$PPU_PROFILE_SKILL/scripts/timeline.py" measure \
  --baseline-command '["python","run_a.py"]' \
  --instrumented-command '["python","run_b.py"]' \
  --workload-identity case-id --warmup 10 --iterations 100 \
  --output fine.perturbation-a-b.json

python "$PPU_PROFILE_SKILL/scripts/timeline.py" measure \
  --baseline-command '["python","run_b.py"]' \
  --instrumented-command '["python","run_c.py"]' \
  --workload-identity case-id --warmup 10 --iterations 100 \
  --output fine.perturbation-b-c.json
```

Every emitted sample includes positive finite `latency_ms`, `correctness: "passed"`,
`synchronized: true`, the exact workload/warmup/iteration and physical-device identity, and a stable
`allocation_identity` such as the authorized Pod UID plus allocation/job id. The helper rejects any
device or allocation drift across the interleaved schedule.

A/B measures end-to-end probe overhead. B/C tests whether nearby density moves common intervals. Do
not import a universal overhead threshold from another architecture; report the measured deltas and
repeat with fewer or moved sites when the conclusion changes materially.

### Close an agent-declared critical path when needed

If the question is whether selected subranges explain an enclosing range, or whether the slowest
owner is stable across captures, declare the semantic parent and components in a plan and analyze
the accepted canonical captures:

```bash
python "$PPU_PROFILE_SKILL/scripts/critical_path.py" \
  --plan critical-path.plan.json \
  --capture attempt-1/fine.timeline.canonical.json attempt-1/fine.timeline.receipt.json \
  --capture attempt-2/fine.timeline.canonical.json attempt-2/fine.timeline.receipt.json \
  --output critical-path.report.json
```

The plan, not the tool, chooses phases, owner-topology comparison, and any material-spread threshold.
The analyzer verifies each canonical hash against its receipt and rejects grid, block, capture-mode,
identity, or selected-site semantic drift. It computes component interval union rather than
double-counting overlap and leaves uncovered time unattributed. Escalate representative owner to
more warps or blocks only when the observed spread or remaining topology ambiguity can change the
optimization decision. See [references/timeline_contract.md](references/timeline_contract.md) for
the plan contract.

## Route C: Optional joint analysis

Choose this only after ACU and fine timeline were each collected and interpreted independently, and
the unresolved question requires their relationship. The fine capture must declare one analysis
owner, one enclosing window, and a non-empty site list. Verify kernel, grid, block, workload, device,
and duration equivalence before interpreting the output.

```bash
python "$PPU_PROFILE_SKILL/scripts/merge.py" \
  --timeline fine.timeline.perfetto.json \
  --timeline-receipt fine.timeline.receipt.json \
  --pm-csv profile.samples.csv \
  --acu-raw-csv profile.raw.csv \
  --acu-metadata profile.extract.json \
  --perturbation fine.perturbation-a-b.json \
  --density-sensitivity fine.perturbation-b-c.json \
  --output-prefix fine.joint
```

The perturbation and density-sensitivity inputs are optional; pass those flags only when the
corresponding validated artifacts exist. Joint merge is fail-closed: it requires decision-grade
timeline and ACU bindings, compares kernel specialization, workload, physical device, runtime,
cache policy, clock configuration, grid, block, and duration, and has no duration-mismatch override.

Interpret the result as four separate claims:

1. owner-local ordering and dependency intervals from the fine timeline;
2. device-global compute, memory, cache, occupancy, and tail phases from ACU;
3. guaranteed overlap first, then possible overlap under the bounded analysis-owner origin offset;
4. probe overhead, density sensitivity, sampling coverage, and remaining ambiguity.

The merger never aligns raw clocks or rescales either run. Full-block duration survival is emitted
only when the manifest explicitly declares all-block coverage, the capture contains one comparable
range per block, and the launch satisfies the validated one-wave condition. Otherwise partial
sampling remains valid and that statistic is omitted.

## Finish

Name the selected route in the result and keep its evidence scope explicit. A valid ACU-only or
timeline-only conclusion is complete without a joint artifact. Restore the probe-free kernel and use
the normal evaluator for final correctness and performance; profile evidence does not replace the
probe-free result.
