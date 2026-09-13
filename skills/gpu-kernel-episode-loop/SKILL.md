---
name: gpu-kernel-episode-loop
description: Run the evidence loop of one long-horizon GPU kernel optimization episode. Use this skill to reconstruct the incumbent, profile and localize a bottleneck, research progressively, plan one coherent direction, implement and repair, validate development correctness and performance, and record every decisive experiment in the episode journal.
---

# GPU Kernel Episode Loop

## When to Use

Use this skill when an orchestrator episode prompt hands you one optimization episode in an isolated
workspace and points here for the evidence loop. It does not apply to Supervisor V0 initialization,
the dedicated Framework Baseline session, or a workspace without an episode journal.

## Evidence ownership

The Episode prompt supplies the execution boundary, mode policy, and framework-escalation rules.
These outrank this skill. Use the Runtime Journal for plans and analysis and Gateway Records for
measurements. No separate plan draft, synthesized plan, or profile-analysis file is required.
`scratch/` is for temporary requests and optional diagnostic files, not durable conclusions.

## Telemetry

Telemetry is best-effort and must not block engineering work. Mark phase boundaries with standalone
commands and keep at most one phase active:

```bash
python3 tools/iteration_trace.py phase-start <profile|research|planning|implementation|correctness|benchmark|recording>
python3 tools/iteration_trace.py phase-end <profile|research|planning|implementation|correctness|benchmark|recording>
python3 tools/iteration_trace.py source-read <gpu_wiki|reference_projects|workspace|public_web> <safe-relative-reference>
```

Never put credentials, private URL parameters, absolute user paths, raw tool output, or transcript
text into telemetry.

## Loop

Repeat this evidence loop until the direction yields a mature candidate or is exhausted. The numbered
steps map onto the telemetry phases above: `profile`, `research`, `planning`, `implementation`,
`correctness`/`benchmark`, and `recording`.

### 1. Reconstruct the incumbent and choose a hypothesis

Read the workspace goal, unmasked `memory/v*.json`, and relevant Directions, Experiments, and Gateway Records through the Runtime tools. Prior-episode summaries
are carried only by canonical memory and are not injected into the episode prompt. Identify attempted
dead ends and open directions from those records, including each record's compact
`experience.experiments`. For PPU, also inspect
`profile_evidence.accepted_ppu_diagnostics` and reuse a conclusion only when its recorded identities
remain comparable and none of its `invalidation_conditions` holds. Start with one falsifiable
hypothesis tied to the current bottleneck.

### 2. Profile and localize

For a PPU target, do not apply the NVIDIA/AMD default below. Read
`skills/ppu-acu-joint-profile/SKILL.md` first and let its PPU-specific per-iteration rule decide
whether new PPU profiler evidence is needed. That route may use source or compiler inspection, the
probe-free benchmark, or still-valid PPU evidence instead of collecting a new profile.

Reuse a profile only when it matches the current Kernel. Otherwise request a profile:

```bash
python3 tools/sandbox.py --kind profile --profile-level sol
```

The Supervisor owns the profiler driver and real input construction. Select an opaque case with
`--profile-shape-id ID`, and use `--profile-source` when source correlation is needed.
For SOL command profiles, `--env PROFILE_WORKLOAD_IDX=2` selects a workload and
`--env PROFILE_ITERS=30` changes driver iterations. No driver is installed in this workspace.

When several shapes or workloads need profiling, run one sandbox command per id in waves of at most
four concurrent jobs and wait for the whole wave before starting the next one. Read the returned
facts or retrieve the stored Gateway Record by ID. No download is required. If a specific diagnostic
file is needed, opt in with a distinct `--sync scratch/profile-<id>` destination for that job.

For generalized Atrex-Bench tasks, choose `--profile-shape-id` from the previous canonical memory's
complete opaque-id `performance.latency_us_by_shape` map. The sandbox privately resolves that id and
injects only its real input case into the ephemeral remote profile job; the driver deletes the case
JSON before importing candidate code. Profile the highest-cost ids and additional ids representing
distinct latency regimes, but do not infer or reconstruct the complete hidden input table.

Extract a concrete bottleneck and source-level target. Use PTX/SASS/TTGIR inspection when compiler
lowering or instruction selection is part of the hypothesis. Do not make speculative optimization
changes before obtaining usable evidence.

When ordinary profiling has isolated one kernel but cannot distinguish a specific in-kernel timing
hypothesis, read `skills/autonomous-gpu-kernel-timeline/SKILL.md` and run its autonomous loop. Use
standalone CUDA/inline PTX through its CUDA backend and CuTe DSL through IKeT. Keep every attempt
under `scratch/timeline/attempt-N`; when the remote command reads backend files, pass that
specific skill path with sandbox `--input` and sync only the attempt output directory.

For PPU, use the routing and capture contracts in the PPU skill linked above.

Timeline instrumentation is a temporary working snapshot in `scratch/`, not a
candidate. Preserve the clean source and each useful instrumented source or reversible patch before
replacing it. After the evidence answers the question, restore or rewrite a probe-free `kernel.py`
before correctness/performance validation, Journal finalization, and handoff. Never submit a
profiling snapshot as the terminal candidate; its latency and failures do not count as promotion attempts
or framework-stall events.

Escalate through the typed profile funnel instead of collecting everything at once: `--profile-level
survey` to enumerate kernels, `sol` (the default) for the bottleneck class, and `deep --kernel-regex
'^<exact_base_function_name>$'` for one named kernel, especially a Triton `@triton.jit` entry. Take
that name verbatim from the survey/SOL result; never guess a substring. Raw `.ncu-rep`/ATT artifacts
stay remote unless `--include-raw-profile` is justified.

On NVIDIA, `summary.txt` carries a `LOCALIZE` line naming the analysis files that pin a symptom to
source lines. Those files exist only on a `--profile-source` run: never pin a source-level claim to a profile
collected without it.

#### Custom diagnostic workloads

Build a diagnostic driver at `scratch/harness/probe.py` and run it through
`--kind dev` when standard profiling cannot express the case — a multi-kernel sequence, a new synthetic
case inside the public domain, or a driver that needs sibling helper modules. It must import `kernel.py` plus the
immutable input module, select a representative workload, warm up, invoke the entry point repeatedly,
and never write memory files. Because it lives below `scratch/harness/`, Python does not put the
workspace root on its import path; add it before importing anything local:

```python
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
```

The sandbox uploads that whole `harness/` directory automatically, so sibling helpers need no extra
flags. Only a file opened dynamically by command code needs a repeatable `--input <relative-path>`
option before `--`, which routes the job through the dev interface.

### 3. Research progressively

Search in this order and stop when one actionable direction is supported:

1. **GPU Wiki through the natural-language front door.** Profile first, then describe the measured
   problem rather than trying to guess query flags. PPU is the exception: start from the decisive
   evidence selected by `ppu-acu-joint-profile`, whether or not it required a new profile:

   ```bash
   python3 tools/sandbox.py --kind wiki-query "<your description>" --brief
   python3 tools/sandbox.py --kind wiki-query --file scratch/research_request.txt
   ```

   Include the true target product and authoritative runtime architecture exactly as supplied. Ask for
   the full product specification and relevant architecture/ISA facts so the response contains isolated
   `hardware_wiki` and `kernel_wiki` records. Also include the operator, framework, shapes, dtypes,
   profile numbers, what was already tried, exact failures, competing hypotheses, and the fact that would
   end this line of work. For a PPU iteration that skipped profiling, identify the source, compiler,
   clean-benchmark, or prior PPU evidence used instead. Do not translate the hardware identity or
   pre-compress the prose into keywords.

   Read the compact response before acting: records are keyed by stable id, every `payload` is isolated,
   `store` distinguishes `gpu_wiki` from namespaced `internal_gpu_wiki` records,
   `match.arch` states its reach, and `notes` reports deterministic normalization, widening, truncation,
   or store gaps. Pass `--exclude <ids-already-read>` on later queries and use `--max-bytes` for a hard
   context bound. `sandbox.py --kind wiki-search` and `sandbox.py --kind wiki-hardware` remain available when the
   exact address is already known; never drop architecture scope to manufacture a match.
   Treat the returned content as a hypothesis and verify it against the current source and Gateway
   facts. Do not copy Wiki transport metadata into the Runtime Journal.
2. `reference-projects/` only when the local wiki is insufficient.
3. Public primary sources only when local sources do not answer the question.

After repeated rejected episodes, expand across DSLs targeting the same architecture instead of
repeating local parameter tweaks. Record the evidence-to-action chain in the Experiment analysis.

### 4. Plan a coherent direction

Propose the Direction through `update-direction` with its hypothesis, rationale, structured `plan`,
`success_criteria`, and `stop_conditions`. Start it before exploration. Use existing Gateway facts
and relevant research to choose concrete edits, risks, and rollback criteria; no separate plan
generator or file is required. For each measured change, record the actual edit and conclusions
through `record-experiment`, citing the supporting Gateway Records.

The episode may contain multiple related experiments, but they must advance one coherent engineering
direction. Preserve useful intermediate source copies in `scratch/` or retrieve captured Kernels by
ID, so failed sub-steps can be reverted without losing the whole direction. Do not use Git.

### 5. Implement and repair

Modify only candidate source/metadata files allowed by policy. Compile and probe through the sandbox.
On compile or correctness failure, diagnose and repair while the direction remains viable. Do not
publish an intermediate checkpoint as a candidate.

Land one optimization category per edit — vectorized load, swizzle, double buffering, tiling change,
and so on — and attribute each edit as `evidence -> inference -> action`. Do not mix unrelated
refactors, formatting, or cleanup into the same change: a bundled edit makes a regression
unattributable. When the evidence localizes a symptom to specific lines, change those lines only.

### 6. Development correctness and performance

Use the immutable evaluator for development measurements:

```bash
python3 tools/sandbox.py --kind run --version vlong --no-sync
python3 tools/sandbox.py --kind run --version vlong --multi-seed 5 --no-sync
```

All workloads and all additional seeds must pass. Never depend on tensor values, pointer identity,
cached outputs, evaluator ordering, or hidden workload IDs. Shape/dtype/layout dispatch is allowed,
and pre-converting stable weights (transpose, contiguous) is allowed because weights do not change
during evaluation.

For a surprising delta, inspect the per-shape facts and compare the relevant stored Gateway Records.
Do not resubmit an identical operation for identical Kernel bytes: the Supervisor owns repeated
measurement and infrastructure retries. GPU selection belongs to the gateway; never set a local
`CUDA_VISIBLE_DEVICES` to steer it. The supervisor independently compares incumbent and candidate
in one ABBA allocation for promotion.

### 7. Record every decisive experiment immediately

Immediately after each decisive measured change, use the Supervisor-backed Runtime Journal tools
defined in the Episode prompt. Do not batch these writes at the end of the Episode. Register a
Direction before active exploration, record the Experiment with a `gateway_record_ids` list citing
the results used in your analysis, then update the Direction lifecycle. The results already identify
their exact Kernels; the Experiment does not need Kernel IDs or a before/after pair.

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

## Leaving the loop

Leave the loop as soon as one coherent candidate passes the full development correctness check and
has credible performance evidence, or as soon as the direction is exhausted or blocked. Then follow
the episode prompt's terminal contract for finalizing the journal and publishing the handoff. For a
PPU full episode, include `outcome.accepted_ppu_diagnostics` using the schema in
`skills/ppu-acu-joint-profile/SKILL.md`; retain only evidence that still applies to the terminal
probe-free kernel. Each retained row must bind an accepted decision-grade artifact by path, SHA-256,
schema, and evidence id. This is optional when no reusable PPU profiler evidence exists.
