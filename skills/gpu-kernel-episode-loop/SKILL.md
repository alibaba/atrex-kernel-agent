---
name: gpu-kernel-episode-loop
description: Run the evidence loop of one long-horizon GPU kernel optimization episode. Use this skill to reconstruct the incumbent, profile and localize a bottleneck, research progressively, plan one coherent direction, implement and repair, validate development correctness and performance, and record every decisive experiment in the episode journal.
---

# GPU Kernel Episode Loop

## When to Use

Use this skill when an orchestrator episode prompt hands you one optimization episode in an isolated
Git worktree and points here for the evidence loop. It does not apply to the V0 baseline session
(`gpu-kernel-baseline`) or to a workspace without an episode journal.

## Episode bindings

The episode prompt supplies a concrete value for every ALL-CAPS bracketed name below. Substitute them
before running any command; never invent a path. Lowercase bracketed names in the command examples are
values you fill in from the campaign, or a choice among the listed alternatives, not bindings.

| Placeholder | Meaning |
| --- | --- |
| `<PROFILE_DIR>` | profile output directory for this episode |
| `<PLAN_DRAFT>` | evidence draft for the canonical version |
| `<PLAN_FILE>` | generated plan for the canonical version |
| `<PLAN_GENERATOR>` | backend-native plan generator invocation |
| `<JOURNAL_CLI>` | episode journal command prefix |
| `<JOURNAL_PATH>` | episode journal path, already shell-quoted |

The episode prompt's ownership rules, execution boundary, mode policy, and framework-escalation
directive outrank this skill. Where they conflict, follow the prompt.

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

Read the workspace goal, unmasked `memory/v*.json`, prior plans/profiles, and the prior-episode
evidence block in the episode prompt. Identify attempted dead ends and open directions. Start with one
falsifiable hypothesis tied to the current bottleneck.

### 2. Profile and localize

Reuse a profile only when it matches the current committed kernel. Otherwise profile through the
sandbox using the vendor-appropriate tooling. Both wrappers run `python <file>`, so the profiled file
is the immutable `profile_driver.py` seeded next to `kernel.py` — never `kernel.py` itself, which the
evaluator only ever imports:

```bash
# NVIDIA
python tools/sandbox.py --kind profile --sync <PROFILE_DIR> -- \
  bash tools/profile_nvidia.sh profile_driver.py --output-dir <PROFILE_DIR> --source

# AMD
python tools/sandbox.py --kind profile --sync <PROFILE_DIR> -- \
  bash tools/profile_kernel.sh profile_driver.py --output-dir <PROFILE_DIR>
```

`profile_driver.py` imports the current `kernel.py`, builds real inputs from the campaign's immutable
ground truth (`definition.json` + `workload.jsonl`, or `shapes.json` + `input.py`), warms up, and then
invokes the candidate repeatedly. Select what it drives with environment variables rather than editing
it — it is a protected path and a candidate that modifies it is rejected:

```bash
PROFILE_ITERS=30 PROFILE_WORKLOAD_IDX=2 python tools/sandbox.py ...   # SOL: one workload
PROFILE_ITERS=30 PROFILE_SHAPE_ID=3 python tools/sandbox.py ...       # Atrex-Bench: one shape
```

Extract a concrete bottleneck and source-level target. Use PTX/SASS/TTGIR inspection when compiler
lowering or instruction selection is part of the hypothesis. Do not make speculative optimization
changes before obtaining usable evidence.

Escalate through the typed profile funnel instead of collecting everything at once: `--profile-level
survey` to enumerate kernels, `sol` (the default) for the bottleneck class, and `deep --kernel-regex
'^<exact_base_function_name>$'` for one named kernel, especially a Triton `@triton.jit` entry. Take
that name verbatim from the survey/SOL result; never guess a substring. Raw `.ncu-rep`/ATT artifacts
stay remote unless `--include-raw-profile` is justified.

On NVIDIA, `summary.txt` carries a `LOCALIZE` line naming the analysis files that pin a symptom to
source lines. Those files exist only on a `--source` run: never pin a source-level claim to a profile
collected without it.

#### When the seeded driver cannot represent the work

Build a local fallback driver at `<PROFILE_DIR>/harness/profile_driver.py` and profile that file
instead when the seeded driver cannot express the case — a multi-kernel sequence, a shape outside the
ground-truth set, or a driver that needs sibling helper modules. It must import `kernel.py` plus the
immutable input module, select a representative workload, warm up, invoke the entry point repeatedly,
and never write memory files. Because it lives below `<PROFILE_DIR>/harness/`, Python does not put the
workspace root on its import path; add it before importing anything local:

```python
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
```

The sandbox uploads that whole `harness/` directory automatically, so sibling helpers need no extra
flags. Only a file opened dynamically by command code needs a repeatable `--input <relative-path>`
option before `--`, which routes the job through the dev interface.

### 3. Research progressively

Search in this order and stop when one actionable direction is supported:

1. `gpu-wiki/docs/` and `gpu-wiki/reference-kernels/`, filtered by real architecture, vendor, DSL,
   operator, and profiler symptom. Use the architecture-scoped index before any broad grep:

   ```bash
   python3 gpu-wiki/scripts/query.py --arch <arch> --vendor <nvidia|amd> \
     --area docs --section kernel-opt --symptom <controlled-symptom>
   python3 gpu-wiki/scripts/query.py --arch <arch> --vendor <nvidia|amd> \
     --area docs --dsl <dsl> --operator <operator> --section ref-docs --section pitfalls
   python3 gpu-wiki/scripts/query.py <operator-or-mechanism> --arch <arch> \
     --vendor <nvidia|amd> --dsl <dsl> --area reference-kernels --kind kernel
   ```

   Open the returned pages and follow their local links. `--symptom` accepts a controlled vocabulary:
   `compute-bound`, `memory-bound`, `low-sm-utilization`, `pipeline-stalls`, `register-pressure`,
   `tail-effect`, `moe-load-imbalance`. Omit `--area` only when combined docs/reference results are
   useful. Narrow reference hits with `--source`, `--status`, or `--kind`; test/build/package files
   require `--include-auxiliary`. Retry uncertain spellings with `--fuzzy` while keeping the same
   architecture/vendor/DSL filters. Unknown filters fail closed — never drop `--arch` to turn an empty
   result into a successful-looking one. On NVIDIA SM90/SM100, `gpu-wiki/3rdparty/` and the
   `KernelWiki` skill are last-resort local sources, used only after the indexed pages fall short.

   `--arch` takes an index token, which is not always the runtime arch string: `sm_90`, `sm_100`, and
   `sm_103` are rejected as `unknown-arch`. Pass the underscore-free form (`sm90`, `sm100`, `sm103`,
   `sm120`, `gfx942`, `gfx950`) or the family name (`hopper`, `blackwell`, `cdna3`, `cdna4`), and run
   `--list-arch` when a token is uncertain. An `unknown-arch` error is a token problem, never a reason
   to retry without `--arch`.
2. `reference-projects/` only when the local wiki is insufficient.
3. Public primary sources only when local sources do not answer the question.

After repeated rejected episodes, expand across DSLs targeting the same architecture instead of
repeating local parameter tweaks. Record source paths and the evidence-to-action chain.

### 4. Plan a coherent direction

Write or update `<PLAN_DRAFT>` with profile evidence, research findings, concrete edits, risks,
rollback points, and measurable acceptance criteria. Then produce `<PLAN_FILE>` with the
backend-native plan generator `<PLAN_GENERATOR>`.

The episode may contain multiple related experiments, but they must advance one coherent engineering
direction. Checkpoint useful intermediate states so failed sub-steps can be reverted without losing
the whole direction.

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
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --no-memory
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --multi-seed 5 --no-memory
```

All workloads and all additional seeds must pass. Never depend on tensor values, pointer identity,
cached outputs, evaluator ordering, or hidden workload IDs. Shape/dtype/layout dispatch is allowed,
and pre-converting stable weights (transpose, contiguous) is allowed because weights do not change
during evaluation.

Before trusting a large delta — especially a regression beyond roughly 30% — re-run the same command
on the same sandbox hardware and compare. GPU selection belongs to the gateway; never set a local
`CUDA_VISIBLE_DEVICES` to steer it. Repeated development measurements are not promotion authority;
the supervisor reruns incumbent and candidate in one ABBA allocation.

### 7. Record every decisive experiment immediately

Immediately after each decisive experiment, append it to the single episode journal. Do not batch
these writes at the end of the episode: every append refreshes the non-canonical `memory/live.json`
progress view in the incumbent workspace.

```bash
<JOURNAL_CLI> append --path <JOURNAL_PATH> \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

## Leaving the loop

Leave the loop as soon as one coherent candidate passes the full development correctness check and
has credible performance evidence, or as soon as the direction is exhausted or blocked. Then follow
the episode prompt's terminal contract for finalizing the journal and publishing the handoff.
