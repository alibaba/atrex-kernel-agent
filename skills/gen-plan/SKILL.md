---
name: gen-plan
description: Generate a structured implementation plan from an evidence draft. Validate paths, obtain configured independent Codex and Qoder reviews, synthesize available advice against repository evidence, preserve the draft, and produce testable acceptance criteria and validation steps.
---

# Generate Plan

This repository-native skill is adapted from the `gen-plan` flow in PolyArch Humanize. It turns an
episode draft into a complete implementation plan without modifying source code or starting the
implementation.

## Arguments

- `--input <path>`: required draft document.
- `--output <path>`: required new plan document.
- `--direct`: generate a one-shot plan without asking questions.
- `--discussion`: ask for decisions that materially change the plan. This is the default when no
  mode is supplied.

`--direct` and `--discussion` are mutually exclusive.

## Hard boundaries

- During this workflow, persist only the requested plan file. Consultation helpers may use
  automatically removed process scratch for isolation.
- Do not edit source, run implementation tasks, create commits, or start another workflow.
- Enabled `ask_codex` and `ask_qoder` consultations are read-only. They are non-persistent by
  default; an explicitly configured long Codex or Qoder reviewer session remains read-only and
  campaign-private. Give every enabled reviewer the same candidate proposal and bounded evidence
  packet, isolate both from the project, and disable Qoder tools.
- Preserve every requirement, constraint, measurement, search result, and rejected direction from
  the draft. The structured plan must be a superset of the draft.
- Keep the original draft verbatim at the bottom of the plan between the template markers.

## Workflow

Execute these phases sequentially.

### 1. Validate input and output

From the campaign workspace, run:

```bash
bash skills/gen-plan/scripts/validate-gen-plan-io.sh \
  --input <input> --output <output> <mode>
```

Stop on a nonzero exit. The script reports the resolved input, output, template, and mode. It never
creates the output file.

### 2. Check relevance

Read the draft and quickly inspect the workspace README, goal, current kernel, prior memory, and any
paths named by the draft. Reject only a draft that is clearly unrelated to this repository. Be
lenient with informal drafts and mixed languages.

### 3. Analyze the draft

Build an evidence-to-action chain:

1. Identify the measured bottleneck or failure.
2. Connect the evidence to a concrete inference.
3. Select exactly one coherent optimization category for the plan.
4. Identify the smallest concrete file changes that test that inference.
5. Define correctness, performance, rollback, and stop conditions.

Before consulting either reviewer, populate
`skills/gen-plan/templates/candidate-proposal-template.md` in automatically removed process scratch.
This frozen candidate proposal is not the final plan. It must state the selected evidence-to-action
chain, exactly one optimization category, target paths and symbols, the expected mechanism, scope
constraints, rejected directions, validation and falsification conditions, and unresolved
assumptions. Do not persist the candidate proposal in the campaign workspace.

Check the draft for unclear scope, contradictions, missing dependencies, infeasible changes, and
quantitative targets. Treat numeric performance targets as trends unless the draft explicitly marks
them as hard acceptance thresholds.

In `direct` mode, make conservative assumptions and record unresolved material choices under
`Pending Decisions`; do not pause for questions. In `discussion` mode, ask only questions whose
answers materially change scope, correctness, or acceptance.

### 4. Obtain configured independent reviews

The campaign independently configures Codex and Qoder for fast and full episodes. It probes a
reviewer only when that reviewer is first enabled for the current episode mode, then caches the
availability decision in private runtime state. A reviewer disabled by configuration or by its
availability probe must not be retried; retain the helper's `disabled` status and recorded reason.
Campaign restarts reuse cached availability decisions.

After completing the initial analysis, freeze one evidence packet and use the bundled review helper
before finalizing the plan direction. Give every enabled reviewer the candidate proposal, original
draft, and the same small set of directly relevant text files, normally `README.md`, `kernel.py`,
the latest canonical memory entry, and source or profile summaries cited by the draft. The candidate
is the primary review target; the draft and context are evidence for testing its claims. Never
include credentials, raw secrets, unrelated files, or large binary profile artifacts.

```bash
bash skills/gen-plan/scripts/ask-reviewers.sh \
  --input <input> \
  --proposal <temporary-candidate-proposal> \
  --context README.md \
  --context kernel.py
```

`--proposal` is required and must name the non-empty frozen candidate proposal in process scratch.
Add other `--context` arguments only when they materially affect the plan. The helper starts the
enabled external reviewers concurrently so neither can see or anchor on the other's response.
When proposal, draft, and context would exceed Qoder's five-attachment limit, the helper folds all
context files into one labeled temporary bundle and gives that identical bundle to every enabled
reviewer. Do not manually remove evidence or issue a second full review call. For an eligible
transient failure, the helper retries only the failed reviewer once; it does not rerun a successful
reviewer and does not retry quota, authentication, timeout, disabled, or missing-CLI failures.
Enabled external reviewer processes always use maximum reasoning effort; episode/session settings,
reviewer effort environment variables, and legacy `--reasoning-effort` arguments cannot lower it.
By default each external review is ephemeral. `--long-reviewer-session codex` or
`--long-reviewer-session qoder` resumes one campaign-private, read-only reviewer thread across
episodes while continuing to send the complete current candidate proposal, draft, and bounded
context on every call. Long Claude reviewer sessions are not implemented and fail explicitly.
Session state lives under `.atrex_long_horizon/` and must never enter a candidate commit. Because
`qodercli` only resolves resumable sessions within the current working directory's project, a
persistent Qoder reviewer runs from a dedicated directory alongside its state file, which also keeps
it isolated from the candidate project.
Each review returns its backend-specific summary marker followed by the same five assessment
sections:

- `CODEX_SUMMARY` or `QODER_SUMMARY`
- `RISKS`
- `MISSING_REQUIREMENTS`
- `DIRECTION_RECOMMENDATIONS`
- `VALIDATION_RECOMMENDATIONS`
- `QUESTIONS_OR_ASSUMPTIONS`

If the current episode backend is Codex or Qoder and its matching
`ATREX_PLAN_REVIEW_*_ENABLED` value is not `0`, first review the frozen candidate proposal and retain
that backend's review in the current session using the same sections. Only then run the helper; it
skips the matching nested process and obtains any other enabled backend's independent review. Mark
the retained review status `current_codex_session` or `current_qoder_session`. Do not revise it after
seeing the external review; resolve new information only during synthesis. If the matching reviewer
is disabled, do not create an in-session substitute review; let the helper record `disabled`.

After the helper's selective retry, if an enabled reviewer is unavailable, times out, or fails, do
not fabricate its advice or rerun the successful reviewer. In `direct` mode, continue with the
available review and conservative analysis, recording each status and failure reason. If every
enabled reviewer fails, continue using only the primary analysis and label the result as unreviewed.
In `discussion` mode, ask whether to retry only the failed reviewer or continue with partial or no
independent review. A reviewer explicitly marked `disabled` is not a failure and must not trigger a
retry question. If every reviewer is disabled, continue with primary analysis and label the plan as
intentionally unreviewed.

### 5. Synthesize available reviews and generate the plan

Compare all available reviews only after the helper has completed. When both are enabled, treat
agreement as a useful confidence signal, not proof, and resolve disagreement from the original
draft and repository evidence rather than by majority vote. Evaluate every recommendation as
follows:

1. Adopt a suggestion only when it strengthens the selected evidence-to-action chain, closes a
   correctness gap, or makes validation more deterministic.
2. Reject suggestions that contradict measured evidence, violate campaign constraints, or introduce
   another optimization category.
3. Defer suggestions that are plausible but need evidence outside the current direction.
4. Resolve conflicting suggestions explicitly, stating the evidence that selected one or rejected
   both.
5. Convert unresolved reviewer questions into conservative assumptions or pending decisions
   according to the selected direct/discussion mode.

Record how the frozen candidate changed after review. Every material correction must identify the
reviewer and supporting evidence; if the candidate remains unchanged, state why the reviews did not
justify a change.

Available reviewers are advisory, not authoritative. The final plan must remain a superset of the
human draft and must still contain exactly one optimization category.

Use `skills/gen-plan/templates/gen-plan-template.md` as the output schema. Replace every placeholder
with concrete content. The plan must include:

- the goal and the profile/research evidence that motivates it;
- Codex and Qoder consultation status (including configured-disabled status), available material
  findings, agreements, disagreements, and the suggestions adopted, rejected, or deferred with
  reasons;
- exactly one optimization category and its evidence-to-inference-to-action chain;
- acceptance criteria in `AC-N` form, each with positive and negative tests;
- upper and lower scope boundaries plus allowed and prohibited choices;
- exact target paths and a dependency-ordered implementation sequence;
- full-workload correctness, multi-seed correctness, and comparable performance validation;
- measurable success, rollback, and direction-exhaustion conditions;
- any assumptions or pending decisions; and
- the original draft, unchanged, at the bottom.

Use milestones, phases, and steps rather than time estimates. Refer to code by path and symbol, not
by line range. Plan terminology such as `AC-N`, `Milestone`, and `Phase` belongs in the plan only and
must not be prescribed as implementation naming.

### 6. Review and write

Before writing, verify that the plan:

- does not omit or contradict the draft;
- uses available reviews selectively and records the disposition of material suggestions and
  conflicts;
- records the frozen candidate and the evidence-based changes made after review;
- proposes only one attributable optimization category;
- names concrete files and validation commands;
- distinguishes correctness from performance evidence;
- has deterministic, measurable acceptance and rollback conditions; and
- contains no implementation changes made during planning.

Write the complete plan to the validated output path, then read it back and fix any remaining
placeholder, inconsistency, or missing draft content. Report the output path, optimization category,
both consultation statuses, adopted-suggestion count, acceptance-criteria count, and pending-decision
count.

## Validation exit codes

| Exit code | Meaning |
| --- | --- |
| 0 | Validation passed |
| 1 | Input file not found |
| 2 | Input file is empty |
| 3 | Output directory does not exist |
| 4 | Output path already exists |
| 5 | Output directory is not writable |
| 6 | Invalid arguments |
| 7 | Plan template is missing |

## Reviewer consultation exit codes

| Exit code | Meaning |
| --- | --- |
| 0 | Consultation completed, or nested invocation intentionally skipped for the matching backend |
| 1 | Draft, candidate proposal, or context input is invalid |
| 2 | Helper arguments or environment configuration are invalid |
| 3 | Reviewer response is missing one or more required review sections |
| 124 | Reviewer consultation timed out |
| 127 | Reviewer CLI could not be found or started |

`ask-reviewers.sh` reports the exit status of each child consultation in its structured output and
returns successfully once both attempts finish, allowing direct mode to retain a surviving review.
