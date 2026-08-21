# <Plan Title>

## Goal

<State the concrete optimization outcome and the constraint it must preserve.>

## Evidence and Direction

- Evidence: <profile, benchmark, source, or prior-episode observation>
- Inference: <what the evidence implies about the bottleneck>
- Optimization category: <exactly one category>
- Action: <smallest coherent implementation direction>

## Dual Review

### Candidate Proposal Reviewed

- Frozen hypothesis: <candidate evidence-to-inference claim>
- Frozen optimization category: <exactly one category>
- Frozen proposed action: <candidate target paths, symbols, and expected mechanism>

### Codex Findings

- Consultation status: <completed, current_codex_session, or unavailable with reason>
- Material findings: <concise Codex risks, gaps, direction, and validation advice>

### Qoder Findings

- Consultation status: <completed, current_qoder_session, or unavailable with reason>
- Material findings: <concise Qoder risks, gaps, direction, and validation advice>

### Cross-Reviewer Synthesis

- Agreements: <where both reviews align and the supporting repository evidence>
- Disagreements resolved: <conflict, selected disposition, and decisive evidence>
- Candidate changes after review: <evidence-based corrections, or why no change was justified>
- Adopted suggestions: <suggestion, source reviewer or both, and supporting evidence>
- Rejected or deferred suggestions: <suggestion, source reviewer or both, disposition, and reason>

## Acceptance Criteria

- AC-1: <Deterministic correctness or implementation criterion>
  - Positive tests (expected to pass):
    - <test and expected result>
  - Negative tests (expected to fail or reject an invalid candidate):
    - <test and expected rejection>
- AC-2: <Deterministic performance or evidence criterion>
  - Positive tests (expected to pass):
    - <test and expected result>
  - Negative tests (expected to fail or trigger rollback):
    - <test and expected rejection>

## Path Boundaries

### Upper Bound

<Most comprehensive acceptable implementation without broadening the optimization category.>

### Lower Bound

<Minimum implementation that still tests the hypothesis and satisfies every acceptance criterion.>

### Allowed Choices

- Can use: <allowed technologies, patterns, or dispatch inputs>
- Cannot use: <prohibited dependencies, fallbacks, hidden dispatch, or evaluator coupling>

## Target Changes

| Path | Intended change | Evidence link | Rollback point |
| --- | --- | --- | --- |
| `<path>` | <concrete change> | <evidence/inference> | <condition> |

## Dependencies and Sequence

1. <Milestone or step>
   - <dependency-ordered action>
2. <Milestone or step>
   - <dependency-ordered action>

## Validation

### Correctness

- <full-workload command and expected result>
- <multi-seed command and expected result>

### Performance

- <comparable benchmark/profile command and measurable criterion>
- <repeatability or attribution check>

### Rollback and Stop Conditions

- <when to revert an experiment>
- <when the direction is exhausted>

## Pending Decisions and Assumptions

- <`None`, or a concrete assumption/decision with its impact>

## Implementation Notes

- Implementation code and comments use domain terminology, not plan markers such as `AC-`,
  `Milestone`, `Step`, or `Phase`.

--- Original Design Draft Start ---

<Insert the original draft verbatim.>

--- Original Design Draft End ---
