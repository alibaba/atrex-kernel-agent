# GPU Kernel Optimizer — Agent Constraints

This file defines hard behavioral constraints for the optimization workflow.
For the full stage-by-stage workflow, read the skill files referenced below.

## Workflow Stage Delegation Rules

When executing `skills/gpu-kernel-profile-optimizer/SKILL.md`:

### Stage 2 (Evidence-Driven Search and Planning)

- The main agent MUST launch a research subagent for Stage 2.
- The subagent MUST follow `skills/gpu-kernel-research/SKILL.md` exactly.
- The main agent SHALL NOT perform evidence search or write the plan directly.
- The main agent SHALL NOT call gpu-wiki search, read reference-projects, or web search for optimization knowledge — this is the subagent's job.

### Stage 4 (Performance, Correctness, and Quality Gate)

- The main agent MUST launch a subagent for Stage 4 validation.
- The main agent SHALL NOT run correctness tests, measure performance, or write the iteration report directly.
- The main agent SHALL NOT fabricate performance numbers or skip validation.

## Prohibited Main Agent Actions

- DO NOT perform search/plan writing that belongs to Stage 2.
- DO NOT run validation/measurement that belongs to Stage 4.
- DO NOT skip the subagent delegation by inlining these tasks.
- DO NOT proceed to Stage 3 without receiving the subagent's plan output.
- DO NOT proceed to Stage 5 without receiving the subagent's quality gate result.

## Skill References

- Full optimization workflow: `skills/gpu-kernel-profile-optimizer/SKILL.md`
- Research subagent contract: `skills/gpu-kernel-research/SKILL.md`
