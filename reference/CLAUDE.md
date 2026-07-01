# GPU Kernel Optimizer — Agent Constraints

This file defines hard behavioral constraints for the optimization workflow.
For the full stage-by-stage workflow, read the skill files referenced below.

## Workflow Stage Delegation Rules

When executing `skills/gpu-kernel-profile-optimizer/SKILL.md`:

### Workflow Integrity Constraint (MANDATORY)

- The agent MUST strictly follow every step defined in the workflow, in the exact order specified.
- NO step may be skipped, abbreviated, or merged with another step under any circumstance.
- Each stage's entry conditions, execution steps, and exit criteria MUST be fully satisfied before proceeding.
- If a step appears unnecessary for the current context, the agent MUST still execute it and document why the result was trivial — skipping is NEVER permitted.

### Stage 2 (Evidence-Driven Search and Planning)

- The main agent MUST launch a research subagent for Stage 2.
- The subagent MUST follow `agents/gpu-kernel-research.md` exactly.
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

## Third-Party Library Prohibition

- Kernel optimization code MUST be implemented from scratch — DO NOT introduce any third-party libraries or external dependencies.
- All optimization logic, data structures, and algorithms used in kernel implementations MUST be self-written.
- Referencing third-party code for learning is permitted, but the final implementation MUST NOT depend on or copy from external libraries.
- If a kernel requires utility functions (e.g., memory management helpers, math primitives), they MUST be implemented inline or as project-local utilities — NEVER imported from external packages.

## Hardware Architecture Constraints

- **blackwell-geforce is NOT blackwell**: `blackwell-geforce` (sm120) and `blackwell` (sm100) are completely different architectures. Do NOT conflate them or assume they share the same optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: The sm103 hardware architecture is similar to sm100 (both belong to the Blackwell data-center family), but is completely different from sm120 (Blackwell GeForce / consumer). When searching for reference kernels or optimization knowledge for sm103, prefer sm100/blackwell sources — NEVER use sm120/blackwell-geforce sources as a substitute.

## Skill References

- Full optimization workflow: `skills/gpu-kernel-profile-optimizer/SKILL.md`
- Research subagent contract: `agents/gpu-kernel-research.md`
