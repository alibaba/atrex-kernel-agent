# Framework baseline

{{BASELINE_OBJECTIVE}}

## Context

- Workspace: `{{WORKSPACE}}` (your cwd)
- Reference version: v{{PREV}}; candidate version: v{{N}}
- Hardware: `{{ARCH}}`
{{DSL_CONTEXT}}
- Additional constraints: {{NOTES}}
{{PLUGINS}}
{{REFERENCE_KNOWLEDGE}}
{{CORRECTNESS_GUIDANCE}}
{{SHAPE_DISCLOSURE}}

## Workflow

1. Read `README.md`, `CLAUDE.md`, `WORKSPACE.md`, `skills/gpu-measurement/SKILL.md`, the current candidate source, and the public contract/reference files. Read immutable `metadata.json` when present.
2. Research only implementation-critical missing facts. When `KernelWiki` is enabled, follow `skills/KernelWiki/SKILL.md`; its Wiki query is a knowledge request, not a Kernel measurement. Otherwise use the public contract and supplied static/compiler evidence instead of inventing facts.
3. Work like a normal Episode: implement one candidate at a time in the existing SourceProvider layout using only its declared editable paths, and use local Git to preserve each testable checkpoint. Before every Kernel-bound Runtime request, commit the complete candidate and leave the worktree clean.
4. Use Runtime Evaluate to establish correctness. Use subset Evaluate, Check, Profile, Disassemble, Dev, or Command when a smaller diagnostic or probe resolves a concrete uncertainty. Inspect the returned result and saved Gateway Record, then keep, repair, or restore the candidate from evidence. If this is a recovery Session, continue from the candidate already on disk instead of starting over.
5. Stop optimizing after a compliant candidate passes the standard full Evaluate. Performance measurements are useful diagnostics, but Episode 0 does not require a speedup over V0: do not keep optimizing a correct V1 merely to improve latency.
6. Leave the complete candidate at a clean Git HEAD and submit `episode-report` exactly as in a normal Episode. Use `status=candidate_ready`, `handoff_reason=direction_terminal`, and cite the passing full Evaluate through the enabled Experiment or direct Gateway Record fields. Active local Directions may remain `in_progress` for later Episodes. The Supervisor runs the normal final-candidate gates but does not require ABBA for Episode 0. If technically blocked, block every active local Direction and submit the ordinary blocked report with the concrete blocker.
