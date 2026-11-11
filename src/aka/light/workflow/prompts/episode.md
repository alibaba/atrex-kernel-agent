# Kernel optimization episode {{EPISODE}}

## Context

- Workspace: `{{WORKSPACE}}`
- Agent-visible comparison base: `{{BASE_COMMIT}}`; measurable candidate identity: clean Git HEAD
- Campaign record slot: v{{VERSION}}
- Hardware: `{{ARCH}}`
{{DSL_CONTEXT}}
- Additional constraints: {{NOTES}}
{{PLUGINS}}
{{REFERENCE_KNOWLEDGE}}
{{RUNTIME_WAIT_POLICY}}
{{RESUME_DIRECTIVE}}
{{GOAL_DIRECTIVE}}
{{ACCEPTANCE_REQUEST}}
{{CONVERSION_DIRECTIVE}}
{{PPU_DIRECTIVE}}

## Workflow

1. Read `CLAUDE.md`, `WORKSPACE.md`, root `PROGRESS.md`, `skills/runtime-records/SKILL.md`, and the current Kernel. Open only relevant Direction and Experiment files. Read or query the relevant `records/<gateway_record_id>.json` directly, selecting only the fields, Shape, or bounded text needed; never print a complete Record. Read a specific historical Conversation only when Git, PROGRESS, research files, and Records cannot recover a decision. Prior analysis is fallible; Gateway Records are measurement authority.
2. Choose the unresolved path with the highest plausible full-workload latency impact and applicability, not the easiest edit. A Direction is a durable optimization program around a bottleneck, workload regime, or architecture objective and normally spans multiple candidates and Experiments. A single implementation, parameter sweep, capability probe, or mechanism test is an Experiment, not a new Direction. Bound implementation risk with an Experiment rather than discarding higher-upside work.
3. Use local Git to manage the Kernel development history and preserve the measured best. Before every Kernel-bound Runtime request, commit the complete candidate and leave the worktree clean. Run the smallest measurement that resolves the current question, inspect its saved Record, then keep, repair, or restore the candidate from evidence.
4. Repeat `probe → candidate → measurement → revert/keep` cycles while a high-value step remains. A checkpoint or failed candidate is not a reason to stop.
5. Submit `episode-report` only when a useful candidate checkpoint has a passing standard full Evaluate, an evidence-backed pivot is ready, pooled search must publish a reusable measured checkpoint to peers, or a concrete blocker prevents further work. Ordinary `direction_terminal` handoff may preserve `in_progress` Directions for a later Episode. Use `candidate_ready` only for a changed candidate. If the comparison base remains best, submit `pivot`; never add a nominal edit to create a diff. `cross_trajectory_share` is available only when this Prompt begins with a `Search location` header. Follow the installed handoff contract and stop only after Runtime accepts the report.

Keep reasoning concise and act once the next step is testable.

{{RESEARCH_MODULE_RULES}}

{{HANDOFF_CONTRACT}}

Write `scratch/episode-report.json` and submit it with `python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json`. Correct a rejected report only while its handoff condition remains true. Report acceptance is a handoff, not promotion; independent verification follows.
