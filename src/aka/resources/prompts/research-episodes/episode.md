# Kernel optimization episode {{EPISODE}}

This is an audit segment in one persistent optimization Session that starts from V0. Each Episode
inherits the same conversation, workspace, and Agent Git history.

## Context

- Workspace: `{{WORKSPACE}}`
- Agent-visible comparison base: `{{BASE_COMMIT}}`; candidate identity: clean Agent Git HEAD
- Hardware: `{{ARCH}}`
{{DSL_CONTEXT}}
- Additional constraints: {{NOTES}}
{{PLUGINS}}
{{REFERENCE_KNOWLEDGE}}
{{RUNTIME_WAIT_POLICY}}
{{GOAL_DIRECTIVE}}
{{PPU_DIRECTIVE}}

## Workflow

1. Read `CLAUDE.md`, `WORKSPACE.md`, `PROGRESS.md`, `.agents/skills/runtime-records/SKILL.md`, and the current Kernel. Open only relevant research files and bounded fields from needed Gateway Records. Gateway Records are measurement authority.
2. Choose the unresolved path with the greatest plausible full-workload impact. A Direction is a durable program around a bottleneck, regime, or architecture objective; Experiments record concrete investigative attempts; related candidates or probes may share one Experiment while the question and approach remain unchanged. Bound risk with probes instead of avoiding high-upside work.
3. Manage development with Agent Git. Commit the complete candidate and leave the worktree clean before Kernel-bound Runtime requests. Run the smallest decisive measurement, inspect its Record, then keep, repair, or restore from evidence. When Experiment recording is enabled, combine related trials within one attempt and record its findings promptly when enough evidence exists, especially when moving to a different approach. Do not wait for the next attempt's results or treat recording as a prerequisite for continuing work.
4. Repeat `probe → candidate → measurement → revert/keep` cycles while valuable work remains. A checkpoint or failed candidate is not a reason to stop.
5. Use `submit-kernel` when you judge an exact commit ready to serve as a final implementation for the current mode. It must already have a passing standard full Evaluate with complete per-Shape performance data. Submit when ready, not mechanically after every Evaluate and not only at the end of an Episode. Fast permits declared reproducible compute libraries; Performance requires self-authored core compute while leaving DSL selection to you; Ultra additionally requires the configured DSL.

Keep reasoning concise and act once the next step is testable. For a complex rewrite, first build
the smallest end-to-end candidate that can be checked for correctness and measured. Use focused
probes to resolve implementation blockers; do not wait to settle every layout, pipeline, or
boundary question before implementing. Cost models and local probes guide the trial, but do not
rule out the whole optimization path unless their evidence applies to the proposed end-to-end
design. An unknown mechanism need not delay a high-value candidate trial.

Before abandoning or deferring a high-upside path because of complexity, risk, or estimated low
benefit, use `python3 tools/sandbox.py --kind wiki-query "<mechanism and missing implementation fact>"`
when Wiki is enabled. Name the mechanism and the question, not just the operator: Wiki searches
relevant knowledge and implementations using mechanism names and aliases. Inspect relevant
returned paths, line numbers and snippets for applicability, then use findings to bound a small
end-to-end trial. Source matches are navigation aids, not validated facts. An empty or partial
search does not prove no implementation exists. If Wiki is disabled or unavailable, continue
from available evidence.

For Atrex-Bench tasks, the read-only `atrex-bench/` directory contains the same source used by
remote evaluation. Read `scripts/run_eval.py` and `src/atrex_bench/` within it as needed to
understand timing, warmup, sampling, and correctness checks. Run measurements through Runtime.

{{RESEARCH_MODULE_RULES}}

{{SUBMISSION_CONTRACT}}

Write `scratch/kernel-submit.json` with `{"kernel_commit_id":"<full Agent Git commit>"}` and run `python3 tools/sandbox.py --kind submit-kernel --request-file scratch/kernel-submit.json`. The first call returns a stable `submission_id`. To revise that candidate after Review feedback, fully evaluate the repaired commit and submit `{"submission_id":"<returned ID>","kernel_commit_id":"<new commit>"}`; omit the ID only for an unrelated Submission.
