# EVIDENCE_BACKED_DISTILLATION

Use only `evidence/evidence_manifest.json`, `evidence/performance_trajectory.json`, copied evidence artifacts,
and the hypothesis-only `teacher_gap_analysis.*` files in this workspace.

Generate:

- `journey.md`
- `pitfalls.md`
- zero or more `optimization_cards/*.md`
- `promotion_checklist.md`
- `draft_manifest.json` with schema version 1, evidence level `single-campaign`, and every generated document path.

Rules:

- Every performance number and verified causal claim must cite one or more evidence IDs as `[E-...]`.
- Masked, reverted, exploratory, and policy-violation evidence cannot prove a verified optimization.
- Teacher gap findings remain hypotheses and are not promotion-eligible.
- Do not copy or reconstruct Teacher source code.
- State the exact architecture, framework, workload/shape scope, and known boundaries.
- These are review drafts only. Never edit canonical `gpu-wiki/`.
- Do not access the public web, reference-projects, upstream repositories, or files outside this workspace.
