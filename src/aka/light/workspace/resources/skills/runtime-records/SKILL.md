---
name: runtime-records
description: Manage research Directions and Experiments, inspect durable evidence, and review exact Kernel commits.
allowed-tools: "Bash Read"
---

# Runtime records

Use `python3 tools/sandbox.py`. Runtime owns immutable measurements, exact measured source, and
commit review; the Agent owns optimization objectives, plans, and interpretations.

Start with `PROGRESS.md`. Open `frontier.json` only for the complete accepted-submission frontier, then only relevant
Direction, Experiment, and `records/<gateway_record_id>.json` files. Projections are read-only.
Gateway Records are measurement authority; all analysis is fallible.
`research-library/` is a read-only, visibility-filtered copy of the versioned Campaign Library. Local identities remain writable
only through Runtime; peer identities are always read-only. Synchronous pooled search freezes peer
revisions at an Epoch watermark, while asynchronous pooled search projects the latest peer revision
immediately.
Read or query `records/<gateway_record_id>.json` directly with ordinary file operations. Select only
the fields or bounded text needed; do not print an entire Record or long stdout/stderr into the Session.
Records observed by this trajectory remain local by default. A lineage sees Records cited by the
Direction/Experiment revisions currently visible in its own projection, plus passing standard full
Evaluates. Accepted submissions and winners therefore include their complete Evaluate evidence.

## Research loop

- A Direction is a durable optimization program organized around a consequential bottleneck, workload regime, or architecture objective, not necessarily an immediate source edit. It should normally span multiple candidates and, when enabled, multiple Experiments. A single implementation, parameter sweep, capability probe, or mechanism test belongs under an existing Direction rather than becoming one. Do not use Directions for Runtime/evaluator maintenance or infrastructure work. Check duplicates and rank by expected benefit and applicability. Multiple local Directions may be `in_progress`; the latest started one is the focus, while queue reordering does not change focus.
- Choose work from the Local work queue. The Shared direction pool is discovery-only: absorb
  compatible peer ideas into one local derivative and cite every source ID in
  `derived_from_direction_ids`.
- Revise a proposed or active local Direction using its stable ID and complete replacement claim.
  Restart deferred or blocked work first. Completed and abandoned Directions are terminal; continue
  them through a derived Direction.
- Direction lifecycle is `proposed --start--> in_progress`. Only `in_progress` may be completed,
  abandoned, deferred, or blocked. Direction records an optimization objective, not a single
  supported/refuted hypothesis; concrete falsifiable work and outcomes belong in Experiments.
- `propose` and revisions made while a Direction remains proposed use `review_not_requested`. `start`
  runs the admission Review; active revisions and closures are reviewed again. Every archived state
  is published to peers with its actual review quality; only eligible evidence may support research
  conclusions.
- Record decisive work as an Experiment with `name`, concrete `intent`, and bounded
  `interpretation`. Records and archived references are independent optional evidence. Keep measured
  facts in Records. A note without a Record is not promotion evidence. Revise only
  `local_writable` Experiments; extend `peer_read_only` evidence with a new local Experiment.
- Use local Git for candidate history. Imported peer history is available through remote-tracking refs such as `atrex/T2`; inspect it with `git log`/`git diff`, or continue it with `git switch -c <branch> atrex/T2`. Runtime moves the ref at its configured publication boundary without changing your HEAD. Commit or restore local work before switching.
- Continue bounded probe/edit/measurement/revert-or-keep cycles while the selected path has a
  concrete high-value step. A supported Experiment does not complete its Direction; compare the
  remaining strategy branches after every Experiment. Close a Direction only when the optimization
  program is genuinely completed or exhausted, strategically deferred, concretely blocked, or
  reasonably abandoned.
- Runtime returns the full Review once with the mutation result. `archive_status` reports storage,
  `claim_review_quality` reports `review_not_requested`, `review_unavailable`, or scientific support, and
  `evidence_eligible` controls whether the record may support a research conclusion, not visibility.
  `runtime_action` is workflow guidance; `scientific_next_check` is the evidence follow-up. Only projected
  `claim_review_quality`, `scope_safe_interpretation`, and `scientific_next_check` remain Agent-visible afterward; full receipts and reviewer
  conversations are Supervisor-private. Partial or `review_unavailable` claims are shared for context and their
  cited Records follow claim visibility, but they still cannot support research conclusions.

Use `submit-kernel` when you judge an exact commit ready to serve as a final implementation for the
current mode. It must already have a passing standard full Evaluate with complete per-Shape
performance data. Submit when ready, not mechanically after every Evaluate. Fast permits declared reproducible compute libraries;
Performance requires self-authored core compute with Agent-selected DSL; Ultra additionally enforces
its configured DSL.

- [Research mutations and report schemas](references/journal.md)
- [Saved results and source](references/records.md)
- New measurements: `skills/gpu-measurement/SKILL.md`
- Knowledge queries, when installed: `skills/KernelWiki/SKILL.md`
