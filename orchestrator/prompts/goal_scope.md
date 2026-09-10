## Episode mode: goal

Own the goal of substantially improving this episode's incumbent across the complete supported
workload. The campaign has completed at least 50 optimization episodes and stalled for more than
three consecutive episodes. This is an extended autonomous engineering episode: use the full
exploration scope to break the plateau and deliver the best validated implementation you can.

You are authorized to investigate multiple independent or interacting directions, replace algorithms,
redesign layouts and schedules, change fusion or dispatch strategies, and substantially refactor or
rewrite `kernel.py`. Plan a roadmap covering the whole optimization goal, with competing hypotheses,
experiments, checkpoints, and measurable acceptance criteria. Reorder, combine, abandon, and replace
directions as evidence develops. Make these engineering decisions autonomously within the campaign
contract. Plans and reviews use goal scope throughout this episode.

There is no per-episode wall-clock deadline or fixed experiment quota. The supervisor allows at
least 20 same-session continuations to recover an incomplete handoff on resumable backends. Use
successive research, profile, implementation, repair, and validation cycles while concrete,
evidence-backed work remains. A failed direction is an experiment outcome: pivot to another direction
inside this worktree. A first small improvement is a checkpoint: preserve the best candidate and
continue investigating remaining high-value opportunities. Reprofile after architectural changes
and validate interactions between combined optimizations. The global campaign budgets still govern
whether another episode can start.

Finish with `candidate_ready` when the roadmap's worthwhile directions have been evaluated and the
best combined candidate is fully validated, or when the campaign's explicit optimization target is
met. Restore the best source into the current branch without changing refs, validate that exact
source, and commit it as HEAD before finalizing. Use `pivot` only when the whole goal has no remaining
evidence-backed avenue and no promotable candidate. Use `blocked` for an actual infrastructure or
authority blocker. Summarize the explored directions, refactors, best candidate, and evidence for
stopping in the journal outcome. Never invent a gain or continue repetitive no-evidence sweeps.

This scope grants broad implementation freedom within `kernel.py`; the evaluator contract, protected
files, framework requirements, sandbox execution boundary, and independent promotion checks remain
binding.
