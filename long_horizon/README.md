# Episode supervisor internals

This package implements the native optimization engine used by
`orchestrator/optimize.py`. It is not a separate command-line entry point.

Each canonical optimization version is explored in an isolated Git branch and worktree. A coding
agent may run multiple related profile/research/edit/validate cycles, preserve private checkpoint
commits, and finally publish one structured handoff: `candidate_ready`, `pivot`, or `blocked`.

The supervisor validates the journal and candidate commit, checks production policy, and evaluates
incumbent and candidate in an exact same-allocation ABBA schedule. A strict correctness-passing
improvement is squash-promoted to the incumbent; every other outcome records canonical
`memory/vN.json` evidence without changing the incumbent kernel.

An episode candidate commit contains only `kernel.py`. Plans, profiles, planner discussions,
journals, and handoffs stay uncommitted and are copied into the episode archive before the isolated
worktree is removed.

Runtime state lives under `.atrex_long_horizon/` in generated campaign workspaces. Public options
such as `--handoff-resumes`, `--verify-repeats`, `--verify-run-timeout`, and
`--min-improvement-pct` are parsed directly by `orchestrator/optimize.py`.

Each active episode also exposes ignored `memory/live.json`. It is initialized immediately and
atomically refreshed after every journal append, but it never participates in version selection or
promotion; `memory/vN.json` remains the canonical supervisor-owned record.

Claude and Codex can resume the same session to repair an incomplete handoff; Qoder and Pi use a
single long invocation. Codex token deltas and marker ordering are read incrementally from the
resumable native rollout. Available invocation components must reconcile with cumulative rollout and
`turn.completed` totals before attribution. Reconciled events may form one phase interval across a
resume boundary. If ledger observation fails, consecutive cumulative stdout usage still supplies a
non-duplicated invocation total while phase attribution degrades fail-closed.

## Module responsibilities

- `campaign.py`: episode budgets, recovery, terminal-state processing, and promotion decisions.
- `git_episode.py`: private branch/worktree lifecycle, protected-path checks, squash promotion,
  and canonical outcome commits.
- `session.py`: one long coding-agent invocation plus bounded same-thread recovery for Claude and
  Codex.
- `journal.py` and `protocol.py`: atomic journal/handoff I/O and terminal validation.
- `verifier.py` and `remote_abba.py`: one-allocation incumbent/candidate ABBA execution.
- `store.py` and `telemetry.py`: restart state, archived attempts, and best-effort episode metrics.

`.atrex_long_horizon/state.json` and `active_episode.json` are restart state, while each
`episodes/eNNNN/` directory archives the prompt, journal-derived attempt, worktree snapshot,
verification payload, and telemetry available for that episode. These files are intentionally
excluded from campaign commits. Accepted evidence is also written to committed
`memory/long_horizon_eNNNN.json` and the canonical `memory/v<N>.json`.
