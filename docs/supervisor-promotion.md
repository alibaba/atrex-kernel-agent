# Exact-commit submission and winner selection

Light no longer has a Supervisor-owned candidate Git history, Episode handoff, automatic terminal
ABBA, or promotion commit. The formal output of a trajectory is an accepted `submit-kernel`
receipt for an exact Agent Git commit.

## Source and identity boundary

- Agent Git is the only public source history and Kernel identity namespace.
- Runtime may initialize the common V0 commit and import peer history under managed refs.
- Runtime never commits, rewrites, rolls back, seals, or promotes Agent work.
- MeasurementStore archives the canonical single-file, source-tree, or manifest artifact measured
  for each Agent commit together with immutable Gateway Records.
- The controller repository remains private task/evaluator/V0 state; it is not a candidate mirror.

`supervisor-git.json`, private candidate refs, and Agent-to-Supervisor commit mappings do not exist
in the current workflow.

## Submission lifecycle

1. The Agent develops and checkpoints candidates in its own Git history.
2. Runtime measurements bind exact source artifacts to reachable Agent commit IDs.
3. The Agent runs a standard full Evaluate for a candidate commit.
4. The Agent calls `submit-kernel` with that full commit ID.
5. Runtime reuses the saved Evaluate, checks correctness and complete per-Shape performance, and
   reviews the final source contract and applicable Production policy.
6. Runtime records the accepted or rejected receipt and refreshes `PROGRESS.md` and the frontier.

There is no performance threshold at submission time. Winner selection considers only accepted
submissions and reads their performance from MeasurementStore rather than trusting Agent prose.

## Multi-trajectory search

Each trajectory keeps its best accepted submission. When more than one trajectory is visible,
Runtime also publishes a global best accepted submission. Peer publication imports reachable Git
history and matching measurement/research records; the receiving Agent can inspect an imported
commit directly with ordinary Git. One-trajectory runs omit the redundant global-winner section.

Episode is an audit/session segment, not a promotion unit. A normal Session stop needs no report.
The next segment resumes the same provider Session and Agent Git history when supported.
