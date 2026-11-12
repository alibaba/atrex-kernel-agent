# Supervisor-owned Episode handoff and promotion

Light gives each trajectory one stable Agent-visible Git repository across optimization
Episodes. The Supervisor separately owns per-Episode trusted worktrees, source seals,
private Journals, and the promotion audit. Agent commits and branches are engineering
state, not measurement proof: the agent submits an `episode-report` with evidence IDs and an
explicit terminal/share/blocker reason. A clean Agent `HEAD` is the canonical Kernel ID; after
the evidence gates pass, the Supervisor imports that exact commit object under
`refs/atrex/accepted/<kernel_commit_id>` and verifies its source instead of recommitting it.

## Boundary and retained workflow

Direction and Experiment recording are enabled by default and independently selectable. A
candidate report selects an Experiment when that module is enabled, or cites
`gateway_record_ids` directly when it is disabled. Direction lifecycle requirements apply only
when that module is enabled. Every combination requires exact-source evidence and the same
acceptance/promotion gates. Optimization starts from measured V0 by default; the dedicated
framework-baseline stage is an explicit compatibility option.

Native mode is a cooperative separation of directories. Select `--profile light-sandbox`
on Linux to enforce the draft/private-worktree boundary through Bubblewrap.
A managed Session requires valid private Git provenance before it can receive a capability.

## Lifecycle

1. CandidateWorkspace opens or initializes the trajectory's stable Agent repository and
   refreshes Episode-scoped capabilities, public inputs, and shared evidence in place.
2. The agent edits the permitted source layout and requests measurements through the Runtime.
3. The Journal validates the report and passing full Evaluate evidence for that exact source.
4. Acceptance obtains or reuses mandatory correctness and enabled policy/numerical evidence.
5. The Workspace imports the exact Agent commit into a protected Supervisor ref, verifies that
   it contains the measured source, and materializes those bytes in the private worktree.
6. The report is persisted before controller projections and handoff publication.
7. The Workflow obtains policy-matched native ABBA and promotes only a strict improvement.

Invalid reports remain repairable. Identical accepted reports can be replayed after a
publication interruption; conflicting terminal reports cannot replace them.
Post-report draft edits do not change the sealed candidate.

| Mechanism | Current owner |
| --- | --- |
| Stable Agent repository and publication | `aka.light.workspace.episode` |
| Report transaction | `aka.light.workflow.journal_service` |
| Source snapshot and private Git sealing | `aka.light.workspace.sources`, `git` |
| Recorded ABBA and scoring | `aka.light.acceptance.recorded_verifier`, `verifier` |
| Promotion audit and repair | `aka.light.acceptance.promotion_audit`, `audit_recovery` |

## Agent-visible workspace

Native Light Campaigns normalize the editable Kernel boundary to `kernel.py + impl/`.
Both files and directories are part of one source tree, and `kernel.py` is an editable
evaluator entry point rather than a fixed adapter. Legacy `single_file` and manifest
providers remain import/execution compatibility mechanisms, not a second Agent-facing
version model. The agent also sees public instructions, reserved HTTP clients, custom writable
`skills/` and `tools/`, `scratch/`, eligible historical conversations, and optional
read-only memory summaries.

The managed trajectory is a real, persistent Agent-owned Git repository. Its clean `HEAD`
commits are engineering checkpoints and canonical Kernel IDs, while the Supervisor's private Journal, immutable
measurement artifacts, promotion audit and evaluator checkout remain outside that repository.
Custom Skills/tools can carry across Episodes independently of kernel promotion.

Snapshots and commits bind **all selected candidate files**, including source deletions,
headers, and build settings. Before importing the Agent commit, the Supervisor rechecks that
its source equals the frozen measured snapshot and that it tracks no non-Kernel workspace
files. Runtime observations and supplemental probe inputs are excluded from candidate
identity. The later incumbent promotion commit may additionally contain version/audit facts,
but it does not replace the accepted candidate's Kernel ID. See [candidate layouts](native-cuda.md).

## Acceptance and reuse

| Phase | Authoritative evidence |
| --- | --- |
| Candidate report | Passing standard full Evaluate for exact source; direct record citations or an enabled selected Experiment |
| Correctness | Native five-case contract, or the supported evaluator's full-workload gate |
| Production policy | Applicable framework/source review over exact candidate files |
| Promotion | Same-allocation native ABBA for committed incumbent and sealed candidate |

Light promotes on geometric-mean latency reduction, strictly above the configured
threshold. `performance_score` is diagnostic. A standard Evaluate, profile, partial-shape
test, or standalone timing cannot replace the ABBA gate.

Trusted acceptance can reuse a completed record only when the entire required task identity
matches. Agent duplicates return the existing record ID instead of silently launching another
sample. Unknown, unavailable, cancelled, or inconsistent results are not completed verdicts.

Each Episode prompt provides its effective ABBA command and incumbent commit. Runtime resolves the
reachable commit and privately exports the correct single-file or source-tree layout so source and
policy identities match.

SOL full-workload correctness is supported with explicit `single_file`, but Light's
native ABBA promotion path requires Agate Atrex-Bench. Explicit SSH supports that native
Atrex-Bench ABBA path for `source_tree` and CUDA-manifest candidates; SOL campaigns that
need a non-native comparison path still require [Legacy verification](legacy-workflow.md).

## Promotion evidence and recovery

Promotion audits live under the private Supervisor scope in
`promotions/long_horizon_eNNNN.json`. The Git promotion carries an
`AKA-Promotion-Audit: sha256:...` trailer binding the audit bytes. Source and authoritative
version facts are committed together; generated `memory/vN.json` is optional.

Recovery checks the committed parent, Episode/version/branch, source, and audit digest.
Missing or corrupt evidence pauses recovery rather than resetting promotion state.
Stop the campaign before applying the exact repair command printed by the error:

```bash
python3 -m aka.light.acceptance.audit_recovery \
  --workspace WORKSPACE --promotion-commit FULL_HEAD \
  restore --file /path/to/original-audit.json
```

If the audit is missing and no backup exists, the operator can explicitly record an
unverifiable audit rather than fabricate a successful gate:

```bash
python3 -m aka.light.acceptance.audit_recovery \
  --workspace WORKSPACE --promotion-commit FULL_HEAD \
  acknowledge-missing --reason 'Operator explanation'
```

Restore requires matching original bytes. Acknowledgement cannot waive corruption or
identity mismatch. Repair requires the active checkpoint's promotion binding and does
not change its counters or outcome.

## Verification, limits and rollback

See [Light verification](optional-light-modules.md#verification-scope) for source-sealing,
report replay, HTTP, Git, and recovery checks, and [live source-layout evidence](native-cuda.md)
for scoped GPU validation. Preserve private records when restoring a matching deployment.
Use a new workspace when changing implementation or source contract; do not rewrite locks.
