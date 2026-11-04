# Supervisor-owned Episode handoff and promotion

Light gives each optimization Episode a persistent Git-free draft. The Supervisor owns
the real worktree, source seals, private Journal, and promotion audit. The agent submits
an `episode-report` with evidence IDs, never a Git commit as proof of measurement.

## Boundary and retained workflow

The default report-only workflow cites `gateway_record_ids`. Optional research recording
uses a selected Experiment and closed Directions. Both paths require exact-source evidence
and the same acceptance/promotion gates. V0 and the optional framework baseline are separate
initialization stages.

Native mode is a cooperative separation of directories. Select `--profile light-sandbox`
on Linux to enforce the draft/private-worktree boundary through Bubblewrap.
A managed Session requires valid private Git provenance before it can receive a capability.

## Lifecycle

1. CandidateWorkspace materializes or resumes the draft.
2. The agent edits the permitted source layout and requests measurements through the Runtime.
3. The Journal validates the report and passing full Evaluate evidence for that exact source.
4. Acceptance obtains or reuses mandatory correctness and enabled policy/numerical evidence.
5. The Workspace seals the validated snapshot using private Git operations.
6. The report is persisted before controller projections and handoff publication.
7. The Workflow obtains policy-matched native ABBA and promotes only a strict improvement.

Invalid reports remain repairable. Identical accepted reports can be replayed after a
publication interruption; conflicting terminal reports cannot replace them.
Post-report draft edits do not change the sealed candidate.

| Mechanism | Current owner |
| --- | --- |
| Draft construction and publication | `aka.task.candidate_workspace` |
| Report transaction | `aka.workflow.light.host.journal` |
| Source snapshot and private Git sealing | `aka.task.source.formats`, `aka.task.candidate_workspace` |
| Recorded ABBA and scoring | `aka.workflow.light.long_horizon.recorded_verifier`, `verifier` |
| Promotion audit and repair | `aka.workflow.light.long_horizon.promotion_audit`, `audit_recovery` |

## Agent-visible workspace

The source layout is task-selected: `source_tree` has a fixed adapter and editable roots;
`cuda_manifest` has CUDA sources and a manifest; `single_file` has one Python candidate.
The agent also sees public instructions, reserved HTTP clients, custom writable
`skills/` and `tools/`, `scratch/`, eligible historical conversations, and optional
read-only memory summaries.

There is no real Git directory, private Journal, promotion audit, or evaluator checkout
in the managed draft. New Episodes start with empty scratch; same-Episode recovery keeps it.
Custom Skills/tools can carry across Episodes independently of kernel promotion.

Snapshots and commits bind **all selected candidate files**, including source deletions,
headers, and build settings. The seal is constructed from validated snapshot bytes,
not a later mutable draft read. Runtime observations and supplemental probe inputs are
excluded from candidate identity. See [candidate layouts](native-cuda.md).

## Acceptance and reuse

| Phase | Authoritative evidence |
| --- | --- |
| Candidate report | Passing standard full Evaluate for exact source; direct record citations or an enabled selected Experiment |
| Correctness | Native six-case contract, or the supported evaluator's full-workload gate |
| Production policy | Applicable framework/source review over exact candidate files |
| Promotion | Same-allocation native ABBA for committed incumbent and sealed candidate |

Light promotes on geometric-mean latency reduction, strictly above the configured
threshold. `performance_score` is diagnostic. A standard Evaluate, profile, partial-shape
test, or standalone timing cannot replace the ABBA gate.

Trusted acceptance can reuse a completed record only when the entire required task identity
matches. Agent duplicates return the existing record ID instead of silently launching another
sample. Unknown, unavailable, cancelled, or inconsistent results are not completed verdicts.

Each Episode prompt provides its effective ABBA command and incumbent retrieval command.
For tree/native-manifest sources, `kernel-read` restores a directory; single-file sources
restore a file. Use the selected command and baseline path so source and policy identities match.

SOL full-workload correctness is supported with explicit `single_file`, but Light's
native ABBA promotion path requires Agate Atrex-Bench. Use
[Legacy verification](legacy-workflow.md) for SOL/SSH campaigns needing that path.

## Promotion evidence and recovery

Promotion audits live under the private Supervisor scope in
`promotions/long_horizon_eNNNN.json`. The Git promotion carries an
`AKA-Promotion-Audit: sha256:...` trailer binding the audit bytes. Source and authoritative
version facts are committed together; generated `memory/vN.json` is optional.

Recovery checks the committed parent, Episode/version/branch, source, and audit digest.
Missing or corrupt evidence pauses recovery rather than resetting promotion state.
Stop the campaign before applying the exact repair command printed by the error:

```bash
python3 -m aka.workflow.light.long_horizon.audit_recovery \
  --workspace WORKSPACE --promotion-commit FULL_HEAD \
  restore --file /path/to/original-audit.json
```

If the audit is missing and no backup exists, the operator can explicitly record an
unverifiable audit rather than fabricate a successful gate:

```bash
python3 -m aka.workflow.light.long_horizon.audit_recovery \
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
