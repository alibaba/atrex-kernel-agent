# Runtime research records and Kernel submission

Light keeps one trajectory-persistent Agent Git repository. There is no durable Episode Journal.
Direction/Experiment revisions live in the Campaign Research Store and versioned Research Library;
every `submit-kernel` decision is an immutable Campaign-level `submissions/` record. Ending a
Session updates the lineage `state.json` audit boundary only; there is no handoff report or
terminal promotion transaction.

Research validation reads the central Research Store directly through an in-memory snapshot.
Current Light creates no temporary or persistent `research-transaction.json`; Campaign and
Episode state exists only in the Supervisor-private Campaign directory.

## Agent operations

| Command | Purpose |
| --- | --- |
| `update-direction` | Propose or update a durable optimization program |
| `prioritize-direction` | Reorder the local Direction queue |
| `record-experiment` | Record one concrete test and its evidence |
| `submit-kernel` | Review one exact Agent Git commit as a formal submission |

Direction and Experiment modules can be enabled independently. Measurements are always recorded
under `records/`. Runtime projects enabled research state into `PROGRESS.md`,
`research-library/directions/<direction_id>/vN/`,
`research-library/experiments/<experiment_id>/vN/`, `records/`, and `frontier.json`.

## Exact-commit submission

Commit the complete candidate and first run a standard full Evaluate for that exact clean commit.
Then write:

```json
{"kernel_commit_id":"0123456789abcdef0123456789abcdef01234567"}
```

and run:

```bash
python3 tools/sandbox.py --kind submit-kernel --request-file scratch/kernel-submit.json
```

Acceptance requires all of the following:

1. the commit is reachable from local Agent Git or an imported peer ref;
2. the commit contains one complete evaluator-compatible source artifact;
3. MeasurementStore has a standard full Evaluate for that exact commit;
4. correctness passed;
5. per-Shape performance is complete and parseable;
6. final source/DSL/dependency policy passes when applicable.

Submission launches no Evaluate or ABBA, changes no Git ref, performs no rollback or promotion,
and does not end the Session. Rejected submissions return actionable diagnostics and remain in
`submissions/` alongside accepted decisions.

## Authority and persistence

Agent Git commit IDs are the only public Kernel identities. MeasurementStore archives canonical
source artifacts and Gateway Records keyed by those IDs. The controller workspace remains the
private task/V0 authority, but Runtime does not recommit Agent source or maintain an
Agent-to-Supervisor Git map.

Accepted submissions drive the lineage winner, global winner for multi-trajectory search, Pareto
projection, and peer sharing. Unsubmitted measurements remain available as research evidence but
do not become winners.

For exact Direction and Experiment schemas, see the packaged
[`runtime-records` operation reference](../src/aka/light/workspace/resources/skills/runtime-records/references/journal.md).
