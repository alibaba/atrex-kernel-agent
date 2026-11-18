# Light Workflow internals

This package is private to `aka.workflow.light.plugin`, not a separate startup command.
The checkout's `docs/optional-light-modules.md` describes composition, options and supported
migration paths. The upstream Workflow does not import these mechanics.

`campaign.py` coordinates Episode budgets and continuation.
`session.py` borrows the selected public Agent service; it does not construct another backend.
Workspace setup is delegated to the selected CandidateWorkspace implementation. Light does not
create per-Episode Git worktrees: each trajectory reuses one persistent Agent repository.
The compatibility-named `journal.py` plugin configures validated research and submission
operations. It creates no Episode Journal: central ResearchStore, SubmissionStore and
MeasurementStore are the durable authorities. Direction and Experiment requests are validated
against an in-memory snapshot of the central store; Light neither writes nor migrates a
`research-transaction.json` compatibility file.
`../measurements.py` requests or reuses evidence through the public `aka.experiment.runner` and the Campaign Measurement Store.

The Agent receives one trajectory-persistent local Git repository. Ending a Session closes an
audit segment; it does not submit, seal or promote source. `submit-kernel` is the only formal
Kernel submission action. It accepts an exact Agent Git commit only when MeasurementStore already
contains a passing full Evaluate with complete per-Shape performance for that commit and the
final-source review passes. Accepted submissions drive lineage/global winners and sharing.
Each Submission has one stable Runtime-issued ID and immutable `vN` revisions, so repaired commits
can continue the same Reviewer Session without overwriting or revoking earlier decisions.

MeasurementStore keeps canonical source artifacts and immutable Gateway Records. Supervisor state
archives facts and source bytes, but it does not own a second candidate Git history, mirror Agent
commits or maintain an Agent-to-Supervisor Git mapping.

Supervisor-private evidence is never installed as an Agent tool. Only reviewed clients, prompts,
Skills and visibility-filtered Runtime projections enter the persistent workspace.
