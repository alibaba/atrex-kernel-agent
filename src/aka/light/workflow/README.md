# Light Workflow internals

This package is private to `aka.light.workflow.plugin`, not a separate startup command.
The checkout's `docs/optional-light-modules.md` describes composition, options and supported
migration paths. The upstream Workflow does not import these mechanics.

`campaign.py` coordinates Episode budgets and continuation.
`session.py` borrows the selected public Agent service; it does not construct another backend.
Workspace operations are delegated to the selected CandidateWorkspace implementation.
`journal.py` and `protocol.py` consume controller projections of the private Runtime Journal.
`recorded_verifier.py` requests or reuses evidence through the selected Tool Runtime.

The Agent receives one trajectory-persistent local Git repository. Ending a Session closes an
audit segment; it does not submit, seal or promote source. `submit-kernel` is the only formal
Kernel submission action. It accepts an exact Agent Git commit only when MeasurementStore already
contains a passing full Evaluate with complete per-Shape performance for that commit and the
final-source review passes. Accepted submissions drive lineage/global winners and sharing.

MeasurementStore keeps canonical source artifacts and immutable Gateway Records. The controller
workspace remains the private task/V0 authority, but it does not mirror Agent commits or maintain
an Agent-to-Supervisor Git mapping.

Controller-side implementation files, evidence and Git metadata are never installed as Agent
tools. Only the reviewed clients, prompts and Skill manifests are published to the draft.
