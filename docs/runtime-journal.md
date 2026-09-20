# Runtime Journal and Episode Reports

Long-running optimization needs both immutable measurement facts and the Agent's evolving interpretation. Gateway Records already bind exact Kernel source to results; this interface adds durable research Directions, structured Experiments and terminal reports that reference those records instead of repeating measurement values in prose.

The users are optimization Agents recording and retrieving evidence, and operators inspecting or recovering interrupted Episodes. Successful writes are persisted before acknowledgement. Validation errors identify the bad fields or lifecycle state so the Agent can correct a report within the same Session. This reduces reliance on an end-of-Episode reconstruction; it does not make Agent-authored analysis scientifically authoritative.

## Scope and compatibility

The controller registers each Long Horizon Episode with its trusted number, workspace, base Commit and branch. Only its optimizer Sessions receive Journal access. Setup, Framework Baseline and auxiliary reviewers retain their existing interfaces. Agents cannot register an Episode, choose a Journal path, or select another Campaign in an HTTP request.

Long Horizon uses the Campaign's public registration method to start Runtime and bind the Episode. Fast Episodes resume with the trial count saved in controller state. Re-registering an existing workspace binding must preserve `minimum_experiments`; a mismatch raises an explicit controller error with the bound/requested values, leaving the existing binding unchanged.

The existing local Journal remains supported and is still the default in Episode prompts. The mounted `runtime-records` Skill offers the new path. Choose one interface before recording Experiments; a Runtime write is rejected if the local Journal already contains experiments or has been finalized. Legacy Journals are not automatically migrated to Direction/Experiment IDs.

Protocol mixing returns HTTP 400 with `journal_protocol_conflict` and `repairable=false`: changing request arguments cannot switch this Episode to Runtime Journal. The Agent must continue the legacy append/finalize and handoff commands from its Episode prompt, preserving existing evidence. An already-finalized Journal only needs its remaining handoff steps; do not erase history to bypass the rejection.

Keep Setup, Fast/Full, plan/profile artifacts, Wiki attribution, Phase Markers, Git operations, verification and promotion unchanged. The Agent still creates its candidate Commit. Runtime report validation reads that Commit but never commits, resets or promotes Git state. Fast Episodes retain their required experiment/evaluation counts and permitted-operation rules. The new interface's explicit behavior changes are evidence-linked Direction lifecycle checks and permission to submit an empty pivot/blocked report when no exploration was started.

## Lifecycle and failure paths

```mermaid
flowchart TD
    C["Long Horizon controller<br/>register_episode"] --> S["Session capability<br/>workspace + private Journal binding"]
    A["Agent: tools/sandbox.py"] --> S
    S --> L["execute_journal<br/>Session/Episode locks + private file lock"]
    L --> V["Validate fields, lifecycle and visible IDs"]
    V -->|Invalid input; no mutation| E["400 + code/message/next_action<br/>Correct and resubmit"]
    E --> A
    V --> M["MeasurementStore.read<br/>Validate cited facts and Kernel binding"]
    M --> J["Atomic private Journal write"]
    J -->|Direction/Experiment| ACK["ID acknowledgement"]
    J -->|Accepted Episode Report| P["Publish legacy Journal projection<br/>then handoff.json"]
    P --> G["Existing Long Horizon verification and promotion"]
    P -->|Publication failure| R["503; keep durable report<br/>Retry identical report after storage repair"]
    L -->|Private evidence unreadable| I["Infrastructure error<br/>Do not invent evidence or rerun GPU work"]
```

`orchestrator/supervisor_runtime.py` authenticates, binds and dispatches. `supervisor/journal.py` validates requests, resolves measurement records and saves the private Episode document. `direction_lifecycle.py` replays legal state transitions; `direction_genealogy.py` validates optional ancestry. `tools/sandbox.py` only parses arguments and sends HTTP. Per-Episode locks serialize multiple Sessions, and OS file locks reject concurrent writers from another controller without blocking all Campaign requests.

## Objects and API

| Object | Content | Ownership |
| --- | --- | --- |
| Direction | Hypothesis, rationale, plan, lifecycle, explicit assessment and ancestry | Agent proposes/interprets; Runtime validates and persists |
| Experiment | Change, Gateway Record IDs, evidence narrative, analysis and action | Agent-authored interpretation; referenced measurements validated by Runtime |
| Gateway Record | Exact request/Kernel binding and saved projected response | Existing Measurement Store |
| Episode Report | Terminal status, summary, selected Experiment, existing candidate Commit or blocker | Agent submits; Runtime validates; Supervisor independently decides acceptance |

All calls use `POST /v1/journal/execute` and the existing Session bearer. The Agent CLI is:

| Command | Input | Successful output |
| --- | --- | --- |
| `update-direction` | `--request-file FILE` | `status`, `direction_id` |
| `record-experiment` | `--request-file FILE` | `status`, `experiment_id` |
| `list-directions`, `list-experiments` | `--output-path scratch/FILE` | `status`, `file`, `count`; index written to file |
| `load-direction`, `load-experiment` | `--record-id ID` | Complete public record as JSON |
| `episode-report` | `--request-file FILE` | `status: accepted`, message |

The HTTP payload uses `operation` plus exactly one of `request`, `file`, `direction_id`, or `experiment_id`. Operations are `direction_update`, `experiment_record`, `directions_list`, `experiments_list`, `direction_load`, `experiment_load`, and `episode_report`. Successful HTTP responses use the existing `exit_code/stdout/stderr` envelope; the CLI prints the JSON in stdout. Malformed input returns a structured HTTP 400, private-state/publication failures a 503; post-write failures must not be reported as safe input retries. Transport loss can leave a write's outcome unknown.

Field-by-field examples are maintained in [the mounted Journal reference](../skills/runtime-records/references/journal.md); saved measurement readers are covered in [the record reference](../skills/runtime-records/references/records.md). Complete encoded request bodies are capped at 2 MiB. Journal request files are capped at 2,093,056 bytes (2 MiB minus a 4 KiB envelope reserve); the client also checks the final encoded body to account for JSON expansion. Oversized requests are rejected locally with instructions to shorten text or lists, before sending HTTP. Private Episode Journals and exported indexes are capped at 16 MiB. Long text and list fields have smaller per-field limits reported by validation.

### Directions and evidence

Propose any number of useful ideas, but start at most three distinct Directions in one Episode and only one at a time. Complete/abandon/block/defer requires a started Direction, explicit supporting Experiment IDs and `hypothesis_status=unresolved|supported|refuted`. Closed Directions can be restarted or reassessed; previous events remain unchanged. Lifecycle completion does not imply a proven hypothesis.

Each Experiment cites at least one visible Kernel-bound Evaluate/ABBA/Profile/Dev/Check/Disassemble record. Failed diagnostics may document an unresolved blocker; Env/Wiki and unbound Dev are not Kernel evidence. Supported/refuted closure requires a completed observation in every selected Experiment. Runtime checks records and bindings, not causal relevance. Late evidence can attach to a closed Direction without silently reopening it or replacing selected support. Optional Wiki attribution follows the existing diagnostic-only schema.

The Journal's private evidence reader derives full-evaluation eligibility from stored request options without carrying the raw request/options/inputs into its evidence view. Journal responses and compatibility reports are assembled separately; the eligibility flag is internal, not an Agent-facing field.

Queries fold the current Journal plus finalized earlier Runtime Journals in this Campaign, including non-winning explorations. Unfinished previous Episodes, future Episodes and unrelated Campaigns are excluded. Stable global Direction/Experiment IDs and immutable ancestry avoid reconstructing identity from Episode-local indexes. List outputs are snapshots, not automatically refreshed files.

### Reports and old consumers

No Direction may remain in progress at handoff. `candidate_ready` requires a current-Episode selected Experiment, a passing full Evaluate record for byte-identical `kernel.py`, and the Agent's existing full `candidate_commit`. The existing worktree validator checks HEAD, branch, source match and permitted diff. Custom-input, Shape-subset, correctness-only, Profile or ABBA alone are insufficient for that report binding; independent verification remains unchanged.

The private report is saved first. Runtime then publishes a schema-1 compatibility Journal for existing Long Horizon consumers and atomically publishes `handoff.json` last. It maps Experiment IDs to the old selected index and derives evaluation fields from Gateway Records. Canonical memory remains Supervisor-generated. Identical accepted reports are idempotent and can repair a failed publication; conflicting terminal reports and subsequent mutations are rejected. Report rejection before persistence leaves the Journal open and writes no handoff.

Ordinary report validation failures remain HTTP 400 with `repairable=true`. `error.message` identifies the invalid field, lifecycle state or evidence requirement, and `error.next_action` explains how to correct and resubmit. Field-shape errors also list missing, unexpected and allowed fields. Correct the report or the indicated prerequisite, then call `episode-report` again in the same Session; only an accepted report ends the Episode. This is distinct from a `journal_protocol_conflict`, which requires continuing the existing legacy workflow rather than changing report arguments.

The workspace's `runtime_journal_projection` flag is only a claim. Completion checks, including interrupted-handoff recovery, read the existing private Journal from the controller's Campaign scope and verify its schema, Episode/base Commit/branch identity, terminal state, finalization and candidate Commit. Empty pivot/blocked reports are accepted only after that verification and only when the private Journal also has no Experiments. Missing, corrupt, unfinalized or mismatched private state fails closed; validation never creates a Journal to authorize the exception. Legacy terminal validation remains strict by default.

## Storage, limits and rollback

Private documents live beside the existing Measurement Store at `<Supervisor scope>/journals/episodes/eNNNNNNNN/journal.json`; they are outside the Agent workspace and hidden in Bubblewrap mode. Native mode still has the operator UID's filesystem authority. The legacy Journal/handoff projection is visible compatibility state, not a replacement private fact store or a new security claim about native mode. Requests never accept private storage paths. Scratch exports use descriptor-based publication so symlinks cannot redirect writes outside the allowed workspace.

Before mutations, Runtime reads the legacy Journal without following symlinks. A symlink, invalid path component or non-regular file returns a repairable 400 with workspace-path repair instructions; no Journal mutation is applied. A missing legacy Journal remains valid. Storage-access failures, including I/O errors, remain non-repairable 503 infrastructure blockers rather than requests to rewrite evidence.

Atomic document replacement favors simplicity but rewrites one bounded Episode document per update. History queries read finalized Journals and validate referenced records; very long Campaigns may need indexing later. Interpretation still depends on Agent submissions, and damaged history fails closed rather than silently discarding evidence. There is no automatic retention cleanup or retry of ambiguous mutations.

Regression fixtures are kept outside the repository. They exercise a local fake GPU Gateway plus real HTTP/client, Git and filesystem paths: durable write/query, cross-Episode history, source/result binding, invalid-report repair, exact report replay, conflicting reports, privacy, scoped concurrent requests, legacy projection/validation and unchanged measurement APIs. These checks do not claim real-model optimization gains or production GPU qualification.

Local verification covered 29 Journal regressions, 102 existing measurement/Gateway regressions and 4 existing Wiki-attribution checks. In particular, the compatibility projection passes the existing Long Horizon completion check and memory compiler, Fast-mode minimums remain enforced, and candidate reports reject correctness-only or Shape-subset evidence. Private-request canaries are absent from the internal evidence view, Agent query responses and compatibility report; deriving eligibility preserves custom-input and Shape-subset exclusions. No new test modules are included in the source tree.

An additional 15 projection-trust regressions cover forged workspace flags, absent/corrupt/unfinalized private records, identity and terminal-field mismatches, omitted private Experiments, valid empty outcomes, strict legacy validation, unchanged Fast minimums and recovery before Episode registration. These fixtures also remain outside the repository.

Nine legacy-path regressions cover file/parent/dangling symlinks, invalid path components, directories/FIFOs, absent Journals, repair-and-retry, unchanged private state on rejection and infrastructure-error classification.

To roll back, stop the Campaign and preserve private Journals/Measurements, workspace Journals and Git state. Finish an active Runtime-Journal Episode before reverting if possible. Restore a previously validated revision/configuration; old consumers can read the schema-1 terminal projection, but cannot continue private Direction IDs or repair private report publication. Do not switch an in-progress Episode between protocols or discard its evidence to force a restart.
