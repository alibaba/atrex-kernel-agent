# Episode supervisor internals

This package implements the native optimization engine used by
`orchestrator/optimize.py`. Campaign execution uses that entry point; the audit repair command
below is an offline, operator-only maintenance tool.

Each canonical optimization version is explored in an isolated Git branch and worktree. A coding
agent may advance up to three Directions one at a time, run multiple related
profile/research/edit/validate cycles, preserve scratch source copies, and finally submit one report: `candidate_ready`, `pivot`, or `blocked`.

Agent reports use `python3 tools/sandbox.py --kind episode-report --request-file scratch/report.json`,
which calls the Supervisor's HTTP Journal service. A candidate selects `selected_experiment_id`, not
a positional index. The service verifies the cited Gateway measurement against current `kernel.py`;
field-validation failures leave the Journal and handoff unchanged so the Agent can correct and resubmit.

Terminal rechecks, including interrupted-Episode recovery, use the same evidence validation as
submission: the selected Experiment must reference visible Kernel-bound Gateway records, including a
completed, passing Evaluate for the exact candidate source. Direction events are replayed across visible
Episodes with the same transition rules as updates, including one in-progress Direction and at most
three distinct starts per Episode. Historical closed Directions may be reassessed or restarted. Invalid
persisted histories are operator-facing Runtime state errors, not instructions for the Agent to rewrite
private files. These checks do not submit GPU jobs or replace the policy and ABBA gates below.

The Supervisor validates the Journal and selected measurement, commits the matching current Kernel,
and checks production policy. Verification requests the Runtime's shared same-allocation ABBA
service: reuse a completed matching record, otherwise measure and persist it. A strict correctness-passing
improvement is squash-promoted to the incumbent; every other outcome records canonical
`memory/vN.json` evidence without changing the incumbent kernel.

An episode candidate commit contains only `kernel.py`. Temporary requests and optional diagnostics
stay in ignored `scratch/`. Plans and analysis are Journal fields; Gateway results are stored by the
Supervisor, so no separate plan or profile-analysis documents are required. The private Runtime Journal and handoff are copied into the
Episode archive before the isolated worktree is removed.

Each newly materialized Episode starts with empty `scratch/`; resuming an already-started Episode
preserves its files. Completed Episode scratch evidence is archived but is not copied into the next
Episode. Checkout-provided scratch symlinks are unlinked without modifying their targets.

Every post-baseline Episode uses this workflow; there is no fixed five-Trial window or separate
early-Episode comparison path. On resuming an older unfinished Fast Episode, the Supervisor keeps
its worktree, Journal, and measurements but uses the ordinary Prompt and ABBA promotion gate.
Completed historical records are not rewritten. Remove retired `--fast-episodes`, `--fast-trials`,
and `--fast-episode-ask-*` options from old launch commands.

The Supervisor stores Git exclusions in the shared `info/exclude`, not a workspace `.gitignore`.
V0 measurements live in `memory/v0.json`, without a duplicate baseline Markdown report. The accepted
framework baseline is pinned in the private Campaign root's `framework_baseline.json`; the commit
and exact Kernel blob are validated on resume, including from linked Episode worktrees.

Promotion audit snapshots live in the private Campaign root under
`promotions/long_horizon_eNNNN.json`, not in Agent `memory/`. Each audit is durably written before
the promotion Commit, whose message binds the audit's SHA-256. Recovery verifies the digest and
Episode identity, including after a crash between commit and state update. Prepared but uncommitted
audits never count as promoted. Legacy snapshots are read from committed Git objects and copied
privately without rewriting history. Agents receive a read-only view of canonical `memory/vN.json`
only; Supervisor audit files and live progress mirrors are omitted.

### Repair an interrupted promotion

If Git already contains the promotion but its private audit is missing or invalid, resume records
`promotion_audit.status = "audit_unverifiable"` in `active_episode.json` and stops with exit code 2
and repair instructions. It keeps the committed Kernel, Episode worktree, phase, and counters;
it neither rejects the Episode nor silently trusts an unverifiable Gate.

Stop the affected Campaign first. From the repository root, use the exact promotion Commit printed
in the error (the current HEAD, not the candidate branch Commit):

```bash
python3 -m long_horizon.audit_recovery \
  --workspace /path/to/campaign --promotion-commit FULL_HEAD_COMMIT \
  restore --file /path/to/original-audit.json
```

Restore requires the original bytes: equivalent JSON with different formatting does not match the
Commit's SHA-256. Both the digest and Episode identity must match before replacing the private file.
Git history and the canonical report are unchanged.

If the audit is genuinely missing and there is no backup, an operator may explicitly retain the
already committed promotion without claiming its original Gate evidence is verified:

```bash
python3 -m long_horizon.audit_recovery \
  --workspace /path/to/campaign --promotion-commit FULL_HEAD_COMMIT \
  acknowledge-missing --reason 'Backup unavailable; reviewed the committed Kernel and report'
```

This writes a private `promotions/long_horizon_eNNNN.recovery.json` receipt bound to the exact
promotion Commit and Episode, including reason and timestamp. It does not recreate the missing
audit. Corrupt audits, digest/identity mismatches, and symlink paths cannot use this acknowledgement.
The committed Kernel and canonical report must still be intact. Run repair with the same Supervisor
environment/private-root configuration as the Campaign; no Agent, Gateway or model is started.

After either repair, rerun the original Campaign command. Recovery records the existing promotion
exactly once without another GPU evaluation or Git commit. When loss was acknowledged, the Attempt
archive and state retain `promotion_audit.status = "audit_unverifiable"` and
`resolution = "operator_acknowledged_missing"`; `accepted` counts the retained Git promotion, not a
newly verified Gate. A restored valid audit records `status = "verified"`. A receipt cannot authorize
a different promotion Commit.

Runtime state lives under `.atrex_long_horizon/` in generated campaign workspaces. Public options
such as `--handoff-resumes`, `--verify-repeats`, `--verify-run-timeout`, and
`--min-improvement-pct` are parsed directly by `orchestrator/optimize.py`.

Each active Episode also has ignored `memory/live.json`. It is a best-effort projection refreshed
after each Supervisor Journal mutation, but it never participates in version selection or promotion;
`memory/vN.json` remains the canonical supervisor-owned record.

Every evaluator or profiler result is paired with an exact `kernel.py` snapshot in the
Supervisor's private per-worktree evidence store. The Agent sees only the compact correctness, latency,
performance, diagnostic, and record-identity projection needed for its next decision. Gateway
transport failures and terminal infrastructure failures are retried with fresh jobs; candidate
compile or correctness failures are never hidden by that retry policy.

Evaluate, Profile, and Wiki responses use operation-specific bounded projections. Dev keeps the
Agent's own probe stdout rather than interpreting it as evaluator truth, but applies a smaller
context limit and reports when bytes were omitted. Evaluate, ABBA, Profile, Dev, Check, and
Disassemble Gateway records can be read by ID through the facade without another external job;
opaque per-Shape facts remain visible and ABBA identifies both compared Kernels. A stable opaque
Kernel ID supports separate Gateway-record and source views; source is digest-verified privately
before restoration beneath `scratch/`. Exact
pre-projection stdout and stderr remain in the private per-worktree Runtime evidence copied into the
Episode archive.

`orchestrator/optimize.py` also owns one loopback HTTP Runtime for the lifetime of each Campaign.
Each coding-agent invocation receives a short-lived bearer capability bound to that invocation's
exact worktree. The Agent-visible `tools/sandbox.py` is a pure HTTP client, and it and the GPU Wiki
query commands transparently forward through
this Runtime; Agate, private-evaluator, and Wiki-store credentials stay in the supervisor process.
The Runtime replaces Agent-supplied workspace, hardware, timeout, and endpoint values with the
Campaign policy before executing a request, and appends a compact request audit to that private
evidence store. The same capability exposes `update/load/list` operations for Directions and
Experiments plus a repairable terminal-report operation. Agent requests carry hypotheses and opaque
Kernel IDs; the Runtime resolves exact Kernel and Gateway evidence before durably appending it.

Git-backed coding sessions require Linux Bubblewrap: `auto` selects it when available, and
`bwrap` requires it explicitly. There is no unsandboxed fallback for these workspaces; `none` is only
for trusted, non-Git tool tests. Both HOME and CWD point to the persistent, Git-free
Agent view at `/home/agent/workspace` (`~`). Selected Skills,
public task inputs, and credential files are mounted read-only. `tools/`, including `sandbox.py`,
is a writable Episode-local seed copy; same-Episode recovery preserves its edits and deletions.
Shared tool seeds and the Supervisor implementation stay unchanged. Network is shared so
the selected model provider remains reachable, while direct Agate
credentials are absent and host-side Agate invocations remain process-policy violations.
Authoritative Gateway records, evaluation ledgers, and Runtime request audits have no path in the
Agent mount namespace. `.git`, `.orchestrator_mode.json`, and `gpu-wiki/` are absent from the
Agent file view. Gateway reads the live view; report submission and session exit publish candidate
files and diagnostics back to the Supervisor worktree, never control metadata. Mode identity is
stored privately in `optimization-policy.json`. Git operations, candidate commits, and
promotion are Supervisor-only. Agent reports name a selected Experiment, not a commit.
CLI directories such as `~/.claude` and `~/.codex` are writable, session-scoped mounts backed by
private `provider-sessions/<session-key>/` storage. They support same-session recovery without
exposing another Episode's Home. The Supervisor additionally records each physical invocation under
`<campaign-parent>/.atrex-supervisor-runtime/<campaign-key>/workspaces/<scope>/sessions/run-<uuid>/`:

```text
conversation.jsonl          # full initial prompt, messages, tool results, child conversations
token-usage.json            # totals, input/output/cache buckets, per-response and per-Agent usage
provider/
├── stdout.stream-json      # CLI output, without thinking-token estimate noise
├── stderr.log              # complete CLI diagnostics
└── native/                 # native transcript deltas, including subagents
```

Stdout/stderr and the conversation are written while the process runs. Native files and usage are
refreshed every second; on exit the reading view is compacted to remove duplicate Claude
stdout/native content while retaining the original provider files. Retries get a new `run-<uuid>`;
existing native history is excluded from that invocation's usage. These Supervisor-owned copies
survive Episode worktree removal and are not mounted into the Agent view. They do not need a custom
header in native child files and have no fixed small file-count limit.

Native usage capture does not depend on bwrap. With `--agent-sandbox=none` (or `auto`
without bwrap), capture uses the CLI's actual configuration directory, including
`CLAUDE_CONFIG_DIR` and `CODEX_HOME`. It captures only the requested session and its
children, not unrelated sessions or credentials in the same Home. Pre-invocation
native history is excluded on resume. Codex's stream-only partial capture does not
block an available ledger reconciliation in either session entry point; an exact
capture including child usage is never replaced by a root-only total. Sandbox mode
alone neither downgrades complete usage nor upgrades incomplete usage to `exact`.

Capture file failures do not terminate the CLI: each stdout/stderr reader keeps draining to EOF
and retains the original stream in memory for the adapter. A failed raw-output or live-conversation
sink is disabled independently; healthy sinks keep recording. Capture errors are reported in
`token-usage.json`, and usage completeness is not claimed after a capture error. On exit the
conversation can still be reconstructed from retained data if storage is writable again. If final
persistence also fails, the process guard reports the capture failure without replacing the CLI's
actual output, exit code, or timeout status. Missing raw evidence is never presented as complete.

Usage comes from Provider counters, never a local token estimate. Claude uses the last counters per
message ID, includes child responses, and reconciles them against the terminal result without adding
the two bills. Codex uses per-rollout cumulative deltas, with cached input separated from uncached
input. Missing/unreconciled information is `unavailable`/`partial`, not zero; inspect `warnings` and
`capture_errors`. Qoder's reported credits are stored separately from tokens. Provider-managed
system prompts and calls not exported by a CLI cannot be reconstructed. A forced Supervisor kill
retains already-written records but can leave the final state marked `running`.

For native Atrex-Bench tasks, the Runtime also materializes exactly one private evaluator copy per
Campaign process and reuses it for Bootstrap, Agent-requested evaluation, and final verification;
`atrex-bench/` is never created in an Agent worktree. The complete gateway engine lives under the
Supervisor-private `supervisor/` package, which is not mounted into the Agent namespace.

Every canonical record carries a compact copy of all structured Experiments already persisted in
the private Runtime Journal. If the supervisor is terminated while an Episode is active, the next startup
resumes the registered Episode worktree in place, including source edits, Journal, and scratch files. If that worktree is missing or no longer matches
the recorded branch and baseline, recovery falls back to archiving it and recording an
`interrupted` `memory/vN.json`. Recovery remains idempotent across repeated termination.

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
- `../supervisor/journal.py`: private Direction/Experiment persistence, factual binding, queries,
  and repairable terminal-report validation.
- `journal.py`, `report.py`, and `protocol.py`: legacy Journal compatibility plus common terminal
  validation and handoff parsing.
- `verifier.py`: request/reuse authoritative Runtime measurements and apply the promotion threshold.
- `../supervisor/gateway.py` and `remote_abba.py`: shared measurement, deduplication, retry,
  per-Shape aggregation, and same-allocation ABBA execution for Agent and Supervisor requests.
- `store.py` and `telemetry.py`: restart state, archived attempts, and best-effort episode metrics.
- `../orchestrator/supervisor_runtime.py` and `../orchestrator/agent_sandbox.py`: scoped HTTP
  capabilities, privileged Agate/Wiki execution, and Bubblewrap projection.

`.atrex_long_horizon/state.json` and `active_episode.json` are restart state, while each
`episodes/eNNNN/` directory archives the prompt, journal-derived attempt, worktree snapshot,
verification payload, and telemetry available for that episode. These files are intentionally
excluded from campaign commits. Full accepted evidence is stored in the private Campaign root's
`promotions/long_horizon_eNNNN.json`; only the canonical `memory/v<N>.json` is committed with the Kernel.

At normal completion, Python failure, `SIGINT`, `SIGTERM`, or `SIGHUP`, the orchestrator writes an ignored
`trace-retention-manifest.json` in the workspace, covering candidate sources,
canonical memory, episode journals and evaluations, and compact profiler reports.
A separate manifest with the same name in
`<campaign-parent>/.atrex-supervisor-runtime/<campaign-key>/` declares
`framework_baseline.json`, `promotions/long_horizon_eNNNN.json`, `wiki-profile/run.json`, and
`wiki-profile/raw/query_events/<date>/*.json` in place.
Each manifest's paths are relative to its own directory; the completion hook must collect both
roots. Private Wiki events are never copied into the Agent workspace. The manifests explicitly
does not declare coding-agent session JSONL, stdout/stderr logs, temporary
worktrees, caches, or bulk profiler captures. Each manifest also binds the
consumer-supplied `platform`, resolved `arch`, and `sandbox_hardware` to the run;
these values are deployment evidence that cannot be reconstructed reliably from
the workspace name or from a locally scoped `solution.json`. A deployment hook
may use this manifest as producer evidence, but remains responsible for path
validation, secret scanning, archive limits, transport, and retry.
`SIGKILL` cannot be handled by Python; a completion hook must treat a missing
manifest after forcible termination as an interrupted/incomplete run.
