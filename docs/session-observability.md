# Session observability

Coding sessions now save an invocation-scoped conversation and provider usage report while the CLI runs. This is an observation-only addition to the existing Agent runtime and Long Horizon runner: it does not introduce an HTTP service, require Bubblewrap, change Agent prompts, or move Git, evaluation, Journal, or promotion responsibilities.

## Where records live

Long Horizon writes into its existing Episode archive, so records survive removal of the Episode worktree:

```text
<campaign>/.atrex_long_horizon/episodes/eNNNN/sessions/
  run-<uuid>/
    conversation.jsonl
    token-usage.json
    provider/
      stdout.stream-json
      stderr.log
      native/               # selected provider transcripts, if available
```

Other invocations through the Agent runtime, including framework-baseline and policy-review sessions, default to `<workspace-parent>/.atrex-session-traces/<workspace-name>/run-<uuid>/`. The default is outside the candidate Git workspace and does not add trace files to candidate commits.

Set `ATREX_SESSION_CAPTURE_DIR` to override that root for standalone invocations. Relative paths resolve against the invocation workspace. Long Horizon selects its Episode-specific archive directory. Every invocation, including same-session repair/resume, gets a new `run-<uuid>`; the provider session ID can remain the same.

## Conversation

`conversation.jsonl` starts with session metadata and the exact prompt passed to the CLI. During execution it appends provider events and newly discovered native messages. At exit it is atomically rebuilt into a de-duplicated reading view, retaining provider-reported reasoning, messages, tool requests/results, subagent content, and terminal status.

The provider directory retains the corresponding stream and native records. Claude's `system/thinking_tokens` progress notifications are omitted from both persisted views; actual reasoning text is retained. Native subagent JSONL does not need an AKA session header. A resume copies only records added during that invocation, not the previously captured conversation again.

This records what the provider exports. It cannot recover unexported internal prompts, hidden reasoning, or calls that the CLI never reports.

## Usage

`token-usage.json` is refreshed during execution and finalized at exit. It includes:

- Invocation/session identity, timestamps, exit status, timeout status, and available Campaign/Episode/invocation labels.
- Per-response provider counters and their source/agent identity, plus per-agent and total counters.
- Disjoint uncached input, cache-read, cache-write, and output token buckets.
- Separate Qoder credits when the provider emits them; credits are never converted to tokens.
- Reconciliation basis, missing usage, capture warnings, and structured `capture_complete`.

The measurement states are:

| State | Meaning |
| --- | --- |
| `exact` | Available response/native counters reconcile with the provider terminal counters. |
| `partial` | Some counters exist, but coverage or reconciliation is incomplete. |
| `unavailable` | No usable token total was exported; this is not zero usage. |

Repeated stream/native copies of the same response are counted once. Claude cumulative task-progress notifications are not new responses. Codex cumulative rollout counters are reduced to invocation deltas; cached input is not added twice. Root terminal totals are not blindly added to child usage because the terminal may already include children.

Existing Phase Marker ordering is retained. Native-only child usage without a reliable position in the root phase sequence contributes to the total, not to an invented phase attribution. Detailed per-response counters remain available in the usage report even when phase attribution is incomplete.

If Codex ledger reconciliation fails while Capture provides exact usage, that usage is preserved. The invocation observation still reports `codex_ledger_unavailable:<ExceptionType>`, the failed ledger cursor is invalidated, and the invocation does not qualify for usage-verified resume. A successful Capture does not hide ledger failures.

The supported native sources are Claude, Codex, Qoder, and Pi. Existing provider-home configuration is respected; this change does not remount credentials or replace HOME. Qoder no longer receives `--no-session-persistence`, so its native session files can be captured. Optional external reviewer helpers or arbitrary model processes bypassing the shared Agent runtime are not automatically covered unless their calls appear in the selected provider transcripts.

## Failure and privacy boundaries

Capture uses incremental reads, bounded transcript discovery, and no-follow regular-file reads. Default limits are 64 MiB per file, 128 MiB retained per invocation, 2 MiB per line, 4,096 discovered files, and 200,000 retained records. Exceeding a limit records partial coverage; it does not terminate the Agent or reject its result.

Native per-file limits count bytes captured during this invocation, not the absolute offset of a resumed transcript. Pre-invocation history is scanned once under a separate budget with the same limits, shared across history files. History bytes and records never consume the live capture allowance. If that scan is capped, capture still starts at the original snapshot boundary; history is not replayed as new content. Usage reconciliation is marked incomplete, and missing Codex baseline counters are not treated as zero.

These limits apply to diagnostic capture, not to the stdout/stderr returned to the existing Runtime parsers. The returned streams remain complete, including oversized lines, late phase receipts, terminal results, and Codex thread IDs—even when diagnostic limits are exhausted. As with the previous `Popen.communicate()` path, functional output remains buffered in memory without a capture-imposed cap. The final conversation is rebuilt only from the bounded diagnostic copy; finalization does not restore omitted output into the archive.

If a persistence sink fails, the pipe reader keeps draining the child's output to EOF. Setup/finalization failures are logged without replacing the CLI outcome. Existing process timeout, dependency guards, termination, and resume policy remain in place. A forcibly killed observer cannot write a final status; surviving files may still say `running`.

Each capture directory is created with mode `0700`. Only the current session and identified descendants are copied, not unrelated transcripts or credential files. Prompt, reasoning, and tool output may nevertheless contain sensitive content: these are local diagnostic files, not sanitized public artifacts. This PR adds no security isolation against the Agent running as the same OS user. The existing trace-retention manifest and upload policy are unchanged.

## Verification

The CLI import/argument smoke check requires no GPU or model credentials:

```bash
python3 orchestrator/optimize.py --help
```

Regression tests are maintained outside this repository. Local validation uses recorded-format fixtures and real Python subprocesses instead of paid Agent CLIs. It covers native children, duplicate/cumulative counters, resume deltas, phase ordering, scoped shared-home discovery, capture limits, complete functional stdout/stderr, pipe draining after write/parser failures, ledger-failure diagnostics and resume qualification, normal exit, timeout, and Long Horizon invocation archives. The CLI smoke check alone does not verify these behaviors.
