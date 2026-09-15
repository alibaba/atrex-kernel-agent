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

The supported native sources are Claude, Codex, Qoder, and Pi. Existing provider-home configuration is respected; this change does not remount credentials or replace HOME. Qoder no longer receives `--no-session-persistence`, so its native session files can be captured. Optional external reviewer helpers or arbitrary model processes bypassing the shared Agent runtime are not automatically covered unless their calls appear in the selected provider transcripts.

## Failure and privacy boundaries

Capture uses incremental reads, bounded transcript discovery, and no-follow regular-file reads. Default limits are 64 MiB per file, 128 MiB retained per invocation, 2 MiB per line, 4,096 discovered files, and 200,000 retained records. Exceeding a limit records partial coverage; it does not terminate the Agent or reject its result.

If a persistence sink fails, the pipe reader keeps draining the child's output to EOF. Setup/finalization failures are logged without replacing the CLI outcome. Existing process timeout, dependency guards, termination, and resume policy remain in place. A forcibly killed observer cannot write a final status; surviving files may still say `running`.

Each capture directory is created with mode `0700`. Only the current session and identified descendants are copied, not unrelated transcripts or credential files. Prompt, reasoning, and tool output may nevertheless contain sensitive content: these are local diagnostic files, not sanitized public artifacts. This PR adds no security isolation against the Agent running as the same OS user. The existing trace-retention manifest and upload policy are unchanged.

## Verification

No GPU or model credentials are needed:

```bash
python3 -m unittest \
  orchestrator.test_session_capture \
  orchestrator.test_session_capture_failures \
  orchestrator.test_session_discovery \
  orchestrator.test_unsandboxed_usage \
  orchestrator.test_session_integration \
  long_horizon.test_journal_wiki_attribution

python3 orchestrator/optimize.py --help
```

Tests use recorded-format fixtures and real local Python subprocesses in place of paid Agent CLIs. They cover native children, duplicate/cumulative counters, resume deltas, phase ordering, scoped shared-home discovery, capture limits, pipe draining after write/parser failures, normal exit, timeout, and Long Horizon invocation archives.
