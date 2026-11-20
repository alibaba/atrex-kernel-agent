# Session observability

Every managed Agent invocation writes its conversation and token receipts to Supervisor-private
storage. Same-Session continuation creates a new invocation archive without discarding earlier
history. These observations do not change Agent Git or decide submission acceptance.

## Archives

Main Agent invocations live below the lineage's `segments/episodes/eNNNN/sessions/`.
Each invocation keeps `conversation.jsonl`, `token-usage.json`, and captured provider streams.
Review families use `measurements/reviews/<submission-id|direction-id|experiment-id>/` with:

- `home/`: native provider state used to resume the reviewer.
- `workspace/`: current evidence and output directory.
- `vN/result.json`, `conversation.jsonl`, `token-usage.json`: immutable revision evidence.
- `vN/captures/`: complete per-attempt captures, including failed attempts.

Unsupported attempts remain private. Agent-visible research contains the bounded review conclusion,
not reviewer conversations. A transcript export is audit evidence; native provider state is still
required for resume. An unavailable native Session causes a fresh-Session retry, not silent reuse
of another conversation.

## Token accounting

Main-Agent usage and Review usage are separate. Reviews are broken down by Review kind; total
all-path usage sums disjoint invocation receipts. A centrally archived Review discovered through
multiple lineage scopes is counted once. `reasoning_tokens` is reported separately and is normally
already included in output/total tokens; it is not added to the total again.

`exact` means the provider receipts cover the observed invocation. `partial` retains known counters
when evidence is incomplete or contradictory. `estimated` indicates a best-effort transcript
estimate; `unavailable` means no usable total. Unknown counters are `null`, not invented zeroes.
Identical duplicate receipts preserve exactness. Conflicting overlapping receipts select one whole
observation (the larger reported total) and mark it partial; they are neither added nor combined
with component-wise maxima into a synthetic bill.

Provider accounting rules differ: Claude root response usage already includes its known tool/model
path and exported child rows must not be added twice. Codex cumulative counters require an invocation
baseline. Pi stream/native copies are reconciled rather than summed; DSH native/stream identities
prevent repeated response accounting. Per-provider diagnostics remain in the archived receipt.
Accepted submissions and passing full Evaluates retain token checkpoints at their result time.

## Capture and liveness

Capture drains stdout/stderr while the process runs. Heartbeats report live pipe byte counts even
when `Popen.wait()` raises a timeout without buffered output. Process-tree termination snapshots
its tracked process groups under a lock so concurrent child discovery cannot interrupt revocation.

Native transcripts are read incrementally with no-follow regular-file checks. Resume uses an
in-process cache of compact response IDs/cumulative counters and an inode/offset cursor. Only the
new suffix is scanned on later invocations; a controller restart or cache eviction rebuilds the
index. An old malformed line or interrupted suffix does not poison a later invocation's quality.
An incomplete history scan is explicitly partial, so missing IDs cannot silently double-count usage.

Capture limits are 64 MiB per file, 128 MiB per invocation, 2 MiB per line, 4,096 files and 200,000
records. Limits constrain observability only: exceeding one marks capture incomplete and does not
terminate the Agent. Native capture may be incomplete despite functional command stdout being
available. No Episode active-time limit is imposed by capture.

## Configuration and compatibility

Capture is built into shared execution; there is no Workflow capture-off option. Standalone
invocations can use `ATREX_SESSION_CAPTURE_DIR` to select an archive directory. Managed Light
chooses private invocation roots. Old capture-off recipes are historical; keep their corresponding
checkout to reproduce them. A missing/unwritable archive is reported and does not fabricate usage.

## Code map

| Responsibility | Owner |
|---|---|
| Invocation ownership, cancellation and optional provider timeout | [`provider.py`](../src/aka/agent/provider.py) |
| Process streams, heartbeat and termination | [`execution/process.py`](../src/aka/agent/execution/process.py) |
| Live archive and resume index | [`session_capture.py`](../src/aka/agent/session_capture.py) |
| Bounded no-follow reads | [`session_tail.py`](../src/aka/agent/session_tail.py) |
| Provider counters and resume baselines | [`session_usage.py`](../src/aka/agent/session_usage.py) |
| Token evidence merge | [`contracts/agent.py`](../src/aka/contracts/agent.py) |
| Shared disjoint-usage rollup | [`usage_summary.py`](../src/aka/agent/usage_summary.py) |
| Point-in-time checkpoints | [`token_checkpoints.py`](../src/aka/workflow/light/long_horizon/token_checkpoints.py) |

Use the [observability Skill](../skills/atrex-aka-light-observability/SKILL.md) for current state,
Kernel history, Runtime/Agate calls and evidence export. Historical capture-performance reports
remain version-specific measurements, not performance claims about this implementation.
