# Supervisor GPU/Wiki Runtime

The Campaign now owns GPU and Wiki execution. Agent sessions keep the familiar `python3 tools/sandbox.py ...` interface, but that script is a standard-library HTTP client. Packaging, endpoint selection, credentials, private evaluator inputs and result projection run in the Supervisor. Operators still launch `orchestrator/optimize.py`; no separate service command is required.

This is PR3 of the simplified AKA migration. It changes the execution boundary, **not the optimization workflow**. Setup/Framework Baseline, Fast/Full Episodes, phase markers, plans, profiles, Agent Git commits, Journal and acceptance remain unchanged. Kernel/Measurement IDs, cross-Episode deduplication, new retry/aggregation policies, Runtime Journal/Report and Supervisor-owned promotion belong to later changes.

## Lifecycle and authority

```mermaid
flowchart TD
    C["Campaign.start_runtime<br/>Loopback HTTP service"] --> S["run_bounded / session_environment<br/>Bind a fresh capability to this workspace"]
    S --> A["Agent: tools/sandbox.py<br/>or existing query_*.py"]
    A --> V{"SupervisorRuntime<br/>Authorize + validate arguments"}
    V -->|Invalid or revoked| E["Actionable error; no job submitted"]
    V -->|GPU| G["Private snapshot + supervisor/gateway.py<br/>Existing Agate / SSH transport"]
    V -->|Wiki| W["Supervisor query tools<br/>Private query telemetry"]
    G --> P["Bounded result projection<br/>Legacy markers + declared profile files"]
    W --> P
    P --> A
    S -->|Agent exits or is interrupted| R["Revoke capability<br/>Stop in-flight request process groups"]
    C -->|Campaign finally| X["Close listener, drain handlers<br/>Remove temporary snapshots"]
```

`Campaign.agent_environment()` starts the listener lazily. `run_bounded()` creates an invocation-scoped capability against the actual Episode worktree, including when the Campaign uses several disposable worktrees. It removes Supervisor Gateway credentials/configuration and supplies only `ATREX_AKA_RUNTIME_URL` and `ATREX_AKA_RUNTIME_TOKEN`. Auxiliary reviewer/problem-generation sessions receive no GPU/Wiki capability. Agent-created children within the same invocation may use that invocation's capability.

The service listens on `127.0.0.1` with a random port. Its authenticated endpoints are:

| Endpoint | JSON request |
| --- | --- |
| `POST /v1/gateway/execute` | `{"argv": ["--kind", "run", "--no-sync"]}` |
| `POST /v1/wiki/query` | `{"tool": "query_hardware", "argv": ["--list", "products"]}` |

`GET /healthz` is an unauthenticated liveness check without Campaign data. Responses preserve the CLI boundary: `{"exit_code": 0, "stdout": "...", "stderr": "..."}`. The client prints those streams and returns the exit code; Agents do not need to write HTTP requests or handle bearer tokens themselves.

The capability, not an Agent-supplied path, selects the workspace. Legacy full-name endpoint flags are stripped so existing prompts remain valid; prefix abbreviations are rejected, and downstream parsers disable abbreviations. Supervisor-only preflight/health controls and Wiki store overrides are rejected. Wiki `--file` accepts only a bounded regular file within the authorized workspace. Symlinks and `..` traversal cannot grant arbitrary host reads or writes.

## GPU operations

Read the mounted `gpu-measurement` Skill for Agent-facing examples. Existing evaluator commands retain their syntax and result markers:

```bash
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --version v2 --no-memory
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
python3 tools/sandbox.py --kind run --baseline-path scratch/base.py --comparison-repeats 2 --no-sync
python3 tools/sandbox.py --kind profile --profile-level sol --no-sync
python3 tools/sandbox.py --kind dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py
python3 tools/sandbox.py --kind check --sanitize memcheck --no-sync
python3 tools/sandbox.py --kind disassemble --format ptx --no-sync
python3 tools/sandbox.py --kind env
```

| Operation | Behavior and Agent result |
| --- | --- |
| Evaluate (`run`) | Existing evaluator semantics; compact `[test_kernel] RESULT_JSON=` with correctness, latency, per-Shape measurements and actionable diagnostics |
| ABBA (`run --baseline-path`) | Reuses the existing same-allocation AB/BA runner; compact `[sandbox] ABBA_JSON=` with baseline/candidate per-Shape metrics and speedup; does not select or promote a Kernel |
| Profile | Typed NCU/rocprof result, hottest Kernel/resource/SOL facts and requested counters via `[sandbox] PROFILE_JSON=`; legacy Profile commands and declared file synchronization remain supported |
| Dev | Bounded stdout/stderr and command exit status for an explicitly declared GPU probe, not necessarily an evaluation |
| Check | Typed compilation/sanitizer diagnostics via `[sandbox] CHECK_JSON=` |
| Disassemble | Typed resource facts and bounded assembly via `[sandbox] DISASSEMBLE_JSON=` |
| Env | Available Gateway environments or requested capabilities; not an arbitrary endpoint/configuration query |

Profile supports Kernel name/regex, source correlation, launch skip/count, Shape selection and counters. Profile/Check/Disassemble accept `--requirement` and `--deps-mode`. Check/Disassemble/Env require a Gateway exposing those typed APIs; they do not add an SSH implementation or upgrade an older local Gateway server. Unsupported Gateway capabilities remain explicit errors.

Evaluate can use `--input-path` and `--shapes-path` for an exploratory custom-input request, or `--mode correctness_only`; these do not replace the canonical acceptance contract. ABBA uses the complete canonical contract and rejects custom command/input/Shape/seed overrides. Native ABBA batches still use the existing runner/aggregation semantics; this PR adds neither three-repeat median aggregation nor new acceptance criteria.

The moved Gateway retains its existing supported-contract Dev fallback and retry behavior. Invalid explicit custom-input or new typed options fail rather than silently running a different request. There is no HTTP-level blind resubmission: a broken connection can leave an unknown remote outcome. The existing independent Supervisor verifier invokes `supervisor/gateway.py` directly and retains its original full diagnostic format and promotion rules.

## Wiki

The mounted `KernelWiki` Skill describes the same stores and query semantics as the existing Wiki. Agent calls to `gpu-wiki/tools/query_nl.py`, `query_wiki.py` and `query_hardware.py` are forwarded automatically; `wiki-query`, `wiki-search` and `wiki-hardware` aliases are also available through `tools/sandbox.py`.

Retrieval, bridge execution and query telemetry run on the Supervisor. Agents can provide a query or a workspace-local query file, not a host store root or retained bridge workspace. `query_id`/`wiki_id` attribution still uses the existing Journal; no new feedback or Journal protocol is introduced.

## Files, diagnostics and limits

GPU execution uses a regular-file snapshot, not the mutable Agent directory. Provider homes, Git, control state, benchmark checkout and linked assets are excluded; trusted evaluator code/private inputs are supplied separately. Arbitrary Dev commands retain the existing explicit `--input` packaging rules. Publication is restricted to requested paths below `profiles/` or `scratch/`, with no-follow reads and atomic writes. Source, Git and control files cannot be replaced by remote output publication. The legacy evaluator log is appended separately for the existing report compiler.

Completed request diagnostics are stored outside the candidate tree, under `<campaign-parent>/.atrex-supervisor-runtime/<workspace-key>/request-<uuid>.json`. These contain bounded command output, invocation arguments, timestamp and exit code for operator debugging. They are not immutable Measurement Records or a public query API. They may contain private evaluation details; protect them like Session traces. Temporary request snapshots are deleted after completion. No automatic retention cleanup is added.

Limits are 2 MiB per HTTP request, 512 argv entries, 16 active request processes, 4,096 files / 64 MiB per snapshot and 16 MiB per input file. Raw stdout is limited to 4 MiB for safe parsing, stderr to a 128 KiB diagnostic prefix; runaway output spooling is interrupted at 64 MiB. An oversized stdout is an unconfirmed outcome, not a successfully parsed prefix. Agent projections are at most 384 KiB stdout / 16 KiB stderr (Wiki stdout: 256 KiB), with explicit truncation or an error when a structured result cannot fit. Profile/assembly have additional per-field limits. Exact private paths and generalized failure details are withheld; raw arbitrary Dev output is not a semantic data-loss-prevention boundary.

Validation errors return HTTP 400 and a repair hint. Missing/revoked capabilities return 401. Unexpected service failures return 503 with an unknown-outcome warning; raw Supervisor exceptions are not returned. Different capabilities do not share an execution lock. A single capability serializes its requests. Closing a Session revokes its capability; in-flight processes are terminated, allowing the existing Gateway signal handler a bounded cleanup window before forced termination. Campaign close stops the listener and joins handlers. This does not guarantee that a remote job has stopped if cancellation or the network fails.

## Platform, compatibility and rollback

Native Linux/macOS (`--agent-sandbox none`) remains supported. HTTP routing and authorization are still used, but native execution is **not filesystem isolation**: an unsandboxed Agent retains the operating-system access of its user.

With `--agent-sandbox bwrap`, the Agent does not receive the private evaluator checkout, private reference directory, Supervisor storage or Gateway credentials. Existing workspace evaluator copies needed by the independent verifier are masked, and old Agate configuration copies in resumed Provider Homes are also masked. The shared host network permits loopback Runtime and model access. Legacy Git, Journal/phase/recovery grants remain; this is not yet the final all-authority-in-Supervisor architecture. Explicit operator grants remain trusted exceptions.

Standalone operator diagnostics now use the private executable, from the AKA checkout:

```bash
python3 supervisor/gateway.py --workspace /path/to/campaign --hardware REMOTE_GPU \
  --url https://your-gateway --kind run --no-sync -- python3 test_kernel.py --no-memory
```

`tools/sandbox.py` outside a live Agent Session fails explicitly instead of taking over operator credentials. Library users creating `Campaign` objects must call `close_runtime()` in `finally`; `optimize.py` already does so. A Supervisor restart creates a fresh listener/capability for a new Agent invocation; old tokens are not durable recovery credentials. Existing recovery may need to restart the Agent rather than reuse an orphaned process with a dead listener.

There is no switch that silently restores direct Agent Gateway access. To roll back, stop the Campaign and revert this revision to its PR2 base before resuming. Existing Kernel commits, numbered memory and Journal schemas are unchanged. Keep private request diagnostics for investigation. Changing to `--agent-sandbox none` disables filesystem isolation, not HTTP routing.

## Verification

Tests are kept outside the repository in accordance with the project's review convention. Local integration checks exercise the actual HTTP service, thin client and private Gateway subprocess against a fake Agate server: Evaluate, ABBA, Profile, Dev, Check, Disassemble, Env, real local Wiki hardware lookup, authorization/argument/path failures, explicit-input errors, output overflow and in-flight capability revocation. They use no model or GPU allocation.

PR1 Session-capture and PR2 auxiliary publication/probe regressions are also checked. Linux smoke testing uses actual Bubblewrap with a temporary workspace and fake Gateway, checking HTTP/Wiki access and the absence of private files/credentials. This does not establish live Provider authentication, remote GPU compilation/profiling correctness or production performance; those require a separate environment-specific end-to-end run.
