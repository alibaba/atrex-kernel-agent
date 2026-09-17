# Agent workspace isolation

AKA can launch coding-agent sessions inside a Linux Bubblewrap namespace. This isolates the coordinator-side filesystem; it is separate from the Gateway or SSH sandbox that executes GPU jobs. It is opt-in: `--agent-sandbox none` remains the default on Linux and macOS.

This launch boundary was introduced in the second step of the simplified AKA split. The subsequent [Supervisor GPU/Wiki Runtime](supervisor-runtime.md) moves GPU/Wiki requests out of Agent sessions. Setup, Framework Baseline, Fast/Full Episodes, phase markers, Journal, Git ownership and candidate validation remain in place.

## Launch lifecycle

```mermaid
flowchart TD
    ENTRY["CliAgentRuntime / LongSessionRunner"] --> HOME["agent_home.prepare_agent_environment<br/>Session-local Provider Home"]
    HOME --> OBS["Create Codex ledger observer<br/>against the same Home"]
    OBS --> RUN["agent_runtime.process.run_bounded"]
    RUN --> MODE{"agent_sandbox"}
    MODE -->|none| NATIVE["Existing native command"]
    MODE -->|bwrap| VIEW["agent_workspace<br/>Optimizer or auxiliary allowlist"]
    VIEW --> WRAP["agent_sandbox + agent_installations<br/>Mounts and minimal CLI installations"]
    WRAP --> FD["sandbox_launch<br/>Environment via anonymous --args FD"]
    FD --> SPAWN["recovery_processes.spawn_owned_session"]
    NATIVE --> SPAWN
    SPAWN --> CAPTURE["PR1 Session capture<br/>Pipes + native transcripts + usage"]
    CAPTURE --> EXIT["Finalize capture<br/>Publish auxiliary output only on success<br/>Always clean up temporary view"]
    WRAP -->|Unsupported host or invalid grant| FAIL["Fail before starting Agent<br/>No silent unsandboxed fallback"]
```

The command and workspace retain their original absolute paths inside the namespace. Existing prompts, tools, Git worktree links and native-session discovery therefore do not need path rewriting. `HOME`, Provider config roots and XDG directories point to the isolated Home before ledger observers and capture are initialized. Recovery-owned launches pass the anonymous argument FD through their ownership wrapper; the parent closes it after spawning.

Campaign reviewer-state environment paths keep their lexical absolute form only in `bwrap` mode, so mount validation can detect symlinks. Native `none` mode retains the existing `Path.resolve()` behavior, including symlink resolution and `..` normalization. Wiki-profile paths are now Supervisor-only and no longer create Agent mounts.

Direct launches omit Bubblewrap's `--die-with-parent`, preserving the native path's process-lifetime behavior: Supervisor or spawning-thread exit alone does not kill the Agent. This does not guarantee survival of lost output pipes or automatic reattachment. Recovery handoffs retain the flag, binding the sandbox to the durable ownership wrapper rather than the original Supervisor; the existing cleanup guardian remains responsible for orphan cleanup. Timeouts and explicit interruptions still use the existing process-group termination logic in both modes.

## Usage and rollback

Add these options to an existing campaign command:

```bash
python3 orchestrator/optimize.py \
  --op-dir /path/to/operator \
  --platform TARGET_GPU --sandbox-hardware REMOTE_GPU \
  --framework Triton --agent-cli claude \
  --agent-sandbox bwrap
```

The coordinator needs Linux, `bwrap` on `PATH`, and permission to create unprivileged user/mount/PID namespaces. A remote GPU Gateway does not satisfy that requirement. On macOS either keep the native path or run the coordinator inside a Linux VM such as Lima. A container host must permit Bubblewrap namespace/proc mounts; selecting `bwrap` never silently bypasses a failed isolation setup.

| Setting | Meaning |
| --- | --- |
| `--agent-sandbox none\|bwrap` | Native execution or the opt-in namespace; default `none`, also configurable with `ATREX_AGENT_SANDBOX` |
| `--bwrap-executable PATH` | Bubblewrap executable; default `bwrap`, or `ATREX_BWRAP_EXECUTABLE` |
| `--agent-read-only-path PATH` | Explicit extra read-only host path, repeatable; mounted at its original absolute path |

Additional grants are operator-authorized exceptions, not automatic dependency discovery. Broad host-home/repository roots and paths overlapping the writable workspace/session homes are rejected. Custom wrappers, external CA bundles, SSH identities, plugins or installations outside the supported layouts may need a narrowly scoped grant and corresponding client configuration. Do not grant whole credential or transcript directories to make a missing dependency disappear.

To roll back the launch boundary, stop the campaign normally and launch with `--agent-sandbox none`. Existing prompts, accepted commits and evaluation policy remain compatible. Isolated Provider sessions are not automatically imported into the operator's global Home; a recovery that requires native CLI thread state must continue in the mode/Home that created it, or start a fresh Agent invocation using the existing recovery flow.

## Filesystem and credentials

The namespace starts from an empty root, not a read-only copy of the host root. It contains:

- Read-only system runtime directories and selected OS configuration/certificate files, minimal `/dev`, a private PID namespace and `/proc`, and private `/tmp` and `/run`.
- The current Optimizer workspace, writable at its original path, plus read-only repository tools, skills and reference assets needed by the existing symlink-based workspace.
- A writable, per-Session Provider Home under `<workspace-parent>/.atrex-agent-homes/<key>/`. Different workspaces/Sessions receive separate copies; same-Session resume keeps its Home. Homes are outside the candidate Git tree.
- Minimal executable files, recognized npm packages/dependencies and Python installation libraries needed by the CLI. Runtime-managed sessions do not add an Agate installation mount. Discovery never restores an entire first-level directory under the operator Home. Installation aliases pointing into Provider history/auth state or the private Session storage are rejected.

Only selected login/settings files are seeded for the active backend and enabled external plan reviewers. Claude/Qoder/Codex/Pi transcript trees, caches and arbitrary global plugins are not copied. Qoder's `.auth/user` and `machine_id` are included; its entire `.qoder` or `.qodersec` tree is not mounted. Destination directory traversal does not follow Agent-created symlinks. Existing Session-local settings are preserved on resume, and changes never write back to the operator Home.

These are copies, not live credential synchronization: refreshed host login state does not overwrite an existing Session Home. State files are private to the owning OS user and can contain credentials and raw conversations. Retain them for resume, and remove them only after that Session is stopped and no longer needed. PR1 capture remains outside the workspace by default and uses the same isolated Home; do not explicitly relocate capture into an Agent-writable path if it must remain private.

Environment inheritance otherwise follows the existing CLI contract. Model credentials intentionally supplied to the Agent remain usable by it. Gateway credentials are removed before launch; the Agent receives a workspace-scoped HTTP capability instead. New Homes do not copy Agate configuration, and old copies are masked for resumed Runtime-managed sessions. Environment values are passed through an anonymous `bwrap --args FD`, not `--setenv KEY secret` in the process command line. This is not protection against a privileged host operator or another process with equivalent OS credentials.

## Auxiliary sessions

Supervisor-launched auxiliary sessions receive a temporary allowlist view at their original working directory. Inputs are mounted read-only; only declared output files are copied back with bounded, no-follow reads after normal process completion with exit code zero and no policy/environment termination. Timeout, interruption, process failure or guard termination skips publication and leaves existing destination reports unchanged. Other files written in that view are discarded. The existing caller still validates report contents.

Temporary-view cleanup is attempted whether publication succeeds, fails or is skipped. If cleanup also fails while another exception is propagating, it adds a diagnostic note to that exception instead of replacing it. Otherwise it logs a warning with the residual view path without changing the returned stdout, stderr, exit status or timeout flag; a logging failure also cannot replace the result. The temporary view may remain on disk and need later cleanup. A publication error on an otherwise successful session still propagates to the caller.

| Role | Read-only inputs | Returned files |
| --- | --- | --- |
| Public problem generation | `reference.py`, `input.py`, `shapes.json`, `metadata.json` | `agent_problem.json` |
| Production policy review | `review_request.json`, `candidate/` | `dependency_review.json` |
| Baseline exit review | `crash_record.json`, `candidate/` | `resume.json` |
| Baseline correctness review | `context/` | `correctness_review.md` |
| Plan-reviewer availability probe | `availability_probe.md`, `availability_proposal.md` | None; existing stdout protocol |

Availability probes use the Campaign workspace as `cwd` in both native and Bubblewrap modes. Their draft/proposal remain Supervisor-created temporary files, passed as absolute paths; relative packet paths are rejected before launch. Bubblewrap snapshots them under the two declared input names in its read-only view and passes those virtual paths to the helper. No probe files are written to the real Campaign, and neither the temporary source directory nor other Campaign files are mounted. Changing `cwd` does not grant access to undeclared relative-path configuration files.

The allowlist contains at most 4,096 files / 16 MiB; each returned file is limited to 8 MiB. These roles receive no campaign Git, Gateway/private-reference, Wiki-history or recovery-state mounts. Public problem generation is the explicit trusted preprocessing exception that reads exact shapes to create the public contract; optimization sessions do not inherit its input view.

Provider-created child Agents and plan helpers launched from within an Optimizer inherit that Optimizer's namespace, not a separate auxiliary allowlist. Optional persistent Codex/Qoder plan consultations have their own campaign-scoped Homes and state under `.atrex_long_horizon/reviewer-state/`, avoiding both global Provider history and the full Long Horizon archive. Entering this opt-in mode starts a separate persistent reviewer context rather than importing a global native thread.

## Legacy grants and scope limits

The remaining compatibility grants do not change optimization or acceptance:

| Existing behavior retained | Explicit compatibility grant | Later ownership migration |
| --- | --- | --- |
| Agent commits the candidate | Current campaign Git common directory, writable; only user name/email copied from global Git configuration | Supervisor-owned Git/promotion, PR6 |
| Agent calls `tools/sandbox.py` | HTTP client and per-invocation capability; no evaluator/private-input/Gateway-credential grant | Implemented by the Supervisor GPU/Wiki Runtime |
| Agent writes phase/recovery evidence and Journal live mirror | Scoped telemetry, recovery-state, reviewer-state and live-memory directories as needed for append/atomic replace | Runtime records/Journal, PR4–5 |

Writable file grants include the containing directory to support atomic replacement and lock files. Validation checks that actual directory before creating or mounting it: it must not be the operator Home or an ancestor, traverse symlinks, or overlap the `.atrex-agent-homes` root in either direction. The current Session Home is mounted only through the dedicated Home setup, never through these legacy environment grants. Use a dedicated log/state subdirectory rather than placing a granted file directly under the operator Home.

These grants are deliberately narrower than host access but **are not a complete authority boundary against a malicious Agent**. Git and legacy control/evidence paths remain Agent-accessible. The private evaluator/reference paths now stay outside the Agent namespace, while Runtime results use bounded projections. Arbitrary GPU probe code is still untrusted; this is not a semantic data-loss-prevention guarantee for everything a probe could print. Immutable Supervisor ledgers and hidden Git belong to later changes.

The Agent shares the host network for model and loopback Runtime access; Gateway/Wiki/SSH operations now run on the Supervisor. There is no new network allowlist, cgroup resource quota, measurement retry policy, Journal protocol, or promotion rule. Custom operator grants and inherited environment variables remain part of the trusted launch configuration.

## Scratch lifecycle

In `bwrap` mode, a newly materialized Episode starts with an empty `scratch/`. Resuming the pre-execution `preparing` phase also resets it; same-Episode exploration/recovery does not. Only that exact directory is cleared, and a symlink is removed without touching its target. The default native mode retains its existing behavior. No new requirement to write plans or reports under `scratch/` is introduced.

## Verification

Local fixtures are kept outside this repository, following its review convention. Checks cover no-sandbox compatibility, repeated CLI parsing, unsupported-host failure, minimal installation mounts, Provider-state isolation, rejection of host-Home/Session-Home legacy grants, safe Home reseeding, anonymous argument ownership, auxiliary input/output rules and scratch symlinks. Linux subprocess checks use temporary fake Agents, including the real Codex adapter launch path, without model/GPU calls. Lifecycle checks cover direct Supervisor/spawning-thread exit, handoff survival across Supervisor exit, ownership-wrapper death cleanup, and timeout cleanup.

On Lima Ubuntu (Python 3.14.4, Bubblewrap 0.11.1), the isolation suite passes, including actual read-only mounts, hidden host files, Git worktree commits, native Session capture, recovery-wrapper FD propagation and timeout cleanup. PR1's external 111-check regression suite also passes on the native path. Installed Claude 2.1.235, Codex 0.148.0 and Qoder 1.1.28 execute `--version` inside the namespace. Pi is not installed in that VM; live authentication, model calls and GPU end-to-end campaigns were not exercised.

For a no-model smoke check on Linux, run from the repository root:

```bash
python3 - <<'PY'
import os
import sys
import tempfile
from pathlib import Path
from orchestrator.agent_runtime.process import run_bounded

with tempfile.TemporaryDirectory(prefix="aka-sandbox-smoke-") as temporary:
    workspace = Path(temporary) / "workspace"
    workspace.mkdir()
    environment = dict(os.environ, ATREX_AGENT_SANDBOX="bwrap", ATREX_SESSION_CAPTURE="0")
    output, error, code, timed_out = run_bounded(
        [sys.executable, "-c", "from pathlib import Path; Path('probe').write_text('ok'); print('ok')"],
        workspace, 20, environment,
    )
    assert code == 0 and not timed_out, (code, error)
    assert (workspace / "probe").read_text() == "ok"
    print(output.strip())
PY
```
