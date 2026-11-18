# Agent workspace isolation

The default `light` profile runs a native coding-agent process with capture on macOS/Linux.
Select `light-sandbox` for coordinator-side Bubblewrap isolation on Linux.
This is separate from the remote GPU executor. An explicit Bubblewrap selection fails
when the host cannot provide it; it does not fall back to native execution.

## Launch and configuration

```bash
aka optimize --repo-root "$PWD" --profile light-sandbox \
  --problem /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --optimization-mode ultra --framework Cuda --agent-cli claude \
  --workspace-base /path/to/new-runs
```

The coordinator needs the `bwrap` executable and permission to create the required
user/mount/PID namespaces. A remote GPU gateway does not satisfy this local requirement.
Native `light` does not require Bubblewrap. On macOS, use native Light or run the complete
coordinator in a Linux VM. A minimal Lima setup is:

```bash
# macOS host
brew install lima
limactl start --name=aka-light template:default
limactl shell aka-light

# inside the default Ubuntu guest
sudo apt-get update
sudo apt-get install -y bubblewrap git python3 python3-venv
```

Install AKA, the Agent CLI and its credentials inside the guest, then launch
`--profile light-sandbox` there. The GPU workload remains remote in Agate; Lima does not need GPU
passthrough.

Configure custom isolation through the Agent provider, with a composition patch:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "bwrap_executable": "bwrap",
    "proc_mode": "private"
  }}]
}
```

Optional engineering references are selected per optimization launch, independently of the Agent
composition patch. They work with both Agent boundaries:

```bash
aka optimize --repo-root /path/to/agent --profile light-sandbox \
  --problem /path/to/problem --platform L20N --optimization-mode ultra --framework Cuda \
  --agent-reference-paths /knowledge/nvidia \
  --agent-reference-paths /knowledge/cuda-guides
```

The option may be repeated. Use `[NAME=]DIR` to name an entry. Each directory appears at
`references/<name-or-directory-basename>/` inside the Agent workspace. Basenames must be unique. Bubblewrap
mounts the directory read-only. Native `light` creates a symbolic link to the original directory;
because the Agent runs as the coordinator OS user, Light cannot prevent modification of those
original files. Retain an independent copy or backup before exposing knowledge to a native Agent.
The Prompt identifies these directories as optional engineering knowledge rather than task
contract, measurement evidence, or candidate source. References are excluded from Kernel Git,
candidate snapshots, and remote command uploads, and cannot be selected as GPU request files. The
final Kernel must not import or depend on them.
A patch replaces the whole Agent configuration. Preserve the selected sandbox and proc mode.
Session capture is built into shared execution and has no `capture` patch option.
The former `--agent-sandbox`, `--bwrap-executable`, and `--agent-read-only-path` optimizer
flags are not the installed configuration surface. The provider generates its internal
environment from these options.

| Option | Meaning |
| --- | --- |
| `sandbox` | `none` or `bwrap` |
| `bwrap_executable` | Bubblewrap executable, default `bwrap` |
| `proc_mode` | `private` procfs, or the existing complete `host` procfs and PID namespace |
| `timeout_seconds` | Optional provider active-time ceiling; omitted by default |
| `pause_agent_during_requests` | Default `false`: Agent stays runnable during Runtime calls; `true` pauses its process tree and excludes request waits from active time |

## Launch lifecycle

`aka.agent.provider` owns the process policy. It prepares the Session Home before
native transcript discovery and Codex ledger observation, constructs the allowed filesystem
view, passes environment values through an anonymous argument FD, and launches the owned
process tree. Capture observes the same Home throughout Session resume and close.

The persistent Agent workspace keeps its absolute path and its own Git repository. Campaign-private
stores, evaluator inputs and Gateway credentials are outside the isolated optimization Session.
The Agent receives a scoped Runtime capability through the shared Unix-socket ToolHost.

Same-Session resume retains its Home and sandbox selection. A changed Home/config root or
sandbox mode requires a new Session. Recorded campaign continuations restore their frozen
provider options; do not reinterpret a running Session by changing environment variables.

## Filesystem and credentials

Bubblewrap starts from an empty root with selected read-only system/runtime files, minimal
devices, private temporary directories, and its selected proc view. It mounts:

- The managed draft, editable `kernel.py + impl/` source tree, custom Skills/tools, and scratch.
- Public contracts, reserved clients/Skills, and eligible history read-only.
- A writable Session-specific Provider Home outside the candidate Git tree.
- Minimal CLI installation files and explicitly validated extra grants.

Only selected login/settings files are seeded into a Provider Home. Transcript trees,
arbitrary global plugins, and Agate configuration are not copied wholesale. Existing
Session settings are preserved; refreshing host login state does not overwrite them.
Model credentials intentionally supplied to the Agent remain usable by it.

Extra grants reject broad host-home roots, private state, symlink traversal, and unsafe
overlap with workspace/Home boundaries. The Agent shares host networking for model and
loopback Runtime access; this configuration supplies no network allowlist or cgroup quota.
Native mode retains the operating-system authority of the launching user.

## Auxiliary sessions

Claim Review uses an independent workspace and follows the selected Agent sandbox configuration.
With `sandbox: bwrap`, supplied evidence is mounted read-only and `review.json` is writable.
With `sandbox: none`, the Reviewer runs natively in that workspace without a Bubblewrap dependency.
A separate directory organizes inputs and outputs; it is not an OS access boundary in native mode.
Final-source reviews and preprocessing also use declared auxiliary workspace roles.

Public problem preprocessing can read detailed inputs to create the generalized contract.
Optimization Sessions do not inherit that private input view.

## Scratch and reusable state

Episodes resume the same workspace, Git repository, scratch files and Session. Runtime refreshes
managed clients and visibility-filtered evidence in place; it does not reset source, branches,
uncommitted edits or scratch at an Episode boundary.

## Restricted containers

Use `proc_mode: host` as the default for Light inside Aone Sandbox. It preserves Bubblewrap's
filesystem isolation but shares the outer container's PID namespace and complete `/proc` with
the Agent. Ordinary Linux and Lima continue to default to `private`.
The isolated Wiki worker inherits this selected proc mode. Its Python/Claude subprocesses
receive a usable `/proc`, rather than an empty directory; filesystem grants remain scoped
to the prepared request and required installations.

This is the default and required Bubblewrap mode in the current Aone Sandbox environment: its outer
container policy rejects Bubblewrap's normal nested procfs mount. Use an Agent composition patch
that retains the complete `light-sandbox` settings:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "proc_mode": "host"
  }}]
}
```

Pass it as `--profile light-sandbox --patch /path/to/patch.json`. No compiler, compatibility shim or `LD_PRELOAD` configuration is required. Because
the Agent can see every process in the Aone Sandbox, use a dedicated Sandbox and do not treat
host mode as process isolation.

## Verification and rollback

Validate the selected boundary on the deployment host: configured read-only paths must reject
writes, Runtime calls must remain reachable, and cancellation must stop the owned process tree.
Offline tests alone do not establish Linux Bubblewrap, provider authentication or live GPU qualification.

Stop the campaign before deployment/configuration changes. Preserve the original Home,
private store, and frozen selection for recovery. Use a new workspace when changing workflows
or incompatible provider/resource identities.

## Limits of the boundary

`private` creates a PID namespace and private procfs. `host` shares the existing PID namespace
and complete `/proc`; same-UID processes can be inspected or signalled where the host permits it.
Host mode is intended for a dedicated, trusted outer container such as Aone Sandbox. Its read-only
filesystem mounts are not a process-isolation guarantee. Native mode has neither restriction.
Provider-spawned child agents inherit the parent's namespace and filesystem grants; they do not
receive a separately narrowed sandbox. A reviewer has separate evidence and native Session state,
but only Bubblewrap enforces the evidence mounts as read-only. Review outputs are consumed only
after successful process completion, using bounded no-follow regular-file reads and schema checks.
GPU-library imports in development are not treated as final-submission violations; the submission
policy evaluates the implementation that is actually submitted.

Light does not install a host-command or GPU-library import blocker. Reading package
metadata or importing a library does not terminate the Session. Use Runtime for measurements
on the selected worker and their recorded evidence. Agent-owned source and Git remain editable.
Revoking Runtime access still stops its bound Agent invocation; it does not leave an Agent
running with unusable tools.

The Wiki worker receives the selected Wiki resources and configuration, the read-only AKA
package, and the interpreter/CLI installations and DNS/TLS files needed to run them. Those
installation mounts can include shared Python dependencies; this is not per-import isolation.
Native execution does not enforce these filesystem grants in the OS.
