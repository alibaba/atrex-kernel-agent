# Agent workspace isolation

The default `light` profile runs a native coding-agent process with capture on macOS/Linux.
Select `light-sandbox` for coordinator-side Bubblewrap isolation on Linux.
This is separate from the remote GPU executor. An explicit Bubblewrap selection fails
when the host cannot provide it; it does not fall back to native execution.

## Launch and configuration

```bash
aka optimize --repo-root "$PWD" --profile light-sandbox -- \
  --op-dir /path/to/atrex-bench/operator \
  --platform '<gpu_arch>' --framework Cuda --agent-cli claude \
  --workspace /path/to/new-runs
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

Configure custom isolation through the Agent provider, with a composition patch before `--`:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "capture": true,
    "bwrap_executable": "bwrap",
    "proc_mode": "private",
    "read_only_paths": ["/absolute/path/to/required/runtime"]
  }}]
}
```

Replace the illustrative runtime path with an existing narrowly scoped directory, or omit it.
Optional engineering references are selected per optimization launch, independently of the Agent
composition patch. They work with both Agent boundaries:

```bash
aka optimize --repo-root /path/to/agent --profile light-sandbox -- \
  --op-dir /path/to/problem --platform L20N --framework Cuda \
  --agent-reference-dir /knowledge/nvidia /knowledge/cuda-guides
```

The option may be repeated. Each directory appears at
`references/<directory-basename>/` inside the Agent workspace. Basenames must be unique. Bubblewrap
mounts the directory read-only. Native `light` creates a symbolic link to the original directory;
because the Agent runs as the coordinator OS user, Light cannot prevent modification of those
original files. Retain an independent copy or backup before exposing knowledge to a native Agent.
The Prompt identifies these directories as optional engineering knowledge rather than task
contract, measurement evidence, or candidate source. References are excluded from Kernel Git,
candidate snapshots, and remote command uploads, and cannot be selected as GPU request files. The
final Kernel must not import or depend on them.
A patch replaces the whole Agent configuration. Include both sandbox and capture settings
when preserving `light-sandbox` behavior; the provider's standalone capture default is false.
The former `--agent-sandbox`, `--bwrap-executable`, and `--agent-read-only-path` optimizer
flags are not the installed configuration surface. The provider generates its internal
environment from these options.

| Option | Meaning |
| --- | --- |
| `sandbox` | `none` or `bwrap` |
| `capture` | Diagnostic capture, independent of filesystem isolation |
| `bwrap_executable` | Bubblewrap executable, default `bwrap` |
| `read_only_paths` | Explicit extra installation/runtime paths |
| `proc_mode` | `private` procfs, or the existing complete `host` procfs and PID namespace |
| `timeout_seconds` | Per-invocation active-time ceiling; Workflow Episode budget is separate |

## Launch lifecycle

`aka.light.agent.provider` owns the process policy. It prepares the Session Home before
native transcript discovery and Codex ledger observation, constructs the allowed filesystem
view, passes environment values through an anonymous argument FD, and launches the owned
process tree. Capture observes the same Home throughout Session resume and close.

The managed draft keeps its absolute path. The controller worktree, Git common directory,
private Journal, evaluator checkout, and Gateway credentials are not mounted into an isolated
optimization Session. The Agent receives a scoped HTTP Runtime capability instead.

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

Supervisor reviewers and preprocessing calls use a temporary allowlist view: declared
inputs are read-only, and declared outputs are copied back through bounded no-follow reads
only after successful completion without policy/environment termination.
Timeouts, interruption, failure, and guard termination do not publish auxiliary outputs.

Public problem preprocessing is the explicit trusted exception allowed to read detailed
inputs when creating the generalized contract. Optimization Sessions do not inherit that
private input view. Provider-created child agents inherit the optimizer namespace rather
than obtaining a new reviewer grant.

## Scratch and reusable state

Every new managed Light Episode starts with empty scratch, in native and sandbox modes.
Same-Episode continuations preserve it. Candidate source and custom Skills/tools have
separate snapshot rules; reusable state can survive an unpromoted kernel.
Reserved assets are refreshed from trusted sources. See
[reusable workspace](optional-light-modules.md#reusable-workspace).

## Restricted containers

Use `proc_mode: host` as the default for Light inside Aone Sandbox. It preserves Bubblewrap's
filesystem isolation but shares the outer container's PID namespace and complete `/proc` with
the Agent. Ordinary Linux and Lima continue to default to `private`.

This is the default and required Bubblewrap mode in the current Aone Sandbox environment: its outer
container policy rejects Bubblewrap's normal nested procfs mount. Use an Agent composition patch
that retains the complete `light-sandbox` settings:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap",
    "capture": true,
    "proc_mode": "host"
  }}]
}
```

Pass it as `--profile light-sandbox --patch /path/to/patch.json` before the optimizer's `--`
separator. No compiler, compatibility shim or `LD_PRELOAD` configuration is required. Because
the Agent can see every process in the Aone Sandbox, use a dedicated Sandbox and do not treat
host mode as process isolation.

## Verification and rollback

[Light verification](optional-light-modules.md#verification-scope) records local process,
allowlist, auxiliary-publication, and Linux Bubblewrap evidence. These checks do not establish
all-provider authentication or live GPU qualification.

Stop the campaign before deployment/configuration changes. Preserve the original Home,
private store, and frozen selection for recovery. Use a new workspace when changing workflows
or incompatible provider/resource identities.
