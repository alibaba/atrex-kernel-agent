# Run AKA through profile-selected startup

AKA uses the installed `aka optimize` command. It converts the public request fields once,
then uses a launch profile to select Workflow through Bootstrap and Core. Evaluation, Execution,
Resources and recovery run from their installed owning modules.

New `aka optimize` launches use the `light` profile (native Agent processes); choose
`light-sandbox` for Bubblewrap. The `workflow` composition also selects Light;
Legacy is not a fallback or a supported profile. Use a new Campaign directory.
Existing handoff/promotion campaigns need their original checkout and environment;
see [migration and rollback](design.md#migration-and-rollback).

## Checkout-backed installed interface

Use Python 3.10 or later, install the matching distributions as described below, and keep the
checkout resources available:

```sh
aka optimize --repo-root "$PWD" --help
aka optimize --repo-root "$PWD" \
  --problem /path/to/problem --platform '<gpu_arch>' --framework Triton \
  --agent-cli codex --workspace-base /path/to/runs
```

`--problem` selects the problem. It accepts the current native Atrex-Bench layout
(`reference.py`, `input.py`, `shapes.json`, within its owning Bench checkout) or SOL
layout (`definition.json`, `reference.py`, `workload.jsonl`). See [problem
preparation](quickstart.md) for the complete requirements. The AKA checkout path and the
problem path are separate inputs.

The installed parser accepts launch and request options together, without a `--`
separator. Light accepts Native Atrex-Bench and SOL inputs. See
[problem preparation](quickstart.md#prepare-the-problem) and the [SSH backend](ssh-backend.md).
Profile-selected environment values and absence are captured for children and recovery.
The removed `orchestrator/optimize.py` shim is not a compatibility interface; automation must invoke
the installed command.

## Installed command

Build the packages with a Python 3.10+ environment containing the `build` frontend:

```sh
python -m pip install build
python tools/build_distributions.py --output /tmp/aka-dist
python -m pip install --force-reinstall /tmp/aka-dist/*.whl
```

The installed packages require a **matching resource checkout** for declared external resources.
Keep the packages and checkout from the same source
revision; the `0.3.0` package version alone does not identify a source revision or repository release.
Launch reconstruction checks the recorded implementation and explicitly declared resource
identities. Normal startup does not initialize every submodule; initialize only selected resource
submodules before exporting a checkout without Git metadata.

The command can run from any working directory:

```sh
aka --help
aka optimize --help
aka optimize --repo-root /path/to/aka --help
aka optimize --repo-root /path/to/aka \
  --problem /path/to/problem --platform '<gpu_arch>' --framework Triton \
  --agent-cli codex --workspace-base /path/to/runs
```

`python -m aka` accepts the same commands and requires installed command metadata.
`--repo-root` identifies matching resource files; executable code comes from installed
packages. It does not change the working directory. `--problem` selects the task input;
`--workspace-base` selects its campaign parent (default: the current directory).

Launch options are `--repo-root`, `--profile` and repeatable `--patch`. Public request
fields such as `--platform` and `--framework` go in the same argument list. `--settings`
and `--runtime` accept JSON objects using the dataclass field names. Alternatively,
`--workflow-request` accepts a serialized `WorkflowLaunch` JSON **string**, not a path;
it cannot be mixed with new-request fields. Execution composition is selected through
profiles/patches, not removed optimizer `--execution-profile`/`--execution-patch` flags.
Only generic `aka run` and direct `aka execution` retain their respective `--` separator.

Core owns the neutral command dispatcher. Bootstrap contributes `run`, and Optimization
contributes `optimize`, via the `aka.commands` package entry-point group. Help lists
installed contributions without importing them; unknown or duplicate command names fail
before loading a command. There is no implicit default command. The Legacy distribution also contributes
`workflow` and `execution`; `aka --help` lists the installed command set. Source-only use of `python -m aka` requires installed command metadata; there is no
source-tree fallback script.

## Select a startup implementation

A launch profile names a Core composition and a service implementing the neutral
synchronous `Startup.run(Invocation) -> int` protocol. Bootstrap does not require
Application or Workflow specifically. Plugin import and `apply(ctx, config)` only
construct instances and register services; invocation starts after assembly.

For example, install this importable module as `example_workflow.py`:

```python
name = "example-workflow"
provide = ("workflow",)

class Workflow:
    def run(self, invocation):
        print(invocation.argv)
        return 0

def apply(ctx, config):
    ctx.provide("workflow", Workflow())
```

Create `profiles/launch.json`:

```json
{"api_version": 1, "composition": "workflow", "target": "workflow"}
```

Create its Core composition at `profiles/compositions/workflow.json`:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "insert": {"name": "example_workflow", "required": true}}
  ]
}
```

Then run it with Bootstrap installed and the plugin importable:

```sh
aka run --profile /path/to/profiles/launch.json -- example-argument
```

The instance receives the arguments after `--` as `Invocation.argv`. Removing, disabling
or failing the required target does not fall back to a default. `--patch` and `--var
NAME=VALUE` operate on the referenced Core composition. Additional business tokens must
be defined as `ServiceKey` objects in code and registered through installed `aka.tokens`
entry points or supplied programmatically; profile `tokens` selects those registered
names. `required_services` requires corresponding selected tokens. The target's Startup
token is supplied by the host.

Launch profiles can additionally declare `bindings` (string values), `environment`
(names whose values or absence must survive restart) and `resources` (root/paths whose
content and directory membership determine identity). Resource declarations may pin
submodules by relative path and commit; uninitialized directories preserve the original
initialization flow, while initialized checkouts must match the parent gitlink and
declared commit with a clean working tree.

Custom compositions retaining the legacy plugin are supported when they can be
reconstructed consistently. The legacy profile selects `startup`, implemented by
the selected Workflow plugin's startup adapter. Both Legacy and Light already use
Workflow providers. Bootstrap keeps the neutral Startup protocol.
Targets own forwarding `Invocation.environment` to children and persisting the selection
for delayed restarts; they do not retain Core's context or composition owner.

## Programmatic invocation and ownership

For a generic startup, use a launch profile and neutral invocation:

```python
from pathlib import Path
from aka.bootstrap.host import run_profile
from aka.contracts.startup import Invocation

exit_code = run_profile(Path("/path/to/profiles/launch.json"), Invocation(("arg1",)))
```

For the optimization CLI adapter, install the matching packages and provide the resource
checkout. Generic `aka run` does not add business checkouts to `sys.path`:

```python
from pathlib import Path
from aka.bootstrap.workflow_cli import main

exit_code = main(["--repo-root", "/path/to/aka", "--profile", "workflow",
                  "--problem", "/path/to/problem", "--platform", "H20",
                  "--framework", "Triton", "--workspace-base", "/path/to/runs"])
```

This is the same public-request parser as the installed optimizer. `main(None)` reads
`sys.argv[1:]`; passing a list supplies explicit arguments. Bootstrap owns composition
disposal on return or error. Disposal does not itself cancel detached or remote jobs.

Core freezes defaults/interpolation before setup.
Bootstrap captures the target, tokens, effective composition and identity of Core,
Bootstrap, Contracts, plugins and declared resources. Post-setup verification rejects
changes outside the frozen composition, including dynamic `Context.plugin()` children
not recorded in its rows.

## Current child and recovery behavior

### Launch selection environment variables

Bootstrap supplies these two keys in `Invocation.environment`:

| Variable | Value |
| --- | --- |
| `AKA_LAUNCH_SELECTION` | Path to the launch-selection JSON snapshot, not the original profile file |
| `AKA_LAUNCH_DIGEST` | SHA-256 digest of the snapshot's JSON content, encoded as 64 hexadecimal characters |

The snapshot records the selected target, effective plugin composition and configs,
bindings, profile-selected environment values or absence, and implementation/resource
identities. Reconstruction uses this snapshot rather than rereading the original
profile or patches. The digest checks the snapshot content; reconstruction also checks
that the current implementations and declared resources match the recorded identities.

Pass both keys together. If both are absent, no recorded selection was supplied;
explicit resume then fails. A missing or empty key, unreadable file or digest mismatch
fails before plugin setup. Forward the host-provided values rather than constructing
them from a profile path. When merging `Invocation.environment` into a child environment,
set string values and remove keys whose recorded value is `None`.

During an invocation, Bootstrap writes `selection.json` with mode 0600 in its private
temporary directory. The file is removed when that invocation ends, including failure
or interruption. A child must read it while the owning invocation is alive. For a
delayed restart, the startup implementation must persist the same content before the
invocation ends and replace `AKA_LAUNCH_SELECTION` with the durable path; copying the
same content leaves `AKA_LAUNCH_DIGEST` unchanged. The legacy adapter uses
`launch-selection.json` in its recovery directory and saves the pair in
`restart.json`'s `launch_environment`. The durable file remains available after the
original invocation ends.

With the recorded pair supplied in the environment, invoke generic reconstruction as:

```sh
aka run --resume -- startup-argument
```

`--resume` and `--profile` are mutually exclusive. Resume accepts no `--patch` or `--var`
overrides. `run_profile()` and `aka run --profile ...` reject inherited selection keys
instead of silently ignoring the explicit profile; use resume, or clear both keys to
start from a profile. The optimizer detects recorded selections automatically and requires the recorded
`--workflow-request`; profile and patch overrides are rejected.

These variables carry startup composition, not campaign progress or saved business
arguments. Pass the arguments needed by the selected target; that target owns workspace
recovery. Generic resume does not provision a legacy checkout or install internal
adapters. Internal optimizer recovery continues through its recorded internal launcher.

### Recovery compatibility

Resume requires the recorded public request, retained resources and matching implementation
identities. Changing package versions, workflows or resource contents is not an implicit
migration. Use a fresh workspace for an incompatible upgrade; retain the matching old
installation if an old campaign must finish.

The common Experiment retry ledger requires
`schema_version=2`; base-era ledgers are
rejected, and no automatic migration is provided. This restriction concerns continuation
of old in-flight recoveries, not creation of new experiments. It is separate from the
common infrastructure retry default: transient failures proven to precede submission permit
two extra attempts after 5/10 seconds, within a 120-second recovery admission window.
Workflow review-service loops have their own policies.

The removed detached monitor and its restart-schema migration protocol are described in
[the pre-migration guide](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/application-plugin-migration.md#legacy-recovery-records).
That link preserves the old documentation text, not a runnable interface at that SHA.
The monitor belongs to the earlier implementation; this installed command set removes it. Existing
resource locks likewise require their original identities; do not rewrite them to bypass checks.

## Distribution

| Distribution | Ownership | Python |
| --- | --- | --- |
| `atrex-aka-framework` | Neutral dispatcher, Core, Contracts, Bootstrap and shared Agent, Execution, Evaluation, Experiment, Task and Resources services | 3.10+ |
| `atrex-aka-light` | Light Workflow, orchestration, profiles and runtime assets | 3.10+ |
| `atrex-aka-dashboard` | `aka.dashboard`, Light observability tools and the read-only viewer | 3.10+ |

The current release is `0.3.0`. Light pins `atrex-aka-framework==0.3.0`;
Dashboard pins both `atrex-aka-framework==0.3.0` and `atrex-aka-light==0.3.0`.
Framework also requires `packaging>=23`, and Light requires
`atrex-gateway-client>=0.14.30,<0.15`.

These three distributions replace the former Core, Contracts, Bootstrap, Optimization,
Legacy and long-horizon observation distributions. Remove the old distributions before
installing their replacements to avoid overlapping files. Current distributions own
disjoint files. Build tooling stages owned files, builds sdists, then builds wheels
from those sdists.

Dashboard is an independent application, launched with
`python -m aka.dashboard <campaign-or-snapshot>`. It reads Light Campaign evidence
through the packaged observability tools; it is not a Bootstrap plugin and does not
provide an `observation_source` token or startup profile. See the
[dashboard guide](optimization-dashboard.md) for source and installed launch commands.

The native command constructs a public Workflow request once, then dispatches the
selected installed Workflow through Bootstrap. Execution, Resources, Agent, and Workflow
already own their packaged implementations and runtime assets; removed checkout scripts
are not alternative launch interfaces.

See [the framework interface](plugin-framework.md) for dependency, scope, event,
configuration and cleanup semantics.
