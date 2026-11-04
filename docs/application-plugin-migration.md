# Run AKA through profile-selected startup

AKA uses the installed `aka optimize` command. It converts the public request fields once,
then uses a launch profile to select Workflow through Bootstrap and Core. Evaluation, Execution,
Resources and recovery run from their installed owning modules.

For fresh `aka optimize` and `aka workflow` launches, the default profile is now `light`.
It requires the matching `atrex-aka-light` distribution and selects native (not Bubblewrap)
Agent execution and managed Light Workflow services. Use `--profile light-sandbox`
for isolation, or `--profile workflow` for the previous Legacy behavior. Missing Light is
an actionable startup error, not an implicit Legacy fallback. `aka run` remains explicit.
Continuations use their saved selection regardless of this new-launch default; do not
switch a live Legacy workspace to Light by omitting the profile on a manual restart.

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
separator. Their workflow applicability is profile-specific: Light uses unified Episodes,
not Legacy Fast/Full stages. Light's source-tree and ABBA path requires native Agate
Atrex-Bench; select the [Legacy workflow](legacy-workflow.md) for SOL/SSH campaigns. Profile-selected environment values and absence are captured for children and recovery.
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
revision; the `0.2.0` package version alone does not identify a source revision or repository release.
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

When explicit retries are enabled, the common Experiment retry ledger requires
`schema_version=2`; base-era ledgers are
rejected, and no automatic migration is provided. This restriction concerns continuation
of old in-flight recoveries, not creation of new experiments. It is separate from the
common infrastructure retry default, whose empty delay schedule does not retry transient
failures. Workflow-specific loops may have their own policies.

The removed detached monitor and its restart-schema migration protocol are described in
[the pre-migration guide](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/application-plugin-migration.md#legacy-recovery-records).
That link preserves the old documentation text, not a runnable interface at that SHA.
The monitor belongs to the earlier implementation; this installed command set removes it. Existing
resource locks likewise require their original identities; do not rewrite them to bypass checks.

## Distribution

| Distribution | Ownership | Python |
| --- | --- | --- |
| `atrex-aka-framework` | `aka`, neutral dispatcher, `aka.core`, `aka.contracts`, `aka.bootstrap`, generic `aka run` command | 3.10+ |
| `atrex-aka-legacy` | Shared business providers, explicit Legacy Workflow, `optimize` / `workflow` / `execution` commands | 3.10+ |
| `atrex-aka-light` | Light Workflow, orchestration and default Light profiles; common services come from Legacy | 3.10+ |
| `atrex-aka-dashboard` | `aka.dashboard`, read-only observer and independent startup profile | 3.10+ |
| `atrex-aka-observation-long-horizon` | `aka.observation.long_horizon`, workflow-specific observation adapter | 3.10+ |

Framework has no runtime package dependencies. Light depends on `atrex-aka-legacy==0.2.0`.
Legacy and both dashboard/observation distributions depend on `atrex-aka-framework==0.2.0`; Dashboard's `long-horizon` extra adds
`atrex-aka-observation-long-horizon==0.2.0`. Neither optional distribution requires
Legacy. Although the observation adapter imports only Contracts interfaces, its
installation includes the full Framework distribution, including Core, Bootstrap and
the CLI. This larger install footprint is a consequence of consolidating the base
packages into Framework and Legacy; module interfaces remain separate within them.

The consolidated packages replace the old Core, Contracts, Bootstrap and Optimization
distributions. Remove the old distributions before installing their replacements to
avoid overlapping files. The five current distributions own disjoint files and pin
matching dependency versions. Distribution boundaries define installation ownership;
they do not require a separate distribution for every future module. Build tooling
stages owned files, builds sdists, then builds wheels from those sdists.

`aka.observation` deliberately uses an implicit namespace package. Each observation
adapter distribution owns its leaf package, such as `aka.observation.long_horizon`;
there is no shared parent `__init__.py` for independently installed adapters to overwrite.
The long-horizon package's discovery include and the build tool's `OWNERS` entry both
select that leaf. The `aka.contracts` module package belongs to the Framework distribution.

Dashboard injects the versioned `observation_source` seam. Its default profile combines
the viewer with the separately packaged long-horizon adapter, without importing the
optimizer. The optimization profile does not load either plugin. See the
[dashboard guide](optimization-dashboard.md) for source and installed launch commands.

The native command constructs a public Workflow request once, then dispatches the
selected installed Workflow through Bootstrap. Execution, Resources, Agent, and Workflow
already own their packaged implementations and runtime assets; removed checkout scripts
are not alternative launch interfaces.

See [the framework interface](plugin-framework.md) for dependency, scope, event,
configuration and cleanup semantics.
