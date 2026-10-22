# Core Agent resources and Wiki selection

New Workflow campaigns use `aka.resources` and the `agent_resources` service.
Core plugins contribute `ToolSpec`, `InstructionSpec`, workspace resources, Skills,
environment bindings and attribution providers through `ctx.contribute(...)`.
The host freezes the settled Scope, then binds it to one canonical workspace and
campaign before the Agent starts. Core itself has no Wiki or Workflow dependency.

Install the matching `atrex-aka-framework`, `atrex-aka-resources` and
`atrex-aka-legacy` distributions. Add `atrex-aka-wiki` and the separately exported
public Wiki data when selecting public Wiki. Unselected Wiki packages and data
are optional. All these distributions currently declare version `0.1.0`; frozen
selections additionally verify exact code and data digests.

The public default `aka workflow` keeps public Wiki enabled. Explicit selection
uses the existing launcher:

```sh
TASK_PROFILES="$(python -c 'from importlib.resources import files; print(files("aka.legacy.workflow").joinpath("profiles"))')"
# No Wiki package or Wiki data is needed for this check.
aka run --profile "$TASK_PROFILES/workflow-no-wiki.json" --check --
# This root contains the selected gpu-wiki/ data directory.
aka run --profile "$TASK_PROFILES/workflow-public-wiki.json" \
  --var "repo_root=/opt/aka/public-resources" --check --
```

To start a campaign, replace `--check --` with `-- --workflow-request "$TASK_REQUEST_JSON"`.
The request is the complete Workflow launch JSON. Profile names are file paths;
the launcher does not search for arbitrary aliases. A selected missing package,
resource or variable is an error. Existing campaigns resume their frozen selection:

```sh
aka run --resume -- --workflow-request "$TASK_REQUEST_JSON"
```

Resume requires the paired `AKA_LAUNCH_SELECTION` and `AKA_LAUNCH_DIGEST` supplied
by the owning run. It cannot accept a replacement profile, variable, patch or
`--check`. Checkpoint restoration requires the original absolute paths and exact
matching code/data artifacts. Runtime plugin replacement and relocation are not
supported.

Tools consume the frozen plan directly, without booting Workflow:

```sh
aka tools list
aka tools call gpu-wiki.query --input request.json
python -m aka.resources.cli call gpu-wiki.query --input - < request.json
```

The new call returns `aka.tool-result.v1` with `data`, `receipt`, `references`,
`coverage` and `diagnostics`. The old `python -m aka.legacy.resources.cli` and
`PluginRegistry` Python facade remain available: in a Core-bound campaign, calls
return the raw business `data` and old catalog/error shapes. Plugin-directory
overrides cannot replace a frozen Core selection. Explicit standalone legacy
catalogs retain the manifest implementation; loading an independent legacy
catalog together with Core Agent resources is rejected.

Entrypoints have signature `invoke(request, context)` and receive finite JSON
objects. `ToolContext` describes the context mapping: `call_id`, canonical
`workspace`, `campaign_id`, declared `resource_paths`, selected `environment` and
tool `config`. The unbound service offers `freeze`, `bind` and `restore_bound`;
the returned `BoundAgentResources` offers catalogs, calls, rendering, installation,
attribution validation and artifact descriptors. Phase names belong to the
consumer. `common` applies to every declared phase; templates perform only
restricted `{variable}` substitution.

Durable resource bundles live at `.aka/agent-resources/<campaign>/` and contain
`resource-snapshot.json`, `tool-plan.json` and the final v3 `launch-selection.json`.
Each call has a distinct receipt and source index under `.atrex_plugins/`.
Receipts contain identities and bounded source metadata, rather than knowledge
payloads. Tool failures are not retried automatically. Interrupted pending calls
remain unknown. Evidence write/provider failures lower coverage and produce
diagnostics; they do not change correctness, performance or promotion verdicts.
Identity mismatches still prevent execution.

Workspace installation excludes selected runtime artifacts before candidate
commits, preserves native Skill names in all three Agent roots, and accepts only
the canonical repository or its registered episode worktrees. Shared Wiki data
is borrowed and is never deleted by disposal. New resource-bound checkpoints use
Workflow checkpoint v2; legacy checkpoint v1 and continuation v1/v2 readers remain
separate. Keep the original matching engine and artifacts for existing campaigns.

## Light

Install `atrex-aka-light` with the matching runtime distributions. Light uses the
same Core host and Wiki plugin; the Supervisor retains the frozen plan and data.
The Agent receives copied Skills, phase instructions, a tool catalog and a scoped
HTTP capability. Tool execution and reference extraction share Light's process
group, execution slot, request deadline and cancellation control. Runtime plans,
host resource paths and credentials are excluded from Agent projections.

The public `light.json` and `light-sandbox.json` defaults select public Wiki.
Select the no-Wiki profile without installing the Wiki package or data:

```sh
LIGHT_PROFILES="$(python -c 'from importlib.resources import files; print(files("aka.light").joinpath("profiles"))')"
aka run --profile "$LIGHT_PROFILES/light-no-wiki.json" \
  --var "repo_root=/opt/aka" --check --
aka run --profile "$LIGHT_PROFILES/light-public-wiki.json" \
  --var "repo_root=/opt/aka/public-resources" --check --
```

The internal distribution supplies the corresponding no/public/internal/both
profiles and retains both Wiki providers as its default. The selected Agent
sandbox is independent of Wiki selection. Inside a Light session use
`python3 tools/plugin.py list` and
`python3 tools/plugin.py call gpu-wiki.query --input scratch/request.json`.
Calls return the complete `aka.tool-result.v1` envelope. Journal experiments
accept generic `attribution` declarations verified against the Supervisor's
campaign receipts and retain the old Wiki fields as a compatibility projection.

Frozen launch recovery uses the original code/data and selection. Light's own
Episode Journal and sealed candidate recovery remain authoritative; this bridge
does not add Legacy Agent session checkpoint export to Light. Existing manifest
integrations can select `light-legacy-resources.json`; Core mode does not mix an
independently discovered manifest registry into the selected tool catalog.
