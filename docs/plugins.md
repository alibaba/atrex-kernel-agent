# AKA local plugins

AKA loads local plugins from an explicit JSON configuration. GPU Wiki is the default
built-in plugin, exposed as `gpu-wiki.query`. A plugin provides tools, instructions,
resources and optional skills; the orchestrator handles discovery and workspace wiring.

## Two independent extension points

A plugin may provide **tools**, **Skills**, or both. A Skill-only package needs no executable,
input/output schemas, or dummy tool. Resources and instructions support these two contributions.

| Layer | Responsibility |
|---|---|
| `plugin_runtime/` | Validate declarations, discover tools/Skills, execute commands, install links and check locks |
| `orchestrator/plugins.py` | AKA defaults, CLI guidance, Agent discovery paths and campaign context |
| Plugin package | Domain behavior, tool implementation, Skill content and optional scoped instructions |

The reusable runtime imports no orchestrator modules and contains no GPU/Wiki policies, campaign
phases or Agent directory conventions. Another host supplies `HostLayout` with its own state
location, Skill discovery directories and reserved workspace paths. Instruction scope names are
host-defined strings; the runtime does not interpret their business meaning.

Tool implementations use a command argument array with JSON stdin/stdout. Python, Node, shell
scripts and compiled executables share the same interface. No shell expansion is performed on
arguments or user requests. The existing Python `entrypoint` declaration is still accepted.
Plugins execute as trusted local code; resource links are not an execution sandbox.

## Query GPU Wiki

From the repository or an initialized campaign workspace:

```bash
python3 tools/plugin.py list
```

Write `wiki_request.json`:

```json
{
  "request": "Target hardware B200, DSL triton. Optimize operator rmsnorm and retrieve techniques and pitfalls.",
  "max_records": 6
}
```

```bash
python3 tools/plugin.py call gpu-wiki.query --input wiki_request.json
```

`list` returns separate `tools` and `skills` catalogs. Each Skill has a namespaced ID, native
name, description and source path. `--input -` reads JSON from stdin. The response preserves the existing Wiki envelope:
`query_id`, `records`, and `notes`. Canonical `wiki_id` values, payloads and public/internal
store isolation are unchanged. `max_bytes` and `exclude` are also supported input fields.
The standard episode request uses the existing deterministic parser; other prose can still
invoke the Wiki's existing bridge agent. The plugin does not introduce a new model dependency.

The plugin declares both the public store and optional `internal_gpu_wiki` sibling as data
dependencies. A conflicting `ATREX_WIKI_STORE_ROOT` is rejected instead of silently selecting
an undeclared store. To use another local dataset, copy/configure the Wiki plugin's resource
paths (keeping the two stores as siblings) and enable that plugin in a new campaign.

Campaign prompts call this tool. Direct Wiki scripts remain available for standalone use
and maintenance. Mining and admission through `wiki-gate` remain separate from this query tool.

## Enable plugins per campaign

The default is `plugins/default.json`, which enables the bundled GPU Wiki. Pass
`--plugin-config /absolute/path/to/plugins.json` to `orchestrator/optimize.py` to select
another configuration. Paths inside a configuration resolve relative to that file.
For example, a `my-plugins.json` at the repository root can contain:

```json
{
  "schema_version": 1,
  "plugins": [
    {"path": "plugins/gpu-wiki", "enabled": true},
    {"path": "/opt/aka-plugins/local-docs", "enabled": true}
  ]
}
```

Entries are enabled by default. Disabled entries are not loaded and may refer to missing
directories. An empty `plugins` array disables all plugins. A package with neither tools nor Skills is rejected. Required hardware evidence
remains a campaign policy; with no suitable knowledge tool, agents consult available
reference sources and keep missing specifications explicitly unknown.

For a standalone tool invocation using a custom configuration:

```bash
python3 tools/plugin.py --config my-plugins.json list
python3 tools/plugin.py --config my-plugins.json call local-docs.query --input request.json
```

In a campaign, the CLI finds the configuration through the environment or the workspace's
plugin lock. An explicit override must match that lock. Resume with the same configuration.

## Add another plugin

A plugin directory might contain:

```text
local-docs/
├── plugin.json
├── instructions.md
├── query.py
├── input.json
├── output.json
└── data/
```

`plugin.json`:

```json
{
  "id": "local-docs",
  "version": "1.0.0",
  "api_version": 1,
  "tools": {
    "query": {
      "description": "Retrieve local reference facts.",
      "command": ["{python}", "{plugin_root}/query.py"],
      "input_schema": "input.json",
      "output_schema": "output.json",
      "timeout_seconds": 30
    }
  },
  "instructions": {"common": "instructions.md"},
  "resources": {"local-docs-data": "data"}
}
```

`input.json`:

```json
{
  "type": "object",
  "required": ["request"],
  "additionalProperties": false,
  "properties": {"request": {"type": "string", "minLength": 1}}
}
```

`output.json` can begin as `{"type": "object"}` and add domain-specific constraints.
A minimal `query.py` demonstrating the execution contract:

```python
import json
import os
import sys
from pathlib import Path

request = json.load(sys.stdin)
root = Path(os.environ["PLUGIN_ROOT"])
text = (root / "data" / "reference.txt").read_text()
print(json.dumps({"source": "reference.txt", "text": text}))
```

Populate `data/reference.txt`, describe when to use the tool in `instructions.md`, and
add the plugin path to the campaign configuration. No orchestrator branch or tool-specific
registration code is needed. Multiple tools can use separate entrypoint files.

## Skill-only package

```json
{
  "id": "document-review",
  "version": "1.0.0",
  "api_version": 1,
  "skills": {
    "review-document": {
      "path": "skills/review-document",
      "description": "Review a document for clarity and consistency."
    }
  }
}
```

Place the original `SKILL.md` and its supporting scripts/references in that directory.
The runtime links the whole directory without rewriting its content or native name.
`document-review.review-document` is its catalog ID; `review-document` is the native name.
Conflicting installation names fail explicitly rather than silently shadowing another Skill.

## Tool settings

A configuration entry may include `"settings": {"endpoint": "https://example.invalid"}`.
A plugin may declare `"settings_schema": "settings.schema.json"` to validate that object
at load time. Tools read their own configuration from `PLUGIN_SETTINGS_JSON`; `PLUGIN_ROOT`
identifies the package directory. Settings are passed to tool processes, not copied into
instructions or call logs. Locks store a digest so changing settings is detected. Keep
credentials out of command arguments, which are stored as part of the runtime lock.

## Use from another host

```python
from pathlib import Path
from plugin_runtime import HostLayout, PluginRegistry

registry = PluginRegistry(
    Path("plugins.json"),
    layout=HostLayout(state_dir=".extensions", skill_roots=("native/skills",)),
)
registry.install(Path("workspace"))
tools = registry.catalog()
skills = registry.skill_catalog()
instructions = registry.instructions("document-review", DOCUMENT_KIND="proposal")
```

This does not require AKA or an installed coding-agent CLI. The host selects tools to call
and scopes to render; tool and Skill implementations stay in the plugin package.

## Manifest contract

- IDs and tool names use lowercase letters, digits and hyphens, starting with a letter.
  The published tool name is `<plugin-id>.<tool-name>`. Duplicate plugin IDs fail at load.
- `api_version` is the integer `1`; `version` identifies the local plugin release.
- Each tool has a description, a `command` argv array, input/output schema files and an integer
  timeout from 1 to 3600 seconds. `command` supports `{python}` (current interpreter) and
  `{plugin_root}` placeholders. The executable must be available when loading. Arguments are
  passed literally; request fields are never interpolated. `entrypoint` is the compatible Python
  shorthand and is mutually exclusive with `command`. Schema/instruction files stay inside the
  plugin directory. Script/library dependencies belong in the plugin or declared resources.
- `instructions` maps arbitrary scope names to files, with `common` included in every scope.
  The host chooses scope names and template values. AKA currently supplies `setup`, `episode`,
  `fast_episode`, `framework_baseline` and values such as `{{PLATFORM}}`, `{{ARCH}}`,
  `{{FRAMEWORK}}`, `{{OPERATOR}}`; these names are not runtime restrictions.
- `resources` maps workspace names to local source paths, relative to the plugin root.
  Explicit external paths are supported for existing data trees such as `../../gpu-wiki`.
  A value can also be `{"path": "../optional-data", "optional": true, "mount": false}`:
  optional paths may be absent, and `mount: false` fingerprints a dependency without exposing
  another workspace link. Missing-to-present changes are detected by the lock.
  Conflicting mounts and attempts to replace existing workspace files fail.
- `skills` maps native skill names to `{"path": "skill-directory", "description": "When to use it"}`.
  Skills are required by default; `optional: true` is supported. The host chooses installation
  roots. AKA supplies `.claude/skills`, `.qoder/skills`, and `.agents/skills`. Optional skills
  install only when their `SKILL.md` exists. No submodule fetch or package installation is
  triggered for optional skills; an existing KernelWiki checkout remains usable.
- `environment` contributes variables to campaign agent sessions. Values may contain
  `{workspace}` (the incumbent campaign workspace) and `{campaign_name}`. Conflicts between
  plugin declarations or with the runner's reserved context variables fail during loading.

The v1 schema dialect supports `type` (`object`, `array`, `string`, `integer`, `number`,
`boolean`, `null`), `description`, `properties`, `required`, `additionalProperties`
(boolean or schema), `items`, `enum`, `minLength`, `minimum` and `maximum`.
Types are mandatory; arrays require an item schema. `minLength` checks trimmed strings,
so whitespace-only required requests fail. Unknown keywords and constraints used on an
incompatible type fail at load. This is an explicit subset, not full JSON Schema support.

## Runtime and recovery

At initialization the registry validates enabled plugins and installs resources, skills,
and `.atrex_plugins/instructions.md`. The campaign records `.atrex_plugins/lock.json` with
configuration/root paths, host layout, resolved executable paths, settings digests, versions
and SHA-256 fingerprints of plugin code, schemas,
instructions, resources and skills. Git metadata and Python caches are excluded. A changed
plugin set, path, version or content fails resume and invocation checks; restore the original
configuration/code or start a new campaign. Plugin code/data directories should be immutable
during a campaign. These checks detect changes; they do not snapshot or sandbox the files.

Episode worktrees install the same locked plugin set. Tool execution keeps the invoking
episode's current directory. `ATREX_PLUGIN_WORKSPACE` points to the incumbent campaign for
lock checks and call logs, so tools must use their actual current directory for episode work.
Plugin environment contributions such as the existing Wiki profile root are supplied by
the campaign. Standalone calls do not automatically create a campaign or a plugin lock.

Successful tool JSON is returned without an additional envelope after schema validation. Errors have the
shape `{"error":{"code":"tool_failed","message":"..."}}` and a nonzero CLI exit code:

| Code | Meaning |
|---|---|
| `invalid_config` / `invalid_manifest` | Missing plugin or invalid declaration |
| `plugin_changed` | Current plugins differ from the campaign lock |
| `mount_conflict` | A resource would replace workspace content |
| `tool_not_found` | The requested tool is not enabled |
| `schema_validation` | Request violates the tool's input schema |
| `tool_failed` | Tool process could not execute or exited unsuccessfully |
| `tool_timeout` | Tool exceeded its deadline; its process group was terminated |
| `invalid_output` | Tool returned non-JSON or violated its output schema |

An empty Wiki `records` map remains a successful query; inspect its `notes` for scope and
store diagnostics. The wrapper preserves the Wiki's existing partial-store behavior.
Resolved executable paths are locked; externally installed runtime binaries are not content-snapshotted.
Campaign call events under `.atrex_plugins/calls/` record call ID, tool, version, status
and duration, with no request or response payload. Telemetry write failure is diagnostic.
Existing Wiki profiling and experiment `wiki_usage` attribution remain intact. Experiments
with no Wiki query or reconsidered response use `not_queried`; retrieval never implies adoption.

Existing pre-plugin workspaces adopt the configured plugins on their first updated run.
Migration removes only old runtime-created Wiki/KernelWiki symlinks that are no longer
declared; user-owned directories remain untouched. Runtime links and state are Git-ignored.

## Validation

```bash
python3 -m unittest orchestrator.test_plugins long_horizon.test_journal_wiki_attribution
python3 -m unittest discover -s gpu-wiki/tools
```

The integration tests load an unrelated local reference plugin without changes to the
registry, cover enabled/disabled wiring and recovery, and compare a real deterministic
Wiki query against the direct CLI while no bridge executable is available on PATH.
