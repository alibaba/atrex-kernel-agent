# Optimize kernels with DSH

AKA can use [DeepSeek Harness (DSH)](https://github.com/deepseek-ai/deepseek-harness)
as its coding agent. DSH reads the operator, develops the implementation, and
uses AKA's tools to evaluate and refine it. AKA manages GPU execution,
correctness checks, performance comparisons, and promotion of the selected
implementation.

Choose DSH with `--agent-cli dsh`. The adapter is available in the default Light
workflow and the explicit Legacy workflow.

## Get ready

Follow the [Quick Start](quickstart.md) to install AKA, configure GPU access, and
prepare an operator. On the coordinator, install DSH using its
[setup instructions](https://github.com/deepseek-ai/deepseek-harness) and configure
authentication for your chosen provider. Keep `dsh` on `PATH` so AKA can launch it.

```sh
dsh --version
dsh --profile headless --help
```

The adapter uses DSH's headless runner and JSON event stream. An existing DSH
profile works too when it includes the headless runner. Select that profile and
its provider and model in the settings below.

## Start an optimization

This example uses the default headless profile with the official DeepSeek route:

```sh
export ATREX_DSH_SESSION_SETTINGS='{"provider":"deepseek-official","model":"deepseek-flash"}'

aka optimize --repo-root /path/to/aka -- \
  --op-dir /path/to/operator \
  --platform '<gpu-target>' --framework Triton \
  --agent-cli dsh --workspace /path/to/runs
```

Use the GPU target configured for your AKA environment and the framework used by
your operator. The operator and its inputs follow AKA's existing problem format.
See [Inspect outputs](quickstart.md#inspect-outputs) for the retained source,
verification results, and Session observations.

To use a provider already configured in another headless profile, fill in its
profile, provider, and model names:

```sh
export ATREX_DSH_SESSION_SETTINGS='{
  "profile": "my-headless",
  "provider": "my-provider",
  "model": "my-model",
  "reasoning_effort": "default"
}'
```

Then run the same `aka optimize` command. Authentication stays with DSH's normal
provider configuration.

## Choose a profile and model

`ATREX_DSH_SESSION_SETTINGS` accepts a JSON object with these optional fields.
`ATREX_SESSION_SETTINGS` provides the shared fallback.

| Setting | Default | Purpose |
| --- | --- | --- |
| `profile` | `headless` | Select a DSH profile that includes the headless runner. |
| `provider` | `deepseek-official` | Select a provider route configured in DSH. |
| `model` | `deepseek-flash` | Select a model available on that route. |
| `reasoning_effort` | AKA's requested effort | Set a supported effort, or use `default` to follow the provider's configuration. |

For the official DeepSeek route, AKA maps `medium` to `high` and `xhigh` to
`max`. Other requested levels pass through to DSH. An explicit `reasoning_effort`
selects the value sent to the route. `default` is useful for models whose DSH
configuration exposes provider defaults rather than selectable effort levels.

AKA uses `DSH_HOME` when it is set and otherwise selects `~/.dsh`. If your DSH
launcher uses a different home, set its absolute path before starting AKA:

```sh
export DSH_HOME=/path/to/dsh-home
```

Keep the same model settings, DSH home, and AKA composition when continuing a
Campaign. AKA retains the native DSH Session identity for same-session
continuation and checkpoints.

## What to expect during a run

DSH works through the same evaluation, profiling, knowledge, and report tools
used by AKA's other coding agents. The selected AKA profile controls which tools
are available. Skills come from the managed workspace, and GPU requests stay in
the foreground while AKA tracks their progress.

With Session capture enabled, AKA records the conversation, tool activity,
native Session family, and reported token usage. Available counters retain their
input, output, and cache categories; incomplete samples carry a partial or
unavailable measurement label.

AKA launches DSH with its unattended permission preset. The coordinator isolation
option remains part of AKA's configuration. See
[Agent workspace isolation](agent-workspace-isolation.md) for Linux Bubblewrap
requirements and installation grants for custom launchers.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `dsh` is unavailable | Check that the coordinator's `PATH` includes your DSH launcher. |
| Provider or model lookup fails | Use the route and model names from the selected DSH profile. |
| Authentication fails (`MISSING_CREDENTIAL`) | Check that profile's provider authentication with DSH on the coordinator, using the same `DSH_HOME`. |
| `UNSUPPORTED_REASONING_EFFORT` | Choose a level supported by the model, or set `reasoning_effort` to `default`. |
| A Session ends with an error or incomplete event stream | Read the captured provider stderr and Session observations, correct the reported cause, and continue through AKA's normal recovery flow. |

## Adapter internals and verification

AKA owns the optimization workflow and DSH owns the model loop. DSH's Cordis
plugins run inside the DSH process. This adapter connects the two through the
existing AKA Agent interface:

```mermaid
flowchart LR
  W[AKA Workflow] --> A[DSH adapter]
  A <-->|prompt and JSON events| D[DSH headless]
  D --> C[AKA tool clients]
  C --> R[AKA Tool Runtime]
  R --> E[Evaluation, Execution, Wiki, Journal]
```

An adapter-owned overlay configures workspace Skills, foreground shell calls,
and uncompressed native logs under `$DSH_HOME/aka-sessions`. A private,
invocation-scoped connection file carries the AKA Session capability to DSH's
shell children. Managed Campaigns create it after the Runtime mints the
capability and remove it when the invocation finishes.

Successful completion requires both a zero process exit status and a matching
DSH `final` event with `reason.kind=completed`. Native checkpoints retain the
selected Session family and its latest log generation. Usage accounting combines
reported root steps with newly observed child attempts and excludes prior
Session history.

The protocol was verified with DSH `0.2.2-alpha.2`. Run the automated checks from
the AKA checkout:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The opt-in integration check launches the installed DSH CLI against local model
and Runtime fixtures. It exercises editing, Skills, tool calls, capture, token
accounting, credential handoff, and native Session continuation:

```sh
AKA_RUN_DSH_INTEGRATION=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The fixtures use synthetic credentials and loopback services. Integration
coverage currently exercises the native coordinator runner; Linux Bubblewrap
validation is tracked separately. Custom Cordis extensions can build on this
adapter in a later phase.
