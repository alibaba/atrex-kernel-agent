# Light composition and optional research

AKA provides only Light orchestration. Public Agent, Task, Evaluation, Execution, Experiment
and Resources services are shared module owners; there is no Legacy launch profile.

| Profile | Agent boundary |
|---|---|
| `light` | Native process, capture enabled |
| `light-sandbox` | Bubblewrap, capture enabled |

Bubblewrap requires Linux. On macOS use a Lima VM or native execution. Proc mode is
`private` or `host`; Aone Sandbox requires `host`. Final-source policy is independent of
this boundary: Fast, Performance and Ultra work with either profile.

## Research and parallel search

Direction and Experiment are disabled by default. Select them in a composition patch:

```json
{
  "api_version": 1,
  "patches": [{"id": "workflow", "config": {
    "tool_modules": ["directions", "experiments"],
    "search": {"trajectories": 3, "epochs": 5, "episodes_per_trajectory": 1,
               "max_parallel_episodes": 3, "synchronization": "asynchronous"}
  }}]
}
```

An empty `search` object selects serial execution. Use `synchronous` to publish at an Epoch
barrier. Role specialization is disabled by default. Each patch replaces the selected
module configuration, then applies module defaults; when patching Agent isolation, preserve
all intended Agent options explicitly.

```bash
aka optimize --repo-root /path/to/aka --profile light --patch /path/to/research.json \
  --problem /path/to/atrex-bench/data/operator --workspace-base /path/to/runs \
  --platform L20N --optimization-mode ultra --framework cuda --max-iters 5
```

Research history lives in the Campaign library. Each Agent sees its own revisions immediately
and peer revisions according to the publication watermark. Peers can derive new entries but
cannot revise another owner's entry. Records referenced by visible research become visible
with that research, subject to exact source availability. All measurements use one central
Campaign store; projections are not independent authorities.

See [design](design.md), [runtime](supervisor-runtime.md), and the
[launch Skill](../skills/atrex-aka-light/SKILL.md) for lifecycle and operator intake.
