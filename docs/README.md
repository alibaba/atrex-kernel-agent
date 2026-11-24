# Documentation

New optimization launches use **Light**. Start with the Quick Start, then read the architecture
and module options for the selected workflow. The current CLI uses `--problem` and `--workspace-base` without an optimizer `--` separator.
[Installed startup](application-plugin-migration.md) documents the migration boundary.
Historical release notes and validation reports
describe the versions and environments identified in those documents.

## Start and configure

| Guide | Scope |
| --- | --- |
| [Quick Start](quickstart.md) | Install matching packages, prepare an operator, launch Light, inspect outputs |
| [Using DSH](dsh-harness-adapter.md) | Choose DSH as the coding agent, reuse a provider profile, and follow Session progress |
| [Architecture Design](design.md) | Current modules, authority boundaries, persistent Sessions, accepted submissions |
| [Light modules and profiles](optional-light-modules.md) | Provider options, Wiki, search, source imports, budgets, recovery |
| [Installed startup](application-plugin-migration.md) | CLI/profile selection, package ownership, frozen continuations |
| [Native CUDA and source layouts](native-cuda.md) | Source trees, single-file and CUDA manifests, source identity |
| [Agent workspace isolation](agent-workspace-isolation.md) | Native/Bubblewrap selection, Homes, mounts, private/host proc modes |

## Runtime and extensions

| Guide | Scope |
| --- | --- |
| [Tool Runtime](supervisor-runtime.md) | Agent-scoped tool authority, GPU/Wiki tools, public results and deadlines |
| [Measurement records](measurement-records.md) | Source/task identity, deduplication, result queries and GPU recovery |
| [Research and submissions](design.md) | Versioned Direction/Experiment records and exact-commit submission audit |
| [Session observability](session-observability.md) | Shared Agent observations, current configuration and historical capture |
| [Core plugin framework](plugin-framework.md) | Composition, service contracts, lifecycle and ownership |
| [Local plugins](plugins.md) | Skill/tool extensions and Runtime invocation paths |
| [GPU Wiki](../gpu-wiki/README.md) | Current structured stores and standalone knowledge interfaces |
| [Dashboard](optimization-dashboard.md) | Optional viewer, lineage, Kernel, token and call evidence |
| [Community local gateway](local_gateway.md) | Local source-archive Evaluate and ABBA |
| [Light OpenSSH backend](ssh-backend.md) | Native Evaluate/ABBA, GPU diagnostics, vendor selection and isolation modes |

## Release and validation records

- [v0.4.1 release notes](releases/v0.4.1.md)
- [v0.4.0 release notes](releases/v0.4.0.md)
- [v0.3.0 release notes](releases/v0.3.0.md)
- [Evaluator outcomes and diagnostics](evaluator-outcomes/README.md)
- [Local source-tree evaluation and terminal evidence](local-gateway-source-evaluation.md)
- [Historical Session capture cost](session-capture-performance.md)
- [Source-layout GPU evidence](native-cuda.md#agent-optimization-evidence)
- [Light implementation verification](optional-light-modules.md#research-and-parallel-search)
