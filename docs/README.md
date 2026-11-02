# Documentation

New optimization launches use **Light**. Start with the Quick Start, then read the architecture
and module options for the selected workflow. Historical release notes and validation reports
describe the versions and environments identified in those documents.

## Start and configure

| Guide | Scope |
| --- | --- |
| [Quick Start](quickstart.md) | Install matching packages, prepare an operator, launch Light, inspect outputs |
| [Architecture Design](design.md) | Current modules, authority boundaries, Episode lifecycle, promotion |
| [Light modules and profiles](optional-light-modules.md) | Provider options, Wiki, search, source imports, budgets, recovery |
| [Installed startup](application-plugin-migration.md) | CLI/profile selection, package ownership, frozen continuations |
| [Native CUDA and source layouts](native-cuda.md) | Source trees, single-file and CUDA manifests, source identity |
| [Agent workspace isolation](agent-workspace-isolation.md) | Native/Bubblewrap selection, Homes, mounts, projected proc |
| [Supplemental numerical validation](numerical-validation.md) | Optional production risk probes and repair semantics |

## Runtime and extensions

| Guide | Scope |
| --- | --- |
| [Tool Runtime](supervisor-runtime.md) | HTTP authority, GPU/Wiki tools, public results and deadlines |
| [Measurement records](measurement-records.md) | Source/task identity, deduplication, result queries and GPU recovery |
| [Runtime Journal](runtime-journal.md) | Default report-only workflow and optional research annotations |
| [Handoff and promotion](supervisor-promotion.md) | Source sealing, acceptance, ABBA, promotion-audit repair |
| [Session observability](session-observability.md) | Capture configuration, transcripts, usage and failure boundaries |
| [Core plugin framework](plugin-framework.md) | Composition, service contracts, lifecycle and ownership |
| [Local plugins](plugins.md) | Skill/tool extensions and distinct Light/Legacy invocation paths |
| [GPU Wiki](../gpu-wiki/README.md) | Current structured stores and standalone knowledge interfaces |
| [Dashboard](optimization-dashboard.md) | Optional viewer, current memory-history and multi-file export limitations |
| [Community local gateway](local_gateway.md) | Local source-archive Eval/ABBA and Legacy request compatibility |

## Legacy and compatibility

| Guide | Scope |
| --- | --- |
| [Legacy workflow](legacy-workflow.md) | Explicit `--profile workflow`, Fast/Full stages, SOL, OpenSSH and recovery |

## Release and validation records

- [v0.4.0 release notes](releases/v0.4.0.md)
- [v0.3.0 release notes](releases/v0.3.0.md)
- [Evaluator outcomes and diagnostics](evaluator-outcomes/README.md)
- [Local source-tree evaluation and terminal evidence](local-gateway-source-evaluation.md)
- [Historical Session capture cost](session-capture-performance.md)
- [Source-layout GPU evidence](native-cuda.md#agent-optimization-evidence)
- [Light implementation verification](optional-light-modules.md#verification-scope)
