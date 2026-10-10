# Supervisor resources

This directory contains Supervisor-side workspace initialization, evaluator adapters, schemas and
packaging resources. It is not linked into the Agent workspace. The installed engineering constraints
come from `CLAUDE.md`; task-specific README and public inputs are projected separately.

Agent tools and their examples are in `skills/gpu-measurement`, `skills/runtime-records` and
`skills/KernelWiki`. See [workspace design](../docs/design.md#agent-facing-workspace) for visibility,
ownership and lifecycle. Legacy evaluator/report helpers here remain for controller and historical
artifact compatibility, not as additional Agent workflow stages.
