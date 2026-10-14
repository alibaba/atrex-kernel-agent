---
name: aka-execution
description: Prepare requests for the selected AKA Execution entry point, consult Gateway or SSH guidance, and preserve complete execution output and artifacts. Use for handing prepared GPU commands, evaluation, or profiling work to AKA Execution; experiment planning and result interpretation belong to their callers.
---

# AKA Execution

Execute prepared work through the entry point supplied by this AKA run. Use its
supported backend options, request format, defaults, and error behavior.

## Identify the entry point

Read the run's Execution tool description, schema, selected backend, and workspace
information. Use its bound executor and configuration. For standalone
`aka execution`, inspect `aka execution --help` and supply a matching checkout and
prepared workspace. The current legacy implementation is described in
[Legacy requests](references/legacy-request.md).

Continue existing runs with their recorded selection and execution path.
Deterministic host callers retain their existing adapters; loading this Skill
does not add a model call to host execution. When the application exposes an
Execution tool, hand requests to it rather than constructing another executor.

## Load backend guidance

- **Agate / Gateway:** Read the `atrex-gpu-gateway` Skill provided in this
  optimization Agent's environment before preparing backend-specific options.
  See [Backend Skill references](references/backend-skills.md) for discovery and
  version handling. Transfer the relevant parameters into fields supported by
  the Execution entry point. Execution owns submission, polling, cancellation,
  and artifact retrieval. `eval --backend atrex|sol` chooses an evaluation suite;
  Gateway profile / URL chooses an endpoint; neither is the Execution plugin.
- **SSH:** Read the selected SSH Execution guidance. Current target, runtime
  initialization, health, and single-GPU constraints are in
  [Legacy requests](references/legacy-request.md).
- **Local HTTP Gateway:** Use Gateway requests with the selected local endpoint.
  Confirm the service is running and accepts the requested hardware alias.
  The current implementation has no separate direct Local executor.

Backend guidance may describe a newer client than the AKA adapter supports.
Check the actual Execution request surface. Report unsupported options without
silently changing the endpoint, operation, evaluation path, or implementation.

## Prepare the request

Preserve the caller's prepared workspace, argv, inputs, worker environment,
hardware, measurement parameters, timeouts, and required outputs. Put backend
options in the entry point's configuration or request fields. Keep argv argument
boundaries intact and use existing credential resolution outside saved requests.

Follow the original input allowlist, private evaluation protection, and
synchronization rules. Host environment and worker environment keep their
existing meanings. The legacy wrapper's `--sync` names output synchronization
paths; it is not an input manifest.

Commands in the Agate Skill describe backend operations. Arguments after
`aka execution --` are worker commands for the original sandbox. Translate
according to the entry point: do not place a host-side `agate ...` invocation
inside worker argv. The current wrapper retains the sandbox's typed / dev mapping.

## Preserve execution results

Preserve return codes, stdout, stderr, structured output, and synchronized
artifacts, including existing redaction. Preserve exceptions and partial output
on failure. Leave measurement interpretation and acceptance decisions to
Evaluation or the original caller.

Distinguish worker execution, observation, and HTTP timeouts. Keep the original
job identity when submission or observation is uncertain; use the existing
execution path to confirm it instead of submitting duplicate work. Existing host
and backend implementations retain their retry and recovery responsibilities.
Report cancellation acceptance separately from confirmed termination. Legacy
host cleanup adds no new remote termination or cross-version recovery guarantee.
