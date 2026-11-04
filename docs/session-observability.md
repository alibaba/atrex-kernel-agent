# Session observability

The shared Agent publishes normalized invocation observations through
`aka.agent.public_output`. The public Agent interface exposes session history and
conversation events without requiring consumers to parse provider-specific transcript trees.

## Current configuration

`aka.agent.plugin` accepts `session_name`, `sandbox`, `bwrap_executable`, `proc_mode`,
`timeout_seconds`, and `state_directory`. There is no `capture` flag or
`ATREX_SESSION_CAPTURE_DIR` configuration in the current shared Agent.

```json
{"api_version":1,"patches":[{"id":"agent","config":{
  "sandbox":"none","proc_mode":"private","session_name":"optimization"
}}]}
```

Use the patch with the [native launch arguments](quickstart.md#launch-light).
For Bubblewrap, retain `sandbox: bwrap` explicitly. Use `state_directory` only for an
operator-selected persistent Agent state root; it is not an arbitrary transcript export.

## Durable observations

Each invocation records a schema-1 JSON observation under `.aka-agent-observations` in
its observation workspace. It contains runtime/session identity, usage, normalized events,
resume support and observation errors. Files are private, atomically published, and read
with no-follow and hardlink checks. Consumers should use the Agent's observation methods,
not infer a completed session from directory contents or provider log filenames.

Usage and conversation coverage depend on the backend's available output. Missing events
or usage are not evidence of zero work. Observation errors remain visible separately from
the primary invocation result. Workflow history and optional dashboard publication consume
these observations; measurements, source identity and candidate promotion are separate.

## Historical capture and validation

The former capture toggles, transcript archive layout, kill switch, provider-specific
reconciliation rules and validation reports are preserved in the
[pre-migration observability guide](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md).
They describe that implementation and are not current configuration or promises about
all provider-native transcripts. Retain a matching old installation to reproduce those
runs. This documentation update validates CLI/configuration contracts; it does not run
paid coding agents or establish all-provider transcript completeness.

## Historical topic index

The links below preserve the earlier documentation text and saved section links.
`1abdf3f3` identifies the documentation snapshot, not a compatible runtime for the
removed commands. Historical behavior predates the shared-module refactor; use the
implementation revision named in each validation record when reproducing evidence.

<a id="motivation-users-and-observable-benefit"></a>
- [Motivation, users, and observable benefit](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#motivation-users-and-observable-benefit) (historical).

<a id="capture-lifecycle"></a>
- [Capture lifecycle](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#capture-lifecycle) (historical).

<a id="codex-reconciliation-and-failure-paths"></a>
- [Codex reconciliation and failure paths](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#codex-reconciliation-and-failure-paths) (historical).

<a id="code-map"></a>
- [Code map](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#code-map) (historical).

<a id="where-records-live"></a>
- [Where records live](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#where-records-live) (historical).

<a id="disable-capture-and-roll-back"></a>
- [Disable capture and roll back](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#disable-capture-and-roll-back) (historical).

<a id="conversation"></a>
- [Conversation](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#conversation) (historical).

<a id="usage"></a>
- [Usage](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#usage) (historical).

<a id="failure-and-privacy-boundaries"></a>
- [Failure and privacy boundaries](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#failure-and-privacy-boundaries) (historical).

<a id="verification"></a>
- [Verification](https://github.com/alibaba/atrex-kernel-agent/blob/1abdf3f36cbb7797fdedec7924ff40431975df7b/docs/session-observability.md#verification) (historical).
