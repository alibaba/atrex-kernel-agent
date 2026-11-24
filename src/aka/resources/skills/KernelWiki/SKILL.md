---
name: kernel-wiki
description: Query GPU optimization knowledge and hardware facts through the session's Supervisor Runtime.
---

# GPU Wiki

Use the actual architecture returned by `python3 tools/sandbox.py --kind env`
for Wiki queries. Describe the operator, DSL, bottleneck and relevant architecture/ISA
question; distinguish measurements from hypotheses. Include a public product name
only when it is known and a product-specific specification is needed. Do not infer
product identity from a deployment name.

For example, when Env reports `sm_120`:

```bash
python3 tools/sandbox.py --kind wiki-query "Runtime architecture sm_120. Triton normalization is measured DRAM-bound; which fusion strategies and architecture/ISA facts apply?" --brief
python3 tools/sandbox.py --kind wiki-query --file scratch/wiki-question.txt
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --capability sm_120 --list features
```

`wiki-hardware` with no arguments lists products. A bare product name is accepted,
as are a leading `--` separator and a single quoted option string; prefer the explicit
arguments shown above. `--help` returns each tool's supported options.

Use the `tools/sandbox.py --kind wiki-*` commands above from an Agent workspace.
The Supervisor owns stores, the Wiki-only model, timeouts, concurrency and query
telemetry; never override their host paths or provider configuration.

For `repairable: true` with `error.code: "request_not_started"`, wait at least
`retry_after_seconds` before retrying the unchanged query with backoff. The query
executor has not started. Do not blindly retry transport failures or unknown outcomes.
For `wiki_queue_timeout`, `wiki_query_in_progress`, `wiki_query_timeout` or
`wiki_unavailable`, wait at least `retry_after_seconds` and retry the same read-only
query unchanged. Do not submit parallel copies or wrap the call in a shorter shell
timeout. If Wiki remains unavailable, continue with other tools and report the
blocker. A Wiki failure is not a GPU environment failure.

Check coverage before restrictive type filters. Read the returned source, scope,
payload and notes. A zero-match fallback sample is labelled, not advice for your
question. A missing hardware spec is not permission to substitute another GPU.
When the Experiment module is enabled, record `query_id` and canonical `wiki_id` in the existing
Experiment when knowledge materially affects it; retrieval alone is not adoption. When that module
is disabled, use the query normally but do not invent an Experiment or attribution mutation.
Use `applied`, `partially_applied`, `reference_only` or `rejected` for `disposition`;
`used`, `material_use` and `considered` are invalid. See the complete attribution JSON example in
`.agents/skills/runtime-records/references/journal.md`. Correct rejected attribution fields and resubmit
with the existing Gateway Record IDs; do not repeat GPU work.

## Find related source implementations

Use `wiki-query` to find related implementations. Describe the mechanism and missing fact,
for example: `python3 tools/sandbox.py --kind wiki-query "Find split-KV implementations and how partial outputs are reduced on sm_120"`.
Read `source_matches`: each hit includes a file path, line numbers, matched terms and a code
snippet. Open relevant returned files for full context and check applicability. These hits are
source references, not validated Wiki facts; they have no `wiki_id`, so do not invent one for
attribution. A partial or empty search does not prove that a mechanism is absent.
