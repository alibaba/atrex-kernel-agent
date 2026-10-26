---
name: KernelWiki
description: Query GPU optimization knowledge and hardware facts through the session's Supervisor Runtime.
---

# GPU Wiki

In every natural-language Wiki request, state the true target product separately from
the authoritative runtime architecture; never infer either from a scheduler label or
masked device name. If the true product is unavailable, say so rather than substitute
another GPU. Ask for that product's hardware specification and relevant architecture/ISA
facts alongside the operator, DSL and bottleneck. Distinguish measurements from hypotheses.
Do not reduce the question to arbitrary keywords.

```bash
python3 tools/sandbox.py --kind wiki-query "Target B200, runtime sm_100. Triton normalization is measured DRAM-bound; which fusion strategies apply? Include B200 hardware specifications and relevant architecture/ISA facts." --brief
python3 tools/sandbox.py --kind wiki-query --file scratch/wiki-question.txt
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product sm120
```

Use the `tools/sandbox.py --kind wiki-*` commands above from an Agent workspace.
The Supervisor owns stores, the Wiki-only model, timeouts, concurrency and query
telemetry; never override their host paths or provider configuration. Repeating an
identical query in this Episode reuses the same completed response and `query_id`.

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
When research recording is enabled, record `query_id` and canonical `wiki_id` in the existing
Experiment Journal when knowledge materially affects an experiment; retrieval alone is not adoption.
Use `applied`, `partially_applied`, `reference_only` or `rejected` for `disposition`;
`used`, `material_use` and `considered` are invalid. See the complete attribution JSON example in
`skills/runtime-records/references/journal.md`. Correct rejected attribution fields and resubmit
with the existing Gateway Record IDs; do not repeat GPU work.
