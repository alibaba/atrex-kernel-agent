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

The existing `gpu-wiki/tools/query_nl.py`, `query_wiki.py` and
`query_hardware.py` commands are equivalent Session clients. The Supervisor owns
stores, bridge execution and query telemetry; never override their host paths.

For `repairable: true` with `error.code: "request_not_started"`, wait at least
`retry_after_seconds` before retrying the unchanged query with backoff. The query
executor has not started. Do not blindly retry transport failures or unknown outcomes.

Check coverage before restrictive type filters. Read the returned source, scope,
payload and notes. A zero-match fallback sample is labelled, not advice for your
question. A missing hardware spec is not permission to substitute another GPU.
Record `query_id` and canonical `wiki_id` in the existing experiment Journal when
knowledge materially affects an experiment; retrieval alone is not adoption.
