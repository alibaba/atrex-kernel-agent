---
name: KernelWiki
description: Query GPU optimization knowledge and hardware facts through the session's Supervisor Runtime.
---

# GPU Wiki

Describe the operator, DSL, runtime architecture and observed bottleneck. Distinguish
measurements from hypotheses. Do not reduce the question to arbitrary keywords.

```bash
python3 tools/sandbox.py --kind wiki-query "Triton normalization on sm_120; measured DRAM-bound, looking for useful fusion strategies" --brief
python3 tools/sandbox.py --kind wiki-query --file scratch/wiki-question.txt
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product sm120
```

The existing `gpu-wiki/tools/query_nl.py`, `query_wiki.py` and
`query_hardware.py` commands are equivalent Session clients. The Supervisor owns
stores, bridge execution and query telemetry; never override their host paths.

Check coverage before restrictive type filters. Read the returned source, scope,
payload and notes. A zero-match fallback sample is labelled, not advice for your
question. A missing hardware spec is not permission to substitute another GPU.
Record `query_id` and canonical `wiki_id` in the existing experiment Journal when
knowledge materially affects an experiment; retrieval alone is not adoption.
