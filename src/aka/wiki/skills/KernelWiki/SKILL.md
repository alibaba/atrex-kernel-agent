---
name: KernelWiki
description: Query the selected GPU Wiki Core tool and preserve its source attribution.
---

# GPU Wiki

Use the enabled Core Wiki tool exposed by the current host. The public tool is
named `gpu-wiki.query`; an internal Wiki, when selected, has its own separately
listed tool. Discover tools and use the invocation instructions supplied by
the host. Do not call Wiki scripts or stores directly.

Pass a JSON object matching the selected tool's declared input schema. For
example:

```json
{"request":"For B200 and Triton rmsnorm, summarize relevant optimization evidence and hardware facts.","max_records":6}
```

The result is an `aka.tool-result.v1` envelope. Read `data` for the Wiki query
response and retain the returned `query_id`, canonical `wiki_id` values, and
the envelope's `references` when recording knowledge used by an experiment.
Call each enabled Wiki separately when evidence from more than one selected
store is needed; do not infer permission to query a store that is absent from
the tool catalog.

State the true target product, runtime architecture, operator, DSL, shapes and
dtypes separately. Include measured symptoms, failed attempts, and remaining
hypotheses when available. Never substitute a hardware identity. Treat a
labelled fallback as a fallback rather than a matching answer. Retrieval is
read-only and does not by itself mean that an experiment adopted the advice.
