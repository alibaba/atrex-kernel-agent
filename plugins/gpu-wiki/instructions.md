GPU Wiki provides isolated hardware facts and benchmark-falsifiable optimization experience.
Use `python -m aka.legacy.resources.cli call gpu-wiki.query --input wiki_request.json` with a JSON object:

```json
{"request": "Target hardware B200, DSL triton. Optimize operator rmsnorm and retrieve techniques and pitfalls.", "max_records": 6}
```

For detailed questions include the exact true product, authoritative runtime architecture, operator,
DSL, shapes/dtypes, measured symptoms, failed attempts and remaining hypotheses. Never substitute
hardware identities. The standard episode request is parsed deterministically; other prose may use
the Wiki's existing tool-free bridge agent. `max_bytes` limits context; `exclude` omits previously
read IDs. Do not call Wiki scripts directly from campaign workflows.

Read `source`, `store`, `type`, `match.arch`, isolated `payload` and deterministic `notes`.
Copy emitted `query_id` and each materially used record's canonical `wiki_id` exactly; preserve
the `gpu_wiki` namespace. This tool queries only its declared public store. When the independent
`internal-gpu-wiki.query` tool is enabled, call it separately for internal records and retain its
own attribution. A labelled fallback sample is not a matching answer.
An empty `records` map is a successful lookup with no records; execution errors are separate.
Retrieval is not adoption. Preserve Wiki attribution in the existing experiment journal only when
used or reconsidered. Do not copy payloads into telemetry or write canonical memory directly.
