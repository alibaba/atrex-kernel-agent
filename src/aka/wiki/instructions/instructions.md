GPU Wiki provides isolated hardware facts and benchmark-falsifiable optimization experience.
Use the selected Core tool `gpu-wiki.query` with its declared input schema. For example, ask about B200, Triton, and rmsnorm optimization techniques with a six-record limit.

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
