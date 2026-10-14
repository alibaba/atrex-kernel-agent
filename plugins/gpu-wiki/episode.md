## Wiki attribution contract

At episode start, run the required bounded GPU Wiki query once using the campaign's exact operator
identifier rather than paraphrasing it. Additional targeted queries are allowed later when new
evidence creates a materially different question:

Create `wiki_request.json` containing a JSON object with a `request` string, then run:
```bash
python3 tools/plugin.py call gpu-wiki.query --input wiki_request.json
```

Use this exact request string (preserve the operator identifier):

Target hardware {{PLATFORM}}, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve techniques and pitfalls.


GPU Wiki query responses emit a top-level `query_id`, and every returned record emits its own
canonical `wiki_id` in `store::record` form. Copy those fields exactly; never reconstruct either
value from a response mapping key or from prose. Whenever a returned record
materially influences an experiment or is explicitly evaluated and rejected, add `wiki_usage` to
that experiment's journal append. Each row must contain the response's emitted `query_id`, an
actually returned record's emitted `wiki_id`, a disposition of `applied`, `partially_applied`,
`reference_only`, or `rejected`, plus a
short `use` and observable `evidence`. Preserve repeated use in separate experiments; do not dedupe
across the episode. The first experiment must account for the required query as either `declared`
or `no_material_use`; it cannot claim `not_queried`. Every experiment must set
`wiki_usage_status` to `declared` with non-empty usage,
`no_material_use` when Wiki was queried without attributable use, or `not_queried` when it was not
queried. For `declared` and `no_material_use`, include `wiki_query_ids` with every Wiki query considered
by the experiment; omit it for `not_queried`. In later experiments, use `not_queried` when the
experiment neither issued a new query nor reconsidered a previous response; do not carry an earlier
query id forward unless its response informed that experiment. Record `evaluation.correctness`, `evaluation.performance`, optional evaluator latency/hash,
and an explicit decision so attribution can be joined to the experiment outcome.
Malformed Wiki telemetry is diagnostic only: the journal drops bad rows into `wiki_usage_errors`
without invalidating the optimization experiment or its terminal handoff.


When conversion is mandatory, replace the general episode-start request with a request naming
product {{PLATFORM}}, exact runtime architecture {{ARCH}}, and asking for the complete product
specification plus only the matching Triton-to-Gluon conversion guidance. Use the same JSON tool
call above. This counts as the required Wiki query; do not also issue the general request.

Expected conversion records: `nvidia.blackwell.any.converter.blackwell` for `sm_100`/`sm_103`,
`nvidia.hopper.any.converter.hopper` for `sm_90`, `amd.cdna3.any.converter.cdna3` for `gfx94*`,
and `amd.cdna4.any.converter.cdna4` for `gfx95*`. Never substitute a sibling architecture's record.
