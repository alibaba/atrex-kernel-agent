# Schema — `kernel_wiki` (the experience store)

The schema is the **single source of truth** for every record in this store. The
schema gate in `tools/check_kernel_wiki.py` validates every record in the store,
and the index and retrieval tools all sit downstream of it. This store has **no
markdown** — the records themselves are the source of truth.

| File | Role |
|---|---|
| [`schema.json`](schema.json) | The single source of truth (JSON Schema, draft 2020-12). Current version `clean-1.3` |
| [`TEMPLATE.md`](TEMPLATE.md) | Field-by-field human reference, **generated from the schema**; do not hand-edit |
| [`render_template.py`](render_template.py) | `schema.json` → `TEMPLATE.md` renderer |

## Version policy

There is exactly **one** `schema.json`, and **git** distinguishes the versions —
no more filename suffixes like `clean-1.N.schema.json`. The `schema` constant
inside `schema.json` records the current semantic version (currently
`clean-1.3`), a record's `schema` field has to match it, and the schema gate
enforces that. To change the schema, edit `schema.json` and commit; the git
history is the version history.

## The four layers of a record

Every record has four layers. Each serves a different consumer, and they are
**not interchangeable**:

| Layer | For whom | Contents | When served |
|---|---|---|---|
| `retrieval` | the retrieval engine | One shape across all 8 types, hard-filterable: `scope` (vendor/arch/product/dsl/operator_family), `generality`, `signals`, `technique_tags`, `triggers`; plus the engine-side `locator` (source position) and `links` (internal id graph) | `locator`/`links` **stripped** |
| `payload` | the agent | **Self-contained** actionable knowledge, polymorphic by type; code is taken verbatim from the corpus | kept |
| `evidence` | `summary` for the agent, `raw` for **humans only** | `summary`: bottleneck evidence, mechanism metrics, measurement environment; `raw`: de-anonymized provenance plus the absolute geomean | `raw` **stripped** |
| `worth` | agent + ranking | `gain` (expected gain, normalized percentages), `rank` (`score` + `tier`), `track` (counters + corpus prior) | only `rank.score`/`rank.tier`/`gain` are served; `track` and `rank.components` **stripped** |

The **served projection** (`tools/query_wiki.py --emit-json`) strips, by
construction: `retrieval.locator`, `retrieval.links`, `evidence.raw`,
`worth.track` and `worth.rank.components`. Those five are either engine
bookkeeping or human-only provenance, so the agent should not — and does not need
to — see them.

## The payload self-containment contract

One query, one record: without looking back at `retrieval`, resolving any id, or
issuing a second query, an agent can answer why it is reading this record
(`goal`), what problem it solves (`problem`), which approach it improves on
(`trace.builds_on.approach`), and what the change is and how much it buys
(`change`/`mechanism`/`implementation` + `worth.gain`). The self-contained gate
enforces this.

## The eight record types

`strategy` (an optimization step that was kept), `anti-strategy` (an attempt that
was overturned, i.e. negative evidence — and it must be an **established fact**:
a checkable condition + a causal mechanism + a conclusive verdict, all three, see
that section of `TEMPLATE.md`), `reference-kernel` (an implementation you can read
as-is), `technique-card` (corpus-wide success and failure statistics for one
technique), `symptom-card` (symptom → candidate-technique entry point), `doc`
(hardware facts), `numerics-rule` / `dispatch-rule` (reserved). See `TEMPLATE.md`
for the per-type fields.

## The gain model (`worth.gain`)

A list of normalized metrics over a closed vocabulary: `latency` plus the roofline
coordinates (`sol`/`mfu`/`dram_throughput`/`compute_throughput`/
`arithmetic_intensity`) plus `compile_time`/`memory_footprint`. `delta_pct` is
sign-normalized (positive always means an improvement); `pct` is lifted to the top
level so ranking and `--min-gain` can read it directly; **absolute latency never
enters** it (that stays in `evidence.raw.effect`); and every metric must carry a
`source`.

## Regenerating

```bash
python3 schema/render_template.py            # schema.json -> TEMPLATE.md
python3 schema/render_template.py --check     # check TEMPLATE.md is in sync with the schema
python3 tools/check_kernel_wiki.py --full    # 9 gates, schema validation included
```


## How this store differs from the measured one (open-source note)

The `schema` constant is still `clean-1.3`, so the mining skill and the
`wiki-gate` admission tool work the same on both sides. The difference is only in
**what a document-derived record can honestly fill in**:

| Where | Note |
|---|---|
| `evidence.summary.confidence` | Gains a `documented` level: sourced from a curated documentation page, not from a harness measurement |
| `worth.gain` | Uniformly `basis: qualitative` and `pct: null` — documentation carries no measured gain, and we do not invent numbers |
| `anti-strategy.trace` | No longer required: a pitfall recorded in documentation has no traceable measured ladder behind it |
| `scope` vocabulary | Extended to every vendor / arch / DSL this wiki covers |
| `scope.architectures` | New: carries the explicit scope of a page that applies to several architectures at once |
| `technique-card.implementation` | New (optional): this store's technique cards are content cards cut along documentation sections, and they carry that section's code verbatim |

Records mined later from traces and sessions will fill in `worth.gain` and the
measured `evidence` fields as normal.
