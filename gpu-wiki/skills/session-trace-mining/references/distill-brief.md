# Distillation brief (copy verbatim when handing it to a distillation agent, replacing the `<>` placeholders)

## Task

Write each of the `<N>` packets as one `session-trace-1.0` record, land them at the
given paths, and get all sixteen `validate_store.py` gates green.

You **cannot see** the session transcripts, only the packets. That is by design:
any field the packet cannot fill has to stay empty, and may never be filled in
from general knowledge.

## Background

The corpus consists of session transcripts from an AI coding agent (GPU operator
optimization runs). These records are retrieved by another agent to decide how to
optimize a kernel next, so every record must answer four questions on its own: why
read it, what problem it solves, which approach it improves on, and how much gain
to expect.

- Python: `python3` (the `schema` gate needs `jsonschema`)
- scripts: `<SKILL>/scripts`
- packet directory: `/tmp/session-trace-mining/<SET>/packets/`
- output root: `<STORE>` (records go under `packet.target.output_dir`, named
  `<packet.target.id>.json`)
- reference exemplar: `<EXAMPLE_RECORD>` — the one record in this store that
  already passes every gate, at a path of the form
  `<STORE>/records/<type>/<vendor>/<arch>/<dsl>/<family>/<id>.json`. Fill fields
  the way it does. On the first distillation batch there is no exemplar yet: write
  one record, take it through the gates, then hand it out as the exemplar for the
  batches that follow.

## How to use each packet field

| Field | Purpose |
|---|---|
| `evidence_text` | **The only place a number may be quoted from.** Each block carries a `### [T1\|T2\|T3 tool-output line N]` header. T1 = benchmark/profiler output, T2 = the agent reading back notes it wrote itself, T3 = a structured field the agent wrote. |
| `<seg_id>.diff` | The verbatim code change. `implementation.snippet` **must** come line for line from here. |
| `measurement` | before/after, improvement percentage and shape count, already computed by the extractor. Take the numbers for `worth.gain` from it. |
| `narrative` | The run's own account (subject, action_description, profile_evidence, pitfall, open_directions). The main raw material for writing `mechanism` and `next_steps`. |
| `scope` / `target` | Copy straight into `retrieval.scope` and the output path. Do not infer the architecture yourself. |
| `session` | **Copy the whole block through unchanged** into `evidence.raw.session`. Do not alter one character — the provenance gate checks the digests line by line. |

## Hard rules (all mechanically checked by the gates)

1. **Numbers may only come from `evidence_text` or `measurement`.** The gate takes
   every number in `worth.gain` (including `note`, `correctness` and
   `measured_over`) and compares it against the evidence. Only two derivations are
   allowed: a percentage computed from before/after, or a percentage computed from
   `speedup=Nx`. Anything else counts as fabrication.
   **`<seg_id>.diff` is code, not a source of numbers**: it is not in the audit
   pool. An iteration count, a constant or a tile size read out of the diff may not
   be the basis for any number in `worth.gain`; if you want to state those, write
   them into `mechanism`.
2. **`snippet` must come verbatim from `<seg_id>.diff`.** A rewritten fragment is
   worse than none: it looks authoritative and does not compile. Set `format` to
   `"unified-diff (verbatim; '-' = before this step, '+' = after)"`.
3. **No absolute times in `worth.gain`.** A microsecond figure means nothing apart
   from its shape set. Absolute values belong in
   `evidence.raw.effect.new_geomean_us` / `parent_geomean_us` (humans only).
4. **`gain.pct` must equal the `delta_pct` of the primary metric.**
5. **`gain.source_kind` is always `"agent-session"`**, and `basis=measured` is only
   allowed when the packet has a T1 block; otherwise use `reported`.
6. **No paths and no personal names in any agent-visible layer.** `/home/...`,
   `/root/...` and `rollout-*.jsonl` may never reach `payload` or `retrieval`. To
   refer to a source file, say "the kernel source". `evidence.raw` is the only
   exempt layer.
7. **`diff_coverage=blind` may not become a `strategy`.** The packet already states
   its `record_type`; do not change it.
8. **Leave `retrieval.links` as an empty object `{}`** unless the id you reference
   really exists in this store.

## Key points for filling fields

- Take `id` / `type` / `level` (`operator`) / `status` (`active`) from the packet.
  `episode_key` = `<arch>|<dsl>|<operator_slug>|<technique_snake>|<level>`.
- `retrieval.signals.symptoms`: use normalized lower-case underscore symptom names,
  not whole log lines.
- `retrieval.triggers`: state "when should this record be read", at most three,
  as second-person situational sentences.
- `payload.problem.statement`: **say clearly why it is slow**, do not restate the
  title.
- `payload.mechanism`: **causality at the machine level** — why this change makes
  it faster. This is the most valuable field in the whole record, and the easiest
  one to leave hollow. If you cannot state a mechanism, the packet's evidence is
  too thin: say so in the report rather than inventing one.
- `payload.trace.builds_on.approach`: **describe the previous approach in prose**
  (the packet's `narrative.builds_on`); do not write an id.
- `payload.next_steps`: only directions the packet actually contains
  (`narrative.open_directions`).
- `worth.rank`: the fixed placeholder
  `{"score": 0.0, "tier": "provisional", "builder_version": "session-trace-0.1"}`,
  computed later in one pass by `score_records.py`.
- `worth.gain.comparable`: **false**. This is a sibling run and is not comparable
  with the main chain.

## Delivery and self-check

```bash
cd <SKILL>/scripts
STM_SET=<SET> python3 validate_store.py --verbose
```

Iterate until everything is green. **Do not modify any file under `scripts/`**, and
do not relax a gate to get past it. When a gate looks wrong, verify by injection:
break a record deliberately and confirm the gate you expected is the one that
fires.

Finally, hand in a report of ≤400 words that must cover:

1. which records you wrote and of what types;
2. **which fields you left empty because the packet evidence was too thin** (this
   is the main signal for improving the pipeline);
3. **which gates stopped you, and for how many rounds**;
4. which record you consider the least valuable, and why.

Treat a batch that reports "no difficulty at all" as suspect: more likely it did
not check the evidence carefully.
