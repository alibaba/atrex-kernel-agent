# wiki-gate

## Role

The admission gate: the single entry point through which a wiki summary coming
from trace/session mining reaches `kernel_wiki/records/`. It is also the executor
that updates the counters during a bulk trace scan — feedback is not reported
live, it is triggered in bulk by trace analysis.

## The two-step interface

The gate is a **purely deterministic tool** and makes no model calls. Semantic
judgement is left to the caller (an agent, or a trace-analysis script).

### Step 1: `--match` (returns candidates, decides nothing)

```bash
python3 scripts/gate.py --match --input /path/to/incoming.json
```

Output (stdout, a single JSON document):
```jsonc
{
  "status": "ok",
  "incoming": { "id", "type", "change", "mechanism", "goal", "gain_direction", "gain_pct", "shape_contract", ... },
  "candidates_count": 3,
  "candidates": [ { same fields as above } × every record in the same scope ]
}
```

The caller (the agent) reads `candidates` and judges for itself: no match →
insert; a match pointing the same direction → confirm; a match pointing the
opposite direction → conflict.

### Step 2: `--commit` (carries out the decision)

```bash
python3 scripts/gate.py --commit insert   --input incoming.json
python3 scripts/gate.py --commit confirm  --input incoming.json --target <existing_record_id>
python3 scripts/gate.py --commit conflict --input incoming.json --target <existing_record_id>
```

| action | Effect |
|---|---|
| `insert` | Writes into records/ once validation passes, **and appends the entry to `records/index.json`** (otherwise the new record is invisible to the next `--match`, and the main store's 7th gate FAILs as well); prints the path on stdout |
| `confirm` | Increments the original record's `worth.track.counters.verified_effective` by 1 (rewritten in place), deduplicated via `worth.track.confirm_keys`, so repeated reports from the same source record never double-count |
| `conflict` | Writes the complete conflict information into `kernel_wiki/conflicts/`, leaving the original record untouched |

`insert` **rejects an id that already exists** (at the same path, or anywhere in
the index) and points you to `confirm` instead — overwriting would mean deleting
an existing record in order to store one you only believed was new.

### What `insert` validates

The gate claims to be the only way into `records/`, so it runs **the same
record-level validation as the main store**, not the schema alone:

- JSON Schema (clean-1.3)
- Established fact (see below)
- The main store's 4 record-level gates, reusing `tools/check_kernel_wiki.py`
  directly: `anonymization` / `raw-isolation` / `self-contained` /
  `no-cross-reference`

The remaining 4 (`ids` / `coverage` / `relations` / `index`) are store-wide in
nature, and are covered by `insert`'s id-collision check plus a `--full` run after
each batch.

If `tools/check_kernel_wiki.py` cannot be imported, the gate **rejects rather than
skips** — silently running 4 gates fewer is precisely what it exists to prevent.

### Scope filtering of the candidate pool

Four hard filters: `vendor` / `arch` / `dsl` / `operator_family`. On `dsl`, `any`
means "independent of the DSL", so it **matches in both directions**: an incoming
record with `dsl=triton` can see `any` candidates, and an incoming record with
`dsl=any` can see candidates from every DSL. The one-directional version once gave
an `any` record 0 candidates while the main store held a near-verbatim `dsl=triton`
duplicate (+15.5% against +15.78%), which was nearly inserted as new knowledge.

### Admitting a negative record: it has to be an established fact

When inserting an `anti-strategy`, the gate **rejects** any of the following:

- `payload.established_fact` is missing
- all four items of `established_fact.condition` (`sm_arch` / `shape_regime` /
  `dtype` / `toolchain`) are empty — **"on this operator" is not a condition**
- `established_fact.mechanism` is shorter than 40 characters, or the whole
  sentence is nothing but a measurement result ("measured, no gain", "all 3 were
  flat")
- `verdict` is `unstable` / `unknown` — both values have been removed from the
  schema enum, and a run that reached no conclusion is not negative knowledge

A rejected record stays in the mining skill's own staging area and is not deleted:
once it has acquired a condition and a mechanism, or has been independently
reproduced several times, it can go through the gate again.

The criteria in full, the five kinds of statement that count as a mechanism and
the six that do not, and the borderline cases are all in
`skills/wiki-gate/references/established-fact-criteria.md`; the same criteria are
enforced on the main-store side by the `established-fact` gate in
`tools/check_kernel_wiki.py`.

## When it runs

- **New wiki records land**: after opt-trace-mining / session-trace-mining
  produces new records, the agent calls `--match`, judges for itself, then calls
  `--commit`
- **Feedback write-back**: when a periodic trace scan finds that a wiki record was
  adopted by an agent and kept, call `--commit confirm`

There is no "the agent reports live" path. All feedback comes from facts recorded
in trace files.

## Definition of a direction conflict

- one strategy and one anti-strategy (or the reverse)
- two strategies whose `gain.pct` have opposite signs

**A difference in magnitude is not a conflict** (that is ordinary variation across
shapes and hardware conditions).

## Conflict files

Written to `kernel_wiki/conflicts/<timestamp>_<id>.json`, retaining the complete
information about the new wiki record (including shape_contract, bottleneck and
observed_symptom) so the reason for the difference can be investigated later. The
`resolution` field starts as null and is filled in after human verification.

## Design constraints

- **Purely deterministic**: the gate makes no semantic judgement and calls no
  model; it does hard filtering, feature extraction and file IO
- **Idempotent**: on a repeated submission of the same record, the key of
  `confirm`'s increment event contains both ids, so writing it again has no effect
- **Never edits an original record's content**: the gate does not modify an
  existing record's payload or evidence; it only updates counters through
  events.jsonl
- **A conflict does not block**: it flags and exits normally (exit 0); the conflict
  files are a queue for humans to read, not an error
- **Every candidate**: `--match` returns all records in that scope without
  truncating, and the caller may process them in batches

## Dependencies

- The Python standard library
- jsonschema (optional; when missing, only the version constant is validated)
- No model API whatsoever
