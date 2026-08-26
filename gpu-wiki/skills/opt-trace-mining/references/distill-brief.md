# Distillation brief: turn one packet into one clean-1.3 record

You are given one `packets/<seg_id>.json`. You produce **one** JSON record file.
Your job is to **fill fields**, not to write code, not to research, and not to
improve on what the trace measured.

## The packet has two layers, and the boundary is hard

- **`agent_facing`** — everything you may read and quote. Absolute paths,
  addresses, private terms, version ids and workload hashes have already been
  removed; benchmarked shapes appear as stable `shape-N` labels. **This is your
  only source of information.**
- **`provenance`** — the raw version, the trace label and the commit. It may be
  copied **only** into `evidence.raw`. It must never appear in `payload`,
  `retrieval`, `evidence.summary` or `worth.gain`.

`agent_facing.diff_file` / `kernel_file` name a sibling file in the same
directory. Read the code from there.

## Naming and placement

- `id`, `episode_key`, `output_dir`, `retrieval.scope`, `level` and `schema` are
  already in `target`. **Copy them verbatim.** Do not mint an id.
- The file name is `<id>.json`, written under `target.output_dir` inside the
  staging store (`$RTM_STORE`, default `kernel_wiki/staging/`). The `layout` gate
  checks that the directory equals the record's own scope.
- `status` is `"active"`.

## Hard rules — all mechanically checked; a record that breaks one is waste

1. **No dangling reference.** Never write a version id (`v83`), a person, an
   absolute root/home path, an e-mail, a path into a trace (`profiles/v83`,
   `memory/v83.json`), a short or full Git hash, or **any `*.md` path**.
   There is no markdown page in this store, so a page citation is a reference to
   a file that does not exist. Refer to an earlier version as "an earlier step",
   and to a benchmarked shape by the packet's `shape-N` label. **The text in
   `agent_facing` is already clean: quoting it is safe; inventing your own
   version numbers is not.**
2. **`implementation.snippet` is verbatim from the sibling diff / kernel file.**
   Do not rename variables, do not complete a fragment, do not write code. Pick
   the hunk that carries the point, not the whole diff. If the snippet starts
   with `-`/`+`, `format` must be
   `"unified-diff (verbatim; '-' = before this step, '+' = after)"`; a whole
   function is `"source (verbatim)"`.
3. **Every number must already appear in the packet.** `payload`,
   `worth.gain`, `evidence.summary` and `retrieval.signals.metrics` are audited
   against `agent_facing` plus the code file. If you cannot find it there, leave
   it out.
4. **Absolute latency goes only in `evidence.raw.effect`.** `worth.gain` carries
   percentages; a latency entry may carry `delta_pct` only, never `before` /
   `after` / a time `unit`.
5. **Prefer a gap to an invention for `bottleneck`.** Only when
   `agent_facing.profiler.ncu_measured_the_kernel_under_test` is `true` may
   `evidence.summary.bottleneck_evidence.basis` be `"profiler"`. When it is
   `false`, use `"bench-only"`, `"asserted"` or `"none"`, and leave
   `payload.problem.bottleneck` and `retrieval.signals.bottleneck` null. The
   `ncu-attribution` gate checks this.

## Which payload fields, by type

`payload` is `additionalProperties: false`: **one extra key is a FAIL.**

| type | payload required | payload optional |
|---|---|---|
| `strategy` | `goal` `problem` `trace` `change` `mechanism` `implementation` `cost` `next_steps` | `config` `applies_when` |
| `anti-strategy` | `goal` `problem` `attempted` `verdict` `lesson` `established_fact` | `trace` `hypothesis` `observed` `root_cause` `would_retry_if` `implementation` `cost` |
| `reference-kernel` | `goal` `problem` `trace` `implementation` | (none) |

### strategy

- `change` — what this step changed, **verbatim** from
  `agent_facing.what_the_run_said.change`.
- `mechanism` — **why it got faster**, at the machine level: launch count, HBM
  traffic, instruction count, occupancy, bandwidth. No absolute latency here.
- `trace.builds_on.approach` — from `agent_facing.builds_on`. When
  `was_the_untouched_baseline` is true, say so and set `is_baseline: true`.
- `cost.edit_scope` ∈ `{one-line, epilogue, kernel-rewrite, architecture}`,
  `cost.risk` ∈ `{low, med, high}`.
- `next_steps` — from `agent_facing.open_directions`, verbatim.

### anti-strategy

> **Admission bar — all three, or write no record at all**
> A negative record is admissible only as an **established fact**: under a
> checkable condition C, doing X necessarily yields the bad result.
> 1. **Checkable condition** — `established_fact.condition` needs at least one
>    of `sm_arch` / `shape_regime` / `dtype` / `toolchain`. **"on this operator"
>    is not a condition**: failing on one operator is an observation, not a law.
> 2. **Causal mechanism** — `established_fact.mechanism`, at least 40
>    characters, saying why it **necessarily** fails. "measured, no gain",
>    "3 variants all flat", "hw floor confirmed" are measurements, not
>    mechanisms.
> 3. **A verdict that concludes** — one of `accuracy-gate/ceiling`,
>    `api-limitation`, `not-worth-it-here`, `performance-ceiling`. `unknown` and
>    `unstable` were removed from the enum on purpose.
>
> If any of the three cannot be taken from **this packet's own evidence**, emit
> no record. Supplying a mechanism the trace never stated is fabrication.
> Criteria and boundary cases:
> `skills/wiki-gate/references/established-fact-criteria.md`.

- `attempted` — what was tried.
- `lesson` — **the transferable sentence**: not "this does not work" but "under
  condition X, Y is slower because Z, so do W instead". It is the natural-language
  restatement of `established_fact` and must not contradict it.
- `agent_facing.how_it_was_fixed`, when present, belongs in `would_retry_if` or
  `lesson`.
- `worth.gain`: `kind: "none"`, `basis: "qualitative"`, `pct: null`,
  **`metrics: []` and `regressions: []`**. A reverted step has no comparable
  before/after pair; put the regression in prose in `observed` or `lesson`.
  Forcing it into `metrics[]` produces an entry that fails the schema and the
  self-contained gate.
- `agent_facing.split_instruction` decides how many records you write:
  `mechanical` / `curated` → exactly one; `agent` → one record for the **most
  substantial** lever only, naming the others in `lesson` / `would_retry_if`
  (an anti-strategy payload has **no** `next_steps` key).

### reference-kernel

- `implementation.snippet` — the representative kernel function from
  `kernel_file` (the `@triton.jit` / `@gluon.jit` / device entry point that the
  rest calls), not the whole file. `format: "source (verbatim)"`.
- `goal` — what a reader learns from this implementation; `problem` — what the
  operator computes.

## `retrieval`, `worth`, `evidence`

- `retrieval.signals.metrics` is an open dict for the profiler numbers you can
  cite (each still audited). `evidence.summary.mechanism_metrics` is a **closed**
  vocabulary — `launch_count`, `memory_traffic`, `occupancy_limiter`,
  `occupancy_pct`, `registers_per_thread`, `smem_bytes_per_block`, `spill`, each
  a `{before, after, value, saved, delta_pct, unit, quote}` object. Anything else
  goes in `retrieval.signals.metrics`.
- `retrieval.signals.shape_regime` holds the **size** axes (`predicate`,
  `var_axes`, `n_shapes`); `retrieval.scope.shape_signature` holds only semantic
  constants (hidden size, dtype). Never put a size axis in `scope`.
- `worth.rank` — write the placeholder `{"score": 0.0, "tier": "provisional"}`.
  **Do not score your own record**; `score_records.py` computes it with the
  store's own model.
- `worth.track.prior` — **omit a field you do not have; never write `null`.** The
  scorer reads `step_gain_pct: null` as a measured 0% gain and floors the record;
  `n_independent_runs: 1` is dropped for the same reason.
- `worth.gain.comparable` — copy `agent_facing.measurements.comparable_with_the_store`.
  When it is false, say why in `worth.gain.note`: the percentages are relative to
  this run's own baseline.
- `worth.gain.pct` must equal the `delta_pct` of the entry named by
  `worth.gain.primary`, and every entry in `metrics[]` needs `metric`,
  `direction` and `source`.
- `evidence.summary.confidence` ∈ `{measured, inferred, documented}`. A number
  this trace's own harness produced is `measured`; a conclusion you reasoned to
  from partial evidence is `inferred`.
- `evidence.summary.measured_on` is required here: `{"harness": ..., "gpu": ...}`,
  copied from `agent_facing.operator.harness` / `gpu`.
- `evidence.raw` — copy `source_repo`, `version`, `git_commit` from `provenance`
  and put the absolute geomeans in `effect` (which needs its own
  `basis: measured|qualitative`). Nothing else: the profile closes this object.

## Self-check before you stop

1. File name == `<id>.json`, inside `target.output_dir`.
2. Re-read `payload` / `retrieval` / `evidence.summary` / `worth.gain`: no
   version id, no person, no path, no `.md`, no 8-hex hash.
3. Every number traceable to the packet.
4. Every snippet line findable in the sibling file.
5. The `bottleneck` fields agree with `ncu_measured_the_kernel_under_test`.

Then run, from the skill's `scripts/` directory:

```bash
RTM_TRACE=<trace> python3 validate_store.py --verbose
```

Iterate until it is green. **Never edit anything under `scripts/`, and never
weaken a gate to make your record pass.** Report back: which fields the packet
was too thin to fill, and which gate blocked you — that is the signal used to
improve the pipeline. When several agents write into one staging store at once,
ignore gate failures that name records you do not own.

Leave an optional field out rather than guessing it. A gap is honest; an
invention is a record that will mislead an agent months from now.
