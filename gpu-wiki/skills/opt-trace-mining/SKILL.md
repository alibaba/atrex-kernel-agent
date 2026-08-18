---
name: opt-trace-mining
description: Mine a per-kernel optimization trace — a git repository capturing successive versions of one kernel being optimized — into structured, gate-validated optimization-experience records for the GPU kernel wiki. Use when asked to distill an optimization run, a kernel_opt trace directory or a version ladder into wiki records; to report what such a run actually achieved; to build, extend, re-run or validate the staging store behind those records; or to explain how a trace-derived record's number, snippet or provenance was established.
---

# Optimization Trace Mining

## Overview

Turns one optimization trace into `clean-1.3` records: one record per change that
measurably moved a metric, and one per failed lever that turned out to be a fact.
Each record answers which operator, what was wrong, what changed, why it worked at
the machine level, the verbatim code, and how much it bought.

The pipeline is split the way its sibling `session-trace-mining` is: **scripts do
deterministic extraction, an agent does semantic distillation, and mechanical
gates catch fabrication.** The gates are why the output can be trusted. A model
filling fields is the weak link, so nothing enters the store that cannot be
checked against the trace.

What makes this corpus different from the sibling's: **the trace is a git
repository, so every claim can be re-resolved.** Provenance is the trace label
plus the commit, the code is a real diff, and the kept/reverted verdict is in the
commit subject rather than in prose. There is no markdown page anywhere in this
repository, so a record cites the trace and the commit and nothing else — records
are the only source of truth here.

## What one trace looks like

```
<trace>/.git                     commit subjects: `v<N>: ...`, and the verdict
<trace>/kernel.py                the kernel at that commit
<trace>/memory/v<N>.json         per-version measurements       (optional)
<trace>/profiles/v<N>*/          profiler captures              (optional)
<trace>/versions/kernel_v<N>.py  kept-version snapshots         (optional)
<trace>/definition.json          operator name and axes         (optional)
<trace>/solution.json            target hardware, languages     (optional)
<trace>/workload.jsonl           one line per benchmarked shape (optional)
```

Only `.git` is required, and each missing piece removes exactly one capability:
no `memory/` means no numbers, hence no `strategy` records; no `profiles/` means
no record may ever claim a profiler-backed bottleneck. The three sources
disagree, so each is read only for what it is authoritative about — commits for
the verdict, step records for the measurements, captures for the bottleneck.

## Setup

Nothing to configure for a trace that states its own hardware:

```bash
export RTM_TRACE=/path/to/your/trace/kernel_opt_001_your_operator
```

- **scratch** — `/tmp/opt-trace-mining/<slug>/`: parsed jsonl, packets. All
  reproducible from the trace, so none of it is committed. Override with
  `RTM_WORKSPACE`.
- **staging** — `kernel_wiki/staging/`: records plus `reports/<slug>/`. Reviewed,
  then promoted. Override with `RTM_STORE`.

Configure before the first run:

- **`config.TRACES`** holds ONE clearly fictional example entry. Either point
  `RTM_TRACE` at your own trace or register it there. The entry supplies the
  measurement target when the trace itself does not state it.
- **hardware**: `ingest.py` reads `solution.json` / `definition.json` and maps
  the token through `config.TARGET_TABLE`. If it finds nothing and no
  `RTM_ARCH` / `RTM_PRODUCT` is set, it **fails** rather than guessing — a record
  filed under hardware nobody measured is worse than no record, because the
  store's scope filter will serve it to an agent on different hardware.
- **`config.NON_TARGET_KERNELS`** lists kernel names a profiler capture may not
  be about. The default is torch's RNG fill, which is what `ncu` grabs when no
  `--kernel-name` filter was passed. Add your harness's own kernels
  (`RTM_NON_TARGET_KERNELS`).
- **`ATREX_WIKI_DENYLIST`** (optional) points at a file of private substrings,
  one per line, scrubbed out of packets and rejected by the store's own gate. It
  is an environment variable and not a committed list on purpose: a committed
  denylist publishes the names it is meant to hide.

`jsonschema` is needed for the two schema gates; without it they SKIP loudly.

## Pipeline

```bash
S=<skill-dir>/scripts
G=<skill-dir>/../wiki-gate/scripts/gate.py
export RTM_TRACE=/path/to/trace

python3 $S/make_schema.py --check     # the profile matches its patch list
python3 $S/ingest.py                  # trace  -> work/versions.jsonl, profiles.jsonl, meta.json
python3 $S/recon.py                   #        -> reports/<slug>/recon.md   READ THIS FIRST
python3 $S/partition.py               #        -> work/segments.jsonl, reports/<slug>/partition.md
python3 $S/build_packets.py           #        -> packets/<seg>.{json,diff,py}
# distil (see below), then:
python3 $S/validate_store.py --verbose        # 9 store gates + 7 trace gates
python3 $S/validate_store.py --injection-tests
python3 $S/score_records.py           # worth.rank + records/index.json
python3 $S/make_readme.py             #        -> <staging>/README.md
# then, per record, the admission gate:
python3 $G --match  --input <record.json>
python3 $G --commit insert --input <record.json>
```

Re-run the last three after every distillation batch.

### What each stage does, and what it refuses to decide

| stage | deterministic output | what it will not do |
|---|---|---|
| `make_schema.py` | `assets/schema/opt-trace-1.0.schema.json`, this corpus's narrowing of `clean-1.3`, from a declared patch list | invent a dialect: every patch only narrows, so passing the profile implies passing the store's schema |
| `ingest.py` | one row per version: verdict, geomean, per-shape latency, correctness, DSL per commit, which captures are usable | guess hardware, or believe the version's self-reported commit hash |
| `recon.py` | the evidence-density report: citable-number share, usable-capture share, what the live store already holds for this operator | decide anything; it exists so a human decides whether the trace is worth distilling |
| `partition.py` | one segment per record-to-be, with ids allocated above the store's existing maxima | judge whether a dead-end is a fact — it flags, the agent decides, the gate enforces |
| `build_packets.py` | a scrubbed, self-contained packet per segment, plus the diff as a sibling file | let a raw identifier reach the layer the agent reads |
| *(the agent)* | one record per packet | write code, invent numbers, or fill a field the packet does not support |
| `validate_store.py` | 16 gates and their injection tests | pass a record it cannot check against the trace |
| `score_records.py` | `worth.rank` and the staging `index.json`, using the store's own `wiki_score` | let an agent score its own record |
| `make_readme.py` | the reviewer's summary of the staging store | claim the records are in the store |

**Read `reports/<slug>/recon.md` before distilling.** It decides what the product
can honestly be: how many milestones carry a geomean (only those may claim
`basis=measured`), how many captures measured the kernel under test (only those
may back a `bottleneck`), and what the store already covers. A trace with almost
no numbers still has value as mechanism and anti-patterns — do not force a gain
claim onto every record.

## What a segment becomes

| segment | one record is | why |
|---|---|---|
| ratchet milestone | `strategy` | a version that set a new best-so-far. Carries code, so it needs a commit |
| dead-end | `anti-strategy` | **one per failed lever, not one per reverted commit.** A single reverted commit routinely lists three unrelated failures; keeping them together produces a record that matches three queries and answers none |
| curated pitfall | `anti-strategy` | mostly hangs off *kept* versions: the run shipped the change and separately wrote down what had not worked. The reverted path cannot see this knowledge |
| final kernel, mega snapshots | `reference-kernel` | the whole implementation, for reading rather than for a delta |

A trace cannot produce a `technique-card` (a cross-corpus aggregate) or a `doc`
(no measurement), and cannot produce a `generic`-level record: one kernel's
measurement is not evidence for every architecture. The profile enforces all
three.

### Why only a ratchet

A trace's latency series is not a progress curve — improvements and regressions
come in roughly equal numbers, and the large excursions are usually the
measurement environment, which the run itself often says so in prose. So a
per-step delta is meaningless and `ladder.py` selects only versions that set a new
best-so-far, with two guards: a jump beyond `REBASELINE_FACTOR` in either
direction is the harness re-baselining, and a version whose own text says it
changed nothing may take the new floor but may not claim it. `python3 ladder.py`
runs the self-test that pins this on a synthetic non-monotonic series.

## The sixteen gates

`validate_store.py` runs this repository's own `tools/check_kernel_wiki.py`
against the staging root, so all nine of its gates apply —
`schema` · `ids` · `anonymization` · `raw-isolation` · `relations` · `index` ·
`self-contained` · `no-cross-reference` · `established-fact` — and then seven that
only this pipeline can run, because only it has the trace and the packets:

- **profile** — the record satisfies `opt-trace-1.0`, which additionally requires
  the trace provenance triple, `measured_on`, `gain.kind`, and
  `established_fact` on every anti-strategy, and closes `evidence.raw` so a dead
  path cannot be reintroduced.
- **layout** — the directory equals the record's own scope, derived exactly as
  `wiki-gate` derives it on insert. The store's `ids` gate checks the filename but
  not the path.
- **verbatim** — `implementation.snippet` must appear literally in the packet's
  sibling diff or kernel file. The gate and the distiller read the same file on
  purpose. Compared line by line, so a snippet assembled from two hunks passes.
- **no-fabrication** — every number in `payload`, `worth.gain`,
  `evidence.summary` and `retrieval.signals.metrics` must appear in the packet or
  in its code. Code-ish fields are exempt because they are verbatim source.
- **provenance** — `evidence.raw` must name this trace, and its `git_commit` must
  resolve to a commit in it. This is the whole of a record's auditability once the
  packets are deleted.
- **ncu-attribution** — a `basis=profiler` claim must cite a capture that measured
  the kernel under test. A capture taken without a `--kernel-name` filter is
  schema-valid and describes the wrong kernel, so it is actively misleading rather
  than merely empty; without this gate such a record looks well-evidenced.
- **store-overlap** — the id must still be free in the live store, because
  `wiki-gate --commit insert` refuses a duplicate and renumbering after a batch is
  the expensive part. An `episode_key` that already exists is reported, not
  failed: whether it is a rediscovery to `confirm` is the agent's judgement.

**Never weaken a gate to make records pass, and never let a distilling agent edit
`scripts/`.** When a gate looks wrong, prove it fires:

```bash
python3 validate_store.py --injection-tests
```

Each case mutates a copy of a real record — and, where the error lives there, its
packet — and asserts the named gate complains. A gate without an injection test is
how a store ends up falsely green.

One gate the predecessor had is deliberately **gone**: it checked that every
record cited an existing markdown page. This repository has no markdown tree, so
that gate could only be satisfied by writing a citation to a file that does not
exist. Provenance replaced it.

## Distillation

Spawn agents with `references/distill-brief.md` **verbatim**. Batch by record type
so a failure has a small blast radius, and point every agent at the one record
that already passes as the worked example.

Require each agent to run `validate_store.py` itself and iterate to green, and to
report **which fields the packet was too thin to fill** and **which gate blocked
it**. That report is the main signal for improving the pipeline; treat a batch that
reports no difficulties with suspicion. When several agents write into one staging
store concurrently, tell them explicitly to ignore gate failures naming records
they do not own.

An anti-strategy segment whose evidence names neither a checkable condition nor a
cause **must not be written up at all**. `partition.py` marks those with
`fact_precheck`, but the flag is a hint, not a verdict: a regex must narrow and
never judge, so it also flags genuine facts whose wording is unusual, and the
agent resolves it from the packet's own evidence.

## How a record reaches the store

`skills/wiki-gate` is the only writer into `kernel_wiki/records/`. Nothing this
skill produces is served until it has been through the gate, whatever its own
gates say:

1. `gate.py --match --input <record.json>` returns every same-scope candidate plus
   any exact `episode_key` match. It makes no decision.
2. The agent decides: no match → `insert`; a match pointing the same way →
   `confirm` (bumps the existing record's counters, idempotently); a match
   pointing the opposite way → `conflict` (queued for a human, exit 0).
3. `gate.py --commit <action>` executes it. `insert` re-runs the store's
   record-level gates, refuses an id that already exists, writes the record under
   `records/<type>/<vendor>/<arch>/<dsl>/<operator_family>/` and appends the index
   entry.

A record rejected by the gate stays in staging. It is not deleted: once it gains
the condition and the mechanism it lacked, or is independently rediscovered, it
can go through again.

## Porting to another trace archive

Everything is trace-agnostic except three places:

- **`config.py`** — `TRACES` (where your traces live and what hardware they ran
  on), `TARGET_TABLE` (hardware token → vendor/arch/product), and
  `NON_TARGET_KERNELS`.
- **`families.py`** — operator naming: raw directory name → record slug and
  workload family. This is the only file that decides where a record is filed, so
  it is self-contained and self-tested rather than shared: a change in another
  tree would silently refile records. `python3 families.py` checks the slug rules
  and that every family it can emit is still a value the schema allows.
- **`ingest.py`** — the only file that knows how a trace is shaped. A different
  layout means adapting `read_commits` / `read_memory` / `read_profiles`;
  everything downstream sees version rows and never a raw file.

Two things to check on a new archive before trusting the output: whether the
commit-subject grammar carries the verdict at all (if not, nothing can supply the
kept/reverted split), and what fraction of profiler captures measured the kernel
under test — `recon.py` prints both.

## Resources

- `references/distill-brief.md` — the agent brief. Pass it verbatim.
- `assets/schema/opt-trace-1.0.schema.json` — the corpus profile, generated by
  `make_schema.py`; run it with `--show` to read the patch list and the reason for
  each patch.
- `skills/wiki-gate/references/established-fact-criteria.md` — the normative
  admission bar for negative knowledge; `partition.py` imports the same regexes
  the store's gate uses, so triage and enforcement cannot disagree.
- Self-tests worth running after any edit: `python3 families.py`,
  `python3 ladder.py`, `python3 anonymize.py`.
