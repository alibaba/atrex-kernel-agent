---
name: session-trace-mining
description: Mine AI coding-agent session transcripts into structured, gate-validated GPU-kernel optimization records for the wiki. Use when asked to turn vibe-coding sessions, Codex rollout logs, or Claude Code project transcripts into wiki records; to summarise what a kernel-optimization session achieved; to build or extend a session-trace store; or to re-run and validate one. Also use when asked how a session-derived record's number, snippet, or provenance was established.
---

# Session Trace Mining

## Overview

Turns session transcripts into `session-trace-1.0` records: one record per change
that measurably moved a metric, answering which operator, what was wrong, what
changed, why it worked at the machine level, the verbatim code, and the gain.

The pipeline splits three ways: **scripts do deterministic extraction, an agent
does semantic distillation, and sixteen mechanical gates catch fabrication.** The
gates are why the output can be trusted. A model filling fields is the weak link,
so nothing enters the store that cannot be checked against the transcript.

What makes a transcript corpus different from a repository of commits: **there is
no repository to check a citation against.** The transcript *is* the corpus. So
provenance is a set name plus a set-relative path plus a per-line digest, and every
number must trace to a span the packet carried — with the agent's own prose
deliberately excluded from that span set.

## Why every part of it lives here

The same shape of problem — extract deterministically, distil semantically, gate
mechanically — applies to any corpus of optimisation history. What travels between
such pipelines is **design**, not code: the three-way split, the declarative
`make_schema.py` patch list, the packet contract, and the discipline of writing
down every pitfall with the number behind it.

**Nothing is imported.** This skill is self-contained, deliberately:

| concern | here | why not shared |
|---|---|---|
| operator naming, workload families | `families.py` | the layout gate derives a record's directory from these functions, so a change in another tree would silently refile records |
| milestone selection (the ratchet) | `ladder.py` | its thresholds decide this store's record set; and an A/B corpus with no ladder pushes back on rules that have no business knowing about it |
| ranking model | `score.py` | a re-ranked store with no diff to show for it is the worst kind of drift |

Each of the three pins its own behaviour with a `self_test()` that runs standalone
(`python3 families.py`, `python3 score.py`). `score.py` reproduces the curve
published by the shared ranking model in `tools/wiki_score.py` for the inputs this
corpus produces — 4% → 0.35, 20% → 0.66, 99% → 1.0, and the 0.35 feedback band —
so scores stay comparable with the committed store **by verification rather than by
coupling**. If the shared model changes, that self-test is where the divergence
surfaces.

The only paths outside the skill are the two *data* locations declared in
`config.py`: where the product goes (`kernel_wiki/session_trace/<set>/`) and the
dedup / overlap scan over `kernel_wiki/records`. No logic crosses the boundary.

## Setup

One thing has to be configured, because no transcripts ship with this repository:

1. point `STM_ROOT` at the directory holding your own transcript archive;
2. register your sets in `scripts/config.py` `SETS`, replacing the single
   `example-set` placeholder entry. A set is named, never passed as a path, so a
   record's provenance survives moving the archive.

Both output locations are derived and need no configuration:

- **scratch** — `/tmp/session-trace-mining/<set>/`: parsed candidates, segments,
  packets. Reproducible, never committed. Override with `STM_WORKSPACE`.
- **product** — `<gpu-wiki>/kernel_wiki/session_trace/<set>/`: records and the
  reports that justify them. Override with `STM_STORE`. It sits beside the
  committed store but **not** inside `kernel_wiki/records/`: these records use the
  derived `session-trace-1.0` schema, so filing them into the committed store would
  make that store fail its own schema gate. Promoting one means rewriting it
  against `schema/kernel/schema.json`, deliberately and one at a time.

Plain `python3` is enough. The `schema` gate additionally needs `jsonschema`;
without it that one gate reports SKIP and the other fifteen still run.

## Pipeline

```bash
S=<skill-dir>/scripts
export STM_ROOT=/path/to/your/transcript-archive
export STM_SET=example-set             # a key of config.SETS -- register your own

python3 $S/make_schema.py --check      # schema matches its patch list
python3 $S/ingest.py                   # transcripts -> work/versions.jsonl
python3 $S/recon.py                    # -> reports/recon.md   READ THIS FIRST
python3 $S/partition.py                # -> work/segments.jsonl, reports/partition.md
python3 $S/build_packets.py            # -> packets/<seg>.{json,diff}
# distil (see below), then:
python3 $S/validate_store.py --verbose          # sixteen gates
python3 $S/validate_store.py --injection-tests
python3 $S/score_records.py            # worth.rank.score + records/index.json
python3 $S/make_readme.py              # -> kernel_wiki/session_trace/<set>/README.md
```

Re-run the last two after every distillation batch.

**Read `reports/recon.md` before distilling.** It decides what the product can
honestly be: the citable-number share, the diff-coverage share, and how much the
committed store already covers. If a set turns out to have almost no numbers, its
value is mechanism and anti-patterns — do not force a gain claim onto every record.

## Two candidate units

The unit is a property of the corpus, not a preference, and it is declared per set
in `config.SETS`. Getting it wrong yields either nothing or nonsense, so it was
settled by a probe before any schema existed (see `references/lessons.md` §1–2).

| unit | when | what one record is |
|---|---|---|
| `version-ladder` | the run keeps a numbered ladder (`memory/vN.json` + `vN:` commit subjects) | one version, assembled **across the whole set** — one session is one version, so the ladder does not exist inside a single file |
| `ab-comparison` | no ladder | one measured A/B: a variant comparison printed complete in one output, or the same benchmark run either side of an edit |

## Evidence tiers

The single most important design decision. Every span carries a tier, and only
three of five may be cited:

| tier | content | citable | caps `gain.basis` at |
|---|---|---|---|
| T1 | benchmark / profiler stdout | yes | `measured` |
| T2 | the agent reading back its own notes (`cat NOTES.md`, `git log`, `Read memory/vN.json`) | yes | `reported` |
| T3 | an agent-authored structured field | yes | `reported` |
| T4 | agent prose and thinking | **no** | — |
| T5 | the orchestrator prompt | **no** | — |

T4 is excluded because admitting it makes the fabrication gate vacuous: the
agent's invented number becomes its own proof. T5 is excluded because those
prompts state the target percentage, which would license any number near it.

## The sixteen gates

`schema` · `ids` · `layout` · `provenance` · `verbatim` · `no-fabrication` ·
`direction` · `raw-isolation` · `relations` · `index` · `evidence-tier` ·
`diff-coverage` · `unit-normalization` · `wiki-overlap` · `pairing-integrity` ·
`anonymization`

The six that carry the weight:

- **provenance** — re-resolves the set by name, the file by set-relative path, and
  every cited line by `sha256(raw line)[:12]`. Retargeting a citation fails; moving
  the whole archive to another absolute path still passes. A cited line with *no*
  digest is a failure too — without that clause the check silently does nothing,
  which is what the line-shift injection caught.
- **verbatim** — `implementation.snippet` must appear literally in
  `packets/<seg>.diff`. The gate and the distiller read the same file on purpose.
- **no-fabrication** — every number in `worth.gain` must be in the packet's
  `evidence_text`, or derivable from it by one of exactly two closed-form rules
  (before/after, or `speedup=Nx`). Derivation from arbitrary *pairs* of pool
  numbers is deliberately not allowed.
- **unit-normalization** — the delta must be reproducible from the measured levels,
  and no absolute time may appear inside `worth.gain`.
- **wiki-overlap** — nothing the committed store already covers: a colliding
  `(operator, version)` dedup key, a record id, or an `episode_key`. It scans
  `kernel_wiki/records`, and **an empty scan is a failure, not a pass** — an
  overlap gate whose index resolved to nothing would print OK forever.
- **anonymization** — the served layers must name no person, host or corpus path.
  Session transcripts are full of `/home/<user>/...`, and `payload` is what gets
  served. The record `id` is checked separately, because a home path flattened into
  a slug has no slash left for the path pattern to catch.

**Never weaken a gate to make records pass, and never let a distilling agent edit
`scripts/`.** When a gate looks wrong, verify by injection:
`validate_store.py --injection-tests` mutates a record (and, where the error lives
there, its packet) and asserts that the named gate complains. Adding a gate without
an injection test is how a store ends up falsely green — two of the eleven cases
here were asleep on their first run.

## Distillation

Spawn agents with `references/distill-brief.md` **verbatim**, substituting the
placeholders. Batch by set and record type so a failure has a small blast radius,
and point every agent at the one record that already passes as the worked example.

Require each agent to run `validate_store.py` itself and iterate to green, and to
report **which fields the packet was too thin to fill** and **which gate blocked
it**. That report is the main signal for improving the pipeline; treat a batch that
reports no difficulties with suspicion.

When several agents write into one store concurrently, tell them explicitly to
ignore gate failures naming records they do not own.

## Porting to another corpus

Everything is corpus-agnostic except two places:

- **`scripts/config.py` `SETS`** — register the set: its path under the archive
  root, its transcript format, its candidate unit, and default scope. Defaults are
  fallbacks only; `ingest.py` detects hardware and DSL and records which happened
  in `arch_basis` / `dsl_basis`.
- **`scripts/transcripts.py`** — the only file that knows how a session log is
  shaped. A third agent product means one new `parse_*` function returning the same
  `Event` stream, plus a branch in `detect_format`. Everything downstream sees
  events and never a raw line.

Two corpus-specific details that will need attention on a new corpus: how a
long-running benchmark's output is linked back to the command that launched it (in
Codex logs it is a `SESSION_ID=N` handshake), and which label words name a whole
implementation rather than a knob (`partition.IMPL_LABEL_RE`).

**Decide the unit before writing any schema.** It is what ids, pairing,
`dedup_key` and the whole `worth.gain` ladder key on, so getting it wrong means
re-doing the schema, the packets and three gates. Three probes exist for exactly
that decision and should be re-run on a new corpus:

```bash
python3 $S/probe_versions.py <transcript> [...]   # per file: versions, edits, metrics, pairing
python3 $S/probe_set.py <set-root>                # does the ladder exist across the set?
python3 $S/probe_ab.py <set-root>                 # if not, how many measured A/Bs are there?
```

`probe_versions.py` answers "can I see versions in one file"; `probe_set.py`
answers the question that actually matters for a ladder, since one session is one
version; `probe_ab.py` sizes the fallback. Write the pass bars down before running
them, and if a corpus fails its bar, change the unit rather than the bar.

## Resources

- `references/lessons.md` — **read before starting.** Every pitfall found while
  building this, with the measured numbers behind each: why one session is one
  version, why the codex sets cannot be paired before/after, the `s`-for-seconds
  trap, the table column that inherited a unit it did not have, and the two gates
  that were asleep.
- `references/distill-brief.md` — the agent brief template.
- `assets/schema/session-trace-1.0.schema.json` — the record schema, generated by
  `make_schema.py` from `clean-1.3.frozen.json`, which is a pinned byte copy of
  this repository's `schema/kernel/schema.json`.
