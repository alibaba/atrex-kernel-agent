# Lessons: mining GPU-kernel knowledge out of agent session transcripts

Every entry is something that cost time or would have produced a wrong record.
The numbers are measured on real transcript corpora, not estimated. Read this
before touching `ingest.py` or the gates.

None of those corpora ships here, so they are referred to by what they are:

| label | what it is |
|---|---|
| ladder set A | a Claude Code project directory, 98 transcripts, keeps a numbered version ladder (max v59) |
| ladder set B | a second Claude Code project directory, 126 transcripts, ladder reaches v98 |
| codex A/B set | six Codex rollout logs of an FP8 attention kernel, no ladder |
| codex GEMM set | two very large Codex rollout logs (21k and 30k lines), GEMM work |
| claude A/B set | a Claude Code set with no ladder at all (gated-delta-net + causal conv) |

## 1. A session is not a ladder — one session is one version

The first probe asked "how many versions can I recover from this file, each with
an edit and a number nearby". Both bar files failed:

| file | versions seen | paired |
|---|---:|---:|
| ladder set A, one transcript | 13 | **1** |
| codex A/B set, one transcript | 3 | **0** |

The ladder-set number is not a corpus problem, it is a units problem. Those runs
are driven by an orchestrator that constrains each session to **exactly one**
optimisation cycle — profile, pick one lever, edit, validate, bench, record, then
exit. So one session file holds **one** version's work, and the twelve other
versions found in it all came from a single `git log --oneline` dump on line 17 —
that is *history*, not this session's content. Pairing per file therefore had to
fail, and would have failed no matter how good the parser was.

Re-probed at set level, attributing each version to the session that produced it:

| set | files | ladder versions (git-log echoes) | attributed to a session | edits+metrics in one region | with a verbatim diff |
|---|---:|---:|---:|---:|---:|
| ladder set A | 98 | 49 (max v59) | 44 | 38 | **37** |
| ladder set B | 126 | 75 (max v98) | 61 | 59 | **57** |

So the ladder is real, but it only exists **across a whole set**. Consequences
baked into `ingest.py`:

- the corpus-wide `git log --oneline` echoes are the ladder skeleton (subject +
  sha + kept/reverted verdict for every version, including versions whose session
  is not in the archive);
- each session contributes *its own* version's edits and numbers, found by
  `own_version()`: writing `memory/vN.json` (strongest), `git commit … vN:`
  (39/44 and 57/61 of attributions), or `memory_manager.py create|update vN`;
- ~55% of sessions own no version at all (54/98 and 64/126). They are planning,
  profiling and research sessions. They are not failures to parse and must not be
  counted as such — they become packet evidence for the version they discuss.

## 2. The codex sets have no ladder, so they get a different unit

The codex A/B set, all 6 files: `[vN]` headings appear 3 times (all in the one file
that keeps a `NOTES.md` beside the kernel), `git log` version subjects 0 times. But
the same set has 70 code edits with verbatim `@@` diffs and 72 metric-bearing tool
outputs in a single file. Measured A/B yield:

| signal | total |
|---|---:|
| `edit → metric before and after, same monotone region` | **76** |
| arrow form `A → B us` in tool output | 39 |
| one-line variant comparison (`static= … clc= … speedup=…x`) | 3 |

Hence two candidate units, not one: `version-ladder` for the ladder sets and
`ab-comparison` for the codex A/B and claude A/B sets. This split was decided
**before** the schema was written, which is the only reason it cost one afternoon
instead of a rewrite: the unit is what `id`, `dedup_key`, pairing and `worth.gain`
all key on.

`edit_paired=76` is an **upper bound and must not be trusted as a candidate
count**: the probe accepted any timing before and any timing after the edit,
anywhere in the region. The real ingest additionally requires the same benchmark
identity on both sides (same command, same shape set), which will cut it down.

## 3. `turn_id` is missing exactly where you need it

In codex transcripts `turn_id` is present on `turn_context`, `task_started`,
`patch_apply_end`, `task_complete` and `turn_aborted` — and on **none** of the
`response_item` types (`message`, `reasoning`, `custom_tool_call*`). So a tool
output cannot be attributed to a turn by reading its own fields; turns are line
ranges between markers and everything inside is attributed positionally.

## 4. Line order is not history

Codex emits `compacted` / `context_compacted` (4 in one file of the codex A/B set,
8 discontinuities total in the largest) and `thread_rolled_back`; claude has
`file-history-snapshot`/`-delta`. Across one of them, an earlier line can
describe a later state. `regions()` splits the event stream at every
discontinuity and a candidate is only formed inside one region. `turn_aborted`
(6 in the same file) means an `apply_patch` landed but the conclusion never did —
those turns are excluded entirely.

## 5. The agent's own prose must never be citable

The trust tiers in `transcripts.py` exist because of this: if `assistant` text
(T4) enters the no-fabrication haystack, the agent's invented number becomes its
own proof and the gate is decorative. The orchestrator prompt (T5) is excluded for
the mirror-image reason — it states the target percentage and the condition for
committing, which would license any number near that threshold. Measured tier mix
on the largest file of the codex A/B set: T1 1125, T2 29, T4 396, T5 147. Nearly a
third of the text is agent prose, so this is not a theoretical exclusion.

The T1/T2 split is by *command*, not by output: `sed -n '1,260p' NOTES.md`,
`Read memory/vN.json` and `git log` all return text the run wrote about itself.
Those numbers are real quotes but they are the agent's own bookkeeping, so they
cap `gain.basis` at `reported`.

## 6. Strip the harness preamble before pooling numbers

Every codex tool output begins `Script completed\nWall time 8.1 seconds\nOutput:`.
Left in, `8.1` is a citable magnitude in the haystack and would let a record
"source" an 8.1% claim from a stopwatch reading. `CODEX_PREAMBLE` removes it.

## 7. Prefer the harness's own patch to a reconstructed one

Claude `Edit` gives `old_string`/`new_string` (100% present: 252/252, 377/377,
97/97), but the paired result line also carries `toolUseResult.structuredPatch`
with pre-computed hunks whose `lines` already have the ` `/`+`/`-` prefixes.
Rendering that is verbatim; assembling a diff from the two strings is not, and
the edit event records which channel was used (`verbatim: true|false`) so the
gate can tell the difference. `MultiEdit`/`NotebookEdit` do not occur in any of
these corpora (0 across all three claude sets) — do not write code for them.

## 8. Shell writes are a blind channel

Codex `cat >` / `tee` edits produce no `patch_apply_end` (up to 285 per file in
the codex GEMM set; 38 in one file of the codex A/B set). Those candidates carry
`diff_coverage: "blind"` and the `diff-coverage` gate forbids them from being
`strategy` records, because a strategy record is schema-required to carry a
verbatim implementation. Blindness is counted in `reports/recon.md` rather than
silently tolerated.

## 9. AppleDouble files double the apparent corpus

`._*` resource forks and `.DS_Store` are everywhere in an archive that has been
copied through a Mac and a zip: the raw file counts of the two ladder sets are 206
and 344 where the real counts are 98 and 126. `iter_transcripts()` skips them.
Top-level `*.zip` / `*.tar.gz` entries sitting next to the extracted directories
are usually verified duplicates of them — list those in `config.EXCLUDED` by name
so a glob over the archive root can never silently double-count a corpus.

## 10. Never guess a unit from magnitude

ncu CSV reports `gpu__time_duration.sum` in **ns**, the decode markdown tables are
in **ms**, the prefill log lines are in **us**, and all three coexist in one
session. A block with a number and no unit anywhere in it is dropped, not
guessed: guessing wrong is a 1000× error published as fact. The `+3.45%` next to
`0.027928 → 0.026997 ms` is a *ratio* (before/after − 1 = 3.45%), not
(before−after)/before (3.33%), so the derivation check must accept either
denominator and record which one was used in `delta_basis`.

## 11. A/B packets carry no narrative, so `mechanism` has to be rebuilt from code

Both distilling agents on the codex A/B set reported the same gap independently,
and it was their largest: for `ab-comparison` candidates every `narrative` field is
null (`subject`, `action_description`, `expected_impact`, `profile_evidence`,
`open_directions`), so `mechanism` had to be reconstructed from the diff plus the
T1/T2 log text, and `payload.next_steps` came out `[]` on all nine.

Version-ladder packets have the opposite problem — none. There `memory/vN.json`
supplies the run's own account of what it changed, why it expected that to work,
and what it would try next. The gap is structural: a codex A/B has no such
document.

The fix is not to let the agent fill the hole from general knowledge. Either the
agent-authored notes near the comparison get attached as narrative (they exist in
these transcripts at tier T3 — `NOTES.md`, `opt_logs_v2.md`), or A/B records are
accepted as thinner and say so. A guessed mechanism is the one failure mode no
gate can catch.

## 12. The same change measured at several shapes makes near-duplicate records

Three records for CLC-vs-static differing only in sequence length (11.13% / 7.09%
/ 0.54%), and three page-size records sharing one diff and one 45-shape matrix.
Both agents nominated one of their own records as the least valuable for exactly
this reason, and one named the real cost: a trio whose only difference is a number
invites a spurious "page X is worse" reading that the evidence does not support.

Candidates sharing a `(baseline_label, candidate_label)` pair and a diff should be
merged into **one** record carrying several shapes — which is what
`problem.shape_contract.n_benchmarked_shapes` and `measured_over` exist for. The
current `partition.py` emits one per comparison instance and leaves the merge
undone.

## 13. Detect a language from the kernel, never from the harness

`detect_dsl` matched the bare word `triton`, so a CuTeDSL kernel benchmarked with
`triton.testing.do_bench` was filed under `dsl=triton`. A distilling agent caught
it, declined to assert `problem.dsl`, and reported it — the brief's reporting
requirement earning its place. After requiring a real kernel signal
(`@triton.jit`, `tl.*`, a `triton.language` import) and explicitly ignoring
`triton.testing`, all six transcripts of that set classify as `cutedsl`.

## 14. The packet diff is code, not a source of numbers

An agent wrote a note citing the timed-iteration count it had read out of the
packet diff, and `no-fabrication` rejected it: the audit pool is `evidence_text`
plus the extracted `measurement`, and the diff is deliberately not in it. The gate
was right by its own contract, but the failure reads as a contradiction because the
number *was* in the packet. Resolved in the brief rather than in the gate — the
diff is the only legal source for `implementation.snippet` and never a source of
numbers. Widening the pool to include it would let any constant in the code back a
performance claim.

## 15. The deepest hole: a pairing that compares two unrelated things

A comparison whose sides were chosen by print order can be two things that are not
alternatives at all, and **no mechanical check can tell**. A distilling agent found
two in one corpus:

- `[CP_PROFILE] mn=313us carry=66us out=172us` — a three-phase CUDA-event
  breakdown of a single call, read as "phase `mn` versus phase `carry`", 78.9%;
- an approximate path against a vendor library that computes a different result,
  -99.5%, with the direction coming from print order alone.

Both passed every gate silently, including `no-fabrication` (the numbers are real),
`direction` (the sign is consistent) and `unit-normalization` (the arithmetic
checks out). The lie is in the *pairing*, which is a semantic claim.

The fix cannot be a check that the sides are comparable — that needs comprehension.
It is structural: `side_basis: "printed-order"` now forbids a gain claim outright,
so such a candidate becomes an `anti-strategy` with `kind: none` that describes the
finding in prose. On the claude A/B set this turned five strategy records into zero
and seven anti-strategy ones. That is the honest yield of that corpus, and it took
an agent withdrawing its own numbers to establish it.

## 16. `rel_path` and `session_id` must name the same transcript

After the primary-transcript fix (§ above), the two could diverge, and `provenance`
correctly rejected a record whose `session_id` had been scraped from an
`originSessionId:` field *inside a memory file the run had read* rather than from
the transcript being cited. The agent declined to edit the provenance block to make
it pass and reported it instead, which is why the bug was found rather than buried.
The id is now taken from the transcript that `rel_path` points at.

## 17. A gate that is too strict looks identical to one that is too loose

Extending `unit-normalization` to audit `gain.regressions` — a reasonable-sounding
suggestion from a distilling agent — rejected three correct records. A worst-shape
regression (-4.4%) describes a *different subset* of shapes than the headline
geomean (+5%), so its levels are not the headline levels and it cannot reproduce
from them. Reverted to `metrics` only, with the sign left to `direction` and the
magnitude to `no-fabrication`.

Scope a check to the claim it can actually verify. From the outside, a gate
rejecting good records is indistinguishable from a gate catching bad ones; only the
injection test and a reader of the failure message can tell them apart.

## 18. A skill must be self-contained, and the rewrite pays for itself

The first version imported three things from other trees: milestone selection
(`ladder.py`) from a sibling mining pipeline, and `slugify`/`family_of` plus the
shared score model from the wiki's own tooling. The instinct was good — scores and
operator names *should* agree across stores — but the mechanism was wrong on three
counts:

- the **layout gate derives a record's directory** from `slugify`/`family_of`, so a
  change in another tree would silently refile records with no diff here to show it;
- the **ladder thresholds decide which versions become records at all**, so this
  store's contents were hostage to a module with no stake in them;
- a **re-ranked store has nothing to show for it**: scores would move and no commit
  here would explain why.

Re-implemented as `families.py`, `ladder.py`, `score.py`, each with a `self_test()`
that runs standalone. Verified by equivalence rather than by assertion: segment ids
came out **byte-identical for all three record-bearing sets**, and the score
histograms were unchanged (`{0.1:1, 0.4:4, 0.5:4, 0.6:2, 0.9:1}` on the codex A/B
set). `score.py` pins the shared model's published curve — 4% → 0.35, 20% → 0.66,
99% → 1.0 — so comparability survives as a checked property instead of a
dependency.

The rewrite also **found two naming bugs** that the borrowed code could not have
surfaced, because nothing in the other tree tested them against this corpus:

- the operator vote picked the *packaging* repo the sessions happened to run in —
  four of the six transcripts of one set — over the kernel's own name, filing the
  whole set under family `misc`. Fixed by preferring a name that `family_of` can
  classify, on the reasoning that a name mapping to a known workload family is far
  more likely to be an operator than a repo.
- a claude project directory flattened from a home path (`-root-<user>-<project>`)
  slugged straight through into a record id, which is the same leak the
  anonymization gate exists to stop, arriving by a different route. Fixed by
  stripping leading container segments (`root`, `home`, `workspace`, …) in
  `normalise()`.

Both were latent for as long as the logic lived elsewhere. Owning the file is what
made them visible.
