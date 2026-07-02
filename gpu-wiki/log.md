# GPU Wiki Log

Append-only, chronological record of what changed in this wiki and when. Newest
entries go at the top. Each entry starts with a fixed, grep-able prefix so the
history can be sliced with plain tools:

```text
## [YYYY-MM-DD] <op> | <subject>
```

`<op>` is one of `ingest`, `query`, `lint`, `normalize`. List the docs that were
touched and any `RELATIONS.md` updates in the body. See `CLAUDE.md` for the
operations that append here.

```text
# last 5 entries
grep "^## \[" log.md | head -5
# everything ingested
grep "^## \[" log.md | grep " ingest | "
```

---

## [2026-07-01] merge | adopt PR #2 vendor-first reorg + rebuild query

Merged PR #2 (vendor-first + NVIDIA architecture-first restructure; removed
redundant non-kernel-optimization docs). Rebuilt the governance layer for the new
layout: `query.py` now matches by path segment so `--arch blackwell` excludes
`blackwell-geforce` (sm120) and Hopper; `build_index` groups by vendor/architecture;
regenerated `docs/index.md`; updated the `CLAUDE.md` taxonomy, README entry points,
and the `gpu-kernel-research` agent L1 scope. All checkers and unit tests pass.

## [2026-06-30] ingest | architecture-scoped query tool

Added `scripts/query.py` (+ `scripts/test_query.py`): keyword search over `docs/`
filtered by `--arch` / `--vendor` / `--dsl`, derived from the path taxonomy, so a
Blackwell query never returns Hopper or CDNA pages (architecture-neutral and
general pages are always included). Wired it into the `gpu-kernel-research` agent's
L1 retrieval step and the `CLAUDE.md` Query operation, added a README routing row,
and added the unit test to the CI gate.

## [2026-06-30] normalize | page-schema sweep + strict structural gate

Normalized page conventions across `docs/`: renamed every `## Related Docs` /
`## Related Documents` heading to `## Related`, fixed `**Last Updated**` →
`**Last updated**` casing, and added a missing `# H1` title (plus a one-line
summary where the body was table/code-first) to 11 pages. Made `build_index` and
`check_structure` fence-aware so `#` lines inside code blocks are no longer
mistaken for titles, regenerated `docs/index.md`, added
`scripts/test_check_structure.py`, and flipped the CI gate to
`check_structure.py --strict` (title + summary now enforced; missing-`## Related`,
orphans, and RELATIONS staleness remain advisory).

## [2026-06-30] lint | bootstrap structural baseline

Ran `scripts/check-self-contained.py` (green) and `scripts/check_structure.py`.
Baseline structural findings: `missing-related` on most legacy pages,
`missing-summary` on a small set, one `missing-h1`
(`docs/ref-docs/nvidia/common/nvidia-ptx-sync-and-async.md`), zero orphans. These
are tracked as non-blocking warnings until the Phase 2 normalization sweep.

## [2026-06-30] ingest | governance layer (index.md, log.md, schema)

Added the wiki-governance layer: generated `docs/index.md` (flat catalog),
created this `log.md`, rewrote `CLAUDE.md` into the operations schema, and pointed
`AGENTS.md`/`README.md` at them. No content pages moved; directory taxonomy and
relative-link convention unchanged.
