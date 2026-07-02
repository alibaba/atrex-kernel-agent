# GPU Wiki Schema

This is the maintenance schema for the GPU Wiki: how the knowledge base is
structured and how to keep it current. Read [`README.md`](README.md) first for
consumer-facing routing (how to *find* knowledge); this file governs how to
*maintain* it. The wiki is a compounding artifact — knowledge is integrated once
and kept current, not re-derived per task. You write and maintain the wiki; a
human curates sources and asks the questions.

After any change to paths, links, or the index, run the checks in the **Lint**
operation below.

## Three layers

- **Raw sources (immutable).** Ground truth the wiki is distilled from, never
  rewritten by maintenance: runnable code in [`reference-kernels/`](reference-kernels/),
  community wikis in [`3rdparty/`](3rdparty/), upstream project checkouts (referred
  to by environment variable, not copied in), and vendor official documentation.
- **The wiki (maintained).** Everything you write and keep consistent: the pages
  under [`docs/`](docs/), every directory `README.md` catalog, the flat catalog
  [`docs/index.md`](docs/index.md), and the synthesis/conflict map
  [`docs/RELATIONS.md`](docs/RELATIONS.md).
- **The schema (this layer).** The governing configuration: this file,
  [`AGENTS.md`](AGENTS.md), and the routing guide [`README.md`](README.md).

## Navigation surfaces

| Surface | Role |
|---|---|
| [`README.md`](README.md) | Consumer routing: five-tier retrieval (P0–P5), symptom-driven keyword tables |
| directory `README.md` files | Local catalog of one directory's children |
| [`docs/index.md`](docs/index.md) | Flat catalog of every `docs/` page with a one-line summary (generated) |
| [`docs/RELATIONS.md`](docs/RELATIONS.md) | Cross-architecture relations, conflicts, reading paths — the synthesis layer |
| [`log.md`](log.md) | Append-only chronology of ingests, queries, and lint passes |

`docs/RELATIONS.md` owns contradictions and cross-architecture differences.
Record conflicts there; do not restate them elsewhere.

## Operations

### Ingest

When integrating a new optimization report, pitfall, hardware fact, or reference:

1. **Route** to the narrowest existing directory by the vendor-first taxonomy
   `Vendor → Architecture (NVIDIA) / DSL (AMD) → Topic` (e.g. `docs/nvidia/blackwell/…`,
   `docs/amd/flydsl/gfx942/…`, `docs/generic/…`). Reuse a directory;
   do not invent a parallel one.
2. **Write or extend** the page using the template below. A single source often
   touches several pages — update the entity/topic pages it affects, not just one.
3. **Catalog it**: add a row to the nearest directory `README.md`.
4. **Refresh the flat index**: `python3 gpu-wiki/scripts/build_index.py --root gpu-wiki`.
5. **Cross-link**: add a `## Related` entry on the new page and a reciprocal link
   from the pages it relates to (relative markdown links only).
6. **Record conflicts**: if the source contradicts or qualifies existing guidance,
   update `docs/RELATIONS.md`.
7. **Append** an `ingest` entry to [`log.md`](log.md).
8. **Lint** (below).

### Query

When answering a question against the wiki:

1. Start from [`README.md`](README.md) routing, then [`docs/index.md`](docs/index.md)
   to locate candidate pages, then read them.
2. For architecture-scoped search use `scripts/query.py` — e.g.
   `python3 gpu-wiki/scripts/query.py "bank conflict" --arch blackwell` returns
   Blackwell and architecture-neutral pages and never Hopper/CDNA ones (accepts
   sm/gfx/chip aliases; `--vendor` and `--dsl` narrow further). The
   `gpu-kernel-research` agent uses this as its L1 retrieval step.
3. Consult [`docs/RELATIONS.md`](docs/RELATIONS.md) for any cross-architecture or
   cross-DSL task before transferring a conclusion.
4. If the answer is durable (a comparison, a synthesis, a newly discovered
   connection), file it back as a page via **Ingest** so it compounds instead of
   disappearing.

### Lint

Run after path/link/index changes, and periodically as a health check:

1. `python3 gpu-wiki/scripts/check-self-contained.py --root gpu-wiki` — link
   integrity and self-containment (blocking gate).
2. `python3 gpu-wiki/scripts/build_index.py --root gpu-wiki --check` — index is
   in sync with the tree.
3. `python3 gpu-wiki/scripts/check_structure.py --root gpu-wiki` — structural
   warnings: missing title/summary/`## Related`, orphan pages, RELATIONS staleness.
4. Review the warnings and the RELATIONS staleness hint; fix what is actionable.
5. Append a `lint` entry to [`log.md`](log.md).

## Page template

A new page should follow this shape (single H1, one-line summary, one metadata
style, a bottom `## Related` section using relative links):

```text
# <Page Title>

<One sentence: what this page covers.>

**Last updated**: YYYY-MM-DD

## <Body sections...>

## Related
- [<sibling or prerequisite page>](relative/path.md)
- [Cross-architecture differences](../../RELATIONS.md)
```

## Conventions

- One `# H1` title per page; a one-line summary directly under it.
- One metadata style: `**Last updated**: YYYY-MM-DD` (lowercase "updated").
- Cross-references are **relative markdown links** that resolve to a file inside
  the wiki — never `[[wikilinks]]` (the link checker cannot validate those) and
  never bare inline code names (they rot silently).
- Keep one focus per file; put overviews in directory `README.md` files.
- Do not add YAML frontmatter containing both `name:` and `description:`, and do
  not name any file `SKILL.md` — both are reserved for skill packages and are
  rejected by the self-containment gate.
- `docs/index.md` is generated; never hand-edit it.
