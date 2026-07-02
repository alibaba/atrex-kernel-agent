# GPU Wiki Agent Entry

Read [`README.md`](README.md) first for routing (how to find knowledge), and
[`CLAUDE.md`](CLAUDE.md) for the maintenance schema (the three layers and the
Ingest / Query / Lint operations).

Follow the self-containment contract: use relative in-wiki links, treat external
checkouts as explicit environment variables, and after any path or index change
run `python3 gpu-wiki/scripts/check-self-contained.py --root gpu-wiki` and
`python3 gpu-wiki/scripts/check_structure.py --root gpu-wiki`.
