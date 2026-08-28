# GPU Wiki Compatibility Index

Read `README.md` and `AGENTS.md` first. Two independent JSON record stores:

- **Experience** — `python3 gpu-wiki/tools/query_wiki.py` over
  `kernel_wiki/records/`, with explicit `--arch` / `--vendor` / `--dsl` /
  `--type` filters before broad grep. `--arch` accepts what the runtime reports
  (`sm_90`, `gfx942`, `h20`, `b300`) as well as the family name (`hopper`,
  `cdna3`). Scope is a hard boundary, unknown filter values fail closed, and a
  zero-match query returns a labelled random sample, not matches.
- **Facts** — `python3 gpu-wiki/tools/query_hardware.py` over
  `hardware_wiki/records/`, e.g.
  `--product b200 --field peak_compute.bf16.dense`. Exact lookup, no ranking, no
  fallback. An unrecorded part exits 4 with a procedure for obtaining the number;
  never borrow another part's.

Prefer vendor official documentation for API and ISA ground truth. After path,
record, or index changes run `python3 gpu-wiki/tools/check_kernel_wiki.py --full`,
`python3 gpu-wiki/tools/check_hardware_wiki.py`, and
`python3 -m unittest discover -s gpu-wiki/tools`.
