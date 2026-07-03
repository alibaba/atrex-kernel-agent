# ncu_helpers

Bundled helpers for parsing NVIDIA Nsight Compute (`ncu`) reports. Used by
`tools/profile_iter_nvidia.sh` (metrics parsing + stall hotspots) and
`tools/extract_nvidia_asm.py` (`--ncu-rep` mode).

- `analyze_reports.py` — parse a `.ncu-rep` into key/all metrics JSON + text.
- `extract_stall_hotspots.py` — per-source-line stall hotspots from a `--set source` report.
- `ncu_utils.py` — shared helpers; imports the `ncu_report` module that ships with
  Nsight Compute (must be importable at runtime on the CUDA host).

Vendored from the `ncu-report-skill` submodule
(https://github.com/DongyunZou/ncu-report-skill), as checked out under the local
`kernel-design-agents` workspace. Override the lookup with the `NCU_HELPERS` env var
or `--ncu-helpers` flag if you keep a newer copy elsewhere.
