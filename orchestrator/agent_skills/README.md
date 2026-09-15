# Agent Skill assets

These are Agent-facing resources, not developer Skill installations. The exact installed files are
listed in `orchestrator/agent_skill_manifest.py`. The installer does not recursively copy a Skill
directory: new upstream files require an explicit manifest change. Missing listed files, invalid
paths, and symlink sources fail before publishing a new snapshot. This README is not installed.

## Ownership and retained files

| Skill | Agent-facing files and purpose |
| --- | --- |
| `gpu-measurement` | `SKILL.md` routes GPU questions; `references/requests.md` owns GPU request examples. |
| `runtime-records` | `SKILL.md` routes persistence questions; `references/journal.md` owns Direction/Experiment/Report requests; `references/records.md` owns result/source reads. |
| `KernelWiki` | `SKILL.md` documents external knowledge queries. |
| `ncu-report-skill` | Adapted `SKILL.md`; diagnosis, metric-interpretation, and raw-report references; four upstream analysis helpers. No separate measurement or Journal protocol. |
| `autonomous-gpu-kernel-timeline` | Routing `SKILL.md`; CUDA/IKeT references; capture/validation CLI; CUDA header/adapter and its documented runnable `test_backend.cu`; CuTe DSL adapter loaded by the CLI. |
| `ppu-acu-joint-profile` | Routing `SKILL.md`; ACU, bottleneck, recorder, transport, and output-contract references; extraction, evidence, timeline, critical-path, joint-merge and report-validation scripts; PPU adapter/header. |

`gpu-measurement`, `runtime-records`, and `KernelWiki` remain mandatory. Timeline remains the default
optional Skill. `gen-plan` stays repository-side and is not installed. Backend discovery links share
one selected snapshot rather than separate copies.

## NCU adaptation

The third-party NCU submodule is unchanged. Its Agent entrypoint is replaced by this directory's
adapted instructions, which retain bottleneck reasoning, metric semantics, and Python report analysis.
Only `analyze_reports.py`, `extract_stall_hotspots.py`, `plot_timeline.py`, and their shared
`ncu_utils.py` are imported from the submodule. The documentation explains their first-action and
device-specific counter limitations.

The upstream installation README, mandatory two-Profile workflow, standalone harness/report
templates, FlashInfer workload discovery, safetensors loader, and broad hardware tutorial are not
installed. GPU requests use `gpu-measurement`; history and reporting use `runtime-records`.
Raw-report processing is optional and runs via Dev if it needs the worker's Nsight installation.

## Updating a Skill

List every retained file and its dependencies explicitly. Check actual mounted Markdown links,
command parsing, and helper imports; do not mask obsolete instructions with a precedence disclaimer.
Runtime uploads must use explicit inputs; custom GPU commands use Dev. Host installation/Git/service
operations do not belong in the Agent view. Keep domain-specific evidence validation; do not turn it
into a mandatory per-Episode pipeline.

`orchestrator.test_agent_skill_manifest` tests the complete mounted selection, including exclusions,
upstream-file isolation, symlink rejection, helper entrypoints and references. Content-keyed assets
leave running sessions' existing views unchanged; subsequent sessions receive the updated selection.
