# Advanced evaluation

For basic candidates/operator packages, see [reference-dir.md](reference-dir.md).
For SOL formats and problem contracts, see [solution-json.md](solution-json.md) and [custom-sol-problem.md](custom-sol-problem.md).

## Shape parallelism

```bash
# Automatic mode derives the worker budget from current quotas and admission.
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
# Limit concurrent single-GPU workers; 1 forces serial execution.
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --shape-shards 2
```

This applies only to atrex eval and is enabled by default without new Diamond configuration.
The image's bench must pass capability verification. An explicit `--atrex-bench-version` is probed after isolated job installation:
capable versions run in parallel; old, unknown or unverified versions fall back to serial execution.
For a known serial-only version, set `--shape-shards 1` to avoid retaining a multi-GPU budget after fallback.
Upgrade the server before using new fields. Use a compatible client for older servers; do not change idempotency keys to rerun in-flight jobs.

Admitted worker count and shape-shard count are separate. Each shard contains at most 4 shapes; remaining shards queue in Ray.
Original shape IDs, input seeds and evaluation configuration are preserved, and all shards merge into one job result.
An explicit budget over the limit is rejected. Worker terminal evidence is limited to 900;
automatic mode reduces the budget while retaining every shape, whereas explicit excess is rejected.
The ledger's peak GPU count shows the total admitted GPU budget; a CPU driver does not mean the job uses zero GPUs.

`eval_execution` reports requested/actual mode, admitted budget and fallback reason. It is `pending` during isolated-install probing;
the terminal record reflects actual execution. The CLI reports explicit-parallel fallback, the budget remains reserved through cleanup,
and idempotent replay retains the original decision. Unrecognized historical diagnostics may be null; do not resubmit because of that.

The companion bench retries only worker/node-failure shards by default, at most 3 retries or 4 total executions per shard.
Successful shards are reused; code errors are not retried. A distributed terminal failure does not replay the entire batch;
loss of the driver/result cache fails explicitly. v1 capability probing does not replace image/wheel shard-recovery acceptance.
The gateway and bench own scheduling/retry controls; `--set` cannot override them.
Ray Data does not override the shared GPU pool's capacity floor. Environments without safe capacity-control isolation fail before evaluation.

Before rolling back to the original 0.14.19, stop ingress and independently confirm all parallel executions have stopped,
then let the original version finish cleanup and restore serial execution.
This evaluation rollback excludes CDC target handoff; environments switched to CDC must separately meet observation-rollback requirements.

## ABBA baseline comparison

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm \
  --baseline baseline.py --abba-repeats 2 --job-timeout 1800
```

Each group performs complete evaluations in A→B→B→A order within one GPU allocation.
Baseline and candidate expose `class Model` and share reference inputs, dependencies and evaluator version.
Only atrex + full is supported; do not combine with `correctness_only`.
`--abba-repeats` counts complete groups, range 1–8, default 1. `--job-timeout` covers the whole job.

- Individual runs are recorded in `result.abba.sdk_results[].abba.runs`. A valid all-shape speedup requires `result.abba.valid=true`.
- `speedup_geomean > 1` means the candidate is faster. Any failed correctness run or missing valid timing makes overall speedup null.
- `shapes.<id>.paired` reports each group's AB/BA latency differences, the order-balanced mean difference and estimated order effect, in ms. Negative differences favor the candidate.
- `corrected_delta_se_ms` requires at least two groups; with one group it is null. Invalid shapes have `paired=null`.
  Standard error treats complete groups as independent observations. Order balancing does not identify instantiation slot bias or guarantee removal of time drift.

This is ABBA over complete evaluation runs, not alternating individual `forward` calls.
It provides neither statistical-significance claims nor automatic promotion decisions. Use identical baseline/candidate sources for a run-level A/A comparison.

## Multi-file source archives

```bash
agate eval --gpu L20D --reference-dir ./data/op \
  --candidate candidate.tar.gz --candidate-entry src/model.py \
  --baseline baseline.zip --baseline-entry pkg/entry.py \
  --abba-repeats 2 --job-timeout 1800
```

Each side may use ZIP/TAR or a single file, in any combination. Omit baseline options for ordinary eval.
Each tree is limited to 2000 members and 4 MiB unpacked; members must be UTF-8 text and the Python entry point must export `Model`.
Both archives use one OSS direct-upload batch. Workers preserve complete independent source trees after extraction.
HTTP/SDK requests reference the same batch's `oss_files` with `{archive, entry_point}`. Paths may not escape the archive, and links are rejected.
An archive that cannot be opened returns `infra/source_archive_unavailable`.
An existing empty/malformed archive returns `validation/source_archive_invalid`, which is not retried.

## Runner config and evaluator versions

```bash
agate eval --gpu L20N --config eval.json
agate eval --gpu L20N --candidate my_kernel.py --reference-dir data/rms_norm \
  --set validation_mode=correctness_only --set benchmark_mode=cuda_graph_replay
agate eval --gpu L20D --config abba.json --abba-repeats 2 --job-timeout 1800
```

JSON config `input`/`reference_dir` paths resolve relative to the config file; YAML is also supported when PyYAML is installed.
ABBA `baseline_input` resolves relative to the same directory and is inlined; explicit `--baseline` takes precedence.
Other runner parameters are validated by worker-side atrex-bench; misspellings produce `unsupported runner config key`.
The gateway controls `output`/`checkpoint_dir` and ignores values for them in the config file.

The server's current default atrex eval/profile wheel is `0.1.0.post20260917160311`; request/cluster pins take precedence.
Use `--atrex-bench-version VERSION` for an explicit version. Without a request pin, an image wheel is reused only if it exactly matches the cluster/default pin.
Inspect `request_digest.atrex_bench_runtime` in `agate get JOB_ID --spec --json` for version source, image version and job-level installation.
Do not describe the default as resolving the newest index version on every job.

## Placement and measurement confidence

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm \
  --cluster-label stage=preview --node-label ray.io/market-type=spot
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm --lock-clocks
```

Cluster labels match configuration; node labels use native backend scheduling.
Hard constraints with no match do not fall back, and OSS upload references are bound to the same placement conditions.
`JobView.placement` echoes explicit requests. Reading a GPU environment alias shows the selected member's cache, not image consistency across the pool.

Clock locking applies only when the driver sees one GPU; shared multi-GPU environments skip it.
The default target is maximum frequency, overridable by cluster configuration.
Unlock immediately after GPU testing. Exit hooks restore clocks after early failure or SIGTERM; a later job first clears locks left by SIGKILL.
NCU always uses `--clock-control none`; Agate's `--lock-clocks` owns clock management.
Use `environment.clock_sm_pct_of_max`, `clock_throttle_reasons` and per-shape `timing_stats.cv` to judge confidence.
`cv=null` means only one sample, so dispersion is unknown; `runs` is the actual sample count.
A supplied `roofline.json` is required for relative `sol.pct`; unavailable values are null with a `sol.reason`.

## Profile and SOL notes

`profile` defaults to `--level sol`; `survey` finds hotspots, and `deep --kernel-regex REGEX` returns raw-report links.
PPU selects ACU by vendor. Explicit `--profiler ncu` returns 400; use `acu` or omit the profiler.
When optional NCU metrics are inaccessible, existing kernels are retained. Empty captures retain tool exit status and diagnostics;
do not infer candidate failure from them alone. See [collectives.md](collectives.md) for multi-GPU communication,
NCU 2026.1+ rank ordering and context scope.

SOL does not benchmark the reference by default. `--benchmark-reference` is required for meaningful speedup, at roughly twice the runtime;
default zero speedup does not mean no acceleration. A custom problem passing proves only agreement with its supplied reference.
Check passed/total counts, `problem_source=inline` and `problem_digest`, and keep these results separate from official-problem statistics.
