# Requests

Run from the Agent workspace. The Runtime URL and scoped token are injected for
this Session; do not copy them into requests, reports or another workspace.

## Choose the operation

Use the narrowest operation that answers the current question. Diagnostic success does
not replace a passing ordinary full Evaluate for the exact selected source.

| Question | CLI kind / controls | What it establishes |
| --- | --- | --- |
| Is the current candidate correct and faster? | `run --no-sync` | Official correctness and end-to-end timing across the contract. Inspect correctness before using latency. |
| Does a numerical repair work without timing? | Atrex-Bench `run --mode correctness_only --no-sync` | Correctness evidence only; missing latency is not zero latency. |
| How do candidate and baseline compare in one allocation? | `run --baseline-path PATH --comparison-repeats 2 --no-sync` | Paired ABBA evidence. Use the injected acceptance command for exact reuse; this does not replace ordinary full Evaluate. |
| Which GPU resource or Kernel needs attention? | `profile --profile-level sol --no-sync` | Structured counters, not an algorithmic optimality verdict. Narrow to a Kernel/launch for `deep`. |
| Does compilation or a sanitizer reveal a defect? | `check --no-sync`, optionally `--sanitize memcheck` | Diagnostic results; inspect inner errors even if transport completed. |
| What code did the compiler generate? | `disassemble --format ptx --no-sync` | Generated device code, not correctness or performance validation. |
| Can a custom probe resolve one implementation question? | `dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py` | Probe output and exit status, not necessarily an evaluation. |
| What is available on the assigned GPU worker? | `env` | GPU/environment/capability facts, not facts about the Agent host or a Kernel measurement. |
| Has this exact candidate or observation already been measured? | `record-read`, `kernel-read`, `kernel-records`, `kernel-pareto-frontier` | Saved facts/source; no new GPU allocation. See the history section below. |

Read `python3 tools/sandbox.py --kind OPERATION --help` before using unfamiliar controls.
The injected task/evaluator contract and actual Runtime validation take precedence over
generic examples. Do not invent endpoint, hardware, credential or evaluator overrides.

## Submit and wait

Run one command in the foreground. Follow the waiting rules in `CLAUDE.md`: do not add a
shell timeout or `&`, and use the backend's wait/output facility if it backgrounds the call.
Do not change the measured source or launch unrelated work while waiting.
`--no-sync` disables output-file synchronization; it does **not** make execution asynchronous.
Managed Runtime waits pause the Agent/subagents and do not consume the Agent active-time
budget, but remain subject to the request's own deadline. A quiet CLI is not a failure.

Save JSON stdout and stderr separately if needed:

```bash
python3 tools/sandbox.py --kind run --no-sync > scratch/evaluate.json 2> scratch/evaluate.stderr
```

Check the exit status and both files after the invocation ends. Do not use `2>&1` or a
pipeline that hides a nonzero exit code. A connection loss does not prove the GPU job
failed or was never submitted; use the recovery guidance below instead of resubmitting.

## Measurement examples and evaluation policy

Follow the injected Evaluation contract. Both evaluators accept the plain full-workload
command below. SOL-ExecBench runs its official `workload.jsonl` harness through the Dev
compatibility route: do not add `--mode`, custom typed input options, or top-level typed
shorthand controls. That route still performs official correctness and performance evaluation,
not a custom probe.

```bash
# Full workload: Atrex-Bench or SOL-ExecBench.
python3 tools/sandbox.py --kind run --no-sync
# Optional explicit version label (both evaluators).
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --version v2 --no-memory
# Atrex-Bench only: correctness-only diagnostic, not a full Evaluate.
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
# Atrex-Bench only: reduced-seed diagnostic, insufficient for final acceptance.
python3 tools/sandbox.py --kind run --mode correctness_only --multi-seed 0 --no-sync
# Generic AB/BA example; use the injected acceptance command for reuse.
python3 tools/sandbox.py --kind run --baseline-path scratch/incumbent.py --comparison-repeats 2 --no-sync
# Atrex-Bench Typed NCU/rocprof profile; IDs name opaque cases, not exact private shapes.
python3 tools/sandbox.py --kind profile --profile-level sol --no-sync
python3 tools/sandbox.py --kind profile --profile-shape-id 0 --no-sync
python3 tools/sandbox.py --kind profile --profile-level deep --kernel-name my_kernel --profile-source --launch-count 1 --no-sync
# Custom remote probe: explicitly declare files opened dynamically.
python3 tools/sandbox.py --kind dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py
# Diagnostics and available environments (typed Gateway, not SSH).
python3 tools/sandbox.py --kind check --sanitize memcheck --no-sync
python3 tools/sandbox.py --kind disassemble --format ptx --no-sync
python3 tools/sandbox.py --kind env
```

For ABBA, optional `--comparison-run-timeout SECONDS` controls each run and must fit the
configured allocation's complete schedule. The default is up to 120 seconds. Acceptance can
reuse only an exact matching request, including baseline bytes, path, repeats and run timeout.
The injected acceptance block supplies the exact command for the current
configuration, including Shape batch size, and a `kernel-read` command for the committed
incumbent. Use that block instead of this generic example when seeking acceptance reuse.
It also reports Runtime-owned measurement repetitions and allocation timeout; do not emulate
them with repeated Agent requests or environment overrides. Other request/input changes can
prevent reuse. A matching comparison is optional and does not replace the report's full Evaluate.

Full Evaluate and ABBA default to 100 timing iterations and 10 warmup iterations, one Shape
per batch and at most 16 concurrent batches. Promotion minimizes GeoMean latency versus the
incumbent; `performance_score` is diagnostic. The returned `improvement_pct` is
`(speedup - 1) * 100`, matching Runtime (100 us to 80 us means 1.25x and 25%).
This is not the latency-reduction percentage used by the Supervisor's promotion threshold.
Do not loop requests to reproduce the sampling policy.

Atrex-Bench Evaluate defaults to the base case plus five additional correctness cases per Shape.
Full mode keeps its normal performance measurement; extra cases do not add timing runs.
`--mode correctness_only` uses the same six cases without timing. An explicit `--shape-id` smoke
defaults to one case, and `--multi-seed N` can override the case count for diagnostics. Check,
Profile and Dev probes do not gain this seed expansion. Agent ABBA defaults to six cases;
Supervisor retention ABBA explicitly uses `--multi-seed 0` after the six-case candidate gate. For legacy explicit
`test_kernel.py --version v2 --multi-seed N` commands, positive N still implies correctness-only
unless `--mode full` is supplied; an implicit default never disables performance measurement.

For Atrex-Bench `candidate_ready`, the Supervisor reuses a successful full or correctness-only six-case result
when the frozen source, complete Shape contract, evaluator, tolerances, target and environment
match. Otherwise it runs the missing six-case check without timing. You do not need a separate
robustness request after a matching full Evaluate. Custom inputs, Shape subsets, fewer seeds and
old records without the resolved seed policy cannot satisfy this check. A passing full Evaluate
for the exact selected source remains required. With research recording disabled, cite its
`gateway_record_id` directly in the report's `gateway_record_ids`; do not invent an Experiment.
With research recording enabled, the selected Experiment must cite that evidence instead.
Follow the installed `skills/runtime-records/references/journal.md` for the current report schema.
Baseline sessions follow their bounded instructions.

For SOL-ExecBench `candidate_ready`, the Supervisor runs or reuses the official full-workload
evaluation of the frozen candidate and matching workload/configuration. The Atrex-Bench six-case
policy and correctness-only mode do not apply to SOL.

For Atrex-Bench evaluator controls, choose one spelling:

```bash
# Shorthand: no explicit command.
python3 tools/sandbox.py --kind run --multi-seed 3 --timed-runs 100 --no-sync
# Explicit command: evaluator controls belong after test_kernel.py.
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --multi-seed 3 --timed-runs 100 --no-memory
```

Top-level `--version`, `--multi-seed`, `--shape-id` and `--timed-runs` require
`--kind run` without an explicit command. Mixing these top-level controls with a
command is rejected before execution; they are not merged or silently ignored.
Explicit shorthand controls require a compatible typed evaluation route and
cannot silently fall back to Dev. Once a Typed Evaluate starts, admission,
transport or evaluator failure also cannot change it into a Dev job. ABBA still accepts `--version` and
`--timed-runs` and `--multi-seed` without a command; Shape/input overrides remain unsupported.
ABBA uses native Eval only; it never runs a Dev driver. Native ABBA requires an Atrex-Bench
input/Shape contract and an Agate endpoint; SOL/SSH comparison is unsupported. Repeats must
be even, 2–16 measurements per side; 2 means one A/B/B/A block.

## Profile and diagnostics

Profile, Check and Disassemble accept repeatable `--requirement 'package==version'`
and `--deps-mode freeze_installed|no_deps`; these affect only the remote job.
Profile also supports `--profiler`, `--profile-counter`, `--kernel-regex`,
`--profile-shape-id`, `--launch-skip`, and `--top-kernels`.
Collection and parsing run on the GPU worker, including the Dev compatibility
route. Consume the structured Profile result (or reread its Gateway Record).
Do not import/install `ncu_report` on the Agent host. The NVIDIA Dev wrapper
exports `profile.json`, `analysis/metrics.csv` and, with `--source`,
`analysis/source.txt`; these are already parsed/exported remotely using the
NCU CLI and Python's standard library. Binary reports stay remote by default,
including OSS transfers. Missing worker NCU or empty metrics is an explicit
Profile failure; narrow the kernel/launch selection or ask the operator to
provision the worker, not the local Agent environment.
Use `--profile-shape-id ID`, not `--shape-id`, to select an opaque case from a saved result.
Atrex-Bench Profile uses the Typed route by default, including hidden Shapes. The Supervisor supplies
exactly one private case; only its opaque ID and projected metrics are returned to the Agent.
Without a selector it uses the first sorted Shape. `--profile-shape-id` takes precedence over
legacy `--env PROFILE_SHAPE_ID=...`; no environment override is needed.
SSH, custom commands/wrappers, `--input`, and driver-specific `PROFILE_*` controls use Dev
when required. SOL-ExecBench Profile also uses the compatibility route; Atrex-Bench Typed Shape
selectors do not apply. A recognized hidden-Shape driver receives only the selected private case;
Typed-only options fail if the fallback cannot honor them.
For hidden-Shape Typed Profile, `--sync scratch/profile` writes a projected
`scratch/profile/gateway_profile.json`; raw profiler artifacts and private diagnostics are not synchronized.

Profile maps Agate's `dram_pct` to `dram_throughput_pct` and preserves per-Kernel
`traffic` counters: `dram_bytes`, `dram_bytes_read`, `dram_bytes_write`, `l2_bytes`,
`duration_ns` and `achieved_dram_gbps`, when supplied. Missing counters are omitted,
not reported as zero. High `memory_sol_pct` alone does not establish DRAM saturation
or an algorithmic speedup limit; inspect the DRAM/traffic counters and request
subsystem-specific counters when the limiting unit remains unclear.

## Saved results and source

Reuse recorded facts across Episodes. Replace these placeholder IDs with values
returned by the Runtime; these queries do not submit GPU jobs:

```bash
python3 tools/sandbox.py --kind record-read --record-id gateway-0123456789abcdef0123456789abcdef
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-0123456789abcdef0123456789abcdef --output-path scratch/previous.py
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-0123456789abcdef0123456789abcdef
python3 tools/sandbox.py --kind kernel-pareto-frontier
```

Measurements and `record-read` print one JSON object with `operation`, `status`,
`result`, `gateway_record_id`, and `kernel_id` when Kernel-bound. ABBA uses
`operation: evaluate`, `result.comparison`, and an additional `baseline_kernel_id`.
Evaluate uses `result.correct` plus `result.correctness` (`status: PASS|FAIL`,
`rel_err`, `max_abs_err`, `max_rel_err`); unavailable error metrics are null, not zero.
`result.mode` is `full` or `correctness_only`; `result.input_scope` is `contract`
or `custom`. Custom input/Shape overrides and Shape subsets are not full-contract evidence.
Correctness-only results have null latency summaries and an empty per-Shape latency map.
ABBA keeps the per-side aggregate metrics, not raw schedules or repetition arrays.

`record-read` prints the saved public result without executing GPU work. Its exit
code is zero when the read succeeds, even if the saved operation has `status: failed`
or `cancelled`. A completed evaluator can have `result.correct: false`; completion
does not mean the Kernel passed. Old stream-only records are normalized on read.
`kernel-read` copies exact measured source into `scratch/`. `kernel-records` lists
that Kernel's Record IDs, operations and timestamps without loading every result.
Dev probes also produce Record IDs even when their output is not an evaluation.
A duplicate-task error means the result already exists, not that the Kernel failed.

## Interpret completion and failures

Inspect the outer `status`, nested `result`, and available diagnostics together. A process
exit code alone cannot distinguish correctness from transport or a successful historical read.
Do not infer unavailable counters or missing measurements as zero.

| Response | Next action |
| --- | --- |
| Full Evaluate completed, `result.correct: true` | Compare the measured latency and per-Shape changes with your expected effect. Keep, refine or restore based on evidence; promotion still belongs to the Supervisor. |
| Evaluate completed, `result.correct: false` | Inspect correctness diagnostics and opaque Shape/case/output identifiers. Repair the candidate; do not call it a Gateway outage or use its timings as a valid speedup. |
| Dev/Check/Profile/Disassemble failure | Read `result.error` and available `stdout`/`stderr` or diagnostic fields. Compiler/import/file errors require source or declared-input repair. A successful Dev probe is not automatically a passing Evaluate. |
| Request rejected with `issues`, `request_schema`, `recovery` or `error.next_action` | Correct the named field/path or evidence reference. Follow the supplied operation-specific guidance, not guessed flags or IDs. |
| `duplicate_gateway_task` | Read the previous Record ID supplied by the error. Do not relabel the version or change source merely to bypass deduplication. |
| `request_not_started`, explicitly `repairable: true` | No executor started. Wait at least `retry_after_seconds` and retry unchanged with backoff; this exception does not authorize retries for other failures. |
| Infrastructure recovery exhausted | Runtime already owns bounded retries. Follow its final `next_action`; report a genuine blocker instead of declaring the Kernel incorrect or building another retry loop. |
| Connection loss, unknown outcome, missing/revoked Session | Inspect saved records when available and ask the operator to reconcile or restore the Session. Do not resubmit blindly or bypass the Runtime. |

An error may appear on stderr as a request/transport error, or inside a persisted measurement's
`result.error`. Preserve the saved ID and separate the observed failure from your explanation.
`feedback_truncation` means details were omitted; rereading the Record is not a promise of
unbounded/raw logs. Narrow subsequent diagnostics or ask the operator for unavailable details.

## Probe files and returned artifacts

Write multi-line probes as files in `scratch/`, then declare each required extra file with
repeatable `--input`; do not embed a whole script in the command string. The contracted candidate
is supplied by the Supervisor. Extra probe files are separate inputs, not a way to replace the
evaluator or expose hidden cases. Remote jobs start fresh, so do not assume a previous job's
probe files or installed dependencies persist.

Use `--sync scratch/path` for returned files. Keep report-cited artifacts and their referenced
files under `scratch/`; only that diagnostics tree is published to the controller workspace.
Paths must be workspace-relative and must not traverse symlinks or `..`.
Do not request control files or changes to Kernel source via remote outputs.

## Selected Core Wiki tools

Wiki access is determined by the selected Core profile. Discover only the tools available in this
session, then call an enabled Wiki tool using its declared input schema. Use the returned source
references and receipt when recording research; do not assume a public or internal provider is enabled.

```bash
python3 tools/plugin.py list
python3 tools/plugin.py call <enabled-wiki-tool> --input scratch/wiki_request.json
```
