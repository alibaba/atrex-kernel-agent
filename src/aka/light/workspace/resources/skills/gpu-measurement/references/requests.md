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
| How do candidate and baseline compare in one allocation? | `run --baseline-commit COMMIT --comparison-repeats 2 --no-sync` | Paired ABBA evidence. Runtime exports the exact reachable Git commit privately; this does not replace ordinary full Evaluate. |
| Which GPU resource or Kernel needs attention? | `profile --profile-level sol --no-sync` | Structured counters, not an algorithmic optimality verdict. Narrow to a Kernel/launch for `deep`. |
| Does compilation or a sanitizer reveal a defect? | `check --no-sync`, optionally `--sanitize memcheck` | Diagnostic results; inspect inner errors even if transport completed. |
| What code did the compiler generate? | `disassemble --format ptx --no-sync` | Generated device code, not correctness or performance validation. |
| Can a self-contained single-file probe that reads no candidate or other workspace file resolve one question? | `dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py` | Only the explicitly declared probe is uploaded; returns its output and exit status. |
| Does custom code read the candidate, helpers, prior artifacts, or any other workspace state? | `command -- python3 -m scratch.experiment` | Bounded engineering workspace projection uploaded; `scratch/` synchronized back. Runtime control state, Skills and bulk `records/` history are excluded. |
| What is available on the assigned GPU target? | `env` | Runtime selects the Campaign hardware and returns its GPU/environment/capability facts; this is not a fact about the Agent host or a Kernel measurement. |
| Has this exact candidate or observation already been measured? | `records/*.json`, local Git, root `PROGRESS.md` | Saved facts and exact committed source; `PROGRESS.md` includes per-Shape historical winners. No new GPU allocation. |

For Python Command probes below `scratch/`, use module form such as
`python3 -m scratch.probe`. Running `python3 scratch/probe.py` directly makes
`scratch/` the first Python module-search directory, so root candidate modules such as
`kernel.py` may not be importable even though Command uploaded them correctly.

Read `python3 tools/sandbox.py --kind OPERATION --help` before using unfamiliar controls.
The Runtime discovers safe native options from its installed Agate client at startup and
shows them under their original Agate spellings (for example `--profiler acu`,
`--cluster-label KEY=VALUE` or `--shape-shards N`). Updating Agate and restarting the
Supervisor refreshes those options and enum values. Runtime-owned credentials, sources,
uploads, waits, timeouts and evaluator selection are deliberately omitted.
Known Agate result and error schemas receive the Runtime's operation-specific summary and
recovery guidance. Unrecognized response markers, error classes and envelope fields are
returned unchanged so a newer Agate client remains diagnosable; only credentials, private
Supervisor paths and oversized-output truncation are applied universally.
The injected task/evaluator contract and actual Runtime validation take precedence over
generic examples. Do not invent endpoint, hardware, credential or evaluator overrides.

## Submit and wait

Run one command in the foreground. Follow the waiting rules in `CLAUDE.md`: do not add a
shell timeout or `&`, and use the backend's wait/output facility if it backgrounds the call.
Do not change the measured source or launch unrelated work while waiting.
`--no-sync` disables output-file synchronization; it does **not** make execution asynchronous.
The session Prompt states whether Runtime request suspension is enabled. When enabled, waits
pause the Agent/subagents and do not consume the Agent active-time budget; when disabled, they
remain active time. Every request remains subject to its own deadline. A quiet CLI is not a failure.

Measurement commands print one compact receipt; the complete result is already saved under the
receipt's `record_path`. Env instead prints the complete redacted Agate capability response directly
to stdout and creates no Gateway Record. When a measurement has stdout or stderr, the receipt's
`stdout_location` or `stderr_location` names both the Record file and exact JSON field containing
the complete stream. Save a measurement receipt separately only when needed:

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
# Generic AB/BA example; use the injected acceptance command for reuse.
python3 tools/sandbox.py --kind run --baseline-commit <full-git-commit> --comparison-repeats 2 --no-sync
# Atrex-Bench Typed NCU/rocprof profile; use a Shape ID returned by Runtime.
python3 tools/sandbox.py --kind profile --profile-level sol --no-sync
python3 tools/sandbox.py --kind profile --profile-shape-id 0 --no-sync
python3 tools/sandbox.py --kind profile --profile-level deep --kernel-name my_kernel --profile-source --launch-count 1 --no-sync
# Custom remote probe: explicitly declare files opened dynamically.
python3 tools/sandbox.py --kind dev --input scratch/probe.py --no-sync -- python3 scratch/probe.py
# Diagnostics and available environments (Gateway or SSH).
python3 tools/sandbox.py --kind check --sanitize memcheck --no-sync
python3 tools/sandbox.py --kind disassemble --format ptx --no-sync
python3 tools/sandbox.py --kind env
```

Managed `env` takes no target, capability, or refresh override. Runtime supplies the Campaign's
hardware identity and requests its capability view; use the returned architecture/features rather
than querying a different product or guessing from a marketing name. The stdout payload is the
complete redacted Agate Env response, not a compact measurement receipt.

For ABBA, optional `--comparison-run-timeout SECONDS` controls each run. Runtime uses an explicit
positive value unchanged and expands the ABBA allocation to fit the complete schedule. Without
that option, Runtime derives the smaller of the Campaign evaluation-step timeout and the maximum
that fits the existing allocation. If the resulting allocation exceeds Agate's typed-job limit,
Runtime returns the supplied per-run timeout, repeat count, required allocation, and service limit;
it does not clamp the value or submit a Job.
Acceptance can reuse only an exact matching request, including baseline commit/source, repeats and
run timeout.
The injected acceptance block supplies the exact Agent-controlled command for the current
configuration and committed incumbent ID. Use that block instead of this generic example when
seeking acceptance reuse.
Runtime also owns Shape batch size, measurement repetitions and the derived allocation timeout; do not emulate
them with repeated Agent requests or environment overrides. Other request/input changes can
prevent reuse. A matching comparison is optional and does not replace the report's full Evaluate.

Full Evaluate and ABBA default to 100 timing iterations and 20 warmup iterations. Runtime uses
its configured Shape-batch upper bound (default 5, and it
may reduce the effective batch size to fit a request deadline) and configured batch-worker cap
(default 16). These are Supervisor settings, not Agent request fields.
Promotion minimizes GeoMean latency versus the
incumbent; `performance_score` is diagnostic. The returned `improvement_pct` is
`(speedup - 1) * 100`, matching Runtime (100 us to 80 us means 1.25x and 25%).
This is not the latency-reduction percentage used by the Supervisor's promotion threshold.
Do not loop requests to reproduce the sampling policy.

Atrex-Bench Evaluate uses the base case plus four additional correctness cases per Shape.
Full mode keeps its normal performance measurement; extra cases do not add timing runs.
`--mode correctness_only` uses the same five cases without timing. An explicit `--shape-id` narrows
performance coverage but does not weaken Runtime-owned correctness coverage. Agent requests cannot
set `--multi-seed`; Check, Profile and Dev probes do not gain this case expansion. Agent and
Supervisor ABBA use the same five-case correctness policy.

For Atrex-Bench `candidate_ready`, the Supervisor reuses a successful full or correctness-only five-case result
when the frozen source, complete Shape contract, evaluator, tolerances, target and environment
match. Otherwise it runs the missing five-case check without timing. You do not need a separate
robustness request after a matching full Evaluate. Custom inputs, Shape subsets, fewer seeds and
old records without the resolved seed policy cannot satisfy this check. A passing full Evaluate
for the exact selected source remains required. When the Experiment module is disabled, cite its
`gateway_record_id` directly in the report's `gateway_record_ids`; do not invent an Experiment.
When the Experiment module is enabled, the selected Experiment must cite that evidence instead.
Follow the installed `skills/runtime-records/references/journal.md` for the current report schema.
Baseline sessions follow their bounded instructions.

For SOL-ExecBench `candidate_ready`, the Supervisor runs or reuses the official full-workload
evaluation of the frozen candidate and matching workload/configuration. The Atrex-Bench five-case
policy and correctness-only mode do not apply to SOL.

For Atrex-Bench evaluator controls, choose one spelling:

```bash
# Shorthand: no explicit command.
python3 tools/sandbox.py --kind run --timed-runs 100 --no-sync
# Explicit command: evaluator controls belong after test_kernel.py.
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --timed-runs 100 --no-memory
```

Top-level `--version`, `--shape-id` and `--timed-runs` require
`--kind run` without an explicit command. Mixing these top-level controls with a
command is rejected before execution; they are not merged or silently ignored.
`--multi-seed` is Runtime-owned and is rejected in either spelling.
Explicit shorthand controls require a compatible typed evaluation route and
cannot silently fall back to Dev. Once a Typed Evaluate starts, admission,
transport or evaluator failure also cannot change it into a Dev job. ABBA accepts `--version` and
`--timed-runs` without a command; Shape/input/seed overrides remain unsupported.
ABBA uses native Eval only; it never runs a Dev driver. Native ABBA requires an Atrex-Bench
input/Shape contract and an Agate or SSH endpoint; SOL comparison is unsupported. Repeats must
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
Use `--profile-shape-id ID`, not `--shape-id`, to select a case from a saved result. On a
generalized hidden-Shape task, all evaluator Shape IDs are opaque. On an exposed task, consult
`shapes.json` for exact parameters.
Atrex-Bench Profile uses the Typed route by default, including Shapes whose parameters are not
public. The Supervisor supplies exactly one evaluator case and returns its ID and projected metrics;
in hidden mode the returned ID remains opaque.
Without a selector it uses the first sorted Shape. `--profile-shape-id` takes precedence over
legacy `--env PROFILE_SHAPE_ID=...`; no environment override is needed.
Custom commands/wrappers, `--input`, and driver-specific `PROFILE_*` controls use Dev
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

Reuse recorded facts across Episodes by reading or querying the relevant
`records/<gateway_record_id>.json` directly; inspect exact Kernel commits with ordinary Git commands. Runtime projects a Record only
when every referenced commit is reachable from local or imported shared refs and its committed
source matches the private measurement snapshot.

Each measurement response and projected Record contains `operation`, `status`,
`result`, `gateway_record_id`, and `kernel_commit_id` when a single Kernel is bound. ABBA uses
`operation: evaluate`, `result.comparison`, `candidate_kernel_commit_id`, and
`baseline_kernel_commit_id`.
Evaluate uses `result.correct` plus `result.correctness` (`status: PASS|FAIL`,
`rel_err`, `max_abs_err`, `max_rel_err`); unavailable error metrics are null, not zero.
`result.mode` is `full` or `correctness_only`; `result.input_scope` is `contract`
or `custom`. Custom input/Shape overrides and Shape subsets are not full-contract evidence.
Correctness-only results have null latency summaries and an empty per-Shape latency map.
ABBA keeps the per-side aggregate metrics, not raw schedules or repetition arrays.

A completed evaluator can have `result.correct: false`; completion does not mean the Kernel passed.
Search `records/*.json` by commit ID to find its measurements. Use `git show`, `git diff`, `git log`
or a local branch to inspect source; imported peer history is under `refs/atrex/shared/`.
Dev and Command receipts show bounded stdout/stderr previews and byte counts. Their complete redacted
streams remain in the saved Record and can be read directly in bounded ranges.
Dev probes also produce Record IDs even when their output is not an evaluation.
A duplicate-task error means the result already exists, not that the Kernel failed.

## Interpret completion and failures

Inspect the outer `status`, nested `result`, and available diagnostics together. A process
exit code alone cannot distinguish correctness from transport or a successful historical read.
Do not infer unavailable counters or missing measurements as zero.

| Response | Next action |
| --- | --- |
| Full Evaluate completed, `result.correct: true` | Compare the measured latency and per-Shape changes with your expected effect. Keep, refine or restore based on evidence; promotion still belongs to the Supervisor. |
| Evaluate completed, `result.correct: false` | Inspect correctness diagnostics and Shape/case/output identifiers. On generalized hidden-Shape tasks, treat IDs as opaque; on exposed tasks, use `shapes.json`. Repair the candidate; do not call it a Gateway outage or use its timings as a valid speedup. |
| Dev/Check/Profile/Disassemble failure | Read `result.error` and available `stdout`/`stderr` or diagnostic fields. Compiler/import/file errors require source or declared-input repair. A successful Dev probe is not automatically a passing Evaluate. |
| Request rejected with `issues`, `request_schema`, `recovery` or `error.next_action` | Correct the named field/path or evidence reference. Follow the supplied operation-specific guidance, not guessed flags or IDs. |
| `duplicate_gateway_task` | Read the previous Record ID supplied by the error. Do not relabel the version or change source merely to bypass deduplication. |
| `gateway_task_running` | The same call is still executing; this request did not submit another Agate task. Wait for the original response. `remaining_seconds` is its remaining Runtime request budget, not a guaranteed successful completion ETA; null means unknown, and zero means the deadline has elapsed and cleanup is still in progress. No complete Record is available to read yet. |
| `request_not_started`, explicitly `repairable: true` | No executor started. Wait at least `retry_after_seconds` and retry unchanged with backoff; this exception does not authorize retries for other failures. |
| Infrastructure recovery exhausted | Runtime already applied the configured terminal/non-terminal retry policy. Follow its final `next_action`; report a genuine blocker instead of declaring the Kernel incorrect or building another retry loop. |
| Connection loss, unknown outcome, missing/revoked Session | Inspect saved records when available and ask the operator to reconcile or restore the Session. Do not resubmit blindly or bypass the Runtime. |

An error may appear on stderr as a request/transport error, or inside a persisted measurement's
`result.error`. Preserve the saved ID and separate the observed failure from your explanation.
Historical Records can contain `feedback_truncation`, which means an older Runtime omitted
details. New Runtime responses preserve complete redacted diagnostic text; they still do not
expose evaluator inputs, credentials or other private controller material.

## Probe files and returned artifacts

Use Dev only when one explicitly declared probe file is self-contained and reads no candidate or
other workspace file. Do not embed a whole script in the command string. Dev uploads only its
declared inputs; it does not implicitly supply the candidate. If the probe imports or opens the
candidate, a helper, a prior artifact, or any other workspace path, use Command instead. Extra probe
files are not a way to replace the evaluator or expose hidden cases. Remote jobs start fresh, so do
not assume a previous job's probe files or installed dependencies persist.

Use `--sync scratch/path` for returned files. Keep report-cited artifacts and their referenced
files under `scratch/`; only that diagnostics tree is published to the controller workspace.
Paths must be workspace-relative and must not traverse symlinks or `..`.
Do not request control files or changes to Kernel source via remote outputs.

For every workspace-dependent custom experiment, use the workspace Command wrapper. It excludes
bulk root `records/`; if a historical result is required remotely, first copy the selected result
into `scratch/` with a normal local file copy rather than uploading the entire measurement history:

```bash
cp records/gateway-0123456789abcdef0123456789abcdef.json scratch/record.json
python3 tools/sandbox.py --kind command -- python3 -m scratch.probe_mma
```

It uploads a bounded engineering projection, including the committed candidate, custom helpers
and safe Runtime-owned tools. It excludes Git internals, credentials, private Runtime inputs,
bulk measurement history, Skills, control state and nested Runtime submission clients. The returned
`scratch/` tree is merged back into the trajectory workspace. Files outside `scratch/` are
read-only inputs from the Agent's perspective and are not copied back. Every invocation still
uses a fresh remote job; packages, processes and undeclared system changes do not persist.

## Optional Wiki

Wiki availability is campaign-configured. Only use these commands when the session mounts
`skills/KernelWiki/SKILL.md`; read that Skill for query scope and hardware identity.
Do not depend on a `gpu-wiki/` source directory in the Agent workspace. When Wiki is explicitly disabled,
use supplied public knowledge and measured evidence instead.

```bash
python3 tools/sandbox.py --kind wiki-query "Triton reduction on sm_120: measured bandwidth bottleneck"
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product B200
```
