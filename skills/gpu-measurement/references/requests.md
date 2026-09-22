# Requests

Run from the Agent workspace. The Runtime URL and scoped token are injected for
this Session; do not copy them into requests, reports or another workspace.

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

Atrex-Bench Evaluate defaults to the base case plus five additional correctness cases per Shape.
Full mode keeps its normal performance measurement; extra cases do not add timing runs.
`--mode correctness_only` uses the same six cases without timing. An explicit `--shape-id` smoke
defaults to one case, and `--multi-seed N` can override the case count for diagnostics. Check,
Profile, Dev probes and ABBA do not gain this default seed expansion. For legacy explicit
`test_kernel.py --version v2 --multi-seed N` commands, positive N still implies correctness-only
unless `--mode full` is supplied; an implicit default never disables performance measurement.

For Atrex-Bench `candidate_ready`, the Supervisor reuses a successful full or correctness-only six-case result
when the frozen source, complete Shape contract, evaluator, tolerances, target and environment
match. Otherwise it runs the missing six-case check without timing. You do not need a separate
robustness request after a matching full Evaluate. Custom inputs, Shape subsets, fewer seeds and
old records without the resolved seed policy cannot satisfy this check. A passing full Evaluate
remains required in the selected Experiment. Baseline sessions follow their bounded instructions.

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
cannot silently fall back to Dev. ABBA still accepts `--version` and
`--timed-runs` without a command; Shape/seed overrides remain unsupported.

Profile, Check and Disassemble accept repeatable `--requirement 'package==version'`
and `--deps-mode freeze_installed|no_deps`; these affect only the remote job.
Profile also supports `--profiler`, `--profile-counter`, `--kernel-regex`,
`--profile-shape-id`, `--launch-skip`, and `--top-kernels`.
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

Reuse recorded facts across Episodes. Replace these placeholder IDs with values
returned by the Runtime; these queries do not submit GPU jobs:

```bash
python3 tools/sandbox.py --kind record-read --record-id gateway-0123456789abcdef0123456789abcdef
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-0123456789abcdef0123456789abcdef --output-path scratch/previous.py
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-0123456789abcdef0123456789abcdef
```

`record-read` prints the same saved public result and preserves its exit code.
`kernel-read` copies exact measured source into `scratch/`. `kernel-records` lists
that Kernel's Record IDs, operations and timestamps without loading every result.
Dev probes also produce Record IDs even when their output is not an evaluation.
A duplicate-task error means the result already exists, not that the Kernel failed.

Use `--sync scratch/path` for returned files. Keep report-cited artifacts and their referenced
files under `scratch/`; only that diagnostics tree is published to the controller workspace.
Paths must be workspace-relative and must not traverse symlinks or `..`.
Do not request control files or changes to Kernel source via remote outputs.

The existing `gpu-wiki/tools/query_*.py` commands use the same HTTP service
automatically. Aliases are also available:

```bash
python3 tools/sandbox.py --kind wiki-query "Triton reduction on sm_120: measured bandwidth bottleneck"
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product sm120
```
