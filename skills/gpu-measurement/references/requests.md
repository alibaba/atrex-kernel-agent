# Requests

Run from the Agent workspace. The Runtime URL and scoped token are injected for
this Session; do not copy them into requests, reports or another workspace.

```bash
# Existing evaluator invocation (all existing evaluator controls remain valid).
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --version v2 --no-memory
# Equivalent shorthand; correctness-only is also available.
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
# One AB/BA schedule per shape batch, both sides in the same GPU allocation.
python3 tools/sandbox.py --kind run --baseline-path scratch/incumbent.py --comparison-repeats 2 --no-sync
# Typed NCU/rocprof profile; IDs name opaque cases, not exact private shapes.
python3 tools/sandbox.py --kind profile --profile-level sol --no-sync
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
Do not run ABBA in a Fast Episode.

Choose one spelling for evaluator controls:

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

Use `--sync profiles/path` or `--sync scratch/path` for returned files.
Paths must be workspace-relative and must not traverse symlinks or `..`.
Do not request control files or changes to Kernel source via remote outputs.

The existing `gpu-wiki/tools/query_*.py` commands use the same HTTP service
automatically. Aliases are also available:

```bash
python3 tools/sandbox.py --kind wiki-query "Triton reduction on sm_120: measured bandwidth bottleneck"
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product sm120
```
