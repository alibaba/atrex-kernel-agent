# Requests

Run from the Agent workspace. The Runtime URL and scoped token are injected for
this Session; do not copy them into requests, reports or another workspace.

```bash
# Existing evaluator invocation (all existing evaluator controls remain valid).
python3 tools/sandbox.py --kind run --no-sync -- python3 test_kernel.py --version v2 --no-memory
# Equivalent shorthand; correctness-only is also available.
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
# One AB/BA schedule per shape batch, both sides in the same GPU allocation.
python3 tools/sandbox.py --kind run --baseline-path scratch/base.py --comparison-repeats 2 --no-sync
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
