# Specialized requests

The current phase and task constraints take precedence over these optional examples.
Replace example paths and IDs with existing workspace files and returned opaque IDs.

## Correctness and public inputs

```bash
python3 tools/sandbox.py --kind run --mode correctness_only --no-sync
python3 tools/sandbox.py --kind run --mode full --multi-seed 5 --no-sync
python3 tools/sandbox.py --kind run --mode correctness_only --input-path scratch/inputs.py --shapes-path scratch/shapes.json --no-sync
```

Additional seeds check correctness; performance uses the base seed. Custom input Python
defines `_make_inputs`; custom Shapes use a JSON object. Either override can be supplied
independently. Custom-input and correctness-only results cannot establish a standard full
evaluation improvement. Do not reconstruct hidden evaluation Shapes.

## Same-allocation comparison

Place the intended baseline in `scratch/baseline.py` and leave the candidate in
`kernel.py`, then request:

```bash
python3 tools/sandbox.py --kind run --mode full --baseline-path scratch/baseline.py --comparison-repeats 2 --no-sync
```

`--comparison-repeats` is the A/B pair count within a measurement, not the Supervisor's
three task repetitions. The Supervisor handles batching, alternating AB/BA measurements,
and aggregation. Inspect the baseline and candidate identities and per-Shape results.
An exploratory comparison does not replace the Supervisor's final acceptance decision.

## Focused Profile and diagnostics

```bash
python3 tools/sandbox.py --kind profile --profile-level deep --kernel-name my_kernel --profile-source --profile-shape-id 0 --launch-skip 0 --launch-count 1 --top-kernels 5 --no-sync
python3 tools/sandbox.py --kind check --sanitize memcheck --no-sync
python3 tools/sandbox.py --kind disassemble --format ptx --no-sync
```

Profile levels are `survey`, `sol`, and `deep`. Use either `--kernel-name` or
`--kernel-regex`, not both. `--profile-shape-id` selects an opaque evaluator case.
Sanitizers are `memcheck`, `racecheck`, `initcheck`, and `synccheck`; availability depends
on the environment. Assembly formats are `auto`, `sass`, `ptx`, and `isa`.

Results and records already contain measurement facts. If files are needed, replace
`--no-sync` with `--sync scratch/profile-result`; `--include-raw-profile` enables raw
profiler files for legacy inline transfer. Downloads are not a prerequisite for analysis.

For a missing dependency in typed Profile, Check, or Disassemble, the supported declaration
is repeatable `--requirement SPEC` with `--deps-mode freeze_installed|no_deps`, subject to
the task's dependency policy. Installation is confined to that Gateway job. Do not run
package installers yourself, locally or inside a Dev command.

## Custom Dev probes

Create the probe and declare every file or directory it reads, including dynamically
loaded dependencies, before `--`:

```bash
python3 tools/sandbox.py --kind dev --input kernel.py --input scratch/probe.py --sync scratch/probe-output -- python3 scratch/probe.py
```

Have the probe write requested outputs under `scratch/probe-output/` in the GPU job.
Use `--no-sync` instead if stdout is sufficient. Uploads are allowlisted; do not assume
the whole workspace, Skills, or other helper files are present. A successful probe is
evidence only for what its command actually tested, not an authoritative evaluator result.

## Environment inspection

```bash
python3 tools/sandbox.py --kind env --env-gpu GPU_NAME --env-capabilities
```

Use a returned environment name. The task's injected target hardware and architecture
remain authoritative; a scheduler label is not the GPU product or architecture.
