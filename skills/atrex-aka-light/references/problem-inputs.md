# AKA Light problem inputs

Read this file before passing an operator to `--op-dir`. AKA Light accepts two
problem families. Candidate file systems such as `source_tree`, `single_file`, and
`cuda_manifest` describe what the Agent edits after the problem is loaded; they are
not additional problem-input formats.

## Supported problem families

| Problem family | How Light recognizes it | Required operator files | Evaluation runtime |
|---|---|---|---|
| Native Atrex-Bench | `shapes.json` exists and the operator is inside a canonical Atrex-Bench checkout | `reference.py`, `input.py`, `shapes.json` | The owning checkout's `scripts/run_eval.py` and `src/atrex_bench/` |
| SOL-ExecBench | Both `definition.json` and `workload.jsonl` exist | `definition.json`, `workload.jsonl`, `reference.py` | The SOL harness and the workload records |

Light does not accept an arbitrary `reference.py` directory, a custom evaluator
script, or a setup command as a deterministic problem. The Legacy Workflow can load
some of those arrangements, but Light rejects them with:

```text
light Workflow deterministic V0 requires native Atrex-Bench or SOL; use legacy for custom setup
```

Use `--profile workflow` for an intentional Legacy run, or convert the problem to one
of the two layouts below.

A concrete minimum Native input is available under
[`examples/native-atrex-bench-vector-add/`](../examples/native-atrex-bench-vector-add/README.md).
Copy it under a canonical Atrex-Bench checkout before passing it to `--op-dir`.

## Native Atrex-Bench operators

### Directory layout

The operator must be a descendant of the same canonical Atrex-Bench checkout that
owns its evaluator runtime:

```text
atrex-bench-root/
├── scripts/
│   └── run_eval.py                 # required runtime marker
├── src/
│   └── atrex_bench/                # required runtime package
└── data/.../operator-name/         # pass this directory to --op-dir
    ├── reference.py                # required: authoritative operator semantics
    ├── input.py                    # required: input construction/calling contract
    ├── shapes.json                 # required: non-empty evaluator cases
    ├── metadata.json               # optional evaluator metadata and tolerances
    ├── roofline.json               # optional performance context
    ├── valid.py                    # optional benchmark-owned validation support
    └── agent_problem.json          # optional hidden-shape public contract
```

`--op-dir` points to `operator-name`, not to the checkout root. AKA walks its parent
directories until it finds both `scripts/run_eval.py` and `src/atrex_bench/`. A copied
operator directory outside that checkout is not a complete Native problem unless the
same canonical runtime is placed in an ancestor directory.

`reference.py` and `input.py` must obey the evaluator ABI implemented by that checkout.
The practical compatibility test is that the operator can be evaluated by that exact
checkout's `scripts/run_eval.py`; do not infer compatibility only from filenames.

### `shapes.json`

`shapes.json` must be a non-empty JSON object. Each key is a Shape ID. Each value must
be an object with:

- `init_kwargs`: a JSON object or `null`;
- `input_kwargs`: a JSON object.

For example:

```json
{
  "m1024_n4096": {
    "init_kwargs": null,
    "input_kwargs": {
      "m": 1024,
      "n": 4096,
      "dtype": "bfloat16"
    }
  },
  "m4096_n4096": {
    "init_kwargs": {"causal": true},
    "input_kwargs": {
      "m": 4096,
      "n": 4096,
      "dtype": "bfloat16"
    }
  }
}
```

The exact keyword names and values are operator-specific. They must match `input.py`;
AKA validates the container shape but the evaluator owns their semantics.

### Optional evaluator metadata

`metadata.json` may define benchmark metadata, correctness policies, or a distributed
evaluation contract understood by the selected Atrex-Bench revision. For a supported
multi-GPU contract, current main requires:

```json
{
  "hardware": {"required_gpus": 2},
  "benchmark_contract": {
    "distributed_evaluation": {
      "launcher": "torchrun",
      "backend": "nccl",
      "world_size": 2
    }
  }
}
```

`world_size` must be an integer from 2 through 64. If `hardware.required_gpus` is
present, it must equal `world_size`. Without this explicit contract the evaluator is
submitted as a single-GPU problem, even if other metadata mentions multiple GPUs.

### Exposed and hidden evaluator Shapes

Current main exposes evaluator Shapes by default in both Leaderboard and Production.
In that mode the Agent can see `reference.py`, `input.py`, `shapes.json`, and any
present `metadata.json`, `roofline.json`, and `valid.py`.

`--no-expose-evaluator-shapes` requests the hidden-shape flow. It is supported only
for a Native Atrex-Bench operator in Production mode. The Agent then sees
`reference.py`, `input.py`, and `agent_problem.json`, while exact evaluator Shapes and
detailed evaluator metadata stay private.

`agent_problem.json` may be supplied by the benchmark author or generated before
setup. A supplied contract must use schema `atrex.agent_problem.v1` and contain:

- a non-empty string `objective`;
- object fields `evaluation`, `operator_contract`, and `shape_domain`;
- `evaluation.exact_cases` equal to `"private"`;
- `evaluation.development_cases_are_evaluation_cases` equal to `false`;
- `invariants`, a list of non-empty strings;
- `coverage_regimes`, a list of objects;
- optional object fields `workload_profile` and `distribution_profile`;
- optional `development_cases`, each with object-or-null `init_kwargs` and object
  `input_kwargs`.

Development cases must be synthetic and must not duplicate any private evaluator
case exactly. An automatically generated contract is additionally checked for useful
non-empty domain/invariant content and for verbatim leakage of private cases.

## SOL-ExecBench operators

A supported SOL input has this layout:

```text
operator-name/
├── definition.json                 # required SOL definition
├── workload.jsonl                  # required workload records
└── reference.py                    # required reference implementation
```

The simultaneous presence of `definition.json` and `workload.jsonl` selects the SOL
path. `reference.py` remains mandatory. SOL owns input generation, per-workload
correctness, and tolerance behavior through its definition and workload records; it
does not use Native `shapes.json` as its task contract.

Current Light SOL/SSH campaigns require the `single_file` candidate layout, and Light
does not provide the Native Atrex-Bench ABBA fallback for that path. Use the Legacy
Workflow when the campaign requires a verification path that the selected Light/SOL
transport does not implement.

## Initial V0 versus candidate source layout

For a Native Atrex-Bench problem, the default initial V0 is derived from
`reference.py`: AKA copies it into the managed candidate as `kernel.py`. A `kernel.py`
or `impl/` directory placed beside the operator inputs is not automatically selected
as V0.

After problem ingestion, one candidate file system remains fixed for the Campaign:

| Candidate file system | What the Agent edits | Constraints |
|---|---|---|
| `source_tree` (default for Native) | Editable `kernel.py` and optional sources under `impl/` | Requires Native Atrex-Bench Eval; `kernel.py` may contain the full implementation |
| `single_file` | One editable Python `kernel.py` | Cannot be selected with `--framework Cuda`; required for current Light SOL/SSH |
| `cuda_manifest` | Native CUDA sources, headers, and `solution.json` | Requires `--framework Cuda`; Agate supplies the Python binding/launcher |

An advanced Native task may seed a larger source tree explicitly through a candidate
composition patch containing both a source manifest and exactly one of a local
`directory` or Git `repository`. This is not inferred from files in `--op-dir`. The
imported tree must contain UTF-8 source only, include no symlinks or prebuilt binary
artifacts, stay within 4,096 files / 64 MiB total / 16 MiB per file, and must not
already contain `kernel.py`; the manifest's adapter becomes the managed `kernel.py`.

## Preflight checklist

Before launch, verify all of the following:

- `--op-dir` is a directory and contains every required file for exactly one supported
  family;
- JSON files parse successfully and use the schema expected by the selected checkout;
- a Native operator is under a canonical Atrex-Bench root with `scripts/run_eval.py`
  and `src/atrex_bench/`;
- `reference.py`, `input.py`, Shapes, metadata, and the installed evaluator revision
  agree on the calling and correctness contract;
- the requested Agate hardware can run the operator, including its GPU count;
- the selected framework and candidate file system are compatible;
- hidden-shape Production has a valid `agent_problem.json` or permits AKA to generate
  one before setup;
- the Campaign uses a fresh workspace when the problem bytes, mode, visibility, or
  candidate layout intentionally change.

AKA snapshots the accepted problem. Resume restores that pinned snapshot; it must not
silently switch to changed operator bytes or a different problem layout.
