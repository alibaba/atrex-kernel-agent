# Minimal Native Atrex-Bench input

This directory shows the smallest useful Native Atrex-Bench problem accepted by AKA
Light:

- `reference.py` defines the authoritative `torch.nn.Module` semantics;
- `input.py` defines `_make_inputs`, which turns one Shape's `input_kwargs` into CUDA
  tensors passed to `Model.forward`;
- `shapes.json` maps stable Shape IDs to `Model.__init__` and `_make_inputs` arguments.

The example intentionally omits optional `metadata.json`, `roofline.json`, `valid.py`,
and `agent_problem.json`.

This directory is stored in the Skill, not under a canonical Atrex-Bench runtime. Copy
it into the `data/` tree of the Atrex-Bench checkout that will evaluate it:

```bash
cp -R \
  ~/.codex/skills/atrex-aka-light/examples/native-atrex-bench-vector-add \
  /path/to/atrex-bench/data/native-atrex-bench-vector-add
```

The resulting layout must have both runtime markers above the operator:

```text
/path/to/atrex-bench/
├── scripts/run_eval.py
├── src/atrex_bench/
└── data/native-atrex-bench-vector-add/
    ├── reference.py
    ├── input.py
    └── shapes.json
```

It can then be selected with:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent -- \
  --op-dir /path/to/atrex-bench/data/native-atrex-bench-vector-add \
  --platform L20D \
  --agent-cli claude \
  --optimization-mode fast \
  --workspace /path/to/vector-add-campaign
```

Use the installed revision's `aka optimize --repo-root REPO -- --help` before launch.
The command uses the default pooled three-trajectory search and exposed-Shape flow.
To run a bounded serial functional test, pass a launch-level composition patch with
`workflow.config.search` set to an empty mapping, then use `--max-iters 5`; in current
Light that value counts Episode audit segments rather than Kernel versions. A Performance/Ultra run with
`--no-expose-evaluator-shapes` may generate `agent_problem.json` before setup.
