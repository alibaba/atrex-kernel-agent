# Custom SOL problems: a complete example without repository access

Read this when creating a problem, converting an existing operator to SOL or validating the custom-problem path.
Everything below ships with the Skill; the local machine needs only `python3`, `agate` and gateway authentication.

## Generate the example

Save the following as `create_custom_sol_smoke.py` and run it. It creates `custom-scale2-smoke/` in the current directory,
containing native upstream `definition.json`, `workload.jsonl` and `solution.json`:

```python
from pathlib import Path
import json


root = Path("custom-scale2-smoke")
root.mkdir(exist_ok=True)

definition = {
    "name": "custom-scale2-smokev1",
    "op_type": "elementwise",
    "description": "Multiply a float32 vector by two.",
    "axes": {"N": {"type": "var"}},
    "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
    "reference": "import torch\n\n\ndef run(x):\n    return x * 2.0\n",
}

workloads = [
    {
        "axes": {"N": 4096},
        "inputs": {"x": {"type": "random"}},
        "uuid": "custom-scale2-4096",
    },
    {
        "axes": {"N": 1048576},
        "inputs": {"x": {"type": "random"}},
        "uuid": "custom-scale2-1m",
        "tolerance": {"max_atol": 0.001, "max_rtol": 0.001},
    },
]

solution = {
    "name": "custom-scale2-pytorch",
    "definition": definition["name"],
    "author": "agate-smoke-test",
    "description": "Minimal PyTorch candidate for the custom-problem smoke test.",
    "spec": {
        "languages": ["pytorch"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::solve",
        "destination_passing_style": False,
    },
    "sources": [{
        "path": "kernel.py",
        "content": "def solve(x):\n    return x * 2.0\n",
    }],
}

(root / "definition.json").write_text(
    json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(root / "workload.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in workloads),
    encoding="utf-8")
(root / "solution.json").write_text(
    json.dumps(solution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(root.resolve())
```

```bash
python3 create_custom_sol_smoke.py
agate eval --backend sol \
  --solution custom-scale2-smoke/solution.json \
  --problem custom-scale2-smoke \
  --gpu L20N
```

These commands use production by default. The same problem directory also works with the upstream local tool:

```bash
sol-execbench custom-scale2-smoke \
  --solution custom-scale2-smoke/solution.json --json
```

## Acceptance criteria

All of the following must hold:

- Job `status` is `succeeded`.
- `result.summary.passed == result.summary.total`.
- `problem_source` is `inline`.
- `problem_digest` starts with `sha256:` and remains the same when the identical problem is resubmitted.

A custom problem passing means the candidate agrees with the caller's reference.
It does not mean it passed the official 235 problems; do not combine these in one score statistic.

## Converting an existing operator

Map the operator information as follows:

| Existing information | SOL location |
|---|---|
| Reference implementation | Top-level `run()` in `definition.json.reference` |
| Shape variables | `definition.json.axes` |
| Input names, shapes and dtypes | `definition.json.inputs` |
| Output names, shapes and dtypes | `definition.json.outputs` |
| Each real shape and its input-generation method | One row in `workload.jsonl` |
| Candidate implementation and entry function | `solution.json.sources[]` and `spec.entry_point` |

The serialized order of `definition.inputs`, argument order of `reference.run()` and candidate entry argument order must match.
For example, inputs ordered `x, weight` require both functions to accept `(x, weight)`.
Problem names must match `custom-<operator>-<tag>`, all lowercase, with a tag of at least 4 characters.

To retain special input distributions, add `generate_inputs(axes_and_scalars, device)` to the reference,
set `custom_inputs_entrypoint`, and use `type=custom` for the corresponding workload inputs.
Do not mix `custom` and other input types within one workload.
