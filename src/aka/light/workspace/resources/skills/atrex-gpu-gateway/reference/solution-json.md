# solution.json: SOL-ExecBench candidates (`agate eval --backend=sol`)

**This file contains your implementation.** `sources[].content` holds inline source, and `spec.entry_point` names your entry function.
Uploading a solution means uploading your operator, not somebody else's solution.

The gateway forwards the file **verbatim**: `agate eval --backend=sol --solution f.json` does not rewrite it.
An existing upstream solution.json can therefore be submitted unchanged.

## Minimal structure

```json
{
  "name": "my-rmsnorm-v3",
  "definition": "001_rms_norm",
  "author": "daocheng",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.py::rms_norm"
  },
  "sources": [
    { "path": "kernel.py", "content": "import triton\n...complete kernel source..." }
  ]
}
```

Submit it with:

```bash
agate eval --backend=sol --solution solution.json --gpu L20N
```

## Fields

| Field | Required | Meaning |
|---|---|---|
| `name` | Yes | A nonempty name for this submission. |
| `definition` | Conditional | Problem name; required for the image catalog. With `--problem`, it can be derived from `definition.json.name`. Keeping it is recommended for direct upstream-tool compatibility. |
| `author` | Yes | Upstream requires a nonempty value. Omitting it triggers `author Field required` upstream. |
| `spec.entry_point` | Yes | `filename.py::function_name`, resolvable from `sources`; replaces atrex's implicit `class Model` contract. |
| `spec.target_hardware` | Yes | Upstream accepts only `B200` / `LOCAL`, not gateway cluster names. This does not select a cluster; `--gpu` does. Use `["B200"]` as shown. |
| `sources` | Yes | At least one `{path, content}`. `content` is an inline source string, not a filesystem path. |
| `spec.languages` | No | `pytorch`, `triton`, `cute_dsl`, `cutile`, `cudnn`, `cublas`, `cutlass`, `cuda_cpp`, etc. |
| `spec.dependencies` | No | Upstream dependency declarations. Use `-r requirements.txt` to install pip packages. |
| `spec.destination_passing_style` | No | Defaults to `true`: outputs are preallocated and passed as trailing arguments. This differs from the atrex default. |
| `spec.binding` | No | C++/CUDA binding style; upstream defaults to `torch`. |
| `spec.compile_options` | No | For example, `{"cuda_cflags": [...], "ld_flags": [...]}`. |
| `description` | No | Free text. |

## Multiple files

`sources` may contain multiple entries, including non-Python files (`.cu`, `.cpp`, `.h`):

```json
"sources": [
  { "path": "kernel.cu",  "content": "__global__ void rms(...) { ... }" },
  { "path": "binding.cpp", "content": "#include <torch/extension.h>\n..." },
  { "path": "run.py",     "content": "from .binding import rms\ndef run(x, w): ..." }
]
```

`entry_point` names the Python entry among those files, such as `run.py::run`.
Paths must be relative and cannot contain `..`.

**`content` is a JSON string**; escape source newlines, quotes and backslashes.
Generate it with a script to avoid hand-escaping errors:

```python
import json, pathlib
srcs = [{"path": p.name, "content": p.read_text()} for p in pathlib.Path("mykernel").glob("*.py")]
doc = {"name": "my-v3", "definition": "001_rms_norm", "author": "me",
       "spec": {"languages": ["triton"], "target_hardware": ["B200"],
                "entry_point": "kernel.py::rms_norm"},
       "sources": srcs}
pathlib.Path("solution.json").write_text(json.dumps(doc, indent=2))
```

## Problem sources

There are two sources:

1. **Image catalog**: omit `--problem`; `definition` selects one of the built-in 235 problems.
2. **Caller-supplied custom problem**: pass `--problem DIR` containing native upstream `definition.json + workload.jsonl`.
   The image catalog is not searched. See [custom-sol-problem.md](custom-sol-problem.md) for a complete example requiring no repository access.

Image-catalog behavior:

- The gateway has no dataset for submission-time validation. A misspelled problem produces a structured runtime
  `validation/no_such_problem` naming the problem, not HTTP 400.
- `--subset L1|L2|Quant|FlashInfer-Bench` bypasses subset searching in the pod.
  Otherwise all four are searched; zero or multiple matches produce explicit errors rather than ambiguous infrastructure failures.
- Counts: L1=94, L2=82, Quant=33, FlashInfer-Bench=26.

## Important pitfalls

- **`speedup` defaults to 0**. Upstream `benchmark_reference=false` skips reference timing,
  leaving `reference_latency_ms=0.0` and a zero ratio. Use explicit `--benchmark-reference` for meaningful speedup at roughly 2× runtime.
  **Default zero does not mean no acceleration.**
- **`--workload-timeout` is a per-workload budget** (upstream `--timeout`, default 600s).
  `--job-timeout` covers the whole job (default 1800s). A problem runs multiple workloads serially; do not confuse the budgets.
- **`REWARD_HACK` does not fail the job**. It is an upstream verdict returned as evidence; inspect `summary.reward_hack`.
- **`problem_patched: ["hf_id"]`** means the gateway patched the problem so it could run.
  The 26 FlashInfer-Bench problems have empty `hf_id` strings in the image, which the upstream model rejects.
  This originates in dataset publication, not the solution. The gateway sets only this provenance field to null in the job-local copy;
  it does not affect verdicts. The 209 healthy problems do not have this key.

## Choosing the backend

| Situation | Backend |
|---|---|
| A faster implementation of an existing catalog problem | `--backend=sol` (this document) |
| Your own operator with an atrex-bench reference directory | `--backend=atrex`; see [reference-dir.md](reference-dir.md) |
| Your own operator using SOL definition/workload/solution contracts | `--backend=sol --problem DIR`; see [custom-sol-problem.md](custom-sol-problem.md) |
