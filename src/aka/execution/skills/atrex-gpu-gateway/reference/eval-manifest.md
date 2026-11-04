# Solution manifests for atrex eval

The `candidate` field of `POST /v1/jobs/eval` accepts three mutually exclusive representations: the existing Python source string,
the existing `{archive, entry_point}` OSS source-archive reference, or a versioned manifest object.
The CLI uses `--manifest FILE.json`, mutually exclusive with `--candidate`, `--candidate-entry` and `--config`; `--reference-dir` remains required.
This is the atrex eval input format, a different contract from SOL-ExecBench `solution.json` used with `--backend sol --solution`.
The `profile` HTTP API also accepts manifests. Compilation jobs and the legacy `/v1/evals` endpoint do not yet accept them.

## v1 fields

```json
{
  "schema_version": 1,
  "name": "add-one",
  "version": "1",
  "entry_point": "impl/entry.py::Model",
  "source": {
    "type": "inline",
    "files": {
      "impl/__init__.py": "",
      "impl/entry.py": "from .helper import Model\n",
      "impl/helper.py": "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x + 1\n"
    }
  },
  "dependencies": [
    {"id": "torch", "kind": "pip", "requirement": "torch>=2"}
  ],
  "targets": {"hardware": ["H20"], "num_gpus": 1, "shape_ids": ["0", "1"]}
}
```

- Schema version, name, implementation version, source and entry point are required; unknown fields are rejected.
  Python entry points use a relative file path followed by `::Model` and support package-relative imports.
  CUDA entry points use the object described below; the worker compiles them and generates an adapter.
  User code is never imported or compiled in the gateway process.
- With `source.type=inline`, `files` maps paths to UTF-8 source. Each Python file is subject to source-size limits and AST validation.
  All files together are limited to 4 MiB and 2000 files. Absolute paths, traversal and conflicting paths are prohibited.
- With `source.type=oss_archive`, provide `archive` (matching `oss_files[].path`), the archive `sha256`,
  and `files` (the complete mapping from paths to SHA256 hashes of UTF-8 file bytes).
  Source travels directly from OSS to the worker. The worker safely unpacks it, verifies every file and digest,
  then applies source validation and entry-point adaptation. Supply binaries as separate attachments.
- Every dependency has a unique `id` and optional `depends_on` references to other dependency IDs.
  Missing nodes, cycles and duplicate IDs are rejected at submission. `pip` dependencies declare a `requirement`;
  these are merged with top-level `requirements`, deduplicated by text, and use the existing validation/installation path.
  `deps_mode` still defaults to `freeze_installed`; the manifest does not change resolution policy automatically.
- `system` dependencies use `name` and `version` to describe system-library/toolchain prerequisites in the image.
  The gateway neither installs nor validates system packages. `artifact`/`weights` use `path` and `sha256`
  to reference separate OSS attachments. The worker verifies their digests and maps them to matching paths in the source tree,
  where code can locate them relative to `__file__`. Attachment contents do not appear in query metadata.
- Python-entry `build` dependencies support `execution=entry_point`, declaring `toolchain`, `inputs`, `outputs` and dependency edges.
  `inputs` reference sources, declared attachments or earlier build outputs; consuming another node's output requires a dependency edge.
  `outputs` are logical artifact names, not instructions for gateway file writes or automatic output collection.
  The candidate entry point still performs compilation/code generation. Manifest shell commands are not executed, and no new DAG scheduler is introduced.
  Pip, system and attachment preparation cannot depend on an entry-point build that has not run.
  CUDA-entry builds are configured separately in `entry_point.build` and cannot also declare these entry-point build dependencies.
- `git` declares a versioned repository for automatic worker preparation; see the next section. It is not limited to a GPU vendor or repository.
- Optional `targets.hardware` must cover all requested hardware aliases, `num_gpus` must match the request,
  and `shape_ids` must cover all reference shapes. `compile_targets` describes entry-point compilation targets;
  it neither injects compiler arguments nor proves the worker supports the architecture.

## Validation and metadata

`GET /v1/jobs/eval/solution-schema` returns model-generated JSON Schema.
`POST /v1/jobs/eval/validate` accepts a complete eval request, reuses submission-time structural/source validation,
and returns `valid`, `scope=declaration_and_inline_source` and `solution`. It submits no job and downloads nothing from OSS.
Structural errors return `422 request_validation_failed`; semantic/source errors return `400 invalid_source`.
Both identify problems in `details.validation_errors`. This endpoint does not guarantee OSS upload credentials, target resources or environment readiness.

Submission can check the source inventory, paths, dependency cycles, declaration constraints and supplied Python content.
An inline entry point that clearly neither declares nor imports `Model` is rejected.
Actual OSS files, digests and dynamic Python symbols can only be confirmed after worker-side extraction/loading.
Successful submission does not prove a runtime entry point exists. Source-archive/attachment digest errors fail as
`source_archive_invalid` before the SDK starts and are not retried.

The `solution` field in `GET /v1/jobs/{job_id}` exposes declarations without source code, file digests, dependency edges and topological order.
Pip dependencies containing URLs are replaced with `sha256:<digest>` in query and validation results; URLs and temporary credentials are not echoed.
The same information is persisted in `ExecutionSpec.metadata.solution` and retained in each attempt's execution specification.
Legacy strings/archives are projected into minimal source descriptions; undeclared file inventories and digests in legacy archives remain unknown.
`declaration_sha256` fingerprints only the manifest declaration. It is not an artifact cache key: reuse also depends on actual hardware,
toolchain, installed dependencies, environment settings and evaluator version. The first version promises neither cache hits nor shared artifacts.
Manifests currently execute serially: automatic shape parallelism falls back to serial execution, and explicit `--shape-shards` greater than 1 is rejected.
Both ABBA candidate and baseline may use manifests, with separate sources, dependencies and build artifacts.
The evaluator must support `eval_mode=abba`; use `--atrex-bench-version` to select a compatible version if the image's evaluator is older.

## Automatic repository dependencies

Declare a repository and version in `dependencies`, for example CUTLASS:

```json
{
  "id": "cutlass",
  "kind": "git",
  "repository": "https://github.com/NVIDIA/cutlass.git",
  "ref": "v3.9.2",
  "include_dirs": ["include", "tools/util/include"]
}
```

The same contract supports CK or another repository, for example
`{"id":"ck","kind":"git","repository":"https://github.com/ROCm/composable_kernel.git","ref":"rocm-6.3.0","include_dirs":["include"]}`.
Multiple repositories are prepared in dependency order. A repository dependency does not imply that the environment supports its toolchain.
The native entry point currently targets CUDA; HIP compilation/evaluation for CK is outside that entry point's support.
Existing pip installation and attachment preparation precede repository preparation and cannot depend on Git nodes.

- `repository` must be a worker-accessible HTTPS URL without username, password, query or fragment.
  Fetches use Git's HTTPS transport without global Git configuration or credential helpers. Private-repository authentication parameters are not supported.
  The worker must have Git installed. The gateway validates the declaration without fetching repositories in the HTTP process.
- Optional `mirrors` is an ordered list of HTTPS mirror URLs with the same validation rules as `repository`.
  The worker tries mirrors first. An inaccessible mirror, missing version or locked-commit mismatch advances to the next mirror, then the original repository.
  Network/tool infrastructure failures fetching the original repository receive at most three attempts, with 1/2-second intervals;
  version and authentication failures are not retried. Every source uses the same `ref`/`commit`; pinning `commit` is recommended.
  Without mirrors, the worker uses the original repository directly. Callers configure mirror URLs; organizational private addresses are not built in,
  and local credentials are not forwarded. The example generator accepts repeatable `--git-mirror "$REPOSITORY_MIRROR_URL"`.
  An internal mirror must allow direct worker access, otherwise the worker falls back.
- `ref` must be a tag name, `refs/tags/<tag>` or a full 40-character commit; branches and arbitrary refspecs are rejected.
  Tags resolve to actual commits. Optional `commit` requires an exact match with a pinned 40-character SHA.
  Missing versions, pin mismatches and missing include directories fail before compilation.
- `include_dirs` lists repository-relative directories; `.` means the repository root.
  CUDA entry points automatically add them to compilation. Use normal `#include` directives in `.cu` files;
  third-party headers need not be copied into `source.files`.
- Repositories are prepared under `.agate_dependencies/<id>` inside the attempt's source tree, a path reserved for agate.
  Python entry points may also refer to it relative to their source tree; existing source validation still applies.
  Repository files do not consume the candidate's 4 MiB source budget. Each job fetches independently, with no shared mutable directory or cross-job cache.
- Repository install/build scripts are not executed. Submodules are not initialized automatically, and Git LFS objects are not downloaded.
  Declare other repositories separately. Each Git command is limited to 300 seconds, within the overall job execution timeout.
- `prepared_dependencies.candidate` records each repository's declared version, resolved commit, logical directory and include directories.
  `fetched_from` records the actual source; `mirror_failures` records fallback sources and error categories without raw remote diagnostics or credentials.
  `solution.dependencies` retains the submitted declaration. Static validation does not prove a version exists or is downloadable by the worker.
  Declaration/version errors are `validation/repository_invalid`; network or Git-tool availability errors are `infra/repository_unavailable`.
  There is no fallback to another version already in the image.

The bundled `cuda-cutlass` example submits only a CUTLASS-using `.cu` file; agate fetches its dependency:

```bash
python make_manifest.py cuda-cutlass --gpu H20 --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference
```

## Native CUDA candidates without Python

`source.files` may contain only `.cu` files and required native sources/headers. Set `entry_point` to this object:

```json
{
  "type": "cuda",
  "symbol": "launch_add_one",
  "build": {
    "toolchain": "torch_extension",
    "sources": ["impl/kernel.cu"],
    "cflags": ["-O3"],
    "cuda_cflags": ["-O3"]
  },
  "inputs": [{"kind": "tensor", "name": "x", "dtype": "float32"}],
  "outputs": [{"name": "out", "like": "x"}],
  "arguments": [{"ref": "x"}, {"ref": "out"}, {"numel": "x"}]
}
```

The `.cu` host launcher has this signature. It selects grid/block/shared-memory settings and launches the kernel:

```cpp
extern "C" void launch_add_one(const void* x, void* out, int64_t n, cudaStream_t stream);
```

- `build.sources` lists `.cu`/`.cpp`/`.cc`/`.cxx` files from the complete source inventory and must include at least one `.cu`.
  `include_dirs` is relative to the source tree; `cflags`, `cuda_cflags` and `ldflags` are compiler-argument arrays.
  Argument entries cannot contain whitespace or shell metacharacters; shell build scripts are not accepted.
  The toolchain is fixed to `torch_extension`. The worker image needs compatible PyTorch, CUDA toolkit, a C++ compiler and ninja;
  system libraries are not installed automatically.
- `inputs` declares names and types in `Model.forward` argument order; names must match the reference inputs.
  Tensors use `kind=tensor` and support `float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int32`, `int64` and `bool`.
  At runtime they must be contiguous, match the declared dtype and share a CUDA device. Inputs are not implicitly copied or converted.
  Scalars use `kind=scalar` and support `int64`, `float32`, `float64` and `bool`. The first version supports only empty `init_kwargs`.
- `outputs` uses `like` to choose the source tensor for device and default shape/dtype; `dtype` can override the output type.
  Each dimension in optional `shape` is a nonnegative integer or `{"input":"x","dim":0}`; arbitrary expressions are not evaluated.
  Outputs are newly allocated contiguous tensors. A single output is returned directly; multiple outputs form a tuple in declaration order.
  The first version does not declare in-place updates, output aliases or arbitrary Python-container return values.
- `arguments` determines launcher argument order: `{"ref":"x"}` references an input/output,
  while `{"numel":"x"}` and `{"size":"x","dim":0}` pass an `int64_t` element count or dimension size.
  Input tensors become `const void*`, outputs become `void*`, and scalars map to `int64_t`/`float`/`double`/`bool` as declared.
  A final `cudaStream_t` is always appended, using the input device's current PyTorch CUDA stream.
  The launcher must use C linkage and obey this ABI, casting pointers to the declared dtype. The gateway does not infer native function signatures.
- The worker first verifies source/attachment digests, then generates C++ bindings and compiles an extension isolated to the attempt.
  The generated `Model` only loads and invokes the extension in the evaluation process; fresh ABBA processes reuse that attempt's artifacts.
  Compilation counts against the total job budget, not kernel timing. Compilation/extension-loading failures return `code/native_build_failed` without automatic retries.
  Native compilation is not a sandbox: it relies on existing worker isolation and creates no shared cross-job build cache.

Run these commands from the bundled example directory; the candidate archive contains only `impl/kernel.cu`:

```bash
python make_manifest.py cuda-native --gpu H20 --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference

python make_manifest.py cuda-native --gpu H20 --archive --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference \
  --oss-file archives/solution.zip=solution.zip
```

The reference benchmark still uses the existing Python input-generator/reference contract. Only the candidate implementation is Python-free.

## Runnable multi-file examples

The bundled [assets/eval-manifest](../assets/eval-manifest/) directory provides four add-one candidates and a shared reference:
Triton separates its entry point and kernel; CUDA uses a Python entry point to compile separate C++ bindings and `.cu` sources;
`cuda-native` submits only `.cu` and declares compilation/invocation in the manifest; `cuda-cutlass` additionally declares a versioned Git dependency.
The CUDA examples require a CUDA GPU, float32 inputs and an image with compatible PyTorch/CUDA/C++ tools.
Run the following commands from that example directory, using an authorized hardware alias:

```bash
python make_manifest.py triton --gpu H20 --output triton.json
agate eval --gpu H20 --manifest triton.json --reference-dir reference

python make_manifest.py cuda --gpu H20 --archive --output cuda.json
agate eval --gpu H20 --manifest cuda.json --reference-dir reference \
  --oss-file archives/solution.zip=cuda.zip
```

The SDK retains its existing builder, accepting a `.json` path or an in-memory manifest mapping:

```python
import os
from atrex_gateway_client import AkSkAuth, Client, build_eval_request

client = Client(os.environ["AGATE_URL"], auth=AkSkAuth(os.environ["AGATE_AK"], os.environ["AGATE_SK"]))
request = build_eval_request("triton.json", "reference", "H20", shape_shards=1)
print(client.validate_eval(request))
accepted = client.submit_job("eval", request)
print(client.wait(accepted["job_id"]))
```

To migrate a single-file candidate, put its source in `source.files={"kernel.py": source}`, set the entry point to
`kernel.py::Model`, add `schema_version`, `name` and `version`, and replace `candidate` with that object.
Reference data, options, env_vars, polling and cancellation stay unchanged. Callers that do not need migration may continue submitting `.py` files.
