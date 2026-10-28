# AKA workspace file systems

Use the file system pinned in `.atrex_file_system.json`. New tasks default to
`source_tree`; framework selection does not switch the file system.

## Source directory (default)

Keep the fixed `kernel.py` adapter unchanged. Without an import configuration it
imports `Model` from editable `impl/model.py`. Edit/add/delete implementation files
under `impl/`; an imported package instead uses the `editable_roots` in
`.atrex_source.json`. All source files travel together in a source archive.

CUDA implementations may use separate `.cu`, `.cpp` and header files with a
Python loader in the editable directory. Compile through the supported Torch
extension API on Agate; use paths relative to the loader's `__file__`.
Do not import `pathlib` or `os`, which Agate source validation rejects.
The source-directory adapter supplies the Python `Model` entry point; an Agate
native-CUDA manifest is not used in this mode. Do not replace the pinned adapter.
`kernel-read` exports the entire source directory for `--baseline-path`.
Use the managed Evaluate/Profile tools and submit the usual episode report.

## Explicit CUDA manifest

With `file_system=cuda_manifest` and `framework=cuda`, native Atrex-Bench campaigns
on Agate use `.cu` files and `solution.json`. Keep V0's PyTorch reference as the baseline only. Do not embed
CUDA in kernel.py, generate Python bindings, run nvcc on the host, or change the
reference/input/evaluator. SOL and SSH do not support this manifest route.

Write this local manifest (adapt the inputs and launcher to the operator):

```json
{
  "schema_version": 1,
  "name": "add-one",
  "version": "1",
  "source": {"type": "local", "files": ["kernel.cu"]},
  "entry_point": {
    "type": "cuda",
    "symbol": "launch_add_one",
    "build": {"toolchain": "torch_extension", "sources": ["kernel.cu"], "cuda_cflags": ["-O3"]},
    "inputs": [{"kind": "tensor", "name": "x", "dtype": "float32"}],
    "outputs": [{"name": "out", "like": "x"}],
    "arguments": [{"ref": "x"}, {"ref": "out"}, {"numel": "x"}]
  },
  "dependencies": []
}
```

`source.type=local` is an AKA workspace convenience. The Supervisor reads exactly
these regular UTF-8 files, packs an OSS archive, calculates archive/file SHA256,
and sends Agate its standard `source.type=oss_archive` manifest. Do not inline
source, manually upload files, edit generated hashes, or put credentials in JSON.
All other fields follow [Agate manifest v1](eval-manifest.md), including its ABI,
source limits and dependency rules. Unknown fields are errors.

Read [the CUDA ABI and Git dependencies](eval-manifest.md#native-cuda-candidates-without-python)
before implementing the launcher or configuring compiler flags/dependencies.
The runnable source is [kernel.cu](../assets/eval-manifest/cuda-native/kernel.cu).
For external libraries, look for a matching repository in the organization-provided
reference-project mirror group first; use verified mirror URLs in `mirrors`, then
fall back to the upstream repository. Do not put credentials in JSON.

Use the normal managed measurement tool, for example:

```bash
python3 tools/sandbox.py --kind run --no-sync
```

The controller supplies the hardware, environment and evaluator. A full Evaluate
covers correctness and performance. Change JSON and native sources together;
any file, build option or dependency change creates a different Kernel identity.
`kernel-read` exports a directory for a native Kernel, suitable for baseline-path.
Temporary experiments belong in scratch/, not source.files. Leave the final exact
candidate in place and use the usual episode-report flow; the Supervisor owns Git.
