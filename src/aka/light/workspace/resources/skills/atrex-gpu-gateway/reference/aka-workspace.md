# AKA workspace file systems

Use the file system pinned in `.atrex_file_system.json`. New tasks default to
`source_tree`; framework selection does not switch the file system.

## Source directory (default)

The candidate is one editable source tree rooted at `kernel.py + impl/`. `kernel.py`
is the evaluator entry point and may either contain the implementation or import it
from files under `impl/`. Edit/add/delete both the entry point and implementation
files as needed; all source files travel together in one source archive.

CUDA implementations may use separate `.cu`, `.cpp` and header files with a
Python loader in the editable directory. Compile through the supported Torch
extension API on Agate; use paths relative to the loader's `__file__`.
Do not import `pathlib` or `os`, which Agate source validation rejects.
`kernel.py` must continue to supply the evaluator's Python `Model` entry point; an
Agate native-CUDA manifest is not used in this mode.
ABBA accepts a reachable `--baseline-commit`; Runtime exports that entire source directory privately.
Use the managed Evaluate/Profile tools and submit the usual episode report.

## Explicit CUDA manifest

With `file_system=cuda_manifest` and `framework=cuda`, native Atrex-Bench campaigns
on Agate use `.cu` files and `solution.json`. Keep V0's PyTorch reference as the baseline only. Do not embed
CUDA in kernel.py, generate Python bindings, run nvcc on the host, or change the
reference/input/evaluator. Gateway and SSH support this manifest route; SOL does not.

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
ABBA accepts a reachable `--baseline-commit`; Runtime exports its native source tree privately.
Temporary experiments belong in scratch/, not source.files. Leave the final exact
candidate in place and submit its exact Agent Git commit with `submit-kernel` when final-source
policy feedback is useful, after a standard full Evaluate passes for that exact commit. The Agent
owns engineering Git and commits; submission verifies saved evidence but launches no Evaluate or
ABBA and performs no promotion, rollback, or handoff.
