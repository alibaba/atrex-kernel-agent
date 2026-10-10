# Local sandbox GPU integration test

Run the same public test suite on a prepared single-GPU Linux machine or Pod.
The suite has no dependency on a CI provider, a resource scheduler, Ray, object
storage, private operator datasets, or platform SDKs. Provisioning and artifact
transport belong to the caller.

## Prepare

Use a non-root account with Git, Python, PyTorch, Triton, NumPy, PyYAML, Node.js
and the Claude Code CLI installed. The Python environment used to launch the
test must already support the allocated CUDA or ROCm GPU. The suite installs
no dependencies at runtime and requires exactly one visible GPU. Allow the
optimizer to reach a standard Claude API endpoint using these environment
variables:

```bash
export ANTHROPIC_AUTH_TOKEN='<model API token>'
export ANTHROPIC_MODEL='<model name>'
# Set ANTHROPIC_BASE_URL only when using a compatible API endpoint.
git submodule update --init 3rdparty/ncu-report-skill
git clone https://github.com/alibaba/atrex-bench.git /tmp/atrex-bench
git -C /tmp/atrex-bench checkout 22d6a57b98ebb45c8db76e98815158d820c1dca3
```

The pinned evaluator revision is the tested compatibility baseline. Supply its
checkout explicitly; the test neither downloads code nor switches revisions.
The gateway receives a clean environment without the model token. Model access
is available only to the optimizer/agent process. Environment filtering is not
an OS isolation boundary; run generated code in an appropriate disposable
machine or container.

## Run

From the public repository root:

```bash
python tools/local_sandbox_test/runner.py \
  --work-dir /tmp/atrex-local-test \
  --bench-root /tmp/atrex-bench \
  --platform MI308X \
  --timeout 7200
```

Use the optimizer platform matching the allocated hardware, such as `H20` or
`MI308X`. The work directory must be new or empty. The default evidence path is
`WORK_DIR/artifacts`; `--output-dir` can select another empty directory. Runs
start their own localhost gateway on port 8000, so use separate network
namespaces for concurrent tests. The fixture is copied into the work directory;
the public source tree and evaluator checkout are not modified.

The suite checks GPU visibility, a bare-Python sandbox command, Triton JIT,
exact vector-add correctness for lengths 512/1025/4096 and returned JSON
artifacts. It then runs the native production optimizer through V0/V1/V2 with
one fast episode. PyTorch reference timing is measured on that GPU.

After optimization exits, the archived final kernel and solution are evaluated
again by the native harness. A fresh schema-v3 receipt binds all three passing
cases and finite positive timings to their SHA-256 fingerprints. Canonical
files are restored even if this evaluation fails, then the final workspace is
checked separately for correctness. Candidate promotion or speedup is not a
passing requirement. Canonical records must match Git HEAD; incomplete or
violating episodes fail.

`result.json` reports `PASS`/`FAIL`, elapsed time, tested source commit, fixture
hashes, GPU environment and campaign outcome. Logs, canonical memories, episode
archives, exact final candidate and its receipt are retained for review. Exit
code zero means the complete suite passed. `--fault missing-module` injects a
missing import into the smoke subprocess and must fail; it is a diagnostic
negative test, never a passing CI result.

## CPU regressions

These tests need neither GPU nor model credentials:

```bash
python -m unittest discover -s tools/local_sandbox_test -p 'test_*.py' -v
python -m unittest discover -s tools -p test_local_gateway_venv.py -v
```
