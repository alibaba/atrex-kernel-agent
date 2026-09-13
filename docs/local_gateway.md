# Community Local Sandbox

`tools/local_gateway.py` is a small, standard-library-only scheduler for maintaining and testing the
localhost optimization path. It accepts requests from the Supervisor-private gateway engine and
queues local GPU commands FIFO. The Agent's `tools/sandbox.py` remains a pure HTTP facade to the
Campaign Runtime.

## Start the Server

Run it from the repository root in the Python environment that contains the workload's GPU stack:

```bash
python tools/local_gateway.py serve \
  --host 127.0.0.1 \
  --port 8000 \
  --state-dir .atrex-local-gateway
```

If an existing workflow uses a hardware token instead of `local`, register it as an alias without changing
the optimizer arguments:

```bash
python tools/local_gateway.py serve --gpu-alias YOUR_GPU_TOKEN
```

The default `--workers 1` serializes commands for one local GPU. A larger value enables parallel command
execution and should only be used when the machine and workloads can safely share the device.

The state directory contains `jobs.db`, per-job uploaded files, stdout, and stderr. Queued jobs survive a
restart. A job that was running when the server stopped is marked failed on the next start rather than being
silently executed twice.

## Use with the Optimizer

The optimizer continues to use `tools/sandbox.py`, so correctness, performance, and profiler commands have
the same packaging and artifact-return behavior as the remote execution path:

```bash
python orchestrator/optimize.py \
  --op-dir /path/to/operator \
  --platform H20 --framework Triton \
  --sandbox-hardware local \
  --sandbox-url http://127.0.0.1:8000
```

`--framework` can be omitted to start all frameworks supported by the runtime GPU architecture in
parallel. Their sandbox requests still enter this scheduler's FIFO queue, and their local optimizer state
uses flat framework/hardware-suffixed names such as `kernel_opt_<name>_triton_h20`. Production
campaigns use a separate path ending in `_production`.

`--sandbox-hardware local` selects the local executor independently of the logical `--platform` value.
The optimizer does not compare platform and inventory names because inventory may expose an alias or a
desensitized GPU description. It uses the runtime architecture probe for automatic framework dispatch.

## Compatibility Surface

The community scheduler implements:

- `GET /healthz`
- `GET /v1/env`, `GET /v1/env/local`, and `GET /v1/env/local/capabilities`
- `POST /v1/jobs/eval` for native Atrex-Bench evaluation
- `POST /v1/jobs/profile` for supported `ncu`/`rocprofv3` profiling requests
- `POST /v1/jobs/dev`, including uploaded text files, environment variables, timeouts, and idempotency keys
- `POST /v1/jobs/compile` for candidate import/construction, one real Shape launch, optional architecture validation, and optional compute-sanitizer execution
- `POST /v1/jobs/disassemble` for `auto`, `sass`, `ptx`, or `isa` extraction after one real Shape launch has triggered Candidate JIT
- `GET /v1/jobs` with the standard kind/user/status/limit filters
- `GET /v1/jobs/<job_id>`, including `wait=true&timeout=<seconds>` long polling
- `POST /v1/jobs/<job_id>/cancel`
- legacy `GET /v1/evals/<job_id>` polling compatibility

`tools/sandbox.py` prefers typed `eval`, `profile`, `compile`, and `disassemble` requests when the
workspace fits their source contract, and falls back to a self-contained `dev` request for SOL,
aggregate, custom-input, or otherwise unrepresentable commands. Profile and diagnostic requests
accept repeatable PEP 508 dependencies and a per-job dependency policy:

```bash
python tools/sandbox.py --kind check \
  --requirement 'custom-kernel-package==1' \
  --deps-mode no_deps --no-sync
python tools/sandbox.py --kind disassemble --format isa \
  --requirement 'custom-kernel-package==1' \
  --deps-mode freeze_installed --no-sync
```

Dependencies are installed into the individual local job directory and do not modify the optimizer
or host Python environment. The Supervisor augments localhost diagnostics with one evaluator-owned
opaque Shape and the input generator. `check` therefore imports, constructs, and really launches the
Candidate; a requested sanitizer wraps that launch. `disassemble` performs the same launch/JIT before
searching the job-local Torch, Triton, CUDA, and CuteDSL caches for PTX, cubin, fatbin, shared objects,
or HSACO. Missing inputs, tools, or generated binaries produce a structured capability/collection
failure rather than a synthetic success. These probes still do not replace numerical correctness or
the authoritative performance evaluator.

The Agent facade also exposes the read-only environment query and full typed Profile selectors:

```bash
python tools/sandbox.py --kind env
python tools/sandbox.py --kind env --env-gpu local --env-capabilities
python tools/sandbox.py --kind profile --profile-shape-id 0 \
  --kernel-name candidate_kernel --profile-source --launch-skip 1 --launch-count 5
```

Each exact full Evaluate or exploratory ABBA task is accepted once per workspace. Its first request
runs three independent identical logical Agate calls; the Supervisor takes the median per Shape and
recomputes the aggregate result. Evaluate reports its aggregation summary; ABBA returns only final
aggregate and per-Shape Baseline/Candidate values. Raw repetitions and ABBA batching details remain
in private evidence. Repeating the
same Kernel task is rejected before Agate execution and reports the previous `gateway_record_id` so
the existing result can be reused. Read it without another Gateway job:

```bash
python tools/sandbox.py --kind record-read \
  --record-id gateway-<timestamp>-<kernel-prefix>
python tools/sandbox.py --kind record-read \
  --record-id kernel-<timestamp>-<opaque-id> \
  --view gateway-records
python tools/sandbox.py --kind record-read \
  --record-id kernel-<timestamp>-<opaque-id> \
  --view source \
  --output-path scratch/restored-kernel.py
```

The generic reader supports Evaluate, same-allocation ABBA, Profile, Dev, Check, and Disassemble
records. It returns the same operation-specific bounded Agent projection, including opaque
per-Shape latency where that operation produces it. An arbitrary Dev probe remains a Dev record
containing its command, exit code, bounded stdout/stderr, and synchronized paths; it is never
presented as evaluator truth. ABBA identifies Incumbent and Candidate Kernel IDs separately.
Each Candidate has an opaque stable Kernel ID; the Supervisor retains its content digest privately.
The `gateway-records` view lists its Evaluate/Profile/Dev/Check/Disassemble records, while the
separate `source` view verifies and writes exact source beneath `scratch/` without a Gateway job.
ABBA appears in both Kernels' indexes with an explicit
`incumbent` or `candidate` role; arbitrary Dev uses `workspace_snapshot` because source association
does not prove that the probe executed that Kernel.
Changing the Kernel, ABBA Baseline, input domain, or measurement parameters creates a different
task.

An Agent may request an exploratory same-allocation comparison between the current `kernel.py` and a
saved workspace source. Each Shape batch executes an alternating AB/BA schedule inside one allocation;
the Supervisor records exact raw evidence and returns a compact normalized comparison:

```bash
python tools/sandbox.py --kind run --mode full \
  --baseline-path scratch/baseline.py --comparison-repeats 2 --no-sync
```

Typed evaluation also accepts `--mode correctness_only`, `--input-path`, and `--shapes-path` through
the Agent facade. The two paths independently replace the input-generator source and Shape object in
that request. Custom evaluation metadata and roofline data are deliberately omitted because they
belong to the sealed evaluator inputs, not the Agent-authored exploratory workload.

## Security Boundary

This server is not a container or privilege boundary. Uploaded files and commands execute as the account
running the server, and job payloads and output remain in the state directory. The server rejects a
non-loopback bind unless `--allow-remote` is supplied, but that flag does not add authentication or
isolation. Use trusted inputs and keep the default loopback bind.
