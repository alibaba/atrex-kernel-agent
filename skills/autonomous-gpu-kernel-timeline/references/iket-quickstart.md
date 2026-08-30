# CuTe DSL / IKeT route

Use this route only for CuTe DSL. Do not substitute the CUDA header. First confirm `run-iket --help`
works and the target is SM90, SM100, SM103, SM110, or SM120. The instrumented kernel must JIT-compile
inside the profiled process; a cached binary compiled before `run-iket` will not acquire events.

## Choose the smallest useful API

All calls belong inside `@cute.kernel`, never only in the host-side `@cute.jit` wrapper.

```python
@cute.kernel
def kernel(...):
    cute.experimental.iket.mark("kernel_start")
    token = cute.experimental.iket.range_start("load")
    # measured work
    cute.experimental.iket.range_end(token)

    cute.experimental.iket.range_push("compute")
    # naturally nested work
    cute.experimental.iket.range_pop()
```

- `mark(name, payload?)` records a point.
- `range_push(name, payload?)` / `range_pop()` is simplest for visibly nested control flow.
- `range_start(name, payload?)` / `range_end(token, payload?)` is appropriate when the closing
  boundary is elsewhere. Start/end payload presence and type must match.
- `sentinel_token(name)` initializes a loop-carried token when the first close precedes the first
  real start. It emits no event by itself.

Keep paired endpoints in the same warp role and compatible control flow. For warp-specialized code,
put both endpoints inside the role guard:

```python
if warp_idx == producer_warp:
    cute.experimental.iket.range_push("tma_issue")
    # issue TMA work
    cute.experimental.iket.range_pop()

if warp_idx == consumer_warp:
    cute.experimental.iket.range_push("mma_wait")
    # wait that establishes completion
    cute.experimental.iket.range_pop()
```

An event after an asynchronous instruction measures issue-side progress, not completion. Use an
explicit wait boundary when completion is the question. Prefer warp-uniform payloads; IKeT records
the first active lane's value. Keep names at most 32 characters and normally fewer than 30 unique
names. Avoid events in the innermost unrolled loop and do not place IKeT ranges inside
`cutlass.range(..., prefetch_stages=...)`.

## Capture and prove activation

Use fresh, attempt-owned directories. `profile-iket` captures and normalizes before the worker exits,
so native `.pftrace` does not have to cross the sandbox boundary. The wrapper refuses to overwrite an
existing run directory:

```bash
python skills/autonomous-gpu-kernel-timeline/scripts/timeline.py profile-iket \
  --run-dir profiles/episode_N/timeline/attempt-N/iket-run \
  --evidence-dir profiles/episode_N/timeline/attempt-N/evidence \
  --kernel-regex '^exact_generated_kernel_name$' \
  --dictionary profiles/episode_N/timeline/attempt-N/events.json \
  --clean-source profiles/episode_N/timeline/attempt-N/clean_kernel.py \
  --instrumented-source profiles/episode_N/timeline/attempt-N/instrumented_kernel.py \
  --workload-identity '<shape,dtype,layout>' --correctness passed -- \
  python profiles/episode_N/timeline/attempt-N/harness/profile_target.py
```

For a remote campaign command, pass the skill as an explicit sandbox input and sync only the attempt:

```bash
python tools/sandbox.py --kind profile \
  --input skills/autonomous-gpu-kernel-timeline \
  --sync profiles/episode_N/timeline/attempt-N -- \
  python skills/autonomous-gpu-kernel-timeline/scripts/timeline.py profile-iket \
    --run-dir profiles/episode_N/timeline/attempt-N/iket-run \
    --evidence-dir profiles/episode_N/timeline/attempt-N/evidence \
    --kernel-regex '^exact_generated_kernel_name$' \
    --dictionary profiles/episode_N/timeline/attempt-N/events.json \
    --clean-source profiles/episode_N/timeline/attempt-N/clean_kernel.py \
    --instrumented-source profiles/episode_N/timeline/attempt-N/instrumented_kernel.py \
    --workload-identity '<shape,dtype,layout>' --correctness passed -- \
    python profiles/episode_N/timeline/attempt-N/harness/profile_target.py
```

Use `capture-iket` and `export-iket` separately only for local debugging. Export succeeds only when a
non-empty native Perfetto trace exists, the target launch is present, all declared events are observed
with the declared kind, locations are valid, and range timestamps are ordered. It emits the common
deterministic Perfetto JSON gzip, summary, manifest, binary identity, native-capture index, and receipt.
The commands above create exploration evidence. Before requesting `--stage final`, run the immutable
AKA evaluator against the instrumented snapshot and pass its sandbox-owned
`.atrex_long_horizon/evaluations.jsonl` to `export-iket` as `--correctness-evidence`; the text
`--correctness passed` alone cannot produce `decision_grade`.

Do not run IKeT in the same process as NSYS or NCU. `--enabled-cluster X,Y,Z` can reduce output for a
large grid, but that is sampled coverage and must not be described as all-CTA evidence. IKeT time is
quantized to 32 ns, so a zero-duration very short range can be valid.

For overhead, compare the minimal one-site IKeT build with the final event set under `run-iket`, and
separately compare an unprofiled baseline with a build enabled by `CUTE_DSL_COMPILER_OPT=iket` or
`cute.compile(..., options="iket")`. Correctness is still required.
Ten percent is the default target, not a universal hard failure; a higher-overhead trace remains
diagnostic and cannot justify performance differences smaller than its disturbance.

The runnable backend check is `backends/cutedsl_backend/test_iket.py`. Authoritative background:

- NVIDIA CUTLASS `media/docs/pythonDSL/guides/iket_profiling.rst`
- NVIDIA CUTLASS `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`
- Dao-AILab QuACK minimal IKeT example
