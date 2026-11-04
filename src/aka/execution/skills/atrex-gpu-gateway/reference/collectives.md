# Single-node multi-GPU collectives

Both client and server must be version 0.13.6 or newer. `eval`, `profile` and `dev` accept
`--num-gpus N`, default `1`, range `1`–`64`. This declares the number of GPUs allocated on one node;
it does not replicate ordinary eval or dev commands N times. Ordinary atrex/SOL harness semantics are unchanged.
Only Ray environments with atomic allocation support N greater than 1; fixed-pod and local backends explicitly reject it.
Select an alias through `agate env` and ask an administrator to confirm per-node GPU count and image dependencies.

In `eval/profile --collective`, `--collective` explicitly switches the job contract: the CLI submits
`kind=collective` to `/v1/jobs/collective`. Candidate, parameter and result schemas differ from ordinary atrex/SOL evaluation.
The eval/profile verbs retain a common entry for correctness/timing and separate nsys sampling.
This is not an extra operator option for ordinary eval and cannot be mixed with harness options such as `--backend` or `--reference-dir`.

## Minimal run

Run from this Skill directory. Replace `GPU` with an environment alias offering multiple GPUs.
The examples ship with the Skill and require no repository access.

```bash
export GPU=REPLACE_MULTI_GPU_ALIAS
agate eval --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --numel 1048576 --numel 16777216
agate eval --gpu "$GPU" --num-gpus 4 --collective allgather \
  --candidate scripts/collectives/allgather.py --numel 1048576
agate eval --gpu "$GPU" --num-gpus 4 --collective reduce-scatter \
  --candidate scripts/collectives/reduce_scatter.py --numel 1048576
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --numel 1048576
```

These three candidates call NCCL directly to validate the contract and environment; **they are not optimized implementations and promise no speedup**.
Copy a candidate and replace `Model.forward(tensor, group)` with your communication implementation to compare against NCCL under the same configuration.
Do not pass `--reference-dir` or `--backend`. Defaults are float32, 5 warmup iterations and 20 timed iterations.
Adjust with `--dtype float16|bfloat16`, `--warmup-iters`, `--bench-iters` and `--job-timeout`.
Three distinct random inputs are checked on every rank. A shape failing correctness has no performance result, and the CLI exits 1.
Default tolerances are `atol=rtol=0.01`; set different thresholds through HTTP request `options`.

## Candidate contract

The candidate is one Python file defining `class Model(torch.nn.Module)` without constructor arguments.
The runner launches one rank per GPU and owns NCCL process-group initialization, device selection, process startup and cleanup.
The candidate accepts a one-dimensional contiguous CUDA tensor and initialized `group`, returning a tensor on the same device with the same dtype.
It may modify the input in place. Do not initialize/destroy the runner's process group or set GPU visibility yourself.
Custom libraries may be installed through `-r requirements.txt`; source validation still applies.

`--numel` is the element count of the complete logical buffer. Repeat it for up to 32 distinct shapes:

| Operator | Input per rank | Output per rank | Semantics |
|---|---|---|---|
| `allreduce` | numel | numel | SUM |
| `allgather` | numel / N | numel | Concatenate in rank order |
| `reduce-scatter` | numel | numel / N | SUM, then contiguous partitions by rank |

The last two require numel divisible by N. N=1 performs no inter-GPU communication and is suitable only for smoke tests, not communication-benefit claims.

## Reading results and comparing GPU counts

`result.measurements` returns `passed`, `baseline`, `candidate` and `speedup` per shape.
Each iteration synchronizes before timing; timing covers the communication call and completion synchronization, using the slowest rank.
`latency_us` is the median across iterations; `min_us`, `p95_us` and `samples` are also available.
This is synchronized host-side end-to-end latency including Python/allocation overhead, not pure kernel time or nccl-tests CUDA-event timing.

`algorithmic_gbps` and `bus_gbps` both use decimal **GB/s**. Algorithmic bandwidth is complete-buffer bytes divided by latency.
The bus correction factor is `2*(N-1)/N` for allreduce and `(N-1)/N` for the other two operations,
following [NCCL tests bandwidth conventions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
`speedup` is **NCCL latency / candidate latency at the same GPU count and shape**, not N-GPU speedup over one GPU.

Keep dtype, logical numel, candidate and image version fixed; measure 2/4/8 GPUs and retain each job_id:

```bash
for n in 2 4 8; do
  agate eval --gpu "$GPU" --num-gpus "$n" --collective allreduce \
    --candidate scripts/collectives/allreduce.py --numel 16777216 \
    --dtype float32 --warmup-iters 10 --bench-iters 50 > "allreduce-${n}gpu.json"
done
```

Check correctness for every shape before comparing latency and bus bandwidth. More GPUs do not guarantee faster collective communication.
Repeat measurements and record the environment, topology and competing load. Offline tests are not GPU performance evidence; run these commands in the target environment.

## Profiling and raw reports

Collective profiling first runs correctness and timing without a profiler, then profiles separately, using Nsight Systems by default.
`profiled_timings=false` means performance numbers come from the earlier unprofiled run.
The nsys `result.profile.kernels` table aggregates all ranks, baseline and candidate, including validation/warmup;
do not use its total time directly to compute candidate gains. In the raw timeline, measured sampling uses
`collective/<operator>/baseline` and `collective/<operator>/candidate` NVTX ranges, tracing at most 5 iterations per shape.

If the environment has OSS configured, retain the full timeline:

```bash
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --oss-output collective.nsys-rep
```

NCU uses a single-node shmem communicator. The worker must have **NCU 2026.1 or newer**,
support `--communicator shmem` and `--communicator-shmem-num-peers`, and provide compatible drivers,
profiling permissions and shared memory accessible by all ranks. The actual NCU executable is checked before starting.
An older version returns `collective_ncu_shmem_unsupported`; a missing executable returns `collective_ncu_unavailable`.

```bash
# The bundled example calls NCCL; replace nccl with your own communication kernel name.
agate profile --gpu "$GPU" --num-gpus 4 --collective allreduce \
  --candidate scripts/collectives/allreduce.py --profiler ncu --level deep \
  --kernel-regex 'nccl.*' --oss-output collective.ncu-rep
```

`--profiler ncu` defaults to `--level sol` with basic metrics; deep collects full metrics.
Specify exactly one of `--kernel-name` and `--kernel-regex`. The selected communication kernels must be invoked
by **every rank in the same order and count**. The first version does not support asymmetric rank launch sequences.
Sampling is enabled only for candidate execution; correctness, warmup, baseline, input generation and inter-rank barriers are excluded.
Performance comparisons still use the separate unprofiled run.

Repeatable `--counter` overrides basic/full metric sets. `--launch-skip` defaults to 0, maximum 10000;
`--launch-count` defaults to 1, maximum 100; `--top-kernels` defaults to 30, maximum 100.
Launch counts apply to matching kernels, not candidate calls or shapes. Pass a single `--numel` to target one shape.
By default shmem samples the first CUDA context in each group while other ranks cooperate;
**it does not guarantee rank 0 or metrics from every GPU**.
`result.profile.kernels[].identity` retains report process_id, device, context, stream and launch_id.
Same-named kernels are not merged; unavailable identities are null. Download raw `.ncu-rep` files as OSS output attachments.
Each shape executes `min(bench_iters, launch_skip + launch_count)` traced candidate calls, never exceeding `--bench-iters` (maximum 1000).
A candidate call may match multiple or zero kernels, so collecting all `--launch-count` samples is not guaranteed.
For empty samples, inspect selectors, skip/count and actual launch sequences.
Upgrade the gateway to at least 0.13.17 before upgrading the client; older gateways may ignore the new options and run nsys.
`eval --collective` rejects profiling options; use `profile --collective`. `--source` is not yet available for collectives.

Ordinary multi-GPU profiling also supports only `--level survey` (NVIDIA nsys); multi-GPU clock locking is not supported yet.
Missing torch/NCCL/nsys, incorrect visible device count, timeouts and profiling failures produce explicit errors, never fabricated success reports.
