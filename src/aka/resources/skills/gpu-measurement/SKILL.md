---
name: gpu-measurement
description: Discover and use the Campaign-enabled measurement tools through the Supervisor Runtime.
---

# GPU measurement

Use `python3 tools/sandbox.py`; the Supervisor owns hardware, credentials, evaluator inputs,
timeouts, retries, and result persistence. Local development and import/API inspection are allowed;
use Runtime for target-GPU work and official measurements. Do not invoke Agate directly. Initial bring-up uses these same Runtime operations before
the Supervisor performs independent submission review.

## Measurement loop

1. State the uncertainty and the observation that would resolve it.
2. Commit the complete candidate and leave Git clean before a Kernel-bound request. The full HEAD
   becomes `kernel_commit_id`.
3. Read `python3 tools/sandbox.py --help` and choose an enabled operation. Standard full Evaluate
   is authoritative submission evidence. Subset results, comparisons and custom probes are diagnostic.
4. Start the request and collect the same invocation's receipt before using its results. With Agent
   suspension disabled, independent work may continue; use the backend's task/output facility for
   background calls. Keep in-flight request inputs unchanged. Do not add shell timeouts or duplicates.
5. For a measurement, inspect the complete canonical Runtime result. The identical durable result is
   saved at `records/<gateway_record_id>.json`; open it only when later work needs the persisted
   evidence. A completed job can still contain a correctness failure.
6. If Experiment recording is enabled and you use the result to choose the next implementation or
   probe, call `record-experiment` with the observation, measured scope, and next action before
   proceeding. Later evidence can revise it; do not wait for another probe to record this decision.

Use the enabled tools listed by Help and their examples in [Runtime requests](references/requests.md).
Read `--kind OPERATION --help` for exact options; do not transfer flags between operations.
Remote jobs are stateless; only returned artifacts persist.

For Atrex-Bench timing questions, read the workspace's `atrex-bench/scripts/run_eval.py`
and follow its imports into `atrex-bench/src/atrex_bench/` (including `eval/performance.py`
when present). This read-only directory contains the same Atrex-Bench source used by remote
evaluation. Run measurements through Runtime; use its help for effective timing options and
Gateway Records for measurement evidence.

## Results and failures

- Gateway Records are immutable measurement evidence. Reuse exact matches by reading their
  `records/<gateway_record_id>.json` projections directly. A Git commit without a passing Record is
  not performance or correctness evidence.
- Runtime deduplicates only Evaluate, including ABBA: an identical call waits for the active
  evaluation or reuses its normal terminal response, including candidate failures or missing
  metrics. `reused_gateway_record_id` identifies the reused Record; reuse does not mean correctness
  passed. Infrastructure/Gateway failures do not prevent a later retry. Other GPU tools execute
  afresh on every call.
- Follow `error.next_action` and preserved upstream diagnostics. Candidate compile, import,
  correctness, or source-policy errors require repair; they are not infrastructure failures.
- Runtime owns infrastructure retries. Do not build another retry loop. Retry an unchanged request
  only when Runtime explicitly marks it repairable and supplies the required delay. Reconcile an
  unknown outcome before resubmitting.
- Treat aggregate SOL and profiler classifications as clues, not causal proof. Use the returned
  counters and per-Shape evidence to test the hypothesis.

See [request syntax, operation boundaries, result fields, and recovery](references/requests.md), or
run `python3 tools/sandbox.py --kind OPERATION --help`.
