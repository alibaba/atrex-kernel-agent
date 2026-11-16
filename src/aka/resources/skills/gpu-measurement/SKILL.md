---
name: gpu-measurement
description: Discover and use the Campaign-enabled measurement tools through the Supervisor Runtime.
---

# GPU measurement

Use `python3 tools/sandbox.py`; the Supervisor owns hardware, credentials, evaluator inputs,
timeouts, retries, and result persistence. The local workspace is CPU-only: do not run or import the
Kernel locally or invoke Agate directly. Initial bring-up uses these same Runtime operations before
the Supervisor performs independent submission review.

## Measurement loop

1. State the uncertainty and the observation that would resolve it.
2. Commit the complete candidate and leave Git clean before a Kernel-bound request. The full HEAD
   becomes `kernel_commit_id`.
3. Read `python3 tools/sandbox.py --help` and choose an enabled operation. Standard full Evaluate
   is authoritative submission evidence. Subset results, comparisons and custom probes are diagnostic.
4. Run the request in the foreground and wait for the same invocation. Do not add shell timeouts,
   background it, edit the candidate while it runs, or submit parallel duplicates.
5. For a measurement, inspect the complete canonical Runtime result. The identical durable result is
   saved at `records/<gateway_record_id>.json`; open it only when later work needs the persisted
   evidence. A completed job can still contain a correctness failure.

Use an enabled Command for custom remote probes, compilation and profiling. Python scripts below
scratch/ should use module form, for example `--kind command -- python3 -m scratch.probe`.
Jobs are stateless; returned scratch artifacts persist. Typed operations select their managed inputs.
Read `--kind OPERATION --help` for exact options; do not transfer flags between operations.
Env returns the complete redacted capability observation, archived without becoming correctness evidence.

## Results and failures

- Gateway Records are immutable measurement evidence. Reuse exact matches by reading their
  `records/<gateway_record_id>.json` projections directly. A Git commit without a passing Record is
  not performance or correctness evidence.
- Follow `error.next_action` and preserved upstream diagnostics. Candidate compile, import,
  correctness, or source-policy errors require repair; they are not infrastructure failures.
- Runtime owns infrastructure retries. Do not build another retry loop. Retry an unchanged request
  only when Runtime explicitly marks it repairable and supplies the required delay. Reconcile an
  unknown outcome before resubmitting.
- Treat aggregate SOL and profiler classifications as clues, not causal proof. Use the returned
  counters and per-Shape evidence to test the hypothesis.

See [request syntax, operation boundaries, result fields, and recovery](references/requests.md), or
run `python3 tools/sandbox.py --kind OPERATION --help`.
