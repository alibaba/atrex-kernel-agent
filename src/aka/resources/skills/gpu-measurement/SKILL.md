---
name: gpu-measurement
description: Run Kernel evaluation, ABBA comparison, profiling, and GPU probes through the Supervisor Runtime.
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
3. Choose the smallest suitable operation. Standard Evaluate is the authoritative correctness and
   performance path; subset Evaluate, ABBA, Profile, Check, Disassemble, Dev, and Command are
   diagnostic unless the report contract explicitly says otherwise.
4. Run the request in the foreground and wait for the same invocation. Do not add shell timeouts,
   background it, edit the candidate while it runs, or submit parallel duplicates.
5. For a measurement, inspect the complete canonical Runtime result. The identical durable result is
   saved at `records/<gateway_record_id>.json`; open it only when later work needs the persisted
   evidence. A completed job can still contain a correctness failure.

Dev and Command are fresh, stateless remote jobs. Use Dev only for a self-contained single-file
probe that reads no candidate or other workspace file, and declare that probe explicitly. Use
Command whenever custom code reads the candidate, helpers, prior artifacts, or other workspace
state. Neither creates a persistent GPU workspace. Typed operations select managed evaluator inputs
automatically, so use explicit command overrides only when the reference documents them.
Runtime options are operation-scoped. Never copy a flag from one `--kind` to another merely because
Agate or a previous Runtime call accepted it. In particular, `--intent` and `--note` are Dev-only
metadata when the current `--kind dev --help` advertises them; never pass them to Evaluate, Profile,
Check, Disassemble, Command, or Env. Read the selected operation's help before adding optional flags.
Env is read-only discovery. Its complete redacted capability response uses the same canonical
receipt and is archived in the central Store, without becoming correctness/performance evidence.

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
