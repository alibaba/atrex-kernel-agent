## GPU measurements

- Measure candidate correctness and performance:
  ```bash
  python3 tools/sandbox.py --kind run --no-sync
  ```
  Reuse exact matching saved results when available. On `candidate_ready`, the Supervisor
  automatically applies the evaluator-specific acceptance check: six-case correctness for
  Atrex-Bench, or the official full workload for SOL-ExecBench. You do not need to request this
  check manually. If it fails, inspect the returned Record, repair the candidate and its evidence,
  then resubmit. Runtime handles batching, repetitions, aggregation and retries.
- Use Profile or custom Dev probes only when they answer an unresolved optimization question.
  Reuse saved Gateway records instead of repeating an identical completed measurement.
