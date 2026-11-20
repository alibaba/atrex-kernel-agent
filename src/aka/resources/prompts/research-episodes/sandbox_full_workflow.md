## GPU measurements

- Measure candidate correctness and performance:
  ```bash
  python3 tools/sandbox.py --kind evaluate --no-sync
  ```
  Reuse exact matching saved results when available. Runtime handles batching, repetitions, aggregation and retries. Use `submit-kernel` when you judge an exact commit ready to serve as a final implementation for the current mode; it must already have a passing standard full Evaluate with complete per-Shape performance. Submit when ready, not mechanically after every Evaluate.
- Use the enabled diagnostic tools only when they answer an unresolved optimization question; `tools/sandbox.py --help` lists this Campaign's available operations. Reuse saved Gateway records instead of repeating an identical completed measurement.
