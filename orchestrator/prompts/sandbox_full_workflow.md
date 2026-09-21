## GPU measurements

- Use `python3 tools/sandbox.py --kind run --mode full --no-sync` for standard correctness and
  performance. Follow any explicit seed/coverage requirements in this session. Runtime handles
  batching, repetitions, aggregation, and infrastructure retries.
- Use Profile or custom Dev probes only when they answer an unresolved optimization question.
  Reuse saved Gateway records instead of repeating an identical completed measurement.
