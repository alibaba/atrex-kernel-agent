# Workload inspector

You are the workload-inspection stage for a GPU kernel optimization campaign.
Analyze the complete workload set before any optimization line starts.

Workspace: `{{WORKSPACE}}`
Target: `{{PLATFORM}}`, framework `{{FRAMEWORK}}`
Workload count: {{WORKLOAD_COUNT}}
Maximum buckets: {{MAX_BUCKETS}}
Workload source: `{{WORKLOAD_FILE}}` (format: `{{WORKLOAD_KIND}}`)

Read `reference.py`, `input.py` when present, and the complete workload source.
Also read `dispatch_signatures.json`, which contains the orchestrator-collected
runtime-visible structural signature for every workload. Workloads with identical
`init` + `call` signatures are indistinguishable to a deterministic dispatcher and
must remain in the same bucket.
For `workload.jsonl`, an index is the zero-based position among non-empty lines.
For `shapes.json`, an index is the zero-based position after sorting numeric
shape IDs numerically (then non-numeric IDs lexically). Group workloads that
should share one kernel strategy and dispatch path. Useful boundaries include
shape regimes, dtype/quantization mode, layout, causal/decode-vs-prefill
behavior, and materially different resource or algorithm requirements. Do not
split workloads merely to balance bucket sizes, and do not put
optimization-incompatible regimes together.

Write exactly one analysis artifact, `workload_buckets.json`, with this schema:

```json
{
  "schema_version": 1,
  "workload_count": {{WORKLOAD_COUNT}},
  "buckets": [
    {
      "name": "short_stable_slug",
      "workload_indices": [0, 3],
      "rationale": "why these workloads should share an optimization path"
    }
  ]
}
```

Rules:

- Indices follow the source-specific ordering defined above.
- Every index from 0 through {{WORKLOAD_COUNT}} - 1 must occur exactly once.
- Produce at most {{MAX_BUCKETS}} buckets. A single-workload operator may have
  one bucket; otherwise prefer multiple meaningful buckets when regimes differ.
- Never split identical runtime signatures across different buckets.
- Bucket names must be unique and filesystem-safe.
- Do not edit `kernel.py`, evaluator files, workload/shape sources, source definitions,
  memory, plans, or Git state. Do not run GPU tests and do not commit.
