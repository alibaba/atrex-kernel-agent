# Workload inspector

You are the workload-inspection stage for a GPU kernel optimization campaign.
Analyze the complete workload set before any optimization line starts.

Workspace: `{{WORKSPACE}}`
Target: `{{PLATFORM}}`, framework `{{FRAMEWORK}}`
Workload count: {{WORKLOAD_COUNT}}
Maximum buckets: {{MAX_BUCKETS}}
Workload source: `{{WORKLOAD_FILE}}` (format: `{{WORKLOAD_KIND}}`)

This is a deliberately data-minimized inspection workspace. Read only
`dispatch_signatures.json`. It contains the complete information available to the
production dispatcher under visibility policy `{{VISIBILITY_POLICY}}`:

- explicit non-tensor `init` and `call` arguments;
- tensor shape, stride, dtype, layout, and `requires_grad`;
- argument positions and keyword names.

The original `reference.py`, `input.py`, `{{WORKLOAD_FILE}}`, tensor contents, and
evaluator metadata are intentionally absent. Do not search outside this workspace or
attempt to reconstruct/read them. In particular, request-length arrays, offsets,
`query_start_loc`, `seq_lens`, tensor statistics/values, pointers, `.item()`, and
`.tolist()` are not legal bucketing evidence. A production dispatcher cannot obtain
them without a device synchronization or extra undocumented input.

Workloads with identical `init` + `call` signatures are indistinguishable to the
dispatcher and must remain in the same bucket. Every bucket boundary and rationale
must be expressible solely as a predicate over fields present in these sanitized
signatures.
For `workload.jsonl`, an index is the zero-based position among non-empty lines.
For `shapes.json`, an index is the zero-based position after sorting numeric
shape IDs numerically (then non-numeric IDs lexically). Group workloads that
should share one kernel strategy and dispatch path. Useful boundaries include
production-visible tensor shape regimes, dtype, layout, explicit scalar modes,
and materially different resource requirements inferable from those fields.
Do not infer semantic regimes such as decode/prefill from unavailable tensor
contents. Do not split workloads merely to balance bucket sizes, and do not put
optimization-incompatible visible regimes together.

Write exactly one analysis artifact, `workload_buckets.json`, with this schema:

```json
{
  "schema_version": 1,
  "dispatch_visibility_policy": "{{VISIBILITY_POLICY}}",
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
- Use no information other than `dispatch_signatures.json`; each rationale must cite
  concrete production-visible signature fields.
- Bucket names must be unique and filesystem-safe.
- Do not inspect parent/absolute paths, run commands that discover outside files, edit
  the signature file, run GPU tests, or commit. Write only `workload_buckets.json`.
