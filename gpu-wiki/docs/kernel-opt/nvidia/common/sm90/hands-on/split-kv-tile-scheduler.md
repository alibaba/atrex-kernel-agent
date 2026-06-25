# Split-KV Decode and Tile Scheduler

In long-sequence MLA Decode, the KV dimension is split across multiple SMs for parallel computation. A custom tile scheduler is used Kramer for load balancing, and a combine kernel merges the results.

**Source**: FlashMLA `csrc/smxx/decode/get_decoding_sched_meta/`, `csrc/smxx/decode/combine/`

---

## Background

MLA Decode is memory-bound (each query token has only 1 row, but needs to traverse all KVs). When sequence length is large, a single CTA cannot efficiently handle the workload, so the KV must be split across multiple SMs.

## Split-KV Approach

```
Query [1 × head_dim]
KV Cache [seq_len × head_dim]

Split into chunks:
SM_0: KV[0 : chunk_size]       → partial_O₀, partial_lse₀
SM_1: KV[chunk_size : 2*chunk] → partial_O₁, partial_lse₁
SM_2: KV[2*chunk : 3*chunk]    → partial_O₂, partial_lse₂
...

Combine kernel: merge partial results → final O
```

### Tile Scheduler

FlashMLA uses a separate kernel to pre-generate scheduling metadata (`DecodingSchedMeta`) and assign work to each SM:

```cpp
struct DecodingSchedMeta {
    int request_idx;    // Which request
    int split_idx;      // KV split index
    int num_splits;     // Total splits for this request
    int kv_start;       // KV start position
    int kv_end;         // KV end position
};
```

Scheduling strategy:
- Short sequences (< chunk_size): handled by a single SM without splitting
- Long sequences: split by chunk_size and assigned to multiple SMs
- Cross-request load balancing: splits from different requests are interleaved and assigned to SMs

### Combine Kernel

```cpp
// Merge multiple partial attention results
// Use log-sum-exp weighted merge (same principle as FlashInfer cascade merge)
for (int split = 0; split < num_splits; split++) {
    float lse_i = partial_lse[split];
    float lse_max = max(lse_max, lse_i);
}
for (int split = 0; split < num_splits; split++) {
    float weight = exp(partial_lse[split] - lse_max);
    output += weight * partial_output[split];
    total_weight += weight;
}
output /= total_weight;
```

### Programmatic Dependent Launch (PDL)

FlashMLA uses PDL to seamlessly chain the splitkv kernel and combine kernel without returning to the host:

```cpp
// splitkv kernel
cudaTriggerProgrammaticLaunchCompletion();

// combine kernel launch
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attrs[0].val.programmaticStreamSerializationAllowed = 1;
```

## Practical Experience

- chunk_size is typically 256–1024 tokens; too small increases combine overhead, too large wastes SMs
- The kernel that pre-generates scheduling metadata is very lightweight (~a few microseconds) but can significantly improve load balancing
- PDL reduces kernel launch gaps (especially important for decode scenarios, since the kernels themselves are very short)
- The combine kernel is memory-bound, and its overhead is usually negligible

## Related Documentation

- **Persistent Kernel**: [Persistent Kernel and Tile Scheduler](persistent-kernel-tile-scheduler.md) — static tile scheduling
- **Cascade Merge**: [Cascade / State Merge](../../../../generic/hands-on/cascade-state-merge.md) — Triton version of split-KV merge
- **Seesaw Scheduling**: [Seesaw Warpgroup Scheduling](seesaw-warpgroup-scheduling.md) — FlashMLA's compute core
