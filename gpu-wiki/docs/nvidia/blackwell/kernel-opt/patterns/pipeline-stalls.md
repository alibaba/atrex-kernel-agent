# Pipeline Stalls

## Symptom

Nsight Compute shows TMA or tcgen05 units idle despite nominally compute-bound workload. Tensor core utilization drops during specific phases of the kernel. Warp-level profiling reveals threads blocked on `mbarrier.try_wait` more than expected.

## Likely Causes

1. **Insufficient pipeline depth**: 2 stages cannot hide a 3-cycle latency chain
2. **Incorrect mbarrier phase tracking**: Consumer observes stale arrivals, waits for next
3. **Missing `tcgen05.fence::after_thread_sync`**: MMA reads SMEM before TMA transfer fully visible
4. **Single-tile scheduling**: All warps serialized on one tile's softmax/epilogue
5. **Producer over-arrives**: Manual `mbarrier_arrive` after async TMA -- hardware + manual both arrive, next stage gets stale release

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Pipeline stages](../techniques/pipeline-stages.md) | Increase NUM_STAGES (3-5 typical on Blackwell) |
| [Warp specialization](../techniques/warp-specialization.md) | Dedicated warps for TMA/MMA/epilogue eliminate role-switching stalls |
| [Double-buffering](../techniques/double-buffering.md) | TMEM buffer A while MMA runs on buffer B |
| [Ping-pong scheduling](../techniques/ping-pong-scheduling.md) | Two query tiles alternate softmax/MMA (FA4 pattern) |

## Diagnosis Checklist

```
1. Profile with Nsight Compute, check tensor core active cycles
2. Inspect mbarrier wait stalls in warp state breakdown
3. Verify phase tracking increments correctly (each wait should flip parity)
4. Check that TMA uses arrive_expect_tx + mbarrier target (not manual arrive)
5. Ensure tcgen05.fence::after_thread_sync between TMA wait and MMA issue
6. Measure pipeline depth: can you add more NUM_STAGES?
```

## Example Progression (tcgen05 tutorial)

- 1-stage: 62% of cuBLAS (TMA blocks MMA)
- 3-stage pipelined: 70% (hide most TMA latency)
- Warp specialized: 80% (no role switching)
- Add 2-SM MMA: 86% (larger tile, more reuse)
- Persistent + CLC: 98% (eliminate tail effect)

## Caveats

- Too many stages consume SMEM; exceeds 228KB budget
- Phase tracking bugs are notoriously hard to debug -- add assertions in development
- Profile first -- pipeline is a waste of effort on memory-bound kernels
