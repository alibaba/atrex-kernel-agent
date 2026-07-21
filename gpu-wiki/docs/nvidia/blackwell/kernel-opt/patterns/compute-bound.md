# Not Reaching Peak FLOPS

## Symptom

Tensor core utilization below 70%. Memory bandwidth is not saturated. Kernel is compute-bound but not reaching peak FLOPS.

## Likely Causes

1. **Pipeline bubbles**: MMA stalled waiting for data from TMA
2. **Non-matmul overhead**: Softmax, activation functions, reductions consuming cycles
3. **Single-SM MMA tiles too small**: Not fully utilizing available compute
4. **Epilogue blocking mainloop**: TMEM reads blocking next MMA

## Candidate Techniques

| Technique | Effect |
|---|---|
| [2-SM cooperative](../hardware/2sm-cooperative.md) | Double effective MMA tile (m256×n256), 2× compute per cycle |
| [Pipeline stages](../techniques/pipeline-stages.md) | Overlap TMA load with MMA compute |
| [Warp specialization](../techniques/warp-specialization.md) | Dedicated warps for TMA/MMA/epilogue, no stalls |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Overlap epilogue with next tile's MMA |
| [Software exponential](../techniques/software-exp.md) | Distribute non-matmul ops across FMA units (FA4) |

## Example: FlashAttention-4

```
// Problem: Blackwell doubles tensor core throughput but SFU count unchanged
// SFU bottleneck: exp() for softmax
//
// Solution: Software 2^x via Cody-Waite + Horner polynomial
// Distributes across FMA units, multiplying exponential throughput
// Result: 1605 TFLOPS (71% utilization) on B200
```

## Caveats
- 2-SM cooperative requires cluster configuration and identical SMEM layouts
- Pipeline depth tuning is workload-dependent (3-5 stages typical)
- Software-emulated transcendentals trade accuracy for throughput
