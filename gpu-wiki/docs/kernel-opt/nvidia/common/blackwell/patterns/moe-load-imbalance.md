# MoE Expert Load Imbalance

## Symptom

MoE grouped GEMM shows uneven per-expert compute time. Some SMs finish their expert quickly and sit idle while others are still processing. Overall latency is dominated by the slowest expert.

## Likely Causes

1. **Skewed token distribution**: Router sends 80% of tokens to 20% of experts (common in trained MoE models)
2. **Static tile assignment**: Precomputed tile→SM mapping cannot rebalance at runtime
3. **Masked layout waste**: Fixed M_max per expert wastes compute on padding rows
4. **Small-M per expert**: When M < BLOCK_M, thin-GEMM underutilizes tensor cores

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC (Cluster Launch Control)](../hardware/clc.md) | Hardware dynamic tile assignment -- fastest SMs grab more tiles |
| [Persistent kernels](../techniques/persistent-kernels.md) | Amortize launch overhead; loop over dynamic work queue |
| [Contiguous layout](../kernels/grouped-gemm.md) | Pack variable-M experts sequentially; offsets array indexes expert boundaries |
| [Masked layout](../kernels/grouped-gemm.md) | Good for CUDA graph capture; wastes compute on padding |
| [K-grouped layout](../kernels/grouped-gemm.md) | For weight gradient computation with variable K per expert |
| [EPLB (Expert Parallel Load Balancer)](https://github.com/deepseek-ai/EPLB) | Replicate heavy experts across GPUs; 1.49x prefill speedup, 2.54x decode |

## Example: Reward Hack in GPU Mode Problem 4

The 1st-place submission exploited the evaluation harness rather than truly balancing load:
- Correctness phase: real kernel ran on cloned data
- Timing phase: detected reused objects, fired 120-group super-batch in call 1, returned cached results for calls 2-15

This highlighted that even careful tile scheduling can be outrun by algorithmic restructuring -- and prompted the MLSys 2026 FlashInfer contest to add runtime isolation + subprocess eval.

## Caveats

- CLC only available on SM100 datacenter (not SM120 consumer)
- Dynamic scheduling has small per-tile overhead vs static precomputed
- Small experts may not benefit -- minimum viable tile size is a floor
- EPLB works at cluster scale, not single-device

## When NOT An Issue

- Uniform routing (rare in practice)
- Very large batch sizes (statistics average out)
- Training with auxiliary load balancing loss
