# Chunk-Based Parallelism

## Overview

Linear attention variants (GatedDeltaNet, RetNet, Mamba) have O(n) complexity but naive implementations are sequential. Chunk-based parallelism divides the sequence into chunks of size C, computes within each chunk in parallel (matmul-friendly), and propagates state between chunks sequentially.

## Pattern

```python
@triton.jit
def chunk_parallel_linear_attn(Q, K, V, State, Output,
                                chunk_size: tl.constexpr,
                                d: tl.constexpr):
    # Grid: (num_chunks, num_heads, batch)
    chunk_id = tl.program_id(0)

    # Load chunk of Q, K, V
    q = tl.load(Q + chunk_id * chunk_size * d + offsets)  # [C, d]
    k = tl.load(K + chunk_id * chunk_size * d + offsets)
    v = tl.load(V + chunk_id * chunk_size * d + offsets)

    # Intra-chunk: parallel O(C^2) attention-like compute
    scores = tl.dot(q, tl.trans(k))
    o_intra = tl.dot(scores, v)

    # Inter-chunk: sequential state propagation
    state = tl.load(State)  # From previous chunk
    o_inter = tl.dot(q, state)

    # Combine and update state
    output = o_intra + o_inter
    state = update_state(state, k, v)

    tl.store(Output + offsets, output)
    tl.store(State, state)
```

## Size Tradeoff

- **Small chunks (C=32)**: low latency decode, fewer intermediate materializations
- **Large chunks (C=256-512)**: better tensor core utilization, higher throughput for prefill
- **TFLA (Tiled FLA)**: two-level tiling allows arbitrary chunk sizes via recursive tiling

## When To Use

- Linear attention (O(n) complexity)
- Recurrent state models (Mamba, GatedDeltaNet, Delta Rule)
- Hybrid architectures mixing linear + full attention (Qwen3-Next uses 3:1 ratio)
