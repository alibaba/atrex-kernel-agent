# Gated Delta Net -- Linear Attention

## Overview

Gated Delta Networks (GatedDeltaNet) replace standard O(n^2) attention with an O(n) linear attention mechanism that uses a delta rule for error-correcting memory updates and exponential gating for adaptive decay. Published at ICLR 2025 by NVlabs, GatedDeltaNet is deployed in production in Qwen3-Next-80B (3:1 hybrid ratio with full attention) and Qwen3.5 (262K context window).

The key advantage is O(1) per-token cost during decoding: the recurrent state is a fixed-size matrix that gets updated with each new token, eliminating the KV cache growth problem entirely.

## Architecture in Qwen3-Next

```
Qwen3-Next-80B: 48 layers
Pattern: 12 x (3 x [GatedDeltaNet -> MoE] -> [Full Attention -> MoE])

Layer distribution:
  - 36 GatedDeltaNet layers (75%): O(n) linear attention
  - 12 Full Attention layers  (25%): Standard GQA for global retrieval
  - All layers followed by MoE: 512 experts, ~19 active per token

Total: 80B parameters, only 3B active per token
```

## Delta Rule Mechanism

Unlike standard linear attention (which uses simple additive updates to the recurrent state), the delta rule performs targeted error-correcting updates:

```python
# Standard linear attention (additive):
#   S_t = S_{t-1} + v_t @ k_t^T
#
# Delta rule (error-correcting):
#   S_t = S_{t-1} + (v_t - S_{t-1} @ k_t) @ k_t^T
#                    ^^^^^^^^^^^^^^^^^^^^^^^^
#                    Error correction term

def delta_rule_step(S, k, v, beta, alpha):
    """
    Single step of gated delta rule update.

    S: recurrent state [d_k, d_v] (the "memory matrix")
    k: key vector [d_k]
    v: value vector [d_v]
    beta: exponential gate (learned, controls decay)
    alpha: delta gate (learned, controls update strength)
    """
    # Retrieve what the current state "thinks" about this key
    v_retrieved = S @ k  # [d_v]

    # Error: difference between true value and retrieved value
    delta = v - v_retrieved  # [d_v]

    # Gated update: decay old state, add error-corrected new info
    S_new = beta * S + alpha * (delta[:, None] @ k[None, :])
    #       ^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       decay         error-correcting update

    return S_new
```

## Chunk-Based Parallel Prefill

During prefill, sequences are divided into chunks that can be processed in parallel. Within each chunk, the inter-token dependencies are resolved via a causal linear recurrence; across chunks, the recurrent state is propagated sequentially.

```python
import triton
import triton.language as tl

@triton.jit
def gated_delta_net_chunk_fwd(
    Q_ptr, K_ptr, V_ptr, Beta_ptr, O_ptr, State_ptr,
    SEQ_LEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,    # e.g., 64 or 128
    D_QK: tl.constexpr,         # qk_dim * d = 4 * 128 = 512
    D_V: tl.constexpr,          # v_dim * d = 8 * 128 = 1024
):
    """
    Chunk-parallel forward pass for GatedDeltaNet.
    Two levels of parallelism:
      1. Across chunks (parallel after inter-chunk state propagation)
      2. Within chunks (parallel via matrix operations)
    """
    chunk_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    chunk_start = chunk_id * CHUNK_SIZE

    # Load recurrent state from previous chunk
    # S shape: [D_QK, D_V]
    S = tl.load(State_ptr + (batch_id * NUM_HEADS + head_id) * D_QK * D_V
                + tl.arange(0, D_QK)[:, None] * D_V
                + tl.arange(0, D_V)[None, :])

    # Intra-chunk computation
    for t in range(CHUNK_SIZE):
        pos = chunk_start + t

        # Load q, k, v, beta for this token
        q = tl.load(Q_ptr + pos * D_QK + tl.arange(0, D_QK))
        k = tl.load(K_ptr + pos * D_QK + tl.arange(0, D_QK))
        v = tl.load(V_ptr + pos * D_V + tl.arange(0, D_V))
        beta = tl.load(Beta_ptr + pos)

        # Output: query the state
        o = tl.sum(S * q[:, None], axis=0)  # [D_V]

        # Delta rule update
        v_retrieved = tl.sum(S * k[:, None], axis=0)
        delta = v - v_retrieved

        # Gated state update
        S = beta * S + delta[:, None] * k[None, :]  # [D_QK, D_V]

        # Store output
        tl.store(O_ptr + pos * D_V + tl.arange(0, D_V), o)

    # Store final state for next chunk
    tl.store(State_ptr + (batch_id * NUM_HEADS + head_id) * D_QK * D_V
             + tl.arange(0, D_QK)[:, None] * D_V
             + tl.arange(0, D_V)[None, :], S)
```

## Streaming Decode Kernel

During autoregressive decoding, each new token only requires one state update -- O(1) per token regardless of context length.

```python
@triton.jit
def gated_delta_net_decode(
    Q_ptr, K_ptr, V_ptr, Beta_ptr, O_ptr, State_ptr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
):
    """
    Single-token decode: O(1) per token.
    The recurrent state replaces the KV cache entirely.
    State size: D_QK * D_V (e.g., 512 * 1024 = 512K floats per head)
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load persistent recurrent state
    state_offset = (batch_id * NUM_HEADS + head_id) * D_QK * D_V
    S = tl.load(State_ptr + state_offset
                + tl.arange(0, D_QK)[:, None] * D_V
                + tl.arange(0, D_V)[None, :])

    # Load new token
    q = tl.load(Q_ptr + tl.arange(0, D_QK))
    k = tl.load(K_ptr + tl.arange(0, D_QK))
    v = tl.load(V_ptr + tl.arange(0, D_V))
    beta = tl.load(Beta_ptr)

    # Query state for output
    o = tl.sum(S * q[:, None], axis=0)

    # Update state with delta rule
    v_retrieved = tl.sum(S * k[:, None], axis=0)
    delta = v - v_retrieved
    S = beta * S + delta[:, None] * k[None, :]

    # Store
    tl.store(O_ptr + tl.arange(0, D_V), o)
    tl.store(State_ptr + state_offset
             + tl.arange(0, D_QK)[:, None] * D_V
             + tl.arange(0, D_V)[None, :], S)
```

## TFLA: Tiled Flash Linear Attention

TFLA adds a second level of tiling within chunks, enabling arbitrarily large chunk sizes. It emits matmuls as inline PTX assembly for both Hopper (WGMMA) and Blackwell (tcgen05).

```cpp
// TFLA: Inline PTX for Blackwell tcgen05 matmul within chunk tiles
// Two levels of parallelism: standard chunkwise + tiling within chunks

// SM100 path: tcgen05.mma for intra-chunk matrix operations
asm volatile(
    "tcgen05.mma.cta_group::1.kind::f16f16f32"
    " [%0], %1, %2;"
    :
    : "l"(tmem_addr), "l"(a_smem_addr), "l"(b_smem_addr)
);
```

## Implementations

Two main implementations exist:

| Implementation | Source | Notes |
|----------------|--------|-------|
| NVlabs/GatedDeltaNet | Reference | Triton kernels, research quality |
| FLA (Flash Linear Attention) | Recommended | Optimized, significantly faster, variable-length support |

## FlashInfer MLSys 2026 Contest

GatedDeltaNet is Track C of the FlashInfer MLSys 2026 contest:
- Parameters: qk_dim=4, v_dim=8, d=128
- Benchmarks: decode `qk4_v8_d128_k_last`, prefill `qk4_v8_d128_k_last`
- Status: decode done for both Hopper and Blackwell; prefill done on Hopper, in progress for Blackwell

## When to Use

- Long-context inference (32K+) where O(n) scaling provides major throughput gains
- Hybrid architectures combining linear attention (75% of layers) with full attention (25%)
- Streaming decode where O(1) per-token cost eliminates KV cache growth

## Caveats

- Triton-based kernels have CPU launch overhead impacting small decode batches; use CUDA graph mode via vLLM
- Recurrent state size (D_QK * D_V per head) can be large -- 512K floats for typical configs
- Quality depends on the learned gating; not a drop-in replacement for standard attention without retraining
- Attention output gating (in Qwen3.5) is required to eliminate Attention Sink and Massive Activation problems

## Sources

- [GatedDeltaNet (ICLR 2025)](https://github.com/NVlabs/GatedDeltaNet)
- [TFLA paper](https://arxiv.org/abs/2503.14376)
- [Qwen3-Next NVIDIA blog](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
- [FlashInfer MLSys 2026 Contest](https://mlsys26.flashinfer.ai/)
