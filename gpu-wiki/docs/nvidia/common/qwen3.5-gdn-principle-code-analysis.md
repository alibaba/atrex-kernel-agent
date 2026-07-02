# Qwen3.5 GDN (Gated Delta Networks): Principle and Code Analysis

Comprehensive analysis of Gated Delta Networks from mathematical foundations through chunk-wise parallel algorithms to production kernel implementation, with code-level walkthrough of the flash-linear-attention implementation.


**Last updated**: 2026-06-30

---

## 1. Overview: From Standard Attention to Linear Attention

Standard Attention has computational complexity O(L^2 d) and IO complexity O(L^2 + Ld). Flash Attention optimizes IO complexity to O(L^2 d^2 M^{-1}) (M = SRAM size) via tiling, but computational complexity remains O(L^2 d). As sequence length grows, arithmetic intensity (ops/bytes) exceeds the hardware compute-bandwidth ratio, and the kernel transitions from memory-bound to compute-bound.

Linear Attention replaces softmax with an approximate kernel function and changes the Q/K/V computation order (compute K^T V first, then multiply by Q), reducing complexity from O(L^2 d) to O(L d^2). In autoregressive scenarios, Linear Attention can be transformed into an RNN recurrent form, maintaining a fixed-size hidden state matrix per step.

**GDN: Gated Delta Networks** combines two improvements over vanilla linear attention:
- **Mamba2/GLA**: Introduces scalar gating decay α_t for fast global memory erasure, but cannot selectively update individual key-value pairs
- **DeltaNet**: Uses the delta rule for precise key-value pair replacement, but lacks fast global memory clearing

GDN introduces both gating decay α and delta update strength β simultaneously, achieving both global forgetting and targeted updates. The cost is a more complex chunk-wise algorithm involving WY decomposition and lower-triangular solving.

---

## 2. Prerequisites

### 2.1 Linear Attention and Mamba2

The recurrent form of linear attention:

S_t = S_{t-1} + v_t k_t^T ∈ R^{d_v × d_k}, o_t = S_t q_t ∈ R^{d_v}

Where S_t is the hidden state matrix accumulating key-value information from the first t steps.

The parallel (matrix) form:

O = (Q K^T ⊙ M) V ∈ R^{L × d_v}

Where M is the causal mask. The distinction from standard attention lies only in the attention weight computation: standard attention uses softmax(QK^T), while linear attention uses QK^T directly.

**With gating decay (Mamba2/GLA)**:

S_t = α_t S_{t-1} + v_t k_t^T, o_t = S_t q_t

Where α_t ∈ (0, 1) is data-dependent scalar decay.

**Chunk-wise parallel training**: Divides input into chunks of size C. Each chunk performs:
- **Hidden state S recurrence**: S_{[t+1]} = γ^C S_{[t]} + V_{[t]}^T K̄_{[t]} (inter-chunk, serial)
- **Output O computation**: O_{[t]} = Q̄_{[t]} S_{[t]}^T + (Q_{[t]} K_{[t]}^T ⊙ Γ_{[t]}) V_{[t]} (intra-chunk, parallel)

The chunk-wise algorithm is exact (chunk size does not affect numerical results). Overall complexity is O(L d^2 + L C d), linear in sequence length L when C is fixed.

### 2.2 DeltaNet: Linear Attention with Delta Rule

DeltaNet provides targeted erasure — only erasing memory related to the current key while leaving other memories intact:

S_t = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T

This can be decomposed into: read old value via S_{t-1} k_t, erase it via -β_t (S_{t-1} k_t) k_t^T, write new value via +β_t v_t k_t^T.

**WY Representation**: The cumulative product of transition matrices ∏(I - β_i k_i k_i^T) can be compactly represented as I - W K^T (Bischof & Van Loan, 1985), enabling efficient chunk-wise computation via forward substitution.

---

## 3. Gated Delta Rule

### 3.1 Formula Definition

S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T

Expanded into three operations:

S_t = α_t S_{t-1} - α_t β_t (S_{t-1} k_t) k_t^T + β_t v_t k_t^T

That is: global decay, erase old value, write new value.

### 3.2 Intuitive Understanding

**Online learning perspective**: Each model variant corresponds to a different objective function:
- Linear Attention: proximity regularizer + new information encoding
- Mamba2: decayed proximity + new information
- DeltaNet: proximity + gradient descent on retrieval error
- Gated DeltaNet: decayed proximity + gradient descent

**TTT (Test-Time Training) perspective**: The hidden state S_t acts as "model parameters." Each token triggers one gradient descent step on loss L = ½||S_t k_t - v_t||^2, where β_t is the learning rate. GDN further adds adaptive weight decay via α_t (analogous to AdamW).

### 3.3 Hardware-Efficient Chunk-Wise Parallel Algorithm

The extended WY representation for GDN:

Ũ_{[t]} = [I + strictLower(diag(β_{[t]}) (Γ_{[t]} ⊙ K_{[t]} K_{[t]}^T))]^{-1} diag(β_{[t]}) V_{[t]}

The only difference from DeltaNet: K K^T is replaced by Γ ⊙ K K^T (decay-aware Gram matrix).

Final chunk-wise algorithm:
- **State recurrence**: S_{[t+1]} = S̄_{[t]} + (Ũ_{[t]} - W̄_{[t]} S_{[t]}^T)^T K̄_{[t]}
- **Output**: O_{[t]} = Q̄_{[t]} S_{[t]}^T + (Q_{[t]} K_{[t]}^T ⊙ M) (Ũ_{[t]} - W̄_{[t]} S_{[t]}^T)

The algorithm contains abundant matrix multiplications, enabling full utilization of GPU Tensor Cores.

---

## 4. Model Integration: Qwen3.5 GatedDeltaNet

### 4.1 Architecture

Qwen3.5 uses a hybrid architecture with GDN layers (linear complexity for long sequences) and Gated Attention layers (precise global attention to compensate linear attention's retrieval limitations).

### 4.2 Key Parameters

```python
class Qwen3_5MoeGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads  # GVA: can be < num_v_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        self.in_proj_qkv = nn.Linear(hidden_size, key_dim * 2 + value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, value_dim, bias=False)  # output gate
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)  # beta
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)  # alpha

        # g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
        # alpha = exp(g) in (0, 1)
        A = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))

        # Depthwise causal conv1d on QKV
        self.conv1d = nn.Conv1d(
            in_channels=key_dim * 2 + value_dim,
            out_channels=key_dim * 2 + value_dim,
            kernel_size=conv_kernel_size,
            groups=key_dim * 2 + value_dim, ...)
```

### 4.3 Forward Pass

1. Project + causal conv1d + SiLU activation
2. Reshape to multi-head form
3. Compute beta (sigmoid) and gate g
4. GVA: repeat Q/K heads when num_k_heads < num_v_heads
5. Call chunk or recurrent algorithm
6. Output gating RMSNorm + output projection

**Alpha parameterization**: Uses log-space (g = log α) to convert cumulative decay products into sums, avoiding numerical underflow. Initial α ≈ 0.00003 (near-memoryless), learned during training.

**Short Conv**: Depthwise causal convolution on QKV provides n-gram features, transforming the online learning objective from "predict self" to "predict next token."

**GVA (Grouped Value Attention)**: Multiple value heads share one Q/K head group. Kernel optimization uses index mapping (i_h // group_size) to avoid the repeat_interleave memory overhead.

---

## 5. Chunk-Wise Algorithm Code Analysis

### 5.1 Forward Pass Pipeline

Six stages, each corresponding to a Triton kernel:

```python
def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, ...):
    # (a) Chunk-local cumulative gate values
    g = chunk_local_cumsum(g, chunk_size=64)
    # (b) Scaled dot product: A = strictLower(diag(beta) * (Gamma * K @ K^T))
    A = chunk_scaled_dot_kkt_fwd(k, g, beta)
    # (c) Solve lower-triangular system: (I + A)^{-1}
    A = solve_tril(A)
    # (d) Compute W and U (extended WY representation)
    w, u = recompute_w_u_fwd(k, v, beta, A, g)
    # (e) Recurrent hidden state h and v_new
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)
    # (f) Compute final output
    o = chunk_fwd_o(q, k, v_new, h, g, scale)
    return g, o, A, final_state, initial_state
```

### 5.2 Cumulative Gate Sum (chunk_local_cumsum)

Computes chunk-local prefix sum of log-space gate values using `tl.cumsum`. Since g = log α, the prefix sum equals the log of cumulative decay: log γ_i = Σ log α_j. Any interval decay ratio γ_i/γ_j is then computed as exp(g_cumsum[i] - g_cumsum[j]).

The REVERSE branch computes reverse cumsum for backward pass gradient propagation.

### 5.3 chunk_scaled_dot_kkt_fwd

Computes K K^T element-wise multiplied by decay factors exp(g_i - g_j), forming the decay-aware Gram matrix.

### 5.4 solve_tril

Forward substitution to solve the lower-triangular system (I + A)x = b. This avoids O(C^3) general matrix inversion since the matrix is strictly lower-triangular with unit diagonal.

### 5.5 recompute_w_u_fwd

Computes W = T K and U = T V where T = (I + strictLower(...))^{-1} diag(β).

### 5.6 chunk_gated_delta_rule_fwd_h

The inter-chunk recurrence kernel: computes v_new = U - W @ S^T and updates S_{[t+1]} = decay(S) + K^T @ v_new.

### 5.7 chunk_fwd_o

The output kernel: O = Q @ S^T + (QK^T ⊙ M) @ v_new (inter-chunk + intra-chunk).

---

## 6. Recurrent Algorithm (Decode)

### 6.1 Fused Recurrent

For autoregressive inference, the recurrent form processes one token at a time:

```python
S = alpha * S  # global decay
S = S - alpha * beta * outer(S @ k, k)  # erase old value
S = S + beta * outer(v, k)  # write new value
o = S @ q  # read output
```

### 6.2 Kernel Implementation

The fused recurrent kernel processes the entire sequence in a single kernel launch, maintaining state in registers/SMEM. Each step is a sequence of element-wise operations and rank-1 updates — fundamentally memory-bound since the outer product (d_v, 1) × (1, d_k) cannot saturate Tensor Cores.

---

## 7. Summary

- GDN combines gating decay (global forgetting) with delta rule (targeted replacement) in a unified recurrence
- The chunk-wise algorithm enables hardware-efficient parallel training with O(Ld^2 + LCd) complexity
- Production implementation requires 6+ specialized Triton/Gluon kernels for the forward pass alone
- Decode (recurrent) mode is inherently memory-bound; prefill (chunk-wise) mode can leverage Tensor Cores via GEMM
- Qwen3.5 adopts GDN as a key architectural component for efficient long-sequence processing


## Related

- [Async Global-to-Shared Memory Copy (CC 8.0+)](async-global-to-shared-copy.md)
- [FlashAttention 1–4: GPU Generational Evolution](flash-attention-1-to-4-gpu-evolution.md)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference](flashinfer-efficient-attention-engine.md)
- [GPU Architecture Deep Dive](gpu-architecture-deep-dive.md)
- [Memory-Bound Kernel Optimization: Hierarchical Reduction](hierarchical-reduction-memory-bound.md)
