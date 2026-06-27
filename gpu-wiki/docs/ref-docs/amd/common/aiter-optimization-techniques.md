# aiter Optimization Techniques in Detail

AMD's official AI inference operator library [aiter](https://github.com/ROCm/aiter) — core optimization patterns and implementation techniques reference.

---

## 1. aiter Project Overview

aiter is a high-performance AI inference operator library maintained by the AMD ROCm team, designed for MI300X (gfx942, CDNA3) and MI350 (gfx950, CDNA4).

### Three-Layer Architecture

| Layer | Technology Stack | Use Cases | Characteristics |
|-------|-----------------|-----------|-----------------|
| **Triton** | Python DSL | Attention, MoE, Fused Kernel | High development efficiency, autotune-friendly |
| **CK (Composable Kernel)** | C++ Template Library | GEMM, Quantized GEMM | Extreme performance, tile-level abstraction |
| **ASM** | Hand-written Assembly (GCN ISA) | Critical GEMM paths | Highest performance, full control over instruction scheduling |

### Covered Operators

- **Attention**: Paged Attention V1/V2, MLA Decode, Lean Attention, Unified Attention
- **GEMM**: FP16/BF16/FP8/INT8/MXFP4 quantized GEMM, Split-K, Grouped GEMM
- **MoE**: Fused MoE (gate+up+down), MXFP4 MoE, MoE Routing
- **Normalization**: RMSNorm, LayerNorm, RMSNorm+Quant fusion
- **Quantization**: FP8/INT8 per-tensor/per-token/per-block, MXFP4 block-scale
- **RoPE**: Fused QKV Split + RoPE, BMM + RoPE + KV Cache

---

## 2. AMD-Specific Optimization Patterns

### XCD Remapping

The MI300X contains 8 XCDs (eXtended Compute Die). Default sequential program ID assignment causes adjacent tiles to land on different dies, resulting in low L2 cache hit rates. XCD remapping ensures that adjacent tiles are mapped to the same XCD:

```python
def remap_xcd(pid, NUM_XCD: tl.constexpr):
    """Remap consecutive pids to adjacent tiles on the same XCD"""
    # pid 0,1,2,...,7 are on XCD 0~7 respectively
    # After remapping, pid 0~N/8 are all on XCD 0, improving L2 locality
    num_pid = tl.num_programs(0)
    pid_per_xcd = (num_pid + NUM_XCD - 1) // NUM_XCD
    xcd_id = pid % NUM_XCD
    local_pid = pid // NUM_XCD
    return xcd_id * pid_per_xcd + local_pid
```

Typical usage: Call `pid = remap_xcd(tl.program_id(0), NUM_XCD=8)` at the entry point of GEMM/Attention kernels.

### exp2 Instead of exp

AMD GPU's `v_exp_f32` is a single-cycle instruction for computing 2^x, whereas `exp(x)` = `exp2(x * log2(e))` requires an additional multiplication. Almost all aiter kernels uniformly use:

```python
LOG2E: tl.constexpr = 1.44269504089
# Replace tl.exp(x)
# Usage in softmax
p = tl.exp2(qk_scaled - m_new)  # Directly maps to v_exp_f32
```

### Cache Modifier

AMD GPUs support fine-grained cache control presetsmaps. aiter uses different modifiers in different scenarios:

| Modifier | Meaning | Use Case |
|----------|---------|----------|
| `.cg` | Cache Global (non-temporal, bypass L1) | Decode scenario where Q is read only once |
| `.wt` | Write-Through | Lean Attention cross-CTA lock writes |
| `.cv` | Cache Volatile (always read from memory) | Lean Attention cross-CTA lock synchronization reads |

```python# decode attention: Q is only used once, does not pollute L1
# lean attention: cross-CTA signal synchronization
tl.store(lock_ptr, value, cache_modifier=".wt")     # write side
flag = tl.load(lock_ptr, cache_modifier=".cv")       # read side (bypass cache)```

### tl.assume Optimization Hints

```python
tl.assume(stride_qm > 0)
tl.assume(stride_kn > 0)
# Hint compiler that stride is positive, enabling:
# 1. More efficient address calculation (avoid sign extension)
# 2. Loop unroll optimization
# 3. Better vectorization judgment
```

---

## 3. Quantized GEMM Optimization

### Block-Scale Quantization Formats

| Format | Data Type | Scale Granularity | Hardware Requirement |
|--------|-----------|-------------------|----------------------|
| A8W8 | FP8 (E4M3/E5M2) | per-1x128 block | CDNA3+ |
| A4W4 (MXFP4) | MXFP4 | per-1x32 block | CDNA4 only |
| A16WFP4 | FP16 activation + MXFP4 weight | per-1x32 block | CDNA4 only |
| A8WFP4 | FP8 activation + MXFP4 weight | Mixed | CDNA4 only |
| AFP4WFP4 | Dual-sided MXFP4 | per-1x32 block | CDNA4 only |

### Weight Preshuffle

Pre-arrange the B matrix to match the data layout of MFMA instructions, avoiding runtime transposition:### Split-K

K dimension is parallelized across multiple CTAs, suitable for scenarios where K >> M*N (such as decode GEMM):

```python
# unified pid encoding: pack tile index and K-split index together
pid = tl.program_id(0)
pid = remap_xcd(pid, NUM_XCD)

# Decode tile coordinates and split index
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
pid_m = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
pid_k = tl.program_id(1)  # K-split dimension

# Each CTA only computes one shard of K
k_start = pid_k * BLOCK_K * num_k_tiles_per_split
k_end = min(k_start + BLOCK_K * num_k_tiles_per_split, K)

# Write partial results to workspace, subsequent reduce kernel aggregates them
tl.store(workspace + pid_k * stride_ws, acc)
```

### pid_grid L2-friendly Sorting

Similar to GROUP_SIZE_M swizzle, ensures adjacent tiles share data in the L2 cache:

```python
# L2 cache-friendly tile traversal order
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
pid_m = first_pid_m + (pid % GROUP_SIZE_M)
pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
```

### MXFP4 Quantized GEMM (CDNA4)

```python
# CDNA4 uses tl.dot_scaled to implement hardware-accelerated block-scaled MMA
acc = tl.dot_scaled(
    a_fp4, a_scales, "e2m1",   # activation: MXFP4 + per-32 scale
    b_fp4, b_scales, "e2m1",   # weight: MXFP4 + per-32 scale
    acc                         # FP32 accumulator
)
```

---

## 4. Attention Optimization

### Paged Attention V1/V2

**V1**: A single CTA processes all KV blocks across the entire sequence:

```python
# V1: Single CTA traverses all KV pages
for i in range(num_kv_blocks):
    block_idx = block_table[i]
    k = load_paged_kv(k_cache, block_idx)  # Load from paged KV cache
    v = load_paged_kv(v_cache, block_idx)
    qk = tl.dot(q, k)        # [BLOCK_M, BLOCK_N]
    # online softmax update
    m_new = tl.maximum(m, tl.max(qk, axis=1))
    p = tl.exp2((qk - m_new[:, None]) * LOG2E)
    acc = acc * tl.exp2((m - m_new) * LOG2E)[:, None] + tl.dot(p, v)
    m = m_new
```

**V2**: Multiple CTAs partition and parallelize, followed by a reduce kernel:

```python
# V2: Each CTA processes a portion of KV blocks
partition_id = tl.program_id(1)
kv_start = partition_id * KV_BLOCKS_PER_PARTITION
kv_end = min(kv_start + KV_BLOCKS_PER_PARTITION, num_kv_blocks)

# Each partition independently computes partial output + log-sum-exp
# Reduce kernel merges all partition results using LSE
```

**Two QK Computation Paths**:

| Path | Implementation | Applicable Scenarios |
|------|------|----------|
| `wo_dot` (element-wise) | Element-wise multiply-add | Standard MHA (num_heads = num_kv_heads) |
| `w_dot` (tl.dot) | Matrix multiplication | GQA (num_heads > num_kv_heads), multiple Q heads share KV |

### FP8 KV Cache

Per-token dynamic dequantization, each token independently maintains its scale:

```python
# Load FP8 KV and dequantize
k_fp8 = tl.load(k_cache_ptr)             # FP8 E4M3
k_scale = tl.load(k_scale_ptr + token_id) # per-token scale
k = k_fp8.to(tl.float16) * k_scale       # Dequantize
```

### MLA Decode with RoPE

Multi-Latent Attention splits Q into nope (without positional encoding) and pe (with positional encoding) parts:

```python
# MLA decode: Separate nope/pe computation
# Q_nope: [B, H, D_nope]  Q_pe: [B, H, D_pe]
# KV: [B, S, D_nope]      K_pe: [B, S, D_pe]

# Phase 1: nope part (without RoPE)
qk_nope = tl.dot(Q_nope, KV.T)  # [H, S]

# Phase 2: pe part (online RoPE application)
Q_pe_rotated = apply_rope(Q_pe, cos, sin, position)
K_pe_rotated = apply_rope(K_pe, cos, sin, positions)
qk_pe = tl.dot(Q_pe_rotated, K_pe_rotated.T)

# Merge
qk = qk_nope + qk_pe
# online softmax + value accumulation

# Cross-split reduction uses log-sum-exp
lse = m + tl.log2(d)  # log-sum-exp for subsequent reduce
```### Lean Attention

Stream-K persistent kernel with ping-pong scheduling and cross-CTA lock synchronization:

```python
# Ping-pong scheduling: Alternately assign Q-blocks to improve memory locality
# CTA 0 processes Q[0], Q[2], Q[4]...
# CTA 1 processes Q[1], Q[3], Q[5]...
q_block_id = cta_id * 2 + (iteration % 2)

# Cross-CTA lock synchronization (using .wt/.cv cache modifier)
# Producer CTA notifies Consumer after completion
if is_producer:
    tl.store(lock_ptr, 1, cache_modifier=".wt")  # write-through
if is_consumer:
    while tl.load(lock_ptr, cache_modifier=".cv") == 0:  # volatile read
        pass  # spin-wait
```

### Unified Attention

A unified attention kernel supporting sliding window, sink token, and sequence indexing:

```python
# Sliding window tile pruning: Skip tiles outside the window
if SLIDING_WINDOW > 0:
    window_start = query_pos - SLIDING_WINDOW
    if kv_block_end < window_start:
        continue  # Entire tile is outside the window, skip

# Sink token support: Keep first N tokens
if SINK_TOKEN_LENGTH > 0 and kv_pos < SINK_TOKEN_LENGTH:
    in_window = True  # Sink tokens always participate in computation

# Binary search seq_idx (multi-sequence batching)
seq_id = binary_search(seq_starts, kv_pos)
```

### Online Softmax

All attention kernels uniformly use the online softmax mode:

```python
# Initialize
m = -float("inf")   # running max
d = 0.0             # running denominator
acc = tl.zeros(...)  # running output

for block in kv_blocks:
    qk = compute_qk(q, k_block)
    m_new = tl.maximum(m, tl.max(qk, axis=1))
    alpha = tl.exp2((m - m_new) * LOG2E)
    p = tl.exp2((qk - m_new[:, None]) * LOG2E)
    # Rescale old accumulated value
    acc = acc * alpha[:, None]
    d = d * alpha + tl.sum(p, axis=1)
    acc += tl.dot(p.to(v.dtype), v_block)
    m = m_new

# Final normalization
acc = acc / d[:, None]
```

---

## 5. MoE Optimization

### E2E Fused MoE

Gate+up weights are interleaved in storage, enabling reshape without permute:

```python
# Weight interleaving trick: gate and up are arranged alternately
# Original: W_gate[E, N, K], W_up[E, N, K]
# Interleaved: W_interleaved[E, 2N, K] where offs = offs_half + (i % 2) * (N // 2)
# This makes gate/up results naturally adjacent, no need for extra permute

for i in range(num_k_tiles):
    # Compute gate and up simultaneously
    a = tl.load(A_ptr + ...)
    w_gate = tl.load(W_ptr + offs_half + 0 * (N // 2) + ...)
    w_up   = tl.load(W_ptr + offs_half + 1 * (N // 2) + ...)
    acc_gate += tl.dot(a, w_gate)
    acc_up   += tl.dot(a, w_up)

# SiLU activation + element-wise multiply
gate_activated = acc_gate * tl.sigmoid(acc_gate)  # SiLU
intermediate = gate_activated * acc_up

# Down projection uses atomic_add (multiple experts accumulate to same output)
tl.atomic_add(out_ptr + ..., down_result, sem="relaxed", scope="gpu")
```

### MXFP4 MoE (CDNA4)

```python
# CDNA4 MXFP4 MoE: Use dot_scaled + scale unswizzle
N_PRESHUFFLE_FACTOR: tl.constexpr = 32  # CDNA4 scale permutation factor

# Unswizzle scales to match MFMA data layout
scale_offs = (n_base // N_PRESHUFFLE_FACTOR) * scale_stride + k_idx
w_scales = tl.load(scale_ptr + scale_offs)

acc = tl.dot_scaled(
    x_fp4, x_scales, "e4m3",
    w_fp4, w_scales, "e4m3",
    acc
)
```

### MoE Routing

Bitmatrix-compressed expert assignment + 32-bit packed encoding:

```python
# Bitmatrix: Each token uses bitmask to indicate which experts it is assigned to
# More compact than sparse indices

# expt_data 32-bit packing: low 16 bits = expert_id, high 16 bits = block_id
expt_data = tl.load(expt_data_ptr + pid)
expt_id = expt_data & 0xFFFF
block_id = expt_data >> 16

# Directly index to corresponding expert's weights and inputs
w_ptr = W_base + expt_id * expert_stride
```### SiLU Fast Path

```python
# Standard SiLU: x * sigmoid(x) = x / (1 + exp(-x))
# Fast implementation (using exp2 hardware instruction):
s = gate / (1.0 + tl.exp2(-LOG2E * alpha * gate))
# Final: fma(s, linear, s) i.e., s * (linear + 1)
result = s * (linear + 1.0)
```

# Gammas: Per-token expert weight scaling, used for weighted aggregation of expert outputs
gamma = tl.load(gammas_ptr + token_id * num_experts + expt_id)
output = expert_output * gamma  # Weighted and accumulated
---

## 6. Fused Kernel Mode

### BMM + RoPE + KV Cache Fusion

Three-stage grid design, completing BMM, RoPE, and KV cache writes in a single kernel:

```python
# Three-stage grid encoding
phase = tl.program_id(2)  # Third dimension distinguishes phases

if phase == 0:
    # Phase 1: BMM (QKV projection)
    qkv = tl.dot(input, W_qkv)

elif phase == 1:
    # Phase 2: RoPE + cache decode
    q, k = apply_rope(q_raw, k_raw, cos, sin, position)
    # FP8/FP4 on-the-fly activation quantization
    k_fp8 = quantize_fp8(k, k_scale)
    v_fp8 = quantize_fp8(v, v_scale)
    tl.store(k_cache + cache_slot, k_fp8)
    tl.store(v_cache + cache_slot, v_fp8)

elif phase == 2:
    # Phase 3: Cache prefill (batch write multiple positions)
    for pos in range(seq_start, seq_end):
        tl.store(k_cache + pos, k_quantized[pos])
        tl.store(v_cache + pos, v_quantized[pos])
```

### RMSNorm + FP8 Quantization Fusion

Avoid writing norm results back to HBM and then reloading them for quantization:

```python
@triton.jit
def rmsnorm_quant_kernel(X, W, Out, Scale, eps, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(X + row * N + cols)

    # RMSNorm
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    x_norm = x / rms * tl.load(W + cols)

    # Direct quantization (no extra memory round-trip)
    amax = tl.max(tl.abs(x_norm))
    scale = amax / 448.0  # FP8 E4M3 max
    x_fp8 = (x_norm / scale).to(tl.float8e4m3fn)

    tl.store(Out + row * N + cols, x_fp8)
    tl.store(Scale + row, scale)
```

Multiple variants: per-tensor static scale / group-wise dynamic scale / split-K reduce + norm + quant.

### Feed-Forward Fusion

Compute both halves of the gate+up concat weight simultaneously:

```python
# W = [W_gate | W_up], concat on N dimension
# One GEMM computes gate and up simultaneously
for k_tile in range(K // BLOCK_K):
    a = tl.load(A_ptr + ...)
    w_left  = tl.load(W_ptr + ...)                  # gate part
    w_right = tl.load(W_ptr + ... + N_half * stride) # up part
    acc_gate += tl.dot(a, w_left)
    acc_up   += tl.dot(a, w_right)

# activation
activated = silu(acc_gate) * acc_up

# down projection: atomic_add accumulation
# cyclic K-offset reduces atomic conflicts
k_offset = (pid_m * 7) % (K_down // BLOCK_K) #
for k_tile in range(K_down // BLOCK_K):
    k_idx = (k_tile + k_offset) % (K_down // BLOCK_K)
    w_down = tl.load(W_down_ptr + k_idx * BLOCK_K + ...)
    partial = tl.dot(activated_tile, w_down)
    tl.atomic_add(out_ptr + ..., partial, sem="relaxed", scope="gpu")
```

### Fused QKV Split + QK RoPE

```python
# One kernel completes:
# 1. Split Q, K, V from QKV concat tensor
# 2. Apply RoPE to Q and K
# 3. Write to respective output buffers
@triton.jit
def fused_qkv_rope_kernel(QKV, Q_out, K_out, V_out, cos, sin, ...):
    # Split
    q = tl.load(QKV + q_offset)
    k = tl.load(QKV + k_offset)
    v = tl.load(QKV + v_offset)

    # RoPE (only for Q and K)
    q_rotated = q * cos - rotate_half(q) * sin
    k_rotated = k * cos - rotate_half(k) * sin

    tl.store(Q_out + ..., q_rotated)
    tl.store(K_out + ..., k_rotated)
    tl.store(V_out + ..., v)  # V does not need RoPE
```## 7. CK-Based GEMM and Auto-Tuning

### Three Backends

| Backend | Selection Method | Characteristics |
|------|----------|------|
| **CK (legacy)** | `--libtype ck` | C++ template metaprogramming, mature and stable |
| **CK-Tile** | `--libtype ck_tile` | Next-generation tile-level API, more flexible |
| **ASM** | `--libtype asm` | Hand-written assembly, ultimate performance |

### Auto-Tuning Workflow

```bash
# Step 1: Prepare shape configuration (CSV format)
# shapes.csv:
# M,N,K,dtype
# 1,4096,4096,fp16
# 32,4096,11008,fp8

# Step 2: Run tuning script
python tune.py --input shapes.csv \
    --libtype ck_tile \
    --splitK 1,2,4,8 \
    --output tuned_results.csv

# Step 3: Tuning output
# cu_num | kernelId | splitK | timing(us) | tflops | bw(GB/s) | errRatio
# 304    | 12       | 4      | 15.3       | 285.6  | 1420     | 1e-6

# Step 4: JIT compile optimal kernel
python build_tuned.py --config tuned_results.csv --output lib/
```

### Weight Layout

Different backends require different weight pre-processing:

```python# ASM backend: shuffle layout (32, 16)
# Reorder weights in (32, 16) tiles to match MFMA data path

# Flat GEMM (CK): [N/128, K*128] layout```

### Architecture Dispatch

Select pre-compiled kernels at runtime based on GPU architecture:

```python
import torch

def get_gemm_kernel(dtype, M, N, K):
    gpu_arch = torch.cuda.get_device_properties(0).gcnArchName# MI300X: Load CDNA3 optimized .hsaco
        # MI350: Load CDNA4 optimized .co (includes MXFP4 support)```

---

## 8. Related Documentation

- [AMD MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md) -- MFMA instruction naming conventions, register layout
- [AMD GPU Kernel Optimization Framework Overview](amd-kernel-optimization-frameworks.md) -- HIP, CK, FlyDSL, TileLang framework comparison
- [LDS Bank Conflict Optimization](../../../kernel-opt/amd/common/lds-bank-conflict-optimization.md) -- Bank architecture and XOR swizzle for conflict elimination
- [Preshuffle B Layout](../../../kernel-opt/amd/common/hands-on/preshuffle-b-layout.md) -- Pre-arranged weight matrix to avoid runtime layout conversion
- [MoE 2-Stage Fusion](../../../kernel-opt/amd/common/hands-on/moe-2stage-fusion.md) -- Expert GEMM + SiLU fusion, mixed precision
- [Hardware Specification Comparison CDNA3 vs CDNA4](../../../hardware-specs/hardware-comparison-cdna3-cdna4.md) -- Architecture parameter comparison
- Triton Kernel Tuning Parameters -- Tile size, MFMA selection, num_warps
