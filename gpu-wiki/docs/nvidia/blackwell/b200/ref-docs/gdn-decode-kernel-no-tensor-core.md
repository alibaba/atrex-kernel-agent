# B200 GDN Kernel Optimization: Why Decode Cannot Use Tensor Core

A deep analysis of why Gated Delta Networks decode kernels are fundamentally memory-bound and cannot leverage Tensor Cores, contrasting with the compute-bound prefill phase.

---

## Glossary

`tcgen05.mma` — Blackwell 5th-gen Tensor Core instruction; `WGMMA` — Hopper 4th-gen; `Swizzle` — address remapping to eliminate bank conflicts; `Ridge Point` — roofline compute-bandwidth inflection; `AI = FLOPs/byte`; `CuTe DSL` — CUTLASS 4.0 Python interface (used by FlashAttention-4); `cuTile` — CUDA 13.1 (2025-12) Python tile API.

---

## TL;DR

| Phase | Operation Type | Tensor Core | Bottleneck | Best Strategy |
| --- | --- | --- | --- | --- |
| Decode | Matrix-vector | No | Memory bandwidth | CuTe SMEM Swizzle |
| Prefill | Matrix-matrix | Yes | Approaching compute | Chunked tcgen05.mma |

---

## 1. Gated Delta Net Background

A linear attention variant with recurrent state updates at O(L) complexity. State `S` has shape `[V,K] = [128,128]`, maintained independently per head. Per-token computation:

```python
g = exp(-exp(A_log) * softplus(a))   # decay gate
beta = sigmoid(b)                    # update gate
S = g * S
old_v = k @ S                        # [K] × [K,V] → [V]
new_v = beta * v + (1-beta) * old_v
S = S + outer(k, new_v - old_v)      # Delta Rule
o = scale * q @ S                    # [K] × [K,V] → [V]
```

---

## 2. B200 Hardware Specifications (Excerpts)

- Tensor Core instruction evolution: mma.sync (A100, 1x) → wgmma (H100, ~2x) → **tcgen05.mma (B200, 2-4x vs Hopper)**
- BF16 Tensor 2.25 PFLOPS / FP32 CUDA 74.45 TFLOPS / HBM3e 8 TB/s
- Ridge Point: BF16 281 FLOP/byte, FP32 9.3 FLOP/byte

---

## 3. Decode: Why Tensor Core Fails

`S @ k` and `S @ q` are both `[128,128]×[128]` GEMV operations — `N = 1 < 16`, which cannot fill a tcgen05.mma tile. Per-token computation is ~1.05M FLOPs / ~1.05 MB → **AI = 1 FLOP/byte**, completely memory-bound.

Measured bandwidth utilization: B=1 → 0.3%, B=64 → 19%, B=256 → **95%** (CuTe v9).

---

## 4. Prefill: Chunked Recurrence Converts GEMV to GEMM

```python
for start in range(0, L, chunk_size):
    Q_chunk = Q[start:end]            # [C, K]
    O_chunk = S @ Q_chunk.T            # [V,K] × [K,C] → mat-mat! tcgen05.mma applicable
    for t in range(start, end):
        S = g[t] * S + outer(k[t], delta[t])   # sequential portion
```

AI improvement: no chunk 1 → chunk=64 ~7.5 → chunk=128 ~12. State recurrence dependency forces sequential updates, but intra-chunk GEMM can be parallelized.

---

## 5. Technology Stack Comparison

### 5 Version Evolution

| Version | Stack | Key Optimization | Lines | Bandwidth Utilization |
| --- | --- | --- | --- | --- |
| v5 | Triton | auto-tune | ~200 | 35% |
| v7 | Raw CUDA | float4 vectorize | ~650 | 95% |
| v8 | Raw CUDA | Warp specialization | ~650 | 95% |
| v9 | CuTe C++ | SMEM Swizzle | ~400 | 95% |
| v10 | CuTe (advanced) | TiledMMA abstraction | ~350 | 95% |
| v11 | cuTile (planned) | Python Tile API | ~100 | TBD |

### Swizzle Principle

SMEM has 32 banks × 4 bytes; 32 threads hitting the same bank → 32-way conflict with 1/32 throughput. Swizzle<3,3,3> formula:

```
physical = logical XOR ((logical >> 3) & 7)
```

Groups of 8 are cross-mixed with the next group of 8 → 8-way conflict reduced to 1-way, SMEM throughput ×8.

### CuTe DSL (FlashAttention-4 Style)

CUTLASS 4.0's Python-native interface. Uses `@cute.jit`, `cute.make_layout`, `cute.SharedMemory`, `cute.make_tiled_mma`, `cute.tcgen05.MmaF16BF16Op` (Blackwell), `cute.gemm`, `cute.copy(TMA)`, `cp_async_wait_group`, etc. Compilation time reduced from minutes to seconds (20-30x speedup) with near-zero performance loss. **FA4 B200 BF16 ~1600 TFLOPS (71%), 1.3x vs cuDNN, 2.7x vs Triton**.

### cuTile (CUDA 13.1)

NVIDIA's official counterpart to Triton. Pure Python, tile-based abstraction, automatic Tensor Core/TMA usage, compiler auto-tuning, cross-architecture support (Ampere → Blackwell).

```python
@ct.kernel
def gdn_decode_v11(state, q, k, v, out, D=128, V=128):
    batch_id, head_id = ct.bid(0), ct.bid(1)
    state_tile = ct.load(state, (batch_id, head_id), (V, D))
    q_tile = ct.load(q, (batch_id, head_id), (D,))
    o_tile = ct.sum(state_tile * q_tile, axis=1)
    state_tile = state_tile + ct.outer(k_tile, v_tile - o_tile)
    ct.store(state, ..., state_tile)
    ct.store(out, ..., o_tile)
```

---

## 6. Performance Comparison

Decode @ Batch=256 on B200: All versions except Triton achieve ~7600 GB/s (95% bandwidth). **At small batch sizes, CuTe wins (lower launch overhead). At medium batch (=64), Triton surpasses due to auto-tune (Triton 1518 vs CuTe 1302 GB/s). At large batch (=256), CuTe Swizzle locks in a 2.68x advantage.**

---

## 7. Conclusions and Technology Selection

- Not all operators should pursue Tensor Core: Memory-bound decode → optimize bandwidth; Compute-bound prefill (chunked) → tcgen05.mma
- Rapid prototyping: Triton/cuTile; Memory-intensive: CuTe Swizzle abstraction; Compute-intensive: CuTe TiledMMA + tcgen05.mma; Peak performance: Raw CUDA
