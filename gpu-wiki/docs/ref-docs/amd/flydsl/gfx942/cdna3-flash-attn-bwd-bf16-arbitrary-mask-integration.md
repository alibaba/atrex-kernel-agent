# FlyDSL Flash Attention Backward bf16 — Arbitrary Mask Integration on MI308X (gfx942)

## Target hardware

- Chip: AMD MI308X (CDNA3, gfx942)
- Peak bf16 MFMA: 206 TFLOPS
- HBM BW: 5.3 TB/s
- CUs: 80, LDS: 64 KB/CU, VGPR: 512/SIMD

## Algorithm baseline

End-to-end backward pass for flash attention with **arbitrary (non-causal) additive mask**:
- Split dQ + dK/dV design (two separate kernel launches, no atomic_add)
- Mask: (B, 1, S, S) f32 additive format → bit-packed u32 bitmask on host
- Precomputed per-workgroup loop bounds from mask sparsity (skip fully-masked tiles)
- OOB guards in kernel eliminate explicit F.pad, passing unpadded tensors directly

Starting point: optimized dQ (V14, 2.93ms) + dK/dV (V15, 6.52ms) kernels from
[cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md).
This document covers the **API integration layer** that connects those kernels to
real-world inputs with arbitrary masks.

### Test shape

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| Q, K, V | (1024, 8, 316, 64) | bf16 | BHSD, non-contiguous stride |
| Mask | (1024, 1, 316, 316) | f32 | Per-batch, broadcast over heads |
| dO (grad_output) | (1024, 8, 316, 64) | bf16 | Same layout as Q |
| scale | 0.125 (= 1/√64) | — | |
| Theoretical FLOPs (bwd) | 0.5235 TFLOP | — | 5*2*B*H*S^2*D |
| S_padded | 320 (= ceil(316/32)×32) | — | For mask tiling |

## API architecture

```
flash_attn_bwd_build()         → compile + cache dQ and dK/dV kernels
flash_attn_bwd_precompute_mask() → pack_mask_to_u32 + compute_dq/dkdv_loop_bounds
flash_attn_bwd()               → orchestrate both kernels with mask context
```

Key design decisions:
1. **Mask packing on host**: f32 additive → bool (threshold -0.5) → bit-packed u32.
   32× memory reduction (419 MB → 13 MB for this shape).
2. **Loop bounds precomputation**: For each workgroup, find first/last active KV-tile
   (for dQ) or Q-tile (for dK/dV). Skip fully-masked tiles entirely.
3. **OOB guards in kernel**: `arith.select(row_in_bounds, row_idx, 0)` redirects
   out-of-bounds loads to row 0. Mask bits for OOB positions are 0, so contributions
   are zeroed in softmax. Eliminates F.pad (saves ~0.3ms allocation + copy overhead).
4. **Separate LSE/Delta precomputation**: LSE and Delta are computed once (forward pass
   byproduct) and reused across backward calls.

## Optimization journey

### V0 — Naive integration with F.pad (12.8 ms)
Direct F.pad of Q, K, V, dO from (B,H,316,64) to (B,H,320,64) before kernel launch.
Mask packed to padded size. Output F.pad-ed back. Heavy memory allocation in timing loop.

### V1 — Pre-allocate + move F.pad outside loop (12.1 ms, -5.5%)
Move all memory allocation (output buffers, padded tensors) outside timing loop.
Only kernel dispatch + sync in measured region.

### V2 — F.pad inside API, pre-allocated outputs (11.9 ms, -1.7%)
Padding done inside `flash_attn_bwd()` but with contiguous() calls only. Output
tensors allocated at actual size. Simpler user interface.

### V3 — OOB guards, eliminate F.pad entirely (11.7 ms, -1.7%)
Added `arith.select` OOB guards to both kernels for:
- K/V cooperative loads (dQ kernel)
- Q/dO cooperative loads (dK/dV kernel)
- LSE and Delta scalar loads (both kernels)

Pattern per load site:
```
row_in_bounds = arith.cmpi(slt, row_idx, seq_len)
row_safe = arith.select(row_in_bounds, row_idx, arith.index(0))
vec = vector.load(base_ptr + row_safe * stride)
vec_safe = arith.select(row_in_bounds, vec, zero_vec)
```

Pass actual `seq_len` (316) to kernel for memory stride computation.
Grid tiles: `ceil(316/32) = 10 = ceil(320/32)` — unchanged.
Loop bounds still computed with `seq_len_padded` (320) for mask alignment.

### V3 result — Final (11.7 ms, 44.7 TFLOPS)
- dQ kernel: ~4.0 ms (up from 2.93ms causal-only due to sparse mask + API overhead)
- dK/dV kernel: ~7.7 ms (up from 6.52ms)
- API overhead (mask packing, bounds, tensor contiguous): amortized via precomputation

## Final perf vs alternatives

Test shape: B=1024, H=8, S=316, D=64, arbitrary sparse mask (~43% valid Q-rows).

| Implementation | avg (ms) | min (ms) | TFLOPS | Speedup vs PyTorch |
|---|---|---|---|---|
| PyTorch SDPA backward | 35.5 | 35.1 | 14.9 | 1.00× |
| **FlyDSL dQ+dK/dV (this work)** | **11.7** | **11.7** | **44.7** | **3.00×** |
| aiter `mha_bwd` CK-tile (2D bias) | 26.6 | 26.5 | 19.8 | 1.33× |
| aiter `flash_attn_func` (autograd) | 33.8 | 33.3 | 15.7 | 1.05× |

**vs aiter best (mha_bwd): 2.26× faster**

### Capability comparison

| Feature | FlyDSL (this) | aiter CK-tile | aiter Triton | PyTorch SDPA |
|---|---|---|---|---|
| Per-batch mask (B,1,S,S) | ✓ | ✗ (2D only) | ✗ (no mask bwd) | ✓ |
| Mask sparsity exploitation | ✓ (loop bounds) | ✗ | ✗ | ✗ |
| Causal mask | ✓ | ✓ | ✓ | ✓ |
| Arbitrary additive mask | ✓ | Partial (2D) | ✗ | ✓ |

## Remaining bottlenecks

1. **API overhead vs kernel-only**: Kernel-only time (causal, precomputed) is 9.45ms.
   With arbitrary mask (sparse, 43% valid rows), total is 11.7ms. The 2.3ms gap includes:
   - Sparse mask has longer effective loop ranges than causal (more tiles active per row)
   - `torch.zeros_like` output allocation (~0.2ms per tensor × 3)
   - Python dispatch overhead for two kernel launches

2. **Kernel-level ceiling** (from companion doc): lds_pt
   cannot be eliminated, occupancy limited to 4 waves for dK/dV.

3. **Mask precomputation not amortized in single-call API**: `flash_attn_bwd()` without
   `mask_ctx` recomputes mask packing + bounds every call. Use `flash_attn_bwd_precompute_mask()`
   for repeated calls with the same mask.

## What would close the remaining gap

1. **Pre-allocated output buffers** (`flash_attn_bwd_fast` path): Zero + dispatch only,
   saves 3× `torch.zeros_like` allocation overhead.
2. **Fused dQ+dK/dV single-launch**: One HIP dispatch instead of two, saves ~0.1ms
   launch overhead. Complex due to different grid structures.
3. **CK V3–style ISA scheduling**: Cycle-level manual instruction scheduling for better MFMA/VALU/LDS interleaving.

## Sustained recipe (API integration)

1. **Pre-compile kernels** with `flash_attn_bwd_build()` — cache across calls
2. **Precompute mask context** with `flash_attn_bwd_precompute_mask()` — one-time cost
3. **Precompute LSE and Delta** from forward pass output — reuse across backward
4. **Pass actual seq_len to kernel** (not padded) — kernel uses it for memory stride
5. **OOB guards in kernel** (`arith.select` pattern) — avoids F.pad entirely
6. **Bit-packed u32 mask** with threshold -0.5 — 32× bandwidth reduction vs f32 mask
7. **Loop bounds per workgroup** — skip fully-masked tiles, major win for sparse masks
8. For highest throughput: use `flash_attn_bwd_fast()` with pre-padded inputs and
   pre-allocated outputs (zero-overhead path)

## Related docs

- Kernel-level optimization journey: [cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md)
- Forward mask optimization: [cdna3-flash-attention-bf16-mask-optimization.md](cdna3-flash-attention-bf16-mask-optimization.md)
- Integration pitfalls:
- Kernel-level pitfalls:
- Reference API code:
- Reference kernel dQ:
- Reference kernel dK/dV:
