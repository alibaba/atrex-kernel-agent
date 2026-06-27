# TMA Gather Sparse Decode

Blackwell (SM100)'s TMA Gather instruction collects KV data from multiple non-contiguous tokens in global memory in a single operation, specifically designed for non-contiguous access patterns such as sparse attention / paged KV cache.

**Source**: FlashMLA `csrc/sm100/decode/head64/kernel.cuh`

---

## Problem Background

Sparse MLA Decode requires loading non-contiguous tokens from the KV cache:

```
KV Cache [max_seq_len × head_dim]:
Token 0: [████████████████]
Token 1: [████████████████]  ← Not needed
Token 2: [████████████████]  ← Needed
Token 3: [████████████████]  ← Not needed
Token 4: [████████████████]  ← Needed
...

Traditional: Issue TMA copy per token → High TMA call overhead
TMA Gather: Collect multiple tokens in one instruction → Reduce TMA calls
```

## TMA Gather Usage

```cpp
// SM100 TMA Gather: Collect KV data for 4 tokens in a single operation
// Input: Global memory addresses for 4 tokens (can be non-contiguous)
// Output: Continuous shared memory area

ku::tma_gather4(
    smem_kv,           // Target shared memory
    tma_descriptor,    // TMA descriptor
    token_indices,     // Global indices for 4 tokens
    mbarrier           // Synchronization barrier
);

// Wait for TMA to complete
cute::arrive_and_expect_tx(mbarrier, expected_bytes);
```

### Integration with TMEM/UMMA

Blackwell's MLA decode complete pipeline:

```
1. TMA Gather → KV data to SMEM (non-contiguous → continuous)
2. UTCCP → Q data to TMEM
3. UMMA TS → QK^T GEMM (Q from TMEM, K from SMEM)
4. Softmax → In registers
5. UMMA SS → PV GEMM (P from SMEM, V from SMEM)
```

```cpp
// TMEM allocation
cute::TMEM::Allocator1Sm().allocate(512, &tmem_Q);

// Q copy from SMEM to TMEM (UTCCP instruction)
SM100_UTCCP_128dp256bit_1cta::copy(smem_Q, tmem_Q);

// QK GEMM: Q in TMEM, K in SMEM (TS = Tensor-Shared)
ku::utcmma_ts(tmem_Q, smem_K, acc_S);

// PV GEMM: P and V both in SMEM (SS = Shared-Shared)
ku::utcmma_ss(smem_P, smem_V, acc_O);
```

### Dual GEMM Mode

Blackwell supports executing "dual GEMM" (2 parallel matrix multiplications) in a single UMMA instruction, further increasing throughput:

```cpp
// SM100_MMA_F16BF16_WS_TS_NOELECT:
// WS = Warp Specialized, TS = Tensor-Shared
// UMMA 2 batch QK GEMM
```

### E8M0 Scale Conversion

Blackwell has native E8M0 scale conversion instructions (CDNA4 also supports similar functionality):

```cpp
// Blackwell : E8M0 -> BF16 scale pair
__nv_bfloat162 scale = __nv_cvt_e8m0x2_to_bf162raw(e8m0_scale);
```

## Performance Data

| Kernel | GPU | TFLOPS |
|--------|-----|--------|
| Sparse MLA Decode (FP8) | H800 (SM90, no TMA Gather) | 410 |
| Sparse MLA Decode (FP8) | B200 (SM100, TMA Gather) | 350* |
| Sparse MLA Prefill | B200 (SM100) | 1450 |
| Dense MHA Prefill Fwd | B200 (SM100) | 1460 |

*B200 decode has not yet been fully optimized.

## Practical Experience

- TMA Gather combines multiple TMA copies into one, reducing TMA unit scheduling overhead
- The number of tokens for Gather is typically 4 (SM100 hardware limit), matching TMA bandwidth
- Applicable to all non-contiguous access patterns: sparse attention, paged KV cache, token selection
- TMEM moves the accumulator out of the register file, freeing up a large number of VGPRs for scalar operations like softmax
- Dual GEMM mode is suitable for scenarios with small head_dim (e.g., MLA's head_dim=64)

## Related Documents

- **tcgen05/TMEM**: [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) — Blackwell TMEM basics
- **Block-Scaled MMA**: [Block-Scaled MMA](block-scaled-mma.md) — FP8/FP4 quantized GEMM
- **MLA Decode**: [MLA Decode](mla-decode.md) — MLA inference optimization overview
- **Seesaw Scheduling (Hopper)**: [Seesaw Warpgroup Scheduling](../sm90/hands-on/seesaw-warpgroup-scheduling.md) — Hopper version MLA decode
