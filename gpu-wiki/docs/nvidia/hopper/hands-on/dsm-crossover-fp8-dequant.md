# DSM Crossover FP8 Dequantization

FlashMLA's Hopper FP8 Sparse Decode core optimization: leverage Distributed Shared Memory (DSM) within a Cluster, with two CTAs performing FP8→BF16 dequantization in a crossover manner, breaking through the dequantization throughput bottleneck of a single CTA.


**Last updated**: 2026-07-01

**Source**: FlashMLA `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`

---

## Background

In FP8 MLA Decode, each token requires:
1. Loading FP8 KV cache from HBM
2. FP8→BF16 dequantization (~50 cycles/token)
3. WGMMA matrix multiplication (~34 cycles/token)

**Dequantization is the bottleneck** (50 > 34 cycles), with Tensor Cores largely idle waiting for data.

## DSM Crossover Solution

Using **Cluster Size = 2**, two CTAs each handle dequantization for half of the tokens, writing results into each other's shared memory via DSM:

```
CTA 0 (SM_A):                       CTA 1 (SM_B):
┌─────────────────┐                 ┌─────────────────┐
│ 1. load FP8 tokens 0-31 │ │ 1. load FP8 tokens 32-63 │
│ 2. Dequant → BF16      │         │ 2. Dequant → BF16        │
│ 3. st.async → SM_B SMEM│─────────│ 3. st.async → SM_A SMEM  │
│ 4. WGMMA( 64 tokens)│ │ 4. WGMMA( 64 tokens) │
└─────────────────┘                 └─────────────────┘
```

### Key PTX Instructions

```cpp
// asynchronouswrite peer CTA shared memory
// st.async current, writebackgroundcomplete
asm volatile(
    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes"
    ".v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
 :: "r"(dst_smem_addr), // : peer CTA SMEM
 "r"(data0), "r"(data1), // quantization BF16 data
       "r"(data2), "r"(data3),
 "r"(mbar_addr) // peer CTA mbarrier
);

// peer CTA SMEM
// XOR 1 CTA 0 mapping CTA 1,
uint32_t peer_addr = __cluster_map_shared_rank(local_smem_ptr, rank ^ 1);
```

### FP8 Dequantization Implementation

```cpp
// FP8 E4M3 -> BF16, tile-level scale factor
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const __nv_bfloat162 &scale) {
 // 1. FP8 -> FP32
    float4 fp32x4 = (float4)(inputs.low);
 // 2. FP32 -> BF16 + scale
    output.low = __float22bfloat162_rn({fp32x4.x, fp32x4.y}) * scale;
    output.high = __float22bfloat162_rn({fp32x4.z, fp32x4.w}) * scale;
    return output;
}
```

### Scale Differences Between DeepSeek-V3 and V3.2

| Model | Scale Format | Tile Size | Scales per Token |
|------|-----------|----------|-----------------|
| V3/V3.1 (MODEL1) | E8M0 (8-bit) | 64 | 8 |
| V3.2 (V32) | FP32 | 128 | 4 |

## Performance Data

| Strategy | Tensor Core Utilization | TFLOPS (H800) |
|------|-------------------|---------------|
| No crossover (single CTA dequant) | ~40% | ~250 |
| **DSM crossover (2 CTA)** | **~65%** | **410** |

## Practical Experience

- DSM crossover doubles dequantization throughput (2 CTAs dequantize in parallel), making dequantization no longer the bottleneck
- `st.async` is key: asynchronous writes do not block computation, enabling overlap between dequantization and WGMMA
- Cluster Size must be 2 (larger clusters increase scheduling complexity with diminishing returns)
- This pattern can be generalized to any "preprocessing → matrix multiplication" scenario where preprocessing is the bottleneck

## Related

- **Cluster Reduction**: [Cluster-Level Reduction](cluster-level-reduction.md) — Thread Block Cluster fundamentals
- **Seesaw Scheduling**: [Seesaw Warpgroup Scheduling](seesaw-warpgroup-scheduling.md) — FlashMLA's BF16 Dense version
- **FP8 Conversion**: [Vectorized FP8 Conversion](vectorized-fp8-conversion.md) — PTX vectorized FP8 conversion
