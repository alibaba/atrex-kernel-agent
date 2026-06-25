# CUTLASS Quantization and Block-Scaled GEMM

Quantization and block-scaled GEMM implementation in CUTLASS, covering FP8 fast accumulation, SM100/SM120 block-scaled MMA, SM90 blockwise scaling, mixed-input GEMM, runtime data types, and sub-byte type handling. These mechanisms form the core infrastructure for low-precision, high-performance GEMM.

## 1. FP8 Fast Accumulation

**Source File**: `include/cutlass/gemm/collective/fp8_accumulation.hpp`

FP8 tensor core accumulators have limited precision (FP8 TC accumulates internally with limited precision), and GEMM with a long K dimension leads to precision degradation. `GmmaFP8Accumulation` addresses this through **periodic promotion**: periodically merging partial results from the TC accumulator into the FP32 main accumulator.

### 1.1 Core Data Structures

```cpp
template <class EngineAccum, class LayoutAccum>
struct GmmaFP8Accumulation {
 TensorAccum accum_temp_; // TC temporary(rmem resident)
 uint32_t accum_promotion_interval_; // MMA executecount, must promote
 uint32_t mma_count_per_mainloop_iteration_; // k_tile MMA count
 uint32_t mma_count_; // currentexecute MMA count
 uint32_t reset_accum_flag_; // requireszero out TC
};
```

### 1.2 promote_core vs scale_core

Two merge strategies:

| Method | Operation | Instruction | Usage |
|------|------|------|------|
| `promote_core` | `accum(i) += accum_temp(i)` | FADD | Simple accumulation, no scale factor |
| `scale_core` | `accum(i) += accum_temp(i) * scale` | FFMA | Multiply by scale factor then accumulate |

```cpp
// promote: add(FADD )
void promote_core(TensorAccumOrig &accum_) {
  warpgroup_wait<0>();
  for (int i = 0; i < size(accum_); ++i) {
    accum_(i) += accum_temp_(i);
  }
}

// scale: multiply-add(FFMA ), scalar / per-element / scale
void scale_core(TensorAccumOrig &accum_, ElementAccumulator const &scale) {
  for (int i = 0; i < size(accum_); ++i) {
    accum_(i) += accum_temp_(i) * scale;
  }
}
```

`scale_core` has three overloads:
- **Scalar scale**: `accum += temp * scale` (single scale factor)
- **Per-element scale**: `accum += temp * scale(i)` (independent scale per element)
- **Dual scale**: `accum += temp * scaleA(i) * scaleB(i)` (A/B dual scale factors, used for blockwise scaling)

### 1.3 Conditional Triggering and Warp Synchronization

Use `__shfl_sync` to ensure all threads within the warpgroup consistently execute promote/scale:

```cpp
void promote_if_needed(TensorAccumOrig &accum_) {
  mma_count_ += mma_count_per_mainloop_iteration_;
  reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
  if (reset_accum_flag_) {
    promote_core(accum_);
    mma_count_ = 0;
  }
}
```

Flow: Check after every `mma_count_per_mainloop_iteration_` MMA operations; when `accum_promotion_interval_` is reached, execute promote/scale and reset the counter. After the mainloop ends, call `promote_residue_if_needed` / `scale_residue_if_needed` to handle any remaining elements.

### 1.4 Why Periodic Promotion is Needed

The FP8 tensor core's internal accumulator has only about 16-bit precision. When the K dimension is large, continuous low-precision accumulation leads to:
- Large values swallowing small values (catastrophic cancellation)
- Accumulation error growing linearly with K

Periodic promotion to FP32 limits the error to within `accum_promotion_interval` instead of across the entire K dimension.

---

## 2. Block-Scaled MMA (SM100 / Blackwell)

**Source File**: `include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp`

SM100 UMMA instructions natively support block-scaled formats, with scale factors directly participating in MMA computation, eliminating the need for software-mediated promote/scale flows.

### 2.1 Supported Formats

Pass data + scale factor pairs via `ElementPairA` / `ElementPairB`:

| Format | ElementA/B | ElementSF | SFVecSize | Description |
|------|-----------|-----------|-----------|------|
| NVFP4 | `float_e2m1_t` | `float_e4m3_t` | 16 | NVIDIA proprietary FP4 |
| MXFP4 | `float_e2m1_t` | `float_e8m0_t` | 32 | OCP MX FP4 |
| MXFP6 | `float_e2m3_t` / `float_e3m2_t` | `float_e8m0_t` | 32 | OCP MX FP6 |
| MXFP8 | `float_e4m3_t` / `float_e5m2_t` | `float_e8m0_t` | 32 | OCP MX FP8 |### 2.2 Separation of Data and Scale Factors

```cpp
using ElementPairA = ElementPairA_;  // cute::tuple<ElementA, ElementSF>
using ElementPairB = ElementPairB_;  // cute::tuple<ElementB, ElementSF>

// separatedata scale
using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
```

**Four TMA channels**: Data and scale factors use independent TMA descriptors:

```cpp
TMA_A tma_load_a; // A matrixdata
TMA_B tma_load_b; // B matrixdata
TMA_SFA tma_load_sfa; // A scale factors
TMA_SFB tma_load_sfb; // B scale factors
```

### 2.3 Sm1xxBlockScaledConfig and SFVecSize

```cpp
static constexpr int SFVecSize = TiledMma::SFVecSize;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
```

`SFVecSize` defines how many data elements each scale factor covers (e.g., for MXFP8, SFVecSize=32 means 32 FP8 elements share one E8M0 scale factor).

### 2.4 Scale Factor MMA (TiledMMA_SF)

Scale factors are applied via a specialized MMA atom `MMA_ScaleFactor`:

```cpp
using TiledMMA_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
                              Layout<Shape<_1,_1,_1>>,
                              Tile<Underscore,Underscore,Underscore>>;
```

Scale factors are copied from smem to TMEM via UTCCP (smem-to-tmem copy), then automatically applied in the MMA:

```cpp
// mma loop
cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
 tCtSFA(_,_,k_block), // A scale factor (TMEM)
 tCtSFB_mma(_,_,k_block)), // B scale factor (TMEM)
    tCrA(_,_,k_block,read_stage),
    tCrB(_,_,k_block,read_stage),
    accumulators);
```

### 2.5 IsOverlappingAccum

```cpp
static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;
```

When `IsOverlappingAccum=true`, the mainloop and epilogue can execute concurrently: the first MMA iteration is manually unrolled Hours before the MMA, `accumulator_pipeline.producer_acquire` is called to obtain a TMEM buffer, achieving pipeline overlap.

### 2.6 Shared Memory Layout

```cpp
struct TensorStorage : cute::aligned_struct<128, _0> {
  ArrayEngine<SmemAllocTypeA, cosize_v<SmemLayoutA>> smem_A;
  ArrayEngine<SmemAllocTypeB, cosize_v<SmemLayoutB>> smem_B;
  ArrayEngine<ElementSF, cosize_v<SmemLayoutSFA>> smem_SFA;
  ArrayEngine<ElementSF, cosize_v<SmemLayoutSFB>> smem_SFB;
};
```

Data and scale factors share the same pipeline stage and are synchronized via `PipelineTmaUmmaAsync`:

```cpp
static constexpr uint32_t TmaTransactionBytes = ABTmaTransactionBytes + SFTransactionBytes;
```

---

## 3. Blockwise Scaling (SM90 / Software FP8)

**Source file**: `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`

SM90 (Hopper) lacks hardware block-scaled MMA and requires a software implementation: via `GmmaFP8Accumulation`'s `scale_if_needed`, at each scale granularity boundary, the TC accumulator is multiplied by the per-block scale factor and then merged into the FP32 main accumulator.

### 3.1 Scale Granularity Parameters

```cpp
static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
static constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});

// ScalePromotionInterval = K_granularity / K_atom_size
static constexpr int ScalePromotionInterval = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});
```### 3.2 Scale Factor Loading

Scale factors are loaded from gmem to smem using `cp.async` (SM80 async copy):

```cpp
using CopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
using CopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
```

When there are enough scale factors (`>= ScaleTmaThreshold`), the system automatically switches to TMA loading for better efficiency:

```cpp
static constexpr bool IsTmaLoadSFA = ScaleMsPerTile >= ScaleTmaThreshold && ...;
static constexpr bool IsTmaLoadSFB = ScaleNsPerTile >= ScaleTmaThreshold && ...;
```

### 3.3 Per-Block Scale Application

In the MMA mainloop, different scale strategies are selected based on the scale dimensions in the M/N direction:

```cpp
// M=1, N=1: scalar scale
ElementBlockScale scale_ab = tCrSFA_local(_0{});
scale_if_needed(accum_local, accumulation, scale_ab);

// M>1, N=1: vector scale(per-M-block)
scale_if_needed(accum_local, accumulation, tCrSFA_local);

// M=1, N>1: vector scale(per-N-block)
scale_if_needed(accum_local, accumulation, tCrSFB_local);

// M>1, N>1: vector scale(per-MN-block)
scale_if_needed(accum_local, accumulation, tCrSFA_local, tCrSFB_local);
```

---

## 4. Mixed-Input GEMM (SM90)

**Source file**: `include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`

Mixed-input GEMM supports A and B using different data types (e.g., FP8 A × FP16 B), implemented via **GMMA RS mode**: the narrower-precision operand is loaded from smem to registers, undergoes format conversion, and then participates in the MMA.

### 4.1 Dispatch Policy

```cpp
MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule>
```

Uses `RmemA` (A-from-registers) mode: the A operand is first loaded to registers for format conversion, while the B operand is read directly from the smem descriptor.

### 4.2 ElementOptionalTuple

The A or B operand can be a tuple containing optional scale and zero point:

```cpp
// ElementAOptionalTuple = cute::tuple<ElementA, ElementScale, ElementZero>
using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
using ElementScale = ...; // optional scale factor
using ElementZero = ...; // optional zero point

static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^
              cute::is_tuple<ElementBOptionalTuple>::value,
    "Either A OR B must be a tuple.");
```

### 4.3 SwapAB Mechanism

Since GMMA RS mode only supports reading A from registers, when B is the operand being scaled, A and B are swapped internally:

```cpp
static constexpr bool SwapAB = !IsATransformed;
using SwappedElementA = conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
using SwappedStrideA  = conditional_t<!SwapAB, StrideA, StrideB>;
```

---

## 5. Runtime Data Types (SM100)

**Source file**: `include/cute/arch/mma_sm100_desc.hpp`

SM100 UMMA supports selecting data types at runtime without recompiling the kernel.

### 5.1 MXF8F6F4Format Enum

```cpp
enum class MXF8F6F4Format : uint8_t {
  E4M3 = 0,    // float_e4m3_t
  E5M2 = 1,    // float_e5m2_t
  E2M3 = 3,    // float_e2m3_t (6-bit)
  E3M2 = 4,    // float_e3m2_t (6-bit)
  E2M1 = 5,    // float_e2m1_t (4-bit)
 INVALID = 7 // runtime proxy bit
};
```

### 5.2 Format Fields in UMMA Descriptors

```cpp
// UMMA Descriptor bit
a_format_ : 3, // bit [7,10) - A
b_format_ : 3, // bit [10,13) - B
```

### 5.3 IsRuntimeDataType Constraint

```cpp
static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
              (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
              "ElementA and ElementB should be both runtime or both static.");
```Switch types at runtime by modifying the format field of the UMMA descriptor:

```cpp
if constexpr (IsRuntimeDataType) {
  tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
  tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
}
```

Use `type_erased_dynamic_float8_t` / `type_erased_dynamic_float6_t` / `type_erased_dynamic_float4_t` as compile-time proxy types.

---

## 6. SM120 Sub-Byte Types

**Source File**: `include/cutlass/float_subbyte.h`

SM120 (Blackwell GeForce) introduces sub-byte types that require special smem handling.

### 6.1 unpacksmem Types

```cpp
namespace cutlass::detail {
  struct float_e2m1_unpacksmem_t;  // 4-bit, MX FP4
  struct float_e2m3_unpacksmem_t;  // 6-bit, MX FP6 (E2M3 variant)
  struct float_e3m2_unpacksmem_t;  // 6-bit, MX FP6 (E3M2 variant)
}
```

These types are numerically identical to the standard `float_e2m1_t` types, but they instruct CUTLASS to use the **smem unpack** path: TMA loads sub-byte data into smem in a byte-aligned fashion, and the MMA instruction automatically unpacks it on read.

### 6.2 Special smem Allocation

When using the `unpacksmem` type, special smem allocation and TMA internal type adjustments are required:

```cpp
// TMA use MMA element type type
using TmaInternalElementA = conditional_t<IsF8F6F4, ElementAMma, ElementA>;

// sub-byte type smem use uint8_t (byte-aligned)
using SmemAllocTypeA = conditional_t<IsF8F6F4 && sizeof_bits_v<ElementAMma> < 8,
                                      uint8_t, ElementAMma>;
```

### 6.3 Type-Erased Dynamic Variants

Union types used in runtime data type scenarios:

```cpp
union type_erased_dynamic_float6_unpacksmem_t {
  float_e2m3_unpacksmem_t e2m3_unpacksmem;
  float_e3m2_unpacksmem_t e3m2_unpacksmem;
};

union type_erased_dynamic_float4_unpacksmem_t {
  float_e2m1_unpacksmem_t e2m1_unpacksmem;
};
```

---

## Architecture Comparison Summary

| Feature | SM90 (Hopper) | SM100 (Blackwell) | SM120 (GB GeForce) |
|---------|--------------|-------------------|---------------------|
| Block-scaled MMA | Software (GmmaFP8Accumulation) | Hardware UMMA | Hardware UMMA |
| Scale factor loading | cp.async / TMA | TMA + UTCCP to TMEM | TMA |
| Accumulator location | Registers | TMEM | TMEM |
| Runtime data type | Not supported | Supported (MXF8F6F4Format) | Supported |
| Sub-byte types | Not supported | Not supported | unpacksmem_t |
| Minimum precision | FP8 (E4M3/E5M2) | FP4 (E2M1) | FP4 (E2M1) |
| Mixed-input | RS mode + format conversion | Native support | Native support |

---

## Related Documents

- [CUTLASS GEMM Optimization](cutlass-gemm-optimization.md) — GEMM tiling and pipeline strategies
-  — Blackwell architecture CuTeDSL programming
- [Pipeline Patterns](cutedsl-pipeline-patterns.md) — TMA pipeline and warp specialization
- [Epilogue Visitor Tree](cutlass-epilogue-visitor-tree.md) — Includes block-scaled output support
