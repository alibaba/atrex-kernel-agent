# CUTLASS Convolution and Implicit GEMM Implementation Analysis

CUTLASS maps convolution operations to GEMM (implicit GEMM algorithm), eliminating the need Kov explicit im2col matrix construction and instead dynamically computing indices when loading data from global memory to shared memory. This document, based on CUTLASS source code and official documentation, analyzes the core design of convolution implementations across generations from Turing to Blackwell.

## Table of Contents

- [Implicit GEMM Concept](#implicit-gemm-concept)
- [Conv2d/3d to GEMM Mapping](#conv2d3d-to-gemm-mapping)
- [Three Convolution Operators](#three-convolution-operators-fpropdgradwgrad)
- [Layout: NHWC and GEMM Row/Column Major](#layout-nhwc-and-gemm-rowcolumn-major)
- [Stride / Dilation / Padding Handling](#stride-dilation-padding-handling)
- [Tile Iteration and Pointer Updates](#tile-iteration-and-pointer-updates)
- [Depthwise Convolution](#depthwise-convolution)
- [Grouped Convolution](#grouped-convolution)
- [Gather/Scatter Convolution (Example 59)](#gatherscatter-convolution-example-59)
- [Wgrad Optimization: Split-K](#wgrad-optimization-split-k)
- [CUTLASS 3.x's ConvProblemShape](#cutlass-3x-convproblemshape)
- [Hopper/Blackwell Convolution Collective](#hopperblackwell-convolution-collective)
- [Reference File Index](#reference-file-index)

---

## Implicit GEMM Concept

Traditional convolution implementations (im2col + GEMM) require explicit expansion of the input tensor, causing memory expansion equal to the filter size (R x S). Implicit GEMM eliminates this overhead:

```
English description
  Activation [N,H,W,C] --im2col--> ConvMatrix [NPQ, RSC] --GEMM--> Output [NPQ, K]
 ^^ memory R*S

Implicit GEMM:
 Activation [N,H,W,C] ----> SMEM tile
 Filter [K,R,S,C] --directload--> SMEM tile ---> Tensor Core GEMM ---> Output [N,P,Q,K]
 requiresmatrix, on-the-fly compute
```

Core idea: when loading threadblock tiles, compute the correct global memory addresses and out-of-bounds predicates by resolving the mapping from GEMM coordinates to convolution tensor coordinates.

---

## Conv2d/3d to GEMM Mapping

### 2D Convolution (Fprop)

Given `y[n,p,q,k] = sum_{c,r,s} x[n, p*stride_h + r*dilation_h - pad_h, q*stride_w + s*dilation_w - pad_w, c] * w[k,r,s,c]`

Mapped to `C = A * B`:

```
A (activations): row-major [NPQ x RSC]    // GEMM M = N*P*Q, GEMM K = R*S*C
B (filters):     col-major [RSC x K]      // GEMM N = K
C (output):      row-major [NPQ x K]
```

Coordinate Mapping:

```
GEMM_M dimension:
 i = q + Q * (p + P * n) // dimension
 mapping: n = i/(PQ), p = (i%PQ)/Q, q = i%Q

GEMM_K dimension:
 gemm_k = s + S * (r + R * c) // filter dimension
 mapping: c = gemm_k/(RS), r = (gemm_k%RS)/S, s = gemm_k%S
```

### 3D Convolution

Adds the depth dimension D/T/Z:

```
Activation: [N,D,H,W,C]  -> GEMM A: [NZPQ x TRSC]
Filter:     [K,T,R,S,C]  -> GEMM B: [TRSC x K]
Output:     [N,Z,P,Q,K]  -> GEMM C: [NZPQ x K]
```

CuTe's im2col transform makes 3D convolution expressions very natural:

```cpp
// im2col activation layout:
// ((N,(Z,P,Q)), (C,(T,R,S))) => idx    // logical (M, K)
auto xformed_act_layout = make_layout(
  make_shape (make_shape (     N,     Z,   P, Q), make_shape (  C,     T,   R, S)),
  make_stride(make_stride(D*H*W*C, H*W*C, W*C, C), make_stride(_1{}, H*W*C, W*C, C)));
```

---

## Three Convolution Operators: Fprop/Dgrad/Wgrad

CUTLASS implements three convolution operators by swapping the mapping of A/B/C to Activation/Filter/Output:

| Operator | A (GEMM Left Matrix) | B (GEMM Right Matrix) | C (GEMM Output) | Purpose |
|----------|----------------------|-----------------------|-----------------|---------|
| **Fprop** | Activation | Filter | Output | Forward propagation |
| **Dgrad** | Output | Filter | Activation | Backward data gradient |
| **Wgrad** | Output | Activation | Filter | Backward weight gradient |

```cpp
enum class Operator {
 kFprop, // forward: y = conv(x, w)
 kDgrad, // datagradient: dx = conv_transpose(dy, w)
 kWgrad, // weightgradient: dw = conv(x^T, dy)
 kDeconv //
};
```

> Note: CUTLASS coding conventions require not mixing (A,B,C) and (Activation,Filter,Output); choose one naming system.

## Layout: NHWC and GEMM Row/Column Major

CUTLASS convolution recommends the **NHWC** layout (channel-last) because:

1. The Channel dimension is contiguous, allowing 128-bit vectorized loads
2. It naturally maps to the K dimension of GEMM (contiguous)
3. It meets the alignment requirements for Tensor Cores

Optimal performance conditions:
- All tensors are 128-bit aligned
- Channel (C) is a multiple of 32
- Filter count (K) is a multiple of 32

```cpp
// CUTLASS 2.x definition
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutC      = cutlass::layout::TensorNHWC;
```

In CUTLASS 3.x, layouts are expressed via CuTe stride tuples and are no longer restricted to NHWC. The ConvProblemShape tensor defaults to layout-right (NDHWC), but supports arbitrary strides:

```cpp
// CUTLASS 3.x ConvProblemShape
TensorExtent shape_A{};     // [n,d,h,w,c]
TensorStride stride_A{}; // stride, default packed right-major
```

---

## Stride / Dilation / Padding Handling

### Address Calculation Formula

For Fprop, given an output position (p,q) and a filter position (r,s), the input position is:

```
h = p * stride_h + r * dilation_h - pad_h
w = q * stride_w + s * dilation_w - pad_w
```

### Padding Predicate Handling

Out-of-bounds access is handled via a predicate mask—the corresponding position loads 0 instead of actual data:

```cpp
// valid checkactivationtensor
bool valid = (h >= 0 && h < H && w >= 0 && w < W);
```

### Expression in ConvProblemShape

CUTLASS 3.x supports asymmetric padding:

```cpp
struct ConvProblemShape {
 ShapePadding lower_padding{}; // [pad_d, pad_h, pad_w] padding
 ShapePadding upper_padding{}; // [pad_d, pad_h, pad_w] padding
  TraversalStride traversal_stride{};  // [stride_d, stride_h, stride_w]
  ShapeDilation dilation{};            // [dilation_d, dilation_h, dilation_w]
  int groups = 1;
};
```

### Encoding Stride/Dilation via the im2col Transform (CuTe)

In CuTe's layout algebra, stride and dilation are directly encoded in the stride tuple:

```
ZPQ strides = DHW strides * traversal_stride // sampling
TRS strides = DHW strides * dilation // filter
ZPQ shape   = floor((DHW + pad - ((TRS-1) * dilation + 1)) / traversal_stride) + 1
```

---

## Tile Iteration and Pointer Updates

### GEMM K-Dimension Traversal Order

Implicit GEMM iterates along the K dimension (RSC / TRSC), processing one segment of channels at each filter position per iteration:

```
for each threadblock tile along K:
 advance_s // s -> s+1, filter width
  if s == S:
 advance_r // r -> r+1, s = 0, filter height
    if r == R:
 advance_c // c += tile_K * split_k, r = 0, channel
```

### Analytic vs Optimized Iterators

CUTLASS provides two iterator strategies:

**Analytic Iterator** (general but slow): Every `at()` call computes the full convolution coordinate mapping, including div/mod operations.

**Optimized Iterator** (high performance): Precomputes a pointer delta table, and the device side only needs to look up the table and add offsets:

```cpp
// Host compute (Conv2dFpropActivationIteratorOptimizedParams)
inc_next[0] = stride[0] * dilation_w;                          // next S
inc_next[1] = stride[1] * dilation_h - (S-1) * stride[0] * dilation_w;   // next R
inc_next[2] = tile_K * split_k - (R-1)*stride[1]*dilation_h
                                - (S-1)*stride[0]*dilation_w;   // next C batch

// Device : + pointeraddition
void advance() {
  ++filter_s_;
  if (filter_s_ == S) { filter_s_ = 0; ++filter_r_; next_idx = 1; }
  if (filter_r_ == R) { filter_r_ = 0; next_idx = 2; }
  add_byte_offset_(params_.inc_next[next_idx]);
}
```

The optimized iterator also uses fast divmod instead of standard division to map GEMM M to NPQ.

---

## Depthwise Convolution

Depthwise convolution is a special case of grouped convolution, where each channel is convolved independently (`C == K == groups`):CUTLASS provides `default_depthwise_fprop.h` specifically for handling depthwise scenarios. Optimizations include:
- The number of channels processed per CTA equals the N dimension of the CTA tile
- Simplified filter loading (only one filter per channel)
- No cross-channel reduction required

---

## Grouped Convolution

Grouped convolution divides the C and K dimensions into G groups, with each group performing convolution independently:

```
Group g :
  Input channels:  [g * C/G, (g+1) * C/G)
  Output channels: [g * K/G, (g+1) * K/G)
```

CUTLASS supports this through two modes:

- **SingleGroup**: When C/G and K/G are large, each CTA processes a sub-tile of a single group
- **MultipleGroup**: When C/G and K/G are small, a single CTA spans multiple groups

`default_conv2d_group_fprop.h` implements group-aware tile iteration.

---

## Gather/Scatter Convolution (Example 59)

Example 59 demonstrates a CuTe-based Ampere 3D convolution kernel that supports both dense and gather/scatter modes.

### Core Idea

Leveraging CuTe's **composed layout**, index indirection is encoded as an outer layout:

```cpp
// Dense mode: standard im2col layout
auto xformed_act_layout = make_layout(
  make_shape (make_shape (N, Z, P, Q), make_shape (C, T, R, S)),
  make_stride(make_stride(D*H*W*C, H*W*C, W*C, C), make_stride(_1{}, H*W*C, W*C, C)));

// Gather mode: use composed layout + IndexedGather
// layout: logical shape (M,K) mapping (idx_buffer_idx, dense_offset) codomain
auto inner = make_layout(
  make_shape (make_shape (N, Z, P, Q), make_shape (C, T, R, S)),
  make_stride(make_stride(D*H*W*E<0>{}, H*W*E<0>{}, W*E<0>{}, E<0>{}),
              make_stride(E<1>{},        H*W*E<0>{}, W*E<0>{}, E<0>{})));

// layout: (idx_buffer_idx, dense_offset) -> actual
// IndexedGather: addr = base + gather_idx[idx_buffer_idx] + dense_offset
auto outer = make_layout(make_shape(_1{},_1{}),
  make_stride(CustomStride{IndexedGather{idx_buf}, C}, _1{}));

auto gathered_layout = composition(outer, make_arithmetic_tuple(_0{},_0{}), inner);
```

Key advantage: **the kernel code remains completely unchanged**—dense and gather/scatter differ only in input layout, and CuTe's layout algebra handles all index calculations automatically.

### Performance

On the RTX 3080 Ti, gather/scatter convolution performs on par with or better than dense convolution (approximately 32 TFLOPS vs 30 TFLOPS), because statically known shapes eliminate most div/mod computations.

---

## Wgrad Optimization: Split-K

The weight gradient computation is unique in that the reduction occurs along the spatial dimension (NPQ). When NPQ is large, Split-K can be used to parallelize:

```cpp
enum class SplitKMode {
  kNone,
 kSerial, // CTA split
 kParallel // different CTA rowdifferent split, reduce
};
```

Parallel Split-K workflow:
1. Divide the NPQ dimension into `split_k_slices` parts
2. Each CTA computes a partial weight gradient
3. An additional reduction kernel accumulates all partial results

GEMM mapping for Wgrad:

```
A = Output [NPQ x K] -> GEMM A (A )
B = Activation [NPQ x C] -> GEMM B (B )
C = Filter [K x RSC] -> GEMM C (output)

Split-K GEMM M (= NPQ) dimensionsplit
```

---

## CUTLASS 3.x ConvProblemShape

CUTLASS 3.x uses the rank-agnostic `ConvProblemShape` to unify 2D/3D convolutions:

```cpp
template <conv::Operator ConvOp_, int NumSpatialDimensions_>
struct ConvProblemShape {
 static constexpr int RankS = NumSpatialDimensions_; // dimension (2 or 3)
 static constexpr int RankT = NumSpatialDimensions_ + 2; // tensordimension

  conv::Mode mode{};               // kCrossCorrelation / kConvolution
 TensorExtent shape_A{}; // tensor A shape
 TensorStride stride_A{}; // tensor A stride ( packed)
 TensorExtent shape_B{}; // tensor B shape
  TensorStride stride_B{};
  TensorExtent shape_C{};
  TensorStride stride_C{};

 ShapePadding lower_padding{}; // padding
  ShapePadding upper_padding{};
  TraversalStride traversal_stride{};
  ShapeDilation dilation{};
  int groups = 1;
};
```Depending on the `ConvOp`, `shape_A/B/C` automatically maps to the correct Activation/Filter/Output.

---

## Hopper/Blackwell Convolution Collective

### SM90 (Hopper): GMMA SS Warpspecialized

```
include/cutlass/conv/collective/sm90_implicit_gemm_gmma_ss_warpspecialized.hpp
```

Uses Hopper's TMA + WGMMA SS instructions, supporting Conv2d/3d Fprop/Dgrad/Wgrad. The Collective layer handles im2col address computation, and the mainloop reuses the GEMM pipeline structure.

### SM100 (Blackwell): UMMA Warpspecialized

```
include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp
```

Uses Blackwell's UMMA instructions and TMEM accumulator. Key features:

```cpp
template <conv::Operator ConvOp, int Stages, int NumSpatialDims, ...>
struct CollectiveConv<MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<...>, ...> {
 // PipelineTmaUmmaAsync: TMA load UMMA computeasynchronous pipeline
  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShapeMNK>;

 // 1SM 2SM mode
  using AtomThrShapeMNK = Shape<decltype(shape<0>(TiledMma::ThrLayoutVMNK{})), _1, _1>;

 // im2col passed stride tuple
  using StrideA = decltype(detail::sm100_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideB = decltype(detail::sm100_dispatch_policy_to_stride_B<DispatchPolicy>());
};
```

### Device-level Kernel

```
include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp
include/cutlass/conv/kernel/sm90_implicit_gemm_tma_warpspecialized.hpp
include/cutlass/conv/kernel/conv_universal.hpp //
```

---

## Reference File Index

### Documentation

| File | Content |
|------|------|
| `media/docs/cpp/implicit_gemm_convolution.md` | Official Implicit GEMM convolution documentation |

### Core Header Files (`include/cutlass/conv/`)

| File | Content |
|------|------|
| `convolution.h` | Operator/Mode/GroupMode enum definitions |
| `convnd_problem_shape.hpp` | CUTLASS 3.x rank-agnostic problem shape |
| `conv2d_problem_size.h` | CUTLASS 2.x Conv2d problem size |
| `collective/collective_conv.hpp` | CollectiveConv base class + specialization dispatch |
| `collective/sm90_implicit_gemm_gmma_ss_warpspecialized.hpp` | Hopper im2col collective |
| `collective/sm100_implicit_gemm_umma_warpspecialized.hpp` | Blackwell im2col collective |
| `kernel/conv_universal.hpp` | Generic conv kernel wrapper |
| `kernel/default_conv2d_fprop.h` | Conv2d Fprop default config |
| `kernel/default_conv2d_dgrad.h` | Conv2d Dgrad default config |
| `kernel/default_conv2d_wgrad.h` | Conv2d Wgrad default config |
| `kernel/default_depthwise_fprop.h` | Depthwise conv default config |
| `kernel/default_conv2d_group_fprop.h` | Grouped conv default config |

### Examples

| Example | Content |
|---------|------|
| `09_turing_tensorop_conv2dfprop` | Turing INT4 Conv2d fprop |
| `59_ampere_gather_scatter_conv` | CuTe 3D Conv + Gather/Scatter |

### Related Documentation

- [CUTLASS GEMM Optimization](cutlass-gemm-optimization.md)
- CUTLASS Programming Model
- [SM100 CuTeDSL Programming](sm100/blackwell-cutedsl-sm100.md)
