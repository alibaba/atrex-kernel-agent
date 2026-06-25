# CUTLASS Epilogue Visitor Tree (EVT)

CUTLASS's Epilogue Visitor Tree (EVT) is a composable fusion framework based on template metaprogramming, allowing arbitrary element-wise operation trees (such as `D = activation(alpha * acc + beta * C + bias)`) to be expressed during the GEMM epilogue stage, all fused into a single kernel execution. This document covers FusionOperation predefined operations, EVT template recursion patterns, leaf/compute/store nodes, and DAG fusion.

## 1. FusionOperation Predefined Operation Hierarchy

**Source File**: `include/cutlass/epilogue/fusion/operations.hpp`

`FusionOperation` is the base class for all predefined fusion operations, combining different functional features through an inheritance hierarchy.

### 1.1 Inheritance Tree

```
FusionOperation (base)
  |
  +-- ScaledAcc                            D = alpha * acc
  |     |
  |     +-- LinearCombination              D = alpha * acc + beta * C
  |           |
  |           +-- LinCombEltAct            D = activation(alpha * acc + beta * C)
  |           |     |
  |           |     +-- LinCombEltActBlockScaleFactor   + block scale output
  |           |
  |           +-- LinCombTopKSoftmaxCol    D = softmax(top_k(alpha * acc + beta * C))
  |           |
  |           +-- LinearCombinationGroupedWgrad   (grouped conv wgrad)
  |           |
  |           +-- LinCombBlockScaleFactor  D = alpha * acc + beta * C + block scale output
  |           |
  |           +-- LinCombPerRowBias        D = alpha * acc + beta * C + per-row bias
  |           |     |
  |           |     +-- LinCombPerRowBiasBlockScaleFactor   + block scale
  |           |     |
  |           |     +-- LinCombPerRowBiasEltAct  D = act(... + per-row bias)
  |           |           |
  |           |           +-- LinCombPerRowBiasEltActBlockScaleFactor   + block scale
  |           |           |
  |           |           +-- LinCombPerRowBiasEltActAux     + aux store
  |           |           |
  |           |           +-- PerRowLinCombPerRowBiasEltAct   + per-row alpha/beta
  |           |           |
  |           |           +-- ScaledLinCombPerRowBiasEltAct   + scale_a * scale_b
  |           |                 |
  |           |                 +-- ScaledLinCombPerRowBiasEltActAmaxAux  + amax + aux
  |           |
  |           +-- LinCombPerColBias        D = alpha * acc + beta * C + per-col bias
  |           |     |
  |           |     +-- LinCombPerColBiasBlockScaleFactor   + block scale
  |           |     |
  |           |     +-- LinCombPerColBiasEltAct  D = act(... + per-col bias)
  |           |           |
  |           |           +-- LinCombPerColBiasEltActBlockScaleFactor   + block scale
  |           |           |
  |           |           +-- LinCombPerColBiasEltActAux      + aux store
  |           |           |
  |           |           +-- PerColLinCombPerColBiasEltAct    + per-col alpha/beta
  |           |           |     |
  |           |           |     +-- PerColResAddPerColBiasEltAct  + residual add
  |           |           |
  |           |           +-- ScaledLinCombPerColBiasEltAct    + scale_a * scale_b
  |           |                 |
  |           |                 +-- ScaledLinCombPerColBiasEltActAmaxAux  + amax + aux
  |           |
  |           +-- LinCombDeEltAct          D = d_activation(dY, Z)  (backward)
  |                 |
  |                 +-- LinCombDeEltActDePerRowBias    + dBias reduction
```### 1.2 FusionOperation Base Class Metadata

```cpp
struct FusionOperation {
  using ElementOutput = void;
  using ElementCompute = void;

  // Source (C matrix) support
  static constexpr bool IsSourceSupported = false;
  static constexpr bool IsResidualSupported = false;

  // Scaling support
  static constexpr bool IsScaleFactorSupported = false;     // scale_a * scale_b
  static constexpr bool IsPerRowScaleSupported = false;
  static constexpr bool IsPerColScaleSupported = false;

  // Bias support
  static constexpr bool IsPerRowBiasSupported = false;
  static constexpr bool IsPerColBiasSupported = false;

  // Activation support
  static constexpr bool IsEltActSupported = false;

  // Auxiliary output support
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAbsMaxSupported = false;

  // Block-scaled output support (SM100/SM120)
  using ElementBlockScaleFactor = void;
  static constexpr int SFVecSize = 0;
  static constexpr bool IsBlockScaleSupported = false;
};
```

### 1.3 Block-Scaled Output

Block-scaled output generates scale factors in the epilogue, supporting multiple combinations:

```cpp
template<int SFVecSize_, class ElementOutput_, class ElementCompute_,
         class ElementBlockScaleFactor_, ...>
struct LinCombBlockScaleFactor
    : LinearCombination<...> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
};
```

---

## 2. EVT (Epilogue Visitor Tree) Pattern

**Source file**: `include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`

EVT implements tree fusion recursively via the `Sm90TreeVisitor<NodeOp, ChildOps...>` template.

### 2.1 Sm90TreeVisitor Core Mechanism

```cpp
template <class NodeOp, class... ChildOps>
struct Sm90TreeVisitor : Sm90VisitorImpl<ChildOps..., NodeOp> {
  // ...
  template <typename ElementAccumulator, int FragmentSize>
  CUTLASS_DEVICE auto
  visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
        int epi_v, int epi_m, int epi_n) {
    constexpr int Rm1 = sizeof...(ChildOps);
    return cute::detail::tapply(callbacks_tuple,
 // 1. child visit(transform stage)
      [&] (auto& child_callbacks) {
        return child_callbacks.visit(frg_acc, epi_v, epi_m, epi_n);
      },
 // 2. child result node op(apply stage)
      [&] (auto&&... frg_inputs) {
        return get<Rm1>(callbacks_tuple).visit(
          frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
      },
 make_seq<Rm1>{} // transform R-1 (children), apply (node)
    );
  }
};
```

**Execution flow**:
1. **begin** -- Pre-loop initialization (gmem broadcast, etc.)
2. **visit** -- Recursive evaluation per fragment: child results -> node operation
3. **end** -- Post-loop cleanup (gmem reduction, etc.)

### 2.2 Sm90VisitorImplBase

Base infrastructure for all visitors, managing the packing of ops tuples, SharedStorage, and Params:

```cpp
template <class... Ops>
struct Sm90VisitorImplBase {
  using SharedStorage = tuple<typename Ops::SharedStorage...>;
  using Arguments = tuple<typename Ops::Arguments...>;
  using Params = tuple<typename Ops::Params...>;
  tuple<Ops...> ops;
};
```

Provides hand-written template specializations for 1-4 Ops (to avoid aggregate initialization limitations).### 2.3 Consumer Store Callbacks Lifecycle

Each node generates a callbacks object via `get_consumer_store_callbacks`, with the full lifecycle:

| Callback Method | Timing | Typical Use |
|---------|------|---------|
| `begin()` | Before subtile loop | gmem broadcast load |
| `begin_loop(epi_m, epi_n)` | Start of each subtile | smem data preparation |
| `previsit(...)` | Before visit | smem broadcast |
| `visit(frg_acc, epi_v, epi_m, epi_n, ...)` | Each fragment | Element-wise computation |
| `reduce(smem_buffer, sync_fn, ...)` | After visit | smem reduction |
| `postreduce(...)` | After reduce | smem -> smem store (for TMA) |
| `tma_store(...)` | After smem fence | TMA store dispatch |
| `end_loop(epi_m, epi_n)` | End of each subtile | Direct gmem store |
| `end()` | After subtile loop | gmem reduction / final write-back |

---

## 3. Leaf Nodes

**Source file**: `include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp`

### 3.1 Sm90AccFetch

The simplest leaf node, directly returns the value of the MMA accumulator:

```cpp
struct Sm90AccFetch : Sm90VisitorImpl<> {
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementAccumulator, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v, int epi_m, int epi_n) {
 return frg_acc; // directreturns
    }
  };
};
```

### 3.2 Sm90SrcFetch\<Element\>

Loads the C matrix (source matrix), which has been preloaded into smem by epilogue collective operations via TMA and copied to registers:

```cpp
template <class Element>
struct Sm90SrcFetch : Sm90VisitorImpl<> {
  CUTLASS_DEVICE bool is_C_load_needed() const {
 return not is_void_v<Element>; // Element=void C load
  }

  CUTLASS_DEVICE bool is_zero() const {
    return is_void_v<Element>;
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    SrcTensor const& tCrC;  // register-resident C fragment

    visit(...) {
      return recast<Array<value_type, FragmentSize>>(tCrC)(epi_v);
    }
  };
};
```

### 3.3 Sm90ScalarBroadcast

Scalar broadcast, loads a single scalar value via gmem (supports batched different scalars and multi-scalar reduction):

```cpp
// StrideMNL = Stride<_0, _0, dL>: batch dimension stride
// BroadcastCount: scalarreduction
// ReductionFn: multiplies( scale_a * scale_b)
```

### 3.4 Sm90RowBroadcast / Sm90ColBroadcast

Row/column vector broadcast, loaded from gmem to smem via TMA, then broadcast to each element:

```cpp
// StrideMNL = Stride<_0, _1, dL>: N stride
// passed TMA load row vector smem, visit M
// StrideMNL = Stride<_1, _0, dL>: M stride
// N ```

---

## 4. Compute Nodes

**Source file**: `include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp`

### 4.1 Sm90Compute

A general-purpose N-ary element-wise computation node:

```cpp
template<
 template <class> class ComputeFn, // computefunction
 class ElementOutput, // outputtype
 class ElementCompute, // computeaccuracy
 FloatRoundStyle RoundStyle // mode
>
struct Sm90Compute {
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
 visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...) {// 1. inputconversion ElementCompute
// 2. ComputeFn compute
// 3. resultconversion ElementOutput return convert_output(compute_output(cvt_frg_inputs...));
    }
  };
};
````ComputeFn` can be any unary/binary/multi-ary function template, such as `multiplies`, `plus`, `ReLu`, etc.

### 4.2 beta * C + Z Performance Specialization

CUTLASS provides a specialized `Sm90TreeVisitor` for the most common `beta * C + Z` pattern Tracing back the key optimization is **skipping the C load when beta=0**:

```cpp
// : beta * C + Z
template <...>
struct Sm90TreeVisitor<
  Sm90Compute<homogeneous_multiply_add, ...>,
  InputScaleOp,          // beta
  Sm90SrcFetch<Element>, // C
  InputAddOp             // Z
> {
  CUTLASS_DEVICE bool is_C_load_needed() const {
// if beta , requiresload C
return (not scale_op.is_zero() && src_op.is_C_load_needed())  }

// compute Z
Array frg_I = convert_Z(frg_added);
// if C void, beta * C + Z
if constexpr (!is_void_v<ElementSource>) {};
```

This avoids unnecessary TMA loads and smem occupation.

---

## 5. Store Nodes

**Source File**: `include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp`

### 5.1 Sm90AuxStore

Auxiliary output store, which writes intermediate results back to gmem via TMA (requires an independent smem buffer):

```cpp
template <int Stages, class EpilogueTile, class Element, ...>
struct Sm90AuxStore {
  struct SharedStorage {
 array_aligned<Element, size(SmemLayout{})> smem_aux; // independent smem buffer
  };

  struct ConsumerStoreCallbacks {
// input conversion register buffer
tC_rAux_frg(epi_v) = convert_input(frg_input);
return frg_input;  // pass-through

// Register -> Smem (R2S copy)
copy(tiled_r2s, tRS_rAux, tRS_sAux(_,_,_,store_pipe_index));

// Smem -> Gmem (TMA store)
copy(tma_store_aux, bSG_sAux(_,_,_,pipe), bSG_gAux(_,_,_,epi_m,epi_n));
  };
};
```

The specialization of `Stages=0` uses direct gmem store (no smem buffer).

### 5.2 Sm90ScalarReduction

Scalar reduction (e.g., amax), writing back a single value via atomic operations:

```cpp
// visit: fragment register-level reduction
// end: passed atomic reduction gmem
```

### 5.3 Sm90RowReduction / Sm90ColReduction

Row/column vector reduction, supporting multi-level reduction:

1. **Warp shuffle reduction** -- warp-level `__shfl_down_sync` / swap-shuffle reduction
2. **Threadblock smem reduction** -- multi-warp reduction via smem
3. **Gmem atomic / workspace reduction** -- inter-CTA reduction via atomics or workspace
4. **Final reduction** -- the last CTA performs the final reduction and writes back the output

Key optimization: the **Swap Shuffle** algorithm, which optimizes the `O(N * log(N))` workload of the traditional shuffle-down approach so that each thread performs useful work at every step.

---

## 6. SplitTreeVisitor (DAG Fusion)

**Source File**: `include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`

### 6.1 Sm90SplitTreeVisitor

Expresses a DAG (Directed Acyclic Graph) fusion as one shared input tree plus multiple output trees:

```cpp
template <class InputTree, class OutputTree, class... AuxOutTrees>
struct Sm90SplitTreeVisitor : Sm90VisitorImpl<InputTree, AuxOutTrees..., OutputTree> {

  visit(frg_acc, epi_v, epi_m, epi_n) {
 // 1. sharedinput
    Array frg_input = get<0>(callbacks_tuple).visit(frg_acc, ...);

 // 2. resultoutput(side effects)
    for_each(make_seq<sizeof...(AuxOutTrees)>{}, [&](auto I) {
      get<I+1>(callbacks_tuple).visit(frg_input, ...);
    });

 // 3. resultmainoutput
// 3. resultmainoutput
return get<Rm2+1>(callbacks_tuple).visit(frg_input, ...);
```Typical use case: `D = activation(Z)` outputs both `Aux = Z` (the result before activation) and `D`.
### 6.2 Sm90TopologicalVisitor

A more general DAG visitor that supports operation graphs with arbitrary topological ordering:

```cpp
template <class ElementCompute, class EdgeTuple, class... Ops>
struct Sm90TopologicalVisitor : Sm90VisitorImpl<Ops...> {
 // EdgeTuple: Op column
 // by Op, resultcache frg_compute_tuple
};
```

---

## 7. SM100/SM120 Differences

**Source files**:
- `include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp`
- `include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp`
- `include/cutlass/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp`

### 7.1 SM100 (Blackwell)

SM100's FusionCallbacks **directly alias to the SM90 implementation**:

```cpp
template <...>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<...>, Operation, ...>
  : FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<...>, Operation, ...> {
 // SM90
};
```

The SM100 epilogue uses TMEM (Tensor Memory) instead of the register file to hold accumulators. The TMEM-to-smem copy path differs, but the logical interface of EVT nodes remains unchanged.

SM100 has separate `sm100_visitor_compute_tma_warpspecialized.hpp` and `sm100_visitor_store_tma_warpspecialized.hpp`, providing additional block-scaled output store nodes.

### 7.2 SM120 (Blackwell GeForce)

SM120 also aliases to SM90/SM100 implementations, but adds specialization for block-scaled output:

```cpp
// SM120 block-scaled output epilogue
// 1. 32 F32 max
// 2. conversion max UE8 (or UE4M3) store scale factor
// 3. UE8 F32 scale
// 4. MUFU
// 5. F32 , conversion ElementD
```

### 7.3 Architecture Comparison

| Feature | SM90 | SM100 | SM120 |
|------|------|-------|-------|
| Implementation file | `sm90_visitor_*` | `sm100_visitor_*` + SM90 alias | `sm120_visitor_*` + SM90 alias |
| Accumulator source | Register (RMEM) | TMEM | TMEM |
| Base EVT nodes | Native implementation | Inherits SM90 | Inherits SM90 |
| Block-scaled store | Not supported | Supported | Supported |
| Direct store (no smem) | Not supported | `Sm100NoSmemWarpSpecialized` | -- |

---

## EVT Composition Example

Building the EVT tree for `D = ReLU(alpha * acc + beta * C + row_bias)`:

```cpp
using EVT =
  Sm90TreeVisitor<
    Sm90Compute<ReLu, ElementD, float, round_to_nearest>,     // ReLU
    Sm90TreeVisitor<
      Sm90Compute<homogeneous_multiply_add, float, float, round_to_nearest>,  // beta*C + Z
      Sm90ScalarBroadcast<float>,                              // beta
      Sm90SrcFetch<ElementC>,                                  // C
      Sm90TreeVisitor<
        Sm90Compute<multiply_add, float, float, round_to_nearest>,  // alpha*acc + bias
        Sm90ScalarBroadcast<float>,                            // alpha
        Sm90AccFetch,                                          // accumulator
        Sm90RowBroadcast<...>                                  // row bias
      >
    >
  >;
```
Evaluation order (depth-first postorder traversal):
1. `Sm90AccFetch` -> returns acc
2. `Sm90ScalarBroadcast<alpha>` -> returns alpha
3. `Sm90RowBroadcast` -> returns bias
4. `multiply_add(alpha, acc, bias)` -> returns Z = alpha*acc + bias
5. `Sm90ScalarBroadcast<beta>` -> returns beta
6. `Sm90SrcFetch` -> returns C
7. `homogeneous_multiply_add(beta, C, Z)` -> returns beta*C + Z
8. `ReLu(result)` -> returns D

---

## Related Documents

- [CUTLASS GEMM Optimization](cutlass-gemm-optimization.md) -- mainloop design and tile strategies
-  -- Blackwell TMEM epilogue
- [Pipeline Patterns](cutedsl-pipeline-patterns.md) -- epilogue pipeline stages
- [Quantization & Block-Scaled GEMM](cutlass-quantization-block-scaled.md) -- block-scaled input MMA
