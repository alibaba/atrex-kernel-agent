# sm_120 cute 4.4.2 TMA + warp-spec implementation pitfalls

Discovered while landing TMA G2S + PipelineTmaAsync + 1-producer/8-consumer

**Last updated**: 2026-06-30

warp specialisation on sm_120 (NVIDIA RTX PRO 5000 / Blackwell-Geforce) under
CuTeDSL 4.4.2. None of these are documented in the existing flashinfer
`NVFP4QuantizeTMAKernel` (which is the closest reference) because that kernel
relies on class-method scoping that hides them.

## 1. `from __future__ import annotations` clashes with `@cute.struct`

### symptom

Module that contains a `@cute.struct` class fails at class-definition time:

```
TypeError: Struct element only support struct/array/base_dsl scalar,
  but got cute.struct.MemRange[cutlass.Int64, _TMA_NUM_STAGES]
```

### cause

`@cute.struct.__init__` iterates the class annotations as **real Python type
objects** (it uses `cls.__annotations__` and dispatches on `MemRange[T, N]`
specialisations). PEP 563 (`from __future__ import annotations`) converts all
annotations to **lazy strings**, so cute receives `"cute.struct.MemRange[..., N]"`
as a literal string and rejects it.

### fix

Do NOT use `from __future__ import annotations` at the top of any module that
declares a `@cute.struct` class. flashinfer kernels happen to not use it, which
is why this trap is invisible there.

## 2. `@cute.struct class` cannot be passed as `@cute.kernel` argument

### symptom

```
DSLRuntimeError: failed to generate argument #N (SharedStorage)
  for JIT function 'kernel_...'.
  Argument SharedStorage: The DSL attempted to convert it into Dynamic Expression
  (aka MLIR values) but failed.
  Call-site argument value: <cutlass.cute.core.struct object at 0x...>
```

### cause

Even with `SharedStorage: cutlass.Constexpr` annotation, cute 4.4.2 still tries
to lower the struct class into a dynamic SSA value when it appears in the
@cute.kernel signature.

### fix

Reference `SharedStorage` via **lexical scope** instead of as a kernel argument
— the same pattern as flashinfer's `self.shared_storage` (defined on the class,
read inside the @cute.kernel method body).

For module-level kernels (no `self`), define `SharedStorage` in module scope
(but see trap #3 below for why a factory function is also required).

## 3. `cute.struct.MemRange[T, N]` annotations only evaluate correctly at function-call time

### symptom

Defining `@cute.struct class _MyStorage` at module top level fails with:

```
TypeError: Struct element only support struct/array/base_dsl scalar,
  but got cute.struct.MemRange[cutlass.Int64, _MY_CONSTANT]
```

even when `_MY_CONSTANT` is a plain Python `int` (not a Constexpr).

### cause

When the class body executes at module import time, cute 4.4.2 sees
`MemRange[T, N]` as an unspecialised parameterised type rather than a specialised
one. Inside a function call, the same expression specialises correctly.

### fix

Wrap the `@cute.struct` class definition in a module-level **factory function**
that is called at module import time:

```python
def _make_v1_tma_shared_storage_class():
    @cute.struct
    class _V1TmaSharedStorage:
        load_full_mbar:  cute.struct.MemRange[cutlass.Int64, _TMA_NUM_STAGES]
        load_empty_mbar: cute.struct.MemRange[cutlass.Int64, _TMA_NUM_STAGES]
        attn_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _TOTAL_SMEM_ELEMS],
            _BUFFER_ALIGN_BYTES,
        ]
        gate_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _TOTAL_SMEM_ELEMS],
            _BUFFER_ALIGN_BYTES,
        ]
    return _V1TmaSharedStorage

_V1TmaSharedStorage = _make_v1_tma_shared_storage_class()
```

flashinfer hides this because its SharedStorage is defined inside `__call__`
(a method body that runs at call time, not import time).

## 4. `cute.ceil_div(dynamic_int, Int32(N))` fails MLIR legalization

### symptom

```
MLIRError: failed to legalize operation 'cute.derefine'
  (cute.derefine of !cute.tile<"N:1"> -> !cute.tile<"?:1">)
```

at `cute.ceil_div(padded_M, Int32(_TMA_ROW_TILE))`.

### cause

Wrapping a known module-level `int` literal in `Int32(...)` demotes it from a
static tile spec to a dynamic value. cute's `ceil_div` then has to derefine
the static-1 tile to a dynamic-1 tile, which the MLIR pass marked illegal.

### fix

Pass the second argument as a **plain Python int** literal:

```python
num_row_tiles = cute.ceil_div(padded_M, _TMA_ROW_TILE)   # NOT Int32(_TMA_ROW_TILE)
```

Same applies to `cute.local_tile(t, (...), (offset_dynamic, 0))` — tile
dims must be Python int, not Int32-wrapped. (See related lesson on
`make_tiled_copy_tv` / `make_tiled_tma_atom` not accepting Constexpr-typed
shape values, which forced module-level literal hoisting in V1 cp.async too.)

## 5. `cute.compile(launcher)` returns a callable whose runtime signature drops Constexpr args

### symptom

```
DSLRuntimeError: input args/kwargs length does not match runtime function signature!
  input args length: 11
  function signature args length: 8
```

when calling `compiled(...)` with the same argument list as `cute.compile(...)`.

### cause

`cute.compile` bakes Constexpr arguments into the kernel at compile time. The
returned callable has a runtime signature that contains ONLY the dynamic args
(Tensor, Int32, etc.). Passing Constexpr args at runtime is rejected.

### fix

Strip Constexpr args from the call to the compiled callable:

```python
compiled = cute.compile(
    launcher,
    mAttn, mGate, mOut, mScl, Int32(M), Int32(padded_M), Int32(num_blocks), mGS,
    num_sf_blocks_per_row, padded_sf_cols, num_col_chunks,   # Constexpr
)
compiled(
    mAttn, mGate, mOut, mScl, Int32(M), Int32(padded_M), Int32(num_blocks), mGS,
    # NO Constexpr args here
)
```

## evidence + reproduction

All five traps were hit and worked around in commit `71f84d8` of the
`kernel_opt_attn_fp4_fusion` working tree (Path-1 fused sigmoid·gate + NVFP4
quantization on bf16 attn_out + bf16 gate, K=4096, M=6144, sm_120 RTX PRO 5000).

Source:
- `kernel.py` — `@cute.kernel fused_sigmoid_mul_nvfp4_kernel` and
  `@cute.jit fused_sigmoid_mul_nvfp4_launch` show all five workarounds together.
- `cute_helpers.py` — `get_smem_ptr_as_int32` + `ld_shared_v4_u32` (consumer-warp
  PTX inline helpers required for SWIZZLE_128B SMEM addressing).

## affected versions

- CuTeDSL 4.4.2 (nvidia_cutlass_dsl_libs_base 4.4.2)
- ncu 2025.2.1
- sm_120 / sm_120a (Blackwell-Geforce, NVIDIA RTX PRO 5000 / 4000)
- driver 580.105.08

Untested on Hopper (sm_90) — flashinfer's NVFP4QuantizeTMAKernel runs there
without hitting these traps, suggesting the cute IR for MemRange / @cute.struct
may behave differently per arch backend.


## Related

- [Vendoring `flash_attn.cute` on cutlass <4.5: API private-name rename trap](cute-442-vendor-flash-attn-pitfalls.md)
- [GDN Chunk Forward Pitfalls (CuTeDSL, SM120)](gdn-chunk-fwd-pitfalls.md)
- [CuTeDSL GDN Decode on sm_120 — Pitfalls](gdn-decode-pitfalls.md)
- [cute-DSL NVFP4 GEMM pitfalls (sm_120, RTX PRO 5000)](nvfp4-gemm-pitfalls.md)
- [sm_120 trap: `vllm.vllm_flash_attn.flash_attn_varlen_func` has no fast path on Blackwell-Geforce](sm120-flash-attn-vllm-no-fast-path.md)
- [Software Pipeline Depth Optimization](../../../common/software-pipeline-depth-optimization.md)
- [Composable Kernel (CK) Architecture Overview](../../../../amd/common/ck-architecture-overview.md)
