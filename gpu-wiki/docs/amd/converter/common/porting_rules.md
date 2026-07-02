# General Gluon Operations Porting Rules (Applicability of each section is annotated: [Common] = General, [AMD CDNA3] = AMD-specific, [Hopper] = NVIDIA Hopper-specific)


**Last updated**: 2026-06-30

## Basic [Common]
Basic Gluon operations are similar to Triton's, such as `gl.program_id`, `gl.constexpr`, `gl.where`, `gl.cdiv`, `gl.load`, `gl.store` etc.). If basic operations are not implemented in gluon,
fall back to their triton implementations. For example, tl.log, tl.float32... can still be compiled in gluon kernel.

## Tensor Creation [Common]
Gluon Tensor creation operations have an additional field `layout`, such as tl.arange(0, 16) -> gl.arange(0, 16, layout=arange_layout); tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], tl.float32) -> gl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], gl.float32, mfma_layout)

## API Priority Rule (STRICT) [Common]
- ALWAYS use `gl.*` when an equivalent exists (`gl.load`, `gl.store`, `gl.exp`, `gl.where`, `gl.zeros`, `gl.arange`, `gl.cdiv`, `gl.cast`, `gl.expand_dims`, `gl.convert_layout`, `gl.float32`, `gl.int32`, `gl.bfloat16`, `gl.float16`, etc.)
- ONLY fall back to `tl.*` when there is genuinely NO `gl` equivalent (e.g., `tl.log2` if `gl.log2` doesn't exist).
- NEVER use `tl.make_block_ptr`. See "Memory Access Porting Pattern" below.
- For `@triton.heuristics`, keep using `triton` since it's a launcher decorator, not a kernel operation.

## CRITICAL: No Triton Block Pointer APIs [Common]
- DO NOT use `tl.make_block_ptr`, `tl.load` with block pointers, or `tl.store` with block pointers.
- ALL memory accesses must be implemented using manual offset computation with `gl.arange` (with explicit layout), `gl.expand_dims`, and mask construction.
- **AMD CDNA3**: Use `gl.amd.cdna3.buffer_load` / `gl.amd.cdna3.buffer_store`. Offset tensors must be `gl.int32`.
- **Hopper/Ampere**: Use `gl.load` / `gl.store` for pipelined loads into shared memory.
- For scalar accesses on all architectures: use `gl.load(ptr + offset)` / `gl.store(ptr + offset, val)`.

## Memory Access Porting Pattern (CRITICAL) [AMD CDNA3]

When the Triton kernel uses `tl.make_block_ptr` + `tl.load`/`tl.store`, you MUST replace it with manual offset + mask + buffer_load/buffer_store. Here is the canonical pattern:

**Triton (BEFORE):**
```python
p = tl.make_block_ptr(base, (M, N), (stride_m, stride_n), (row_off, col_off), (BLOCK_M, BLOCK_N), (1, 0))
val = tl.load(p, boundary_check=(0, 1))
```

**Gluon (AFTER):**
```python
# Choose layout from TTGIR for the load target (e.g., blocked2 for loads, mma for stores)
row_idx = gl.arange(0, BLOCK_M, layout=slice_load_layout_dim1)
col_idx = gl.arange(0, BLOCK_N, layout=slice_load_layout_dim0)
row_2d = gl.expand_dims(row_idx, axis=1)   # [BLOCK_M, 1]
col_2d = gl.expand_dims(col_idx, axis=0)   # [1, BLOCK_N]
offsets = (row_off + row_2d) * stride_m + (col_off + col_2d) * stride_n
mask = ((row_off + row_2d) < M) & ((col_off + col_2d) < N)
offsets_i32 = gl.cast(offsets, gl.int32)
val = gl.amd.cdna3.buffer_load(ptr=base, offsets=offsets_i32, mask=mask, other=0.0)
# If needed, convert layout: val_mma = gl.convert_layout(val, mma)
```
For stores, use `gl.amd.cdna3.buffer_store(stored_value=val, ptr=base, offsets=offsets_i32, mask=mask)`.

For scalar loads (e.g., `tl.load(ptr + offset)`), use `gl.load(ptr + offset)`.

For 1D loads, use the same pattern but with a 1D `gl.arange` and no `gl.expand_dims`:
```python
idx = gl.arange(0, BLOCK, layout=slice_layout)
offsets = (start + idx) * stride
mask = (start + idx) < bound
offsets_i32 = gl.cast(offsets, gl.int32)
val = gl.amd.cdna3.buffer_load(ptr=base, offsets=offsets_i32, mask=mask, other=0.0)
```

## Memory Access Porting Pattern (CRITICAL) [Hopper]

When the Triton kernel uses `tl.make_block_ptr` + `tl.load`/`tl.store`, you MUST decompose it into manual pointer arithmetic + `gl.load`/`gl.store`. Here is the canonical pattern:

**Triton (BEFORE):**
```python
p = tl.make_block_ptr(base, (M, N), (stride_m, stride_n), (row_off, col_off), (BLOCK_M, BLOCK_N), (1, 0))
val = tl.load(p, boundary_check=(0, 1))
```

**Gluon (AFTER) — Direct load via pointer arithmetic:**
```python
row_idx = gl.arange(0, BLOCK_M, layout=slice_layout_dim1)
col_idx = gl.arange(0, BLOCK_N, layout=slice_layout_dim0)
row_2d = gl.expand_dims(row_idx, axis=1)   # [BLOCK_M, 1]
col_2d = gl.expand_dims(col_idx, axis=0)   # [1, BLOCK_N]
ptrs = base + (row_off + row_2d) * stride_m + (col_off + col_2d) * stride_n
mask = ((row_off + row_2d) < M) & ((col_off + col_2d) < N)
val = gl.load(ptrs, mask=mask, other=0.0)
```

**Gluon (AFTER) — Pipelined load via async_copy (preferred for performance):**
```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy
# Allocate shared memory destination
smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], shared_layout)
# Build pointer tensor
ptrs = base + (row_off + row_2d) * stride_m + (col_off + col_2d) * stride_n
# Async DMA: global → shared (bypasses registers, ~2x faster)
async_copy.async_copy_global_to_shared(smem, ptrs, mask=mask)
async_copy.async_copy_wait()
val = smem.load(target_layout)
```

For stores, use `gl.store(ptrs, val, mask=mask)`.

For 1D loads:
```python
idx = gl.arange(0, BLOCK, layout=slice_layout)
ptrs = base + (start + idx) * stride
mask = (start + idx) < bound
val = gl.load(ptrs, mask=mask, other=0.0)
```

## Dot Product Porting Pattern (CRITICAL for AMD) [AMD CDNA3]

When the Triton kernel uses `tl.dot(a, b)`, you MUST:
1. Store both operands into shared memory using `gl.allocate_shared_memory`
2. Load them with `DotOperandLayout`
3. Call `gl.amd.cdna3.mfma`

```python
# a: [M, K] in some layout, b: [K, N] in some layout
a_smem = gl.allocate_shared_memory(a.dtype, [M, K], shared_layout, value=a)
a_dot = a_smem.load(dot_op0)  # DotOperandLayout(operand_index=0, parent=mma, k_width=4)
b_smem = gl.allocate_shared_memory(b.dtype, [K, N], shared2_layout, value=b)
b_dot = b_smem.load(dot_op1)  # DotOperandLayout(operand_index=1, parent=mma, k_width=4)
acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc_init)
```

The `shared_layout` and `shared2_layout` should be taken from the TTGIR (look for `#shared` and `#shared2` definitions). The `k_width` in `DotOperandLayout` should match the TTGIR's `#ttg.dot_op<{..., kWidth = N}>`.

## Layout Selection from TTGIR (IMPORTANT) [Common]

> NOTE: AMD uses amdg.buffer_load/buffer_store operations, while Hopper uses different async copy and TMA operations.

When translating, inspect the TTGIR to determine which layout each operation uses:
- **Global memory loads**: Look at `amdg.buffer_load` result type (e.g., `#blocked2`, `#blocked1`, `#blocked`)
- **Global memory stores**: Look at `amdg.buffer_store` input type (e.g., `#mma`)
- **Accumulators / MMA results**: Always `#mma` layout
- **1D vectors** (e.g., gate values): Use `gl.SliceLayout(dim, parent_layout)` matching the TTGIR slice
- **Shared memory**: Use the `#shared` / `#shared2` layouts from TTGIR
- **Dot operands**: Use `gl.DotOperandLayout` with `k_width` from TTGIR's `#ttg.dot_op`

The `gl.arange` layout for offset computation should match the layout of the tensor being loaded/stored.
For loads into `#blocked2`, use `SliceLayout(dim, blocked2)`. For stores from `#mma`, use `SliceLayout(dim, mma)`.

## Type Casting [Common]
- `tensor.to(gl.bfloat16)` works in Gluon (inherited from Triton tensor method).
- Alternatively use `gl.cast(tensor, gl.bfloat16)`.
- For dtype references, prefer `gl.float32`, `gl.bfloat16`, `gl.int32` over `tl.*` equivalents.
- For accessing pointer element type, `k.dtype.element_ty` still works.

## Operations Specific to Gluon and AMD [AMD CDNA3]
* gl.amd.cdna3.buffer_load(ptr=ptr, offsets=offsets, mask=mask, other=0.0), this is similar to gl.load, but with a base offset specified by ptr field. offsets must be int32.
* gl.amd.cdna3.buffer_store(stored_value=val, ptr=ptr, offsets=offsets, mask=mask), this is similar to gl.store, but with a base offset specified by ptr field. offsets must be int32.
* gl.amd.cdna3.mfma(input, other, acc), this is similar to gl.dot(input, other, acc)

# Complete Gluon API Reference

## 1. Decorators [Common]

* **`@gluon.jit`** → `@triton.jit`
  * Purpose: Mark a function as a JIT-compiled GPU kernel
  * Difference: Gluon uses `Language.GLUON` and generates `.ttgir` files; Triton generates `.ttir` files

* **`@gluon.constexpr_function`** → `@triton.constexpr_function`
  * Purpose: Mark a compile-time constant function
  * Difference: None, identical

## 2. Core Data Types [Common]

* **`gl.constexpr`** → `tl.constexpr`
  * Purpose: Compile-time constant
  * Difference: None

* **`gl.tensor`** → `tl.tensor`
  * Purpose: Tensor type
  * Difference: Gluon tensors must specify layout via `distributed_type`

* **`gl.distributed_type(element_ty, shape, layout)`** → `tl.block_type(element_ty, shape)` (no direct mapping)
  * Purpose: Define distributed tensor type with explicit layout
  * Difference: **Gluon-specific**, Triton infers layout automatically
  * Parameters:
    ```python
    element_ty: dtype  # Element data type
    shape: List[int]   # Tensor shape
    layout: DistributedLayout  # Layout specification (e.g., BlockedLayout, AutoLayout)
    ```

* **`gl.shared_memory_descriptor`** → No direct mapping
  * Purpose: Shared memory descriptor handle
  * Difference: **Gluon-specific**, for explicit shared memory management

* **`gl.shared_memory_descriptor_type(element_ty, shape, layout, alloc_shape)`** → No direct mapping
  * Purpose: Type for shared memory descriptors
  * Difference: **Gluon-specific**
  * Parameters:
    ```python
    element_ty: dtype          # Element data type
    shape: List[int]           # Logical shape
    layout: SharedLayout       # Shared memory layout
    alloc_shape: List[int]     # Physical allocation shape
    ```

## 3. Primitive Types [Common]

All identical to Triton:
* `gl.void` → `tl.void`
* `gl.int1` → `tl.int1`
* `gl.int8` → `tl.int8`
* `gl.int16` → `tl.int16`
* `gl.int32` → `tl.int32`
* `gl.int64` → `tl.int64`
* `gl.uint8` → `tl.uint8`
* `gl.uint16` → `tl.uint16`
* `gl.uint32` → `tl.uint32`
* `gl.uint64` → `tl.uint64`
* `gl.float8e5` → `tl.float8e5`
* `gl.float8e5b16` → `tl.float8e5b16`
* `gl.float8e4nv` → `tl.float8e4nv`
* `gl.float8e4b8` → `tl.float8e4b8`
* `gl.float8e4b15` → `tl.float8e4b15`
* `gl.float16` → `tl.float16`
* `gl.bfloat16` → `tl.bfloat16`
* `gl.float32` → `tl.float32`
* `gl.float64` → `tl.float64`
* `gl.pointer_type` → `tl.pointer_type`

## 4. Memory Operations [Common]

* **`gl.load(ptr, mask=None, other=None)`** → `tl.load(ptr, mask=None, other=None)`
  * Purpose: Load data from global memory
  * Difference: Gluon returns tensor with layout information

* **`gl.store(ptr, value, mask=None)`** → `tl.store(ptr, value, mask=None)`
  * Purpose: Store data to global memory
  * Difference: None

* **`gl.allocate_shared_memory(element_ty, shape, layout, value=None, _semantic=None):`** → No direct mapping
  * Purpose: Explicitly allocate shared memory
  * Difference: **Gluon-specific**, Triton uses implicit shared memory management
  * Parameters:
    ```python
    element_ty: dtype               # The element data type
    shape: List[int]                # The dimensions of the shared memory
    layout: SharedLayout            # Shared memory layout (e.g., SwizzledSharedLayout)
    value: Tensor, optional         # Initial value to copy into shared memory
    ```

* **`gl.shared_memory_descriptor.load(layout)`** → No direct mapping
  * Purpose: Load data from shared memory descriptor to registers
  * Difference: **Gluon-specific**
  * Parameters:
    ```python
    layout: DistributedLayout  # Target distributed layout for loaded tensor
    ```

## 5. Tensor Creation [Common]

* **`gl.arange(start, end, layout)`** → `tl.arange(start, end)`
  * Purpose: Create sequence tensor
  * Difference: **Gluon requires layout parameter**
  * Parameters:
    ```python
    start: int
    end: int
    layout: DistributedLayout  # Required in Gluon
    ```

* **`gl.full(shape, value, dtype, layout=None)`** → `tl.full(shape, value, dtype)`
  * Purpose: Create tensor filled with value
  * Difference: **Gluon needs layout** (optional, defaults to AutoLayout)
  * Parameters:
    ```python
    shape: List[int]
    value: scalar
    dtype: dtype
    layout: Optional[DistributedLayout] = None  # Defaults to AutoLayout
    ```

* **`gl.zeros(shape, dtype, layout=None)`** → `tl.zeros(shape, dtype)`
  * Purpose: Create zero-filled tensor
  * Difference: **Gluon needs layout**

* **`gl.zeros_like(input, shape=None, dtype=None, layout=None)`** → `tl.zeros_like(input)`
  * Purpose: Create zero tensor with same properties as input
  * Difference: Gluon can specify layout

* **`gl.full_like(input, value, shape=None, dtype=None, layout=None)`** → No direct mapping
  * Purpose: Create filled tensor with same properties as input
  * Difference: **Gluon-specific**

## 6. Tensor Operations [Common]

* **`gl.cast(input, dtype)`** → `tl.cast(input, dtype)`
* **`gl.broadcast(input_tensor1, input_tensor2)`** → `tl.broadcast(input_tensor1, input_tensor2)`
  * Notice: gl.broadcast make two input tensors boradcast to same shape and return both tensors back in a tuple, should favor implicit broadcast over this api
* **`gl.expand_dims(input, axis)`** → `tl.expand_dims(input, axis)`
* **`gl.reshape(input, shape)`** → `tl.reshape(input, shape)`
  * Difference: Gluon supports `can_reorder` parameter, but it's meaningless
* **`gl.permute(input, dims)`** → `tl.permute(input, dims)`
* **`gl.split(input)`** → `tl.split(input)`
* **`gl.join(input_tensor1, input_tensor2)`** → `tl.join(input_tensor1, input_tensor2)`
  * Difference: Gluon cannot join scalars
* **`gl.to_tensor(x)`** → `tl.to_tensor(x)`

* **`gl.convert_layout(input, layout)`** → No direct mapping
  * Purpose: Convert tensor layout
  * Difference: **Gluon-specific**
  * Parameters:
    ```python
    input: tensor
    layout: DistributedLayout  # Target layout
    ```

## 7. Arithmetic Operations [Common]

All identical to Triton:
* `gl.add(x, y)` → `tl.add(x, y)`
* `gl.sub(x, y)` → `tl.sub(x, y)`
* `gl.mul(x, y)` → `tl.mul(x, y)`
* `gl.maximum(x, y)` → `tl.maximum(x, y)`
* `gl.minimum(x, y)` → `tl.minimum(x, y)`

## 8. Math Functions [Common]

All identical to Triton:
* `gl.exp(x)` → `tl.exp(x)`
* `gl.exp2(x)` → `tl.exp2(x)`
* `gl.log(x)` → `tl.log(x)`
* `gl.log2(x)` → `tl.log2(x)`
* `gl.sin(x)` → `tl.sin(x)`
* `gl.cos(x)` → `tl.cos(x)`
* `gl.sqrt(x)` → `tl.sqrt(x)`
* `gl.sqrt_rn(x)` → `tl.sqrt_rn(x)`
* `gl.rsqrt(x)` → `tl.rsqrt(x)`
* `gl.abs(x)` → `tl.abs(x)`
* `gl.fma(x, y, z)` → `tl.fma(x, y, z)`
* `gl.fdiv(x, y)` → `tl.fdiv(x, y)`
* `gl.div_rn(x, y)` → `tl.div_rn(x, y)`
* `gl.erf(x)` → `tl.erf(x)`
* `gl.floor(x)` → `tl.floor(x)`
* `gl.ceil(x)` → `tl.ceil(x)`
* `gl.umulhi(x, y)` → `tl.umulhi(x, y)`

## 9. Reduction Operations [Common]

* `gl.reduce(input, axis, combine_fn)` → `tl.reduce(input, axis, combine_fn)`
* `gl.sum(input, axis=None)` → `tl.sum(input, axis=None)`
* `gl.max(input, axis=None)` → `tl.max(input, axis=None)`
* `gl.min(input, axis=None)` → `tl.min(input, axis=None)`
* `gl.xor_sum(input, axis=None)` → `tl.xor_sum(input, axis=None)`
* `gl.reduce_or(input, axis)` → tl.reduce_or(input, axis)

## 10. Atomic Operations [Common]

All identical to Triton:
* `gl.atomic_add(ptr, val, mask=None)` → `tl.atomic_add(ptr, val, mask=None)`
* `gl.atomic_max(ptr, val, mask=None)` → `tl.atomic_max(ptr, val, mask=None)`
* `gl.atomic_min(ptr, val, mask=None)` → `tl.atomic_min(ptr, val, mask=None)`
* `gl.atomic_and(ptr, val, mask=None)` → `tl.atomic_and(ptr, val, mask=None)`
* `gl.atomic_or(ptr, val, mask=None)` → `tl.atomic_or(ptr, val, mask=None)`
* `gl.atomic_xor(ptr, val, mask=None)` → `tl.atomic_xor(ptr, val, mask=None)`
* `gl.atomic_xchg(ptr, val, mask=None)` → `tl.atomic_xchg(ptr, val, mask=None)`
* `gl.atomic_cas(ptr, cmp, val, mask=None)` → `tl.atomic_cas(ptr, cmp, val, mask=None)`

## 11. Program Control [Common]

* `gl.program_id(axis)` → `tl.program_id(axis)`
* `gl.num_programs(axis)` → `tl.num_programs(axis)`
* **`gl.num_warps()`** → No direct mapping
  * Purpose: Get number of warps
  * Difference: **Gluon-specific**
* **`gl.num_ctas()`** → No direct mapping
  * Purpose: Get number of CTAs
  * Difference: **Gluon-specific**

## 12. Debugging & Assertions [Common]

* `gl.device_print(prefix, *args)` → `tl.device_print(prefix, *args)`
* `gl.device_assert(cond, msg)` → `tl.device_assert(cond, msg)`
* `gl.static_print(*args)` → `tl.static_print(*args)`
* `gl.static_assert(cond, msg=None)` → `tl.static_assert(cond, msg=None)`
* `gl.assume(cond)` → `tl.assume(cond)`

## 13. Advanced Operations [Common]

* `gl.associative_scan(input, axis, combine_fn)` → `tl.associative_scan(input, axis, combine_fn)`
* `gl.gather(input, index, axis)` → `tl.gather(input, index, axis)`
*  `gl.histogram(input, num_bins, mask=None, layout=None)` → `tl.histogram(input, num_bins, mask=None)`
* `gl.inline_asm_elementwise(asm_str, constraints, args, dtype, is_pure, pack)` → `tl.inline_asm_elementwise(...)`
* **`gl.map_elementwise(fn, *args)`** → No direct mapping
  * Purpose: Element-wise mapping
  * Difference: **Gluon-specific**

## 14. Layout Classes (Gluon-Specific) [Common]

* **`gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])`**
  * Purpose: Define blocked distributed layout
  * Parameters:
    ```python
    size_per_thread: List[int]      # Elements per thread per dimension
    threads_per_warp: List[int]     # Threads per warp per dimension
    warps_per_cta: List[int]        # Warps per CTA per dimension
    order: List[int]                # Dimension ordering for partitioning
    cga_layout: List[List[int]] = []  # CTA tiling bases (optional)
    ```

* **`gl.SliceLayout(dim, parent)`**
  * Purpose: Layout for sliced tensor
  * Parameters:
    ```python
    dim: int                    # Dimension to slice
    parent: DistributedLayout   # Parent layout before slicing
    ```

* **`gl.DistributedLinearLayout(reg_bases, lane_bases, warp_bases, block_bases, shape)`**
  * Purpose: Linear distributed layout with explicit bases
  * Parameters:
    ```python
    reg_bases: List[List[int]]    # Register-level distribution bases
    lane_bases: List[List[int]]   # Lane-level distribution bases
    warp_bases: List[List[int]]   # Warp-level distribution bases
    block_bases: List[List[int]]  # Block-level distribution bases
    shape: List[int]              # Tensor global shape
    ```

* **`gl.DotOperandLayout(operand_index, parent, k_width)`**
  * Purpose: Layout for dot operand
  * Parameters:
    ```python
    operand_index: int          # 0 for LHS, 1 for RHS
    parent: DistributedLayout   # Parent MMA layout
    k_width: int                # Elements per 32-bits
    ```

* **`gl.NVMMADistributedLayout(...)`**
  * Purpose: NVIDIA MMA distributed layout

* **`gl.NVMMASharedLayout(...)`**
  * Purpose: NVIDIA MMA shared memory layout

* **`gl.SwizzledSharedLayout(...)`**
  * Purpose: Swizzled shared memory layout

* **`gl.PaddedSharedLayout(...)`**
  * Purpose: Padded shared memory layout

* **`gl.SharedLinearLayout(...)`**
  * Purpose: Shared memory linear layout

* **`gl.AutoLayout()`**
  * Purpose: Automatic layout inference

* **`gl.CoalescedLayout()`**
  * Purpose: Coalesced access layout

* **`gl.set_auto_layout(enabled)`**
  * Purpose: Enable/disable automatic layout

* **`gl.to_linear_layout(layout)`**
  * Purpose: Convert to linear layout

* **`gl.bank_conflicts(layout)`**
  * Purpose: Calculate bank conflicts

## 15. Synchronization [Common]

* **`gl.barrier()`** → No direct mapping
  * Purpose: Thread barrier synchronization
  * Difference: **Gluon-specific**

* **`gl.warp_specialize()`** → No direct mapping
  * Purpose: Warp specialization
  * Difference: **Gluon-specific**

## 16. Utility Functions [Common]

* `gl.cdiv(x, y)` → `tl.cdiv(x, y)`
* `gl.ravel(input)` → `tl.ravel(x, can_reorder=False)`
* `gl.static_range(start, end, step=1)` → `tl.static_range(start, end, step=1)`
* `gl.multiple_of(x, values)` → `tl.multiple_of(x, values)`
* `gl.max_contiguous(x, values)` → `tl.max_contiguous(x, values)`
* `gl.max_constancy(x, values)` → `tl.max_constancy(x, values)`
* `gl.where(condition, x, y)` → `tl.where(condition, x, y)`
* **`gl.dot_fma(a, b, acc)`** → No direct mapping
  * Purpose: Dot product with FMA
  * Difference: **Gluon-specific**
* **`gl.fp4_to_fp(x, dtype)`** → No direct mapping
  * Purpose: Convert FP4 to floating point
  * Difference: **Gluon-specific**

## 17. NVIDIA Hopper Architecture APIs [Hopper]

* **`gl.nvidia.hopper.fence_async_shared(cluster=False)`**
  * Purpose: Fence for asynchronous shared memory operations
  * Parameters:
    ```python
    cluster: bool = False  # Whether to fence across cluster
    ```

* **`gl.nvidia.hopper.warpgroup_mma_init(value)`**
  * Purpose: Initialize warpgroup MMA accumulator
  * Parameters:
    ```python
    value: tensor  # Initial accumulator value
    ```

* **`gl.nvidia.hopper.warpgroup_mma(a, b, acc, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False)`**
  * Purpose: Warpgroup MMA (Tensor Core) operation
  * Parameters:
    ```python
    a: tensor or shared_memory_descriptor  # LHS operand
    b: shared_memory_descriptor            # RHS operand
    acc: tensor                            # Accumulator
    use_acc: bool = True                   # Use initial accumulator value
    precision: Optional[str] = None        # Dot input precision
    max_num_imprecise_acc: Optional[int] = None  # Max imprecise accumulations
    is_async: bool = False                 # Asynchronous operation
    ```

* **`gl.nvidia.hopper.warpgroup_mma_wait(num_outstanding=0, deps=None)`**
  * Purpose: Wait for warpgroup MMA operations
  * Parameters:
    ```python
    num_outstanding: int = 0        # Number of outstanding operations to wait for
    deps: Optional[Sequence[tensor]] = None  # Dependencies to keep alive
    ```

* **`gl.nvidia.hopper.cluster.arrive(relaxed=False)`**
  * Purpose: Arrive at CTA cluster barrier
  * Parameters:
    ```python
    relaxed: bool = False  # Use relaxed semantics
    ```

* **`gl.nvidia.hopper.cluster.wait()`**
  * Purpose: Wait for all CTAs in cluster to arrive

## 18. NVIDIA Ampere Architecture APIs [Hopper, Ampere]

* **`gl.nvidia.ampere.mma_v2(a, b, acc, input_precision=None)`**
  * Purpose: MMA v2 (Tensor Core) operation
  * Parameters:
    ```python
    a: tensor                        # LHS operand with DotOperandLayout
    b: tensor                        # RHS operand with DotOperandLayout
    acc: tensor                      # Accumulator with NVMMADistributedLayout
    input_precision: Optional[str] = None  # Input precision
    ```

* **`gl.nvidia.ampere.async_copy.async_copy_global_to_shared(smem, pointer, mask=None, cache_modifier="", eviction_policy="", volatile=False)`**
  * Purpose: Asynchronously copy from global to shared memory
  * Parameters:
    ```python
    smem: shared_memory_descriptor   # Destination shared memory
    pointer: tensor                  # Source pointer tensor
    mask: Optional[tensor] = None    # Predicate mask
    cache_modifier: str = ""         # Cache modifier
    eviction_policy: str = ""        # Eviction policy
    volatile: bool = False           # Volatile load
    ```

* **`gl.nvidia.ampere.async_copy.mbarrier_arrive(mbarrier, increment_count=True)`**
  * Purpose: Arrive on mbarrier after async copies complete
  * Parameters:
    ```python
    mbarrier: shared_memory_descriptor  # Barrier object
    increment_count: bool = True        # Increment arrival count
    ```

* **`gl.nvidia.ampere.async_copy.commit_group()`**
  * Purpose: Commit current async copy group

* **`gl.nvidia.ampere.async_copy.wait_group(num_outstanding=0)`**
  * Purpose: Wait for async copy groups
  * Parameters:
    ```python
    num_outstanding: int = 0  # Wait until this many or fewer groups in-flight
    ```

* **`gl.nvidia.ampere.mbarrier.allocate_mbarrier(batch=None, two_ctas=False)`**
  * Purpose: Allocate mbarrier
  * Parameters:
    ```python
    batch: Optional[constexpr] = None  # Batch size
    two_ctas: constexpr = False        # Synchronize every other CTA
    ```

* **`gl.nvidia.ampere.mbarrier.init(mbarrier, count)`**
  * Purpose: Initialize mbarrier
  * Parameters:
    ```python
    mbarrier: shared_memory_descriptor  # Barrier object
    count: int                          # Initial count
    ```

* **`gl.nvidia.ampere.mbarrier.invalidate(mbarrier)`**
  * Purpose: Invalidate mbarrier

* **`gl.nvidia.ampere.mbarrier.wait(mbarrier, phase, pred=True, deps=())`**
  * Purpose: Wait for mbarrier phase completion
  * Parameters:
    ```python
    mbarrier: shared_memory_descriptor      # Barrier object
    phase: int                              # Phase index
    pred: bool = True                       # Predicate
    deps: Sequence[shared_memory_descriptor] = ()  # Dependencies
    ```

* **`gl.nvidia.ampere.mbarrier.arrive(mbarrier, pred=True)`**
  * Purpose: Arrive on mbarrier
  * Parameters:
    ```python
    mbarrier: shared_memory_descriptor  # Barrier object
    pred: bool = True                   # Predicate
    ```

* **`gl.nvidia.ampere.mbarrier.MBarrierLayout`**
  * Purpose: mbarrier layout class

## 19. NVIDIA Blackwell Architecture APIs [Blackwell]

* **`gl.nvidia.blackwell.allocate_tensor_memory(dtype, shape, layout, alloc_shape=None)`**
  * Purpose: Allocate Tensor Memory (TMEM)
  * Parameters:
    ```python
    dtype: dtype                         # Element data type
    shape: List[int]                     # Logical shape
    layout: TensorMemoryLayout           # TMEM layout
    alloc_shape: Optional[List[int]] = None  # Physical allocation shape
    ```

* **`gl.nvidia.blackwell.get_tmem_reg_layout(element_ty, shape, layout, num_warps, instr_variant="32x32b", cga_layout=())`**
  * Purpose: Get TMEM-compatible register layout
  * Parameters:
    ```python
    element_ty: dtype                    # Element type
    shape: Sequence[int]                 # Global tensor shape
    layout: TensorMemoryLayout           # TMEM layout
    num_warps: int                       # Number of warps
    instr_variant: str = "32x32b"        # TMEM instruction variant
    cga_layout: Sequence[Sequence[int]] = ()  # CGA layout bases
    ```

* **`gl.nvidia.blackwell.tensor_memory_descriptor.load(layout)`**
  * Purpose: Load from Tensor Memory
  * Parameters:
    ```python
    layout: DistributedLayout  # Destination layout
    ```

* **`gl.nvidia.blackwell.tensor_memory_descriptor.mma(b, layout, red_layout=None)`**
  * Purpose: MMA operation using Tensor Memory
  * Parameters:
    ```python
    b: tensor or tensor_memory_descriptor  # RHS operand
    layout: DistributedLayout              # Result layout
    red_layout: Optional[DistributedLayout] = None  # Reduction layout
    ```

* **`gl.nvidia.blackwell.tma.async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=True)`**
  * Purpose: Asynchronously gather elements using TMA
  * Parameters:
    ```python
    tensor_desc: tensor_descriptor         # Tensor descriptor
    x_offsets: tensor                      # 1D X offsets
    y_offset: int                          # Scalar Y offset
    barrier: shared_memory_descriptor      # Barrier
    result: tensor_memory_descriptor       # Result TMEM
    pred: bool = True                      # Predicate
    ```

* **`gl.nvidia.blackwell.tma.async_scatter(tensor_desc, x_offsets, y_offset, src)`**
  * Purpose: Asynchronously scatter elements using TMA
  * Parameters:
    ```python
    tensor_desc: tensor_descriptor     # Tensor descriptor
    x_offsets: tensor                  # 1D X offsets
    y_offset: int                      # Scalar Y offset
    src: tensor_memory_descriptor      # Source TMEM
    ```

* **`gl.nvidia.blackwell.TensorMemoryLayout(block, col_stride, cta_split_num=None, two_ctas=False)`**
  * Purpose: Tensor Memory layout class
  * Parameters:
    ```python
    block: Tuple[int, int]                      # Contiguous elements per row/col
    col_stride: int                             # Column stride (power of 2)
    cta_split_num: Optional[Tuple[int, int]] = None  # CTA split factors
    two_ctas: bool = False                      # Two-CTA mode
    ```

* **`gl.nvidia.blackwell.TensorMemoryScalesLayout(cta_split_num=None)`**
  * Purpose: Tensor Memory scales layout class
  * Parameters:
    ```python
    cta_split_num: Optional[Tuple[int, int]] = None  # CTA split factors
    ```

## 20. NVIDIA Hopper TMA APIs [Hopper]

* **`gl.nvidia.hopper.tma.make_tensor_descriptor(base, shape, strides, block_shape, layout, padding_option="zero")`**
  * Purpose: Create TMA tensor descriptor
  * Parameters:
    ```python
    base: tensor                    # Base pointer
    shape: List[tensor]             # Tensor shape (dynamic)
    strides: List[tensor]           # Tensor strides (dynamic)
    block_shape: List[constexpr]    # Block shape (static)
    layout: NVMMASharedLayout       # Shared memory layout
    padding_option: str = "zero"    # Padding option ("zero" or "nan")
    ```

* **`gl.nvidia.hopper.tma.async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False)`**
  * Purpose: Async copy from global to shared using TMA
  * Parameters:
    ```python
    tensor_desc: tensor_descriptor         # Tensor descriptor
    coord: Sequence                        # Coordinates
    barrier: shared_memory_descriptor      # Barrier
    result: shared_memory_descriptor       # Result shared memory
    pred: bool = True                      # Predicate
    multicast: bool = False                # Multicast mode
    ```

* **`gl.nvidia.hopper.tma.async_copy_shared_to_global(tensor_desc, coord, src)`**
  * Purpose: Async copy from shared to global using TMA
  * Parameters:
    ```python
    tensor_desc: tensor_descriptor     # Tensor descriptor
    coord: Sequence                    # Coordinates
    src: shared_memory_descriptor      # Source shared memory
    ```

* **`gl.nvidia.hopper.tma.store_wait(pendings)`**
  * Purpose: Wait for TMA store operations
  * Parameters:
    ```python
    pendings: int  # Number of pending operations
    ```

## 21. AMD CDNA3/CDNA4 Architecture APIs [AMD CDNA3]

* **`gl.amd.cdna3.buffer_load(ptr, offsets, mask=None, other=None, cache=None)`**
  * Purpose: AMD buffer load from global memory
  * Parameters:
    ```python
    ptr: pointer to scalar          # Base pointer
    offsets: tensor                 # Offset tensor (int32/uint32)
    mask: Optional[tensor] = None   # Predicate mask
    other: Optional[tensor or scalar] = None  # Default values for masked elements
    cache: Optional[str] = None     # Cache modifier
    ```

* **`gl.amd.cdna3.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)`**
  * Purpose: AMD buffer store to global memory
  * Parameters:
    ```python
    stored_value: tensor            # Value to store
    ptr: pointer to scalar          # Base pointer
    offsets: tensor                 # Offset tensor
    mask: Optional[tensor] = None   # Predicate mask
    cache: Optional[str] = None     # Cache modifier
    ```

* **`gl.amd.cdna3.buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic add
  * Parameters:
    ```python
    ptr: pointer to scalar          # Base pointer
    offsets: tensor                 # Offset tensor
    value: tensor                   # Value to add
    mask: Optional[tensor] = None   # Predicate mask
    sem: Optional[str] = None       # Memory semantic (default: acq_rel)
    scope: Optional[str] = None     # Memory scope (default: gpu/agent)
    ```

* **`gl.amd.cdna3.buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic max (similar parameters as atomic_add)

* **`gl.amd.cdna3.buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic min (similar parameters as atomic_add)

* **`gl.amd.cdna3.buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic AND (similar parameters as atomic_add)

* **`gl.amd.cdna3.buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic OR (similar parameters as atomic_add)

* **`gl.amd.cdna3.buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic XOR (similar parameters as atomic_add)

* **`gl.amd.cdna3.buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None)`**
  * Purpose: AMD buffer atomic exchange (similar parameters as atomic_add)

* **`gl.amd.cdna3.mfma(a, b, acc)`**
  * Purpose: AMD MFMA (Matrix Fused Multiply-Add) operation
  * Parameters:
    ```python
    a: tensor  # LHS operand
    b: tensor  # RHS operand
    acc: tensor  # Accumulator
    ```

## 22. AMD GFX1250 Architecture APIs [AMD GFX1250]

* **`gl.amd.gfx1250.wmma(a, b, acc)`**
  * Purpose: AMD WMMA operation
  * Parameters:
    ```python
    a: tensor  # LHS operand
    b: tensor  # RHS operand
    acc: tensor  # Accumulator
    ```

* **`gl.amd.gfx1250.wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)`**
  * Purpose: AMD scaled WMMA with microscaling formats
  * Parameters:
    ```python
    a: tensor                       # LHS operand
    a_scale: Optional[tensor]       # LHS scale factor
    a_format: str                   # LHS format ("e2m1", "e4m3", "e5m2")
    b: tensor                       # RHS operand
    b_scale: Optional[tensor]       # RHS scale factor
    b_format: str                   # RHS format ("e2m1", "e4m3", "e5m2")
    acc: tensor                     # Accumulator
    ```

* **`gl.amd.gfx1250.get_wmma_scale_layout(dot_operand_layout, shape)`**
  * Purpose: Get scale layout for WMMA scaled operands
  * Parameters:
    ```python
    dot_operand_layout: DotOperandLayout  # Dot operand layout
    shape: List[int]                      # Scale tensor shape
    ```

* **`gl.amd.gfx1250.tdm.make_tensor_descriptor(...)`**
  * Purpose: Create AMD TDM (Tensor Descriptor Memory) descriptor

* **`gl.amd.gfx1250.async_copy.*`**
  * Purpose: AMD async copy operations

* **`gl.amd.gfx1250.mbarrier.*`**
  * Purpose: AMD mbarrier operations

* **`gl.amd.gfx1250.cluster.*`**
  * Purpose: AMD cluster operations

## 23. AMD Layout Classes [AMD CDNA3]

* **`gl.amd.AMDMFMALayout`**
  * Purpose: AMD MFMA layout

* **`gl.amd.AMDWMMALayout`**
  * Purpose: AMD WMMA layout

## 24. AMD Warp Pipeline [Common]

* **`gl.amd.warp_pipeline_stage`**
  * Purpose: Warp pipeline stage management


## Related

- [Triton → Gluon Conversion Guide (NVIDIA Hopper)](hopper-conversion-guide.md)
- [HSACO Offline Launcher Guide](hsaco-offline-launcher.md)
- [Real-time Learning Guide](learning_guide.md)
- [Verification Guide](verification_guide.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
