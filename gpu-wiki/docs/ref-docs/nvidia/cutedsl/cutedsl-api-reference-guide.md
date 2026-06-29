# CuTeDSL API Reference Guide

A module-by-module quick reference for the CuTe DSL interface, designed to maintain consistency with CuTe C++ while providing Python ergonomics across Ampere, Hopper, and Blackwell GPU architectures.

---

## 1. Core Decorators

| Decorator | Purpose | Parameters |
|---|---|---|
| `@cute.jit` | Host-side JIT compiled function | `preprocessor` (default True): auto-expands Python control flow |
| `@cute.kernel` | GPU kernel function | `preprocessor` (default True): auto-expands loops/conditionals to GPU IR |

---

## 2. Compilation and Execution Interface

| Function/Method | Description |
|---|---|
| `cute.compile()` | JIT compilation entry point; supports AOT compilation and caching |
| `kernel().launch(grid, block, cluster, smem)` | Launch kernel with specified grid/block dimensions, cluster configuration, and shared memory size |
| `kernel(grid=..., block=..., smem=...)` | Direct invocation syntax sugar |

---

## 3. Tensor Operations

### 3.1 Tensor Creation and Conversion

| Function | Description |
|---|---|
| `cute.make_tensor(ptr, layout)` | Create tensor from pointer and layout |
| `cute.make_layout(shape, stride)` | Create a Layout |
| `cute.make_layout_tv(tiler, layout)` | Create thread-value layout |
| `cute.make_identity_tensor(shape)` | Create identity tensor (for coordinate computation) |
| `cute.from_dlpack(tensor)` | Import tensor from PyTorch/JAX via DLPack protocol |
| `cute.make_rmem_tensor(shape, dtype)` | Create register memory tensor (replaces legacy `make_fragment`) |
| `cute.make_rmem_tensor_like(tensor)` | Create rmem tensor with same shape as existing tensor |

### 3.2 Tensor Operations

| Method/Function | Description |
|---|---|
| `tensor.load()` / `tensor.store(value)` | Load/Store |
| `tensor.reduce(op, axis, init)` | Reduction (supports `cute.ReductionOp`) |
| `tensor.broadcast_to(shape)` | Broadcast |
| `tensor.reshape(shape)` | Reshape |
| `tensor.to(dtype)` | Type conversion |
| `tensor[...]` | Slicing and index access |
| `cute.zipped_divide(tensor, tiler)` | Vectorized partitioning |
| `cute.composition(layout1, layout2)` | Layout composition |
| `cute.size(tensor/layout, mode=...)` | Get size of specified mode |

---

## 4. Hardware Atoms

| Category | Interface | Description |
|---|---|---|
| **MMA** | `cute.make_tiled_mma(atom)` | Create tiled MMA operation |
| | `tcgen05.MmaF16BF16Op(...)` | Blackwell FP16/BF16 MMA |
| | `cute.gemm(tiled_mma, C, A, B, C)` | Execute matrix multiply |
| **Copy** | `cute.make_tiled_copy(atom)` | Create tiled copy operation |
| | `cp.async` (implicit) | Asynchronous copy instruction |
| | `CopyReduceBulkTensorTileS2GOp` | TMA Reduce operation |
| **TCGEN05** | `tcgen05.copy(...)` | Blackwell copy instruction |
| | `tcgen05.mma(...)` | Blackwell MMA instruction |

---

## 5. Memory Management

| Class/Function | Description |
|---|---|
| `cute.SmemAllocator` | Shared memory allocator |
| `SmemAllocator.allocate(size)` | Allocate shared memory |
| `cute.TmemAllocator` | Tensor Memory allocator (Blackwell) |
| `tiled_mma.make_fragment_A/B/C(...)` | Create MMA fragments (TMEM/SMEM descriptors) |

---

## 6. Synchronization and Pipelining

| Interface | Description |
|---|---|
| `cute.PipelineAsync` | Asynchronous pipeline management |
| `pipeline.sync()` | Simple CTA synchronization |
| `pipeline.NamedBarrier` | Custom barrier (specify participating threads and barrier ID) |
| `cute.arch.barrier` | Low-level barrier instruction (deprecated; use pipeline API) |
| `cute.arch.thread_idx()` | Get thread index |
| `cute.arch.block_idx()` | Get block index |
| `cute.arch.warp_idx()` | Get warp index |

---

## 7. Math and Utility Functions

| Category | Functions |
|---|---|
| Math | `cute.math.exp`, `cute.math.exp2`, `cute.math.log`, `cute.math.log2` (supports fast-math control) |
| Utilities | `cute.repeat`, `cute.repeat_as_tuple` |
| Debug | `cute.printf(...)` (GPU-side print), `cute.print_tensor(tensor)` |
| Types | `cutlass.Int32`, `cutlass.Float32`, `cutlass.Constexpr`, etc. |

---

## 8. Architecture-Specific Interfaces

| Architecture | Specific Support |
|---|---|
| Blackwell (SM100) | `tcgen05` module (MMA, Copy, TME operations), Tensor Memory, Warp Specialization |
| Hopper (SM90) | TMA, GMMA, Warpgroup operations |
| Ampere (SM80) | WMMA, basic async copy |

---

## 9. Data Type Support

- **Floating point:** FP64, FP32, TF32, FP16, BF16, FP8 (e4m3/e5m2)
- **Block-scaled types:** NVFP4, MXFP4, MXFP6, MXFP8
- **Integer:** I8, I4, binary types
- **Custom types:** Supported via `JitArgument` and `DynamicExpression` protocol

---

## 10. Debug and Optimization Interface

| Feature | Environment Variable / Method |
|---|---|
| Print MLIR | `CUTE_DSL_PRINT_IR=1` or `compiled_kernel.__mlir__` |
| Print PTX | `CUTE_DSL_KEEP_PTX=1` or `compiled_kernel.__ptx__` |
| Print CUBIN | `CUTE_DSL_KEEP_CUBIN=1` or `compiled_kernel.__cubin__` |
| Source correlation | `CUTE_DSL_LINEINFO=1` (generates line info for profiling) |
| Log control | `CUTE_DSL_LOG_LEVEL`, `CUTE_DSL_LOG_TO_CONSOLE` |

---

## 11. Full API Documentation

The complete API documentation is available in the official CuTe DSL API Reference.
