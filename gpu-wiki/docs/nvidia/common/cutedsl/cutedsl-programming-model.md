# CuTeDSL Programming Model


**Last updated**: 2026-06-30

## Positioning

CuTeDSL is a Python DSL in NVIDIA CUTLASS 4.x, providing a low-level GPU kernel programming model consistent with CuTe C++. Compilation path:

```
Python kernel -> AST Rewrite -> Tracing -> MLIR IR -> ptxas -> SASS
```

Target users: students learning GPU programming, researchers prototyping, performance engineers tuning kernels.

**Platform requirements:** Linux x86_64, Python 3.10-3.13, NVIDIA Driver ≥ 575.51.03, CUDA Toolkit 12.9+

---

## @jit and @kernel Decorators

### @jit — Host-side JIT Function

```python
@cute.jit
def host_function(a, b):
 # Python or DSL function
 # preprocessor=True (default): automaticconversion Python IR
    pass
```

### @kernel — GPU Kernel Function

```python
@cute.kernel
def gpu_kernel(ptr, layout):
 # compilation GPU
 # launchparameter: grid, block, cluster, smem
    pass
```

### Calling Convention Matrix

| Caller | Callee | Allowed | Behavior |
|--------|---------|------|------|
| Python | `@jit` | ✅ | DSL runtime |
| Python | `@kernel` | ❌ | Error |
| `@jit` | `@jit` | ✅ | Compile-time inlining |
| `@jit` | Python function | ✅ | Compile-time inlining |
| `@jit` | `@kernel` | ✅ | Dynamic invocation via GPU driver |
| `@kernel` | `@jit` | ✅ | Compile-time inlining |
| `@kernel` | `@kernel` | ❌ | Error |

**Key point**: kernels can only be launched from `@jit` functions, not directly from Python or called from other kernels.

---

## Three-Stage Compilation Pipeline

### Stage 1: Pre-Staging (Python AST)

The AST preprocessor rewrites decorated functions, inserting callbacks around control flow structures to explicitly capture program structure.

### Stage 2: Meta-Stage (Python Interpreter)

Executes the rewritten function with proxy tensor arguments:
- Callbacks emit structured IR at control flow boundaries
- Tensor operations are recorded via operator overloading
- Compile-time constants undergo **partial evaluation** and are directly folded into IR

### Stage 3: Object-Stage (Compiler Backend)

1. Progressive lowering to hardware-specific representations
2. Optimization passes (tiling, vectorization, memory promotion)
3. Conversion to PTX/SASS and assembly into device binary

---

## Type System: Constexpr vs Dynamic

Code is **executed twice**: at **meta-programming time** (host CPU, building the kernel) and at **runtime** (GPU execution).

| Type | Phase | Behavior |
|------|------|------|
| `cutlass.Float32` | Runtime (dynamic) | Proxy created at meta-stage, computed on GPU |
| `cutlass.Constexpr` | Compile-time | Value known at meta-stage, compiled into kernel |

```python
@cute.jit
def add_dynamic(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta]", result)             # <Float32 proxy>
    cute.printf("[gpu] %f\n", result)   # 7.000000

@cute.jit
def add_const(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
 print("[meta]", result) # 7.0 (compilation)
    cute.printf("[gpu] %f\n", result)   # 7.000000
```

**Output mechanism:**
- `print()` — executed at meta-stage, inspect shape/stride/tile size
- `cute.printf()` — compiled into kernel, executed at GPU runtime

---

## Control Flow

### Compile-time vs Runtime

| Construct | Runtime (emit IR) | Compile-time (Python execution) |
|------|-------------------|--------------------------|
| `if cutlass.const_expr(...)` | ❌ | ✅ |
| `if pred` | ✅ | ❌ |
| `while cutlass.const_expr(...)` | ❌ | ✅ |
| `while pred` | ✅ | ❌ |
| `for i in cutlass.range_constexpr(n)` | ❌ | ✅ Fully unrolled |
| `for i in range(n)` | ✅ | ❌ |
| `for i in cutlass.range(n)` | ✅ | ❌ |

### Compile-time Metaprogramming

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    ...
    if cutlass.const_expr(do_relu):
 ... # ReLU do_relu=True compilation IR

gemm(..., False) # none ReLU kernel
gemm(..., True) # ReLU kernel
```

### Software Pipelining Loop Attributes

```python
for i in cutlass.range(bound, prefetch_stages=3):
    cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
    use(buffer[i % total_stages])
```The compiler automatically generates the prefetch loop and main loop. **Experimental feature, only supported on SM90+.**

### Dynamic Control Flow Restrictions

1. No `break`/`continue`/`return`/exceptions
2. Variables defined inside dynamic control flow bodies are not accessible from outside
3. Cannot change variable types within dynamic control flow

---

## JIT Cache

### Implicit Cache (Default)

Cache key = hash(MLIR bytecode + DSL source files + shared libraries + environment variables)

- Hit → Skip compilation, reuse executor
- Miss → Compile kernel, store executor

**Note**: The MLIR generation step **always executes** (for consistency verification), only the compilation step is skipped.

### Explicit Cache (cute.compile)

```python
compiled = cute.compile(kernel_fn, *args)
# cache compiled executor, direct
compiled(a, b, c, stream)
```

Completely bypasses the MLIR regeneration overhead of the implicit cache.

### File Persistence

Default path: `/tmp/{user}/cutlass_python_cache` (may be lost on reboot)

| Environment Variable | Purpose |
|----------|------|
| `CUTE_DSL_CACHE_DIR` | Custom cache directory |
| `CUTE_DSL_DISABLE_FILE_CACHING` | Disable file cache, memory cache only |

---

## Framework Integration

### DLPack Implicit Conversion

```python
@cute.jit
def foo(src):
    print(src)  # ptr<f32, generic> o (?,?,?):(?,?,1)

a = torch.randn(30, 20, 32, device="cpu")
foo(a) # automaticconversion, layout ( leading dim stride=1)
```

### DLPack Explicit Conversion

```python
from cutlass.cute.runtime import from_dlpack

t = from_dlpack(x) # layout, copy
t = from_dlpack(x).mark_layout_dynamic #
t = from_dlpack(x).mark_compact_shape_dynamic(0) # dimension
```

Each `from_dlpack` incurs approximately 2-3μs overhead.

### Bypassing DLPack

```python
from cutlass.cute.runtime import make_ptr

a_ptr = make_ptr(cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
mA = cute.make_tensor(a_ptr, layout=layout)
```

---

## CuTeDSL Core API

### Layout Construction

```python
layout = cute.make_layout((4, 4))                         # compact column-major
layout = cute.make_layout((4, 4), stride=(1, 4)) # definition stride
layout = cute.make_ordered_layout((4, 4), order=(1, 0))   # row-major
layout = cute.make_identity_layout((4, 4)) # mapping
```

### Tensor Construction

```python
ptr = cute.make_ptr(Float32, base_ptr, AddressSpace.gmem)
tensor = cute.make_tensor(ptr, cute.make_layout((64, 128), stride=(128, 1)))

# register tensor
rmem = cute.make_rmem_tensor((128, 32), cutlass.Float16)
rmem = cute.make_rmem_tensor_like(src, cutlass.Float32)

# tensor
ident = cute.make_identity_tensor((3, 2))  # [(0,0),(1,0),(2,0),(0,1),...]
```

### TensorSSA (Immutable Value Semantics Tensor)

```python
a = rmem.load()           # TensorSSA
b = cute.full_like(a, 1.0)
c = a + b #
d = c.to(cutlass.Float16) # typeconversion
e = c.reduce(op, init_val, reduction_profile) # bymodereduction
```

### Layout Algebra Functions

```python
cute.composition(L1, L2)           # R(c) = L1(L2(c))
cute.complement(layout, cotarget) # layout
cute.zipped_divide(layout, tiler)  # tiling: (Tile, Rest)
cute.logical_product(block, tiler) # layout
cute.coalesce(layout) # coalesced
cute.right_inverse(layout) # right
cute.left_inverse(layout) # left
```

### Partition Operations

```python
cute.local_partition(target, tiler, index, proj) # by tiler index
cute.local_tile(input, tiler, coord, proj)        # zipped_divide + slice
```

### Swizzle Transformations

```python
swizzle = cute.make_swizzle(b=3, m=0, s=3)  # (BBits, MBase, SShift)
# result = lowbit XOR (highbit << SShift)
```### @cute.struct Decorator

```python
@cute.struct
class Storage:
    mbarA: cute.struct.MemRange[cutlass.Int64, num_stages]
    dataA: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, size_a], 1024]

# use
storage.mbarA.data_ptr # pointer
storage.mbarA.get_tensor(layout, swizzle) # create tensor
```

---

## Autotuning GEMM

### Search Space Parameters

- `mma_tiler_mn`: MMA tile dimensions
- `cluster_shape_mn`: Number of CTAs in a cluster
- `use_2cta_instrs`: Whether to use 2-CTA instructions (Blackwell)
- `use_tma_store`: Whether to use TMA store

### Benchmarking Best Practices

1. 5-10 warmup iterations to stabilize GPU temperature
2. 100-1000 timed iterations
3. Use CUDA events for precise timing
4. `nvidia-smi` Lock SM and memory frequency
5. Remove outliers, use min/avg statistics

### Result Caching

```python
# compilationcache
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{mma_tiler}x{cluster_shape}"
config_kernel_dict[kernel_cache_key] = compiled_kernel

# inputcache
input_kernel_dict[(m, n, k)] = best_compiled_kernel
# power_of_2 decrease key count
```

---

## Limitations

| Limitation | Description |
|------|------|
| Linux only | Windows not supported |
| No convolution | GEMM only |
| 32-bit layout | Shape/Stride only supports 32-bit; 64-bit planned |
| No dependent types | Expression types must be known at compile time |
| No dynamic function return values | Only Constexpr values can be returned |
| Limited OOP | Cannot pass dynamic values through class state (DynamicExpression required) |
| No mixing lru_cache with @jit | DSL objects have context dependencies |
| Python data structures read-only | list/tuple/dict structures cannot be modified at runtime |


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS 4.0 Python Support](cutlass-4.0-python-support.md)
- [PTX Programming Model and Basics](../ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../ptx/ptx-instruction-set.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](cutlass-cute-fundamentals.md)
