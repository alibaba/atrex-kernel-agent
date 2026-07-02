# QuACK Architecture Overview

QuACK ("Quirky Assortment of CuTe Kernels") is a high-performance CUDA kernel library developed by Dao-AILab (Tri Dao), written entirely in CuTeDSL (Python), targeting SM90 (H100), SM100 (B200/B300), and SM120 (GeForce) architectures. The library provides two major kernel categories: **reduction kernels** (RMSNorm, Softmax, Cross Entropy, LayerNorm, TopK) and **GEMM kernels** (with various epilogue fusion variants), all deeply integrated with PyTorch via `torch.library.custom_op`.


**Last updated**: 2026-06-30

## Project Structure and Kernel Types

### Reduction Kernels

`rmsnorm.py`, `softmax.py`, `cross_entropy.py`, and other reduction kernels inherit from `ReductionBase` (`reduction_base.py`), sharing a unified pattern: configure cluster size, build tiled copy, allocate reduction buffer + mbarrier, and launch `@cute.kernel`.

```python
class RMSNorm(ReductionBase):
    def __init__(self, dtype, N, is_layernorm=False):
        super().__init__(dtype, N, stage=2 if is_layernorm else 1)
        self.reload_from = None if N <= (16384 if is_layernorm else 8192) else "smem"
```

Core methods:
- `_threads_per_row()` / `_num_threads()` -- determine thread layout
- `_get_tiled_copy()` -- build 2D tiled copy descriptor
- `_allocate_reduction_buffer_and_mbar()` -- allocate reduction buffer and mbarrier in shared memory

### GEMM Kernels

GEMM adopts a multi-layer design, decoupling the public API, SM version selection, epilogue fusion, and tile configuration layers:

| File | Responsibility |
|------|------|
| `gemm.py` | Public API, parameter validation, SM version routing, `@jit_cache` compilation entry |
| `gemm_interface.py` | Unified interface (cross-SM), autotuner integration, `custom_op` registration |
| `gemm_sm90.py` / `gemm_sm100.py` / `gemm_sm120.py` | SM version-specific implementations |
| `gemm_default_epi.py` + `gemm_*_epi.py` | Epilogue variants (bias, activation, gated, etc.) |
| `gemm_config.py` | `GemmConfig` data class: tile size, cluster dims, swizzle settings |

SM version routing is performed in `_compile_gemm()`:

```python
@jit_cache
def _compile_gemm(a_dtype, b_dtype, d_dtype, ..., device_capacity, ...):
    sm_to_cls = {
        9: GemmDefaultSm90,
        10: GemmDefaultSm100,
        11: GemmDefaultSm100,
        12: GemmDefaultSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
```

Supported epilogue fusion variants include: `gemm_act` (GEMM + activation), `gemm_dact` (GEMM + activation backward), `gemm_gated` (GEMM + gated activation such as SwiGLU), `gemm_dgated`, `gemm_symmetric`, `gemm_sq_reduce` (SmoothQuant reduction), `gemm_norm_act` (GEMM + norm + activation).

## Compilation Pipeline: Python → AST → MLIR → PTX → Binary

QuACK uses CuTeDSL's `@cute.jit` and `@cute.kernel` decorators to compile Python functions into GPU kernels. Compilation path:

1. **Python AST walk** -- CuTeDSL traverses the AST of the decorated function, converting control flow (if/for/while) into structured control flow in MLIR
2. **MLIR lowering** -- Generates MLIR IR, lowered through multiple passes
3. **PTX generation** -- MLIR lowers to PTX (NVIDIA virtual assembly)
4. **JIT linking** -- PTX compiles to binary cubin, loaded and dispatched through the TVM-FFI runtime

The entire compilation process takes approximately 100 ms per kernel. Triggered via `cute.compile()`:

```python
compiled_fn = cute.compile(gemm_obj, *fake_args, options="--enable-tvm-ffi")
```

### TVM-FFI Runtime Dispatch

When `--enable-tvm-ffi` is enabled, compiled artifacts support a two-stage execution model:

- **Compile time**: Builds IR using symbolic tensors (fake tensors), requiring only shape/stride/dtype metadata
- **Runtime**: TVM-FFI extracts pointers from real tensors, validates dimension constraints, and invokes the compiled kernel

Symbolic dimensions are created via `cute.sym_int()`. **Reusing the same `sym_int` instance tells TVM-FFI that these dimensions must match at runtime**:

```python
m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()

# m A D shared -- TVM-FFI row D.shape[0] == A.shape[0]
mA = fake_tensor(a_dtype, (m, k, l), leading_dim=..., divisibility=...)
mD = fake_tensor(d_dtype, (m, n, l), leading_dim=..., divisibility=...)
```## Multi-Layer Cache System

QuACK's cache design reduces the 100ms compilation overhead per kernel to approximately 1ms (disk hit) or approximately 0ns (memory hit) after the first invocation.

### Layer 1: `@jit_cache` Decorator

`@jit_cache` (`cache_utils.py`) is a unified decorator that provides both in-memory caching and persistent disk caching. All `_compile_*` functions are decorated with it:

```python
@jit_cache
def _compile_softmax_fwd(dtype, out_dtype, N):
    # ... build fake tensors ...
    return cute.compile(softmax_op, ...)
```

**In-memory cache (fastest, approximately 0ns)**: A per-function Python `dict`, keyed by function argument tuples. Subsequent calls with the same configuration within the same process hit directly.

**Disk cache (approximately 1ms)**: On a memory miss, checks disk for a cached `.o` (object file). The disk key is the SHA-256 hash of the `(fn.__qualname__, *args, **sorted_kwargs)`. The cache directory contains a **source fingerprint** -- the SHA-256 of all `quack/*.py` files plus the Python/CUTLASS/TVM-FFI versions. Any source change invalidates the entire cache.

**Cache miss flow**:

```
@jit_cache wrapper(dtype, out_dtype, N)
 |-- In-memory dict lookup --> hit ( 0ns)
  |
  |-- hash(fn.__qualname__, *args) --> SHA-256 hex
 |-- Shared lock: check if .o exists --> load via tvm_ffi ( 1ms)
  |
  |-- Cache miss --> call fn(dtype, out_dtype, N)
  |     \-- cute.compile(...) --> MLIR --> PTX --> binary
  |
  \-- Exclusive lock: export_to_c() --> write .o to cache
```

**Concurrency safety**: Multiple processes access the cache in serialized fashion via `FileLock` (`fcntl.flock`). Reads use a shared lock (concurrent readers), while writes use an exclusive lock (double-checking post-acquisition whether the `.o` has already been written by another process).

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `QUACK_CACHE_ENABLED` | `1` | Set to `0` to disable disk caching (in-memory caching remains effective) |
| `QUACK_CACHE_DIR` | `/tmp/$USER/quack_cache` | Override the cache directory location |

### Layer 2: Autotuning Result Cache

The autotuner's benchmark results are cached to disk in JSON format, implemented through Triton's `FileCacheManager` infrastructure. The cache key includes: package version, tuning key (tensor metadata), and all config strings.

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `QUACK_CACHE_AUTOTUNING` | Not set | Set to `1` to enable disk caching of autotuning results |
| `QUACK_FORCE_CACHE_UPDATE` | Not set | Set to `1` to ignore previously cached tuning results |

### Parallel Compilation: Persistent Subprocess Worker Pool

When the autotuner encounters a cache miss, it needs to compile all candidate configs (approximately 44 for GEMM). Two constraints make naive parallelization infeasible:

1. **`cute.compile()` is not thread-safe** -- MLIR uses thread-local state, and `ThreadPoolExecutor` would cause data races
2. **`fork()` segfaults after CUDA initialization** -- the parent process has already called `torch.cuda.init()`, and the CUDA driver handles duplicated by `fork()` are invalid in child processes

Solution: `Autotuner._precompile()` launches persistent worker processes via `subprocess.Popen()` (always new processes, not fork). Each worker:

1. Sets `COMPILE_ONLY = True` (compiles to produce `.o` without launching the kernel)
2. Creates `FakeTensor` matching the parent process's tensor metadata (shape/stride/dtype, without allocating GPU memory)
3. Calls the kernel function to trigger `@jit_cache` compilation and export `.o`
4. Stays alive to amortize the `import quack` overhead (approximately 2–3s), receiving subsequent tasks via stdin

**Quick check optimization**: Before launching workers, compile one config in the main process first. If it completes within 0.5s, it indicates the `.o` cache is warm, and parallel compilation is skipped.

```python
t_check = time.time()
self.fn(*args, **configs[0].all_kwargs())
if time.time() - t_check < 0.5:
    return  # cache is warm, no need for workers
```

## Autotuner

QuACK's autotuner (`autotuner.py`) is adapted from Triton's autotuner and supports searching for the optimal solution across multiple tile/cluster/swap configurations.

### Search Space

`GemmConfig` (`gemm_config.py`) is a frozen dataclass that defines the search dimensions:

```python
@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    pingpong: bool = True
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    swap_ab: bool = False
    max_swizzle_size: int = 8
    device_capacity: int = 9
    use_tma_gather: bool = False
```SM90 and SM100 each have their own config generation functions (`_get_sm90_configs` / `_get_sm100_configs`), combining tile_m/tile_n, cluster_m/cluster_n, and swap_ab to produce the full search space.

### Tuning Process

```python
@autotune(
    configs=[AutotuneConfig(tile_m=128, tile_n=192, ...), ...],
    key=["A", "B"],
    cache_results=True,
)
def gemm_tuned(A, B, out, *, tile_m, tile_n, cluster_m, cluster_n, swap_ab, ...):
    ...
```

1. Compute tuning key from tensor shape/stride/dtype
2. Check disk cache (JSON) for matching results
3. On cache miss: precompile all configs in parallel, then GPU warmup (200ms warmup to avoid artificial advantage for the first config), then benchmark one by one
4. Use `triton.testing.do_bench` to measure median time (warmup=5, rep=25)
5. Select the fastest config, cache it, and use it

### nvidia-matmul-heuristics Integration

Optionally depends on `nvidia-matmul-heuristics` (`nvmmh_heuristic.py`), which provides analytic heuristic algorithms to directly select tile/cluster configurations based on problem shape, without benchmarking. For untuned GEMM (e.g., test scenarios), this avoids the overhead of full autotuning.

### Config Pruning

The Autotuner supports providing the following via the `prune_configs_by` parameter:
- `perf_model` -- performance model estimates runtime, keeping only the top `top_k` best configs
- `early_config_prune` -- rule-based early pruning (e.g., filtering unreasonable configs based on num_stages)

## PyTorch Integration

QuACK achieves seamless integration with eager mode and `torch.compile` through three PyTorch mechanisms.

### `torch.library.custom_op`

The low-level implementation of each kernel is registered as `custom_op`, specifying `device_types="cuda"` and `mutates_args`:

```python
@torch.library.custom_op("quack::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: Tensor, out: Tensor) -> None:
    ...
```

### `register_fake` for `torch.compile`

Each `custom_op` registers a `register_fake` implementation, serving two purposes:

1. **`torch.compile` tracing** -- prevents dynamo from tracing the real implementation (which contains `cute.compile` calls that would cause graph breaks)
2. **`--compile-only` precompile mode** -- when `COMPILE_ONLY=True`, `register_fake` triggers compilation but does not execute the kernel

```python
@_softmax_fwd.register_fake
def _softmax_fwd_fake(x, out):
    from quack.cache_utils import COMPILE_ONLY
    if COMPILE_ONLY and not isinstance(x.size(1), torch.SymInt):
        N = x.size(1)
        dtype, out_dtype = [torch2cute_dtype_map[t.dtype] for t in [x, out]]
        _compile_softmax_fwd(dtype, out_dtype, N)
```

Note the `not isinstance(..., torch.SymInt)` guard: under `torch.compile`, dynamo traces with `SymInt`, which cannot be used rules to compile concrete kernels.

### `torch.autograd.Function` for backward

Higher-level APIs (such as `SoftmaxFunction`, `ChunkedLinearCrossEntropyFunction`) use `torch.autograd.Function` to wrap forward/backward `custom_op`, providing automatic differentiation support:

```python
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ...):
        _softmax_fwd(x, out)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, dout):
        (out,) = ctx.saved_tensors
        _softmax_backward(dout, out, dx)
        return dx, None
```

The `_LinearOps` family of classes in `linear.py` encapsulates forward/backward matmul function configurations for different GEMM variants.

## Variable-Length (Ragged) Sequence Support

QuACK supports variable-length sequences (ragged inputs) without padding, achieved through the TMA descriptor's ptr_shift technique. The core implementation is in `varlen_utils.py`.

### VarlenManager

`VarlenManager` manages offset computation for variable-length sequencesuba, supporting two modes:
- `varlen_m` -- variable-length M dimension (e.g., input sequences of different lengths), using `cu_seqlens_m`
- `varlen_k` -- variable-length K dimension (e.g., KV caches of different lengths), using `cu_seqlens_k`

```python
@mlir_namedtuple
class VarlenArguments(NamedTuple):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None
```### TMA ptr_shift Trick

To handle variable-length sequences without updating the TMA descriptor, QuACK creates a higher-order tensor (extra dim) with a fixed `big_int` dimension. Two approaches:

| Approach | Extra dims | Maximum input rank | Whether pointer is shifted |
|------|-----------|---------------|-------------|
| `ptr_shift=True` (1-extra-dim) | 1 | 4D (4+1=5, TMA maximum) | Yes, offset forward by `big_int * stride * elem_bytes` |
| `ptr_shift=False` (2-extra-dim) | 2 | 3D (3+2=5, TMA maximum) | No (leverages 64-bit address wraparound) |

Both approaches ensure the final address is correct through algebraic operations (extra terms cancel each other out).

### Load vs Store Asymmetry

Key finding: **TMA load validates the `globalAddress` field in the TMA descriptor**. If `ptr_shift=True` causes the base address to point to unmapped GPU memory, TMA load will fail, even if the actual data access address (after applying coordinates) is valid. **TMA store does not have this issue**, because the store path only accesses the computed target address.

Therefore, QuACK's actual strategy is:
- **TMA load** (reading A, B, C): Uses `ptr_shift=False` (2-extra-dim, maximum 3D input)
- **TMA store** (writing D): Can use `ptr_shift=True` (1-extra-dim, maximum 4D input)

This is automatically selected based on the tensor's rank in `offset_batch_A()` / `offset_batch_B()` / `offset_batch_epi()`:

```python
def offset_batch_A(self, mA_mkl, batch_idx):
    if const_expr(self.varlen_k):
        offset = params.cu_seqlens_k[batch_idx]
        ragged_rank = const_expr(cute.rank(mA_mkl))
 if const_expr(ragged_rank == 2): # create ragged tensor
            mA_mk = cute.domain_offset((None, offset), mA_mkl)
        else:
            length = params.cu_seqlens_k[batch_idx + 1] - offset
            ptr_shift = const_expr(ragged_rank == 3)  # rank 3 = 1-extra-dim
            mA_mk = copy_utils.offset_ragged_tensor(
                mA_mkl, offset, length, ragged_dim=1, ptr_shift=ptr_shift,
            )
```

## Key Design Patterns

### `Constexpr` vs Dynamic Parameters

The core distinction in CuTeDSL is **compile-time (Constexpr) vs runtime (Dynamic)** values. QuACK extensively leverages this mechanism for compile-time specialization:

- `cutlass.const_expr()` -- Marks conditions as compile-time evaluated, different branches produce different compilation artifacts
- `cutlass.range_constexpr()` -- Compile-time loop unrolling
- `cutlass.Constexpr[T]` type annotations -- Marks NamedTuple fields as compile-time constants, TVM-FFI runtime skips these fields

```python
if const_expr(self.varlen_m):
 # varlen_m=True compilation
    mA_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx], None), mA_mkl)
```

Dynamic control flow limitations:
- `break` / `continue` / `return` are not supported (inside loops or if statements)
- Variables defined inside control flow are not visible outside
- Variable types cannot be changed within control flow
- All types must be determined at compile time (dependent types not supported)

### `NamedTuple` Kernel Parameters + `@mlir_namedtuple`

QuACK uses `NamedTuple` to organize kernel's structured parameters (epilogue args, varlen args, scheduler args), and adds MLIR value reconstruction capability via the `@mlir_namedtuple` decorator:

```python
@mlir_namedtuple
class EpilogueArguments(NamedTuple):
    mPostAct: cute.Tensor
 act_fn: cutlass.Constexpr[Callable] = None # compilation
    alpha: Optional[Float32 | cute.Tensor] = None
    beta: Optional[Float32 | cute.Tensor] = None
```

`@mlir_namedtuple` adds the `__new_from_mlir_values__` method: iterates over fields, `None` and `StaticTypes` remain unchanged, complex types (`cute.Tensor`, pointer) consume the corresponding number of MLIR values for reconstruction.

### `ParamsBase` Dataclass Parameters

`ParamsBase` is the dataclass version of the parameter base class, used for JIT-level structures. It automatically separates fields into constexpr (static) and non-constexpr (dynamic), extracting only the MLIR values of dynamic fields:

```python
@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values
```### Symbolic Shapes

Create symbolic dimensions for building fake tensors at compile time. Reuse the same symbolic dimension across multiple tensors to declare equality constraints, which TVM-FFI verifies at runtime.

For varlen scenarios, symbolic dimensions need to be reassigned:

```python
if varlen_m:
 m = cute.sym_int # m total_m(column)
 a_m = cute.sym_int if gather_A else m # A differentrow
    mA = fake_tensor(a_dtype, (a_m, k), ...)
    mD = fake_tensor(d_dtype, (m, n), ...)
```

## Two-Phase Test Workflow

QuACK leverages the compile-execute separation mechanism to significantly accelerate testing:

```bash
# Pass 1: rowcompilation kernel(requires GPU memory)
pytest tests/test_softmax.py --compile-only -n 64

# Pass 2: row( .o cache)
pytest tests/test_softmax.py
```

The plugin in `conftest.py`: sets `COMPILE_ONLY=True`, calls `torch.cuda.init()` (before FakeTensorMode), globally enters `FakeTensorMode`, and swallows all test errors (only checking whether compilation succeeds).

## Key File Index

| File | Responsibility |
|------|------|
| `cache_utils.py` | `@jit_cache` decorator, `FileLock`, `COMPILE_ONLY` flag |
| `autotuner.py` | `Autotuner` (benchmark + parallel precompilation), `FileCacheManager` |
| `_compile_worker.py` | Persistent subprocess worker: `FakeTensorMode` + `COMPILE_ONLY` loop |
| `compile_utils.py` | `make_fake_tensor()` -- symbolic CuTe tensor for compilation |
| `gemm_tvm_ffi_utils.py` | `make_fake_gemm_tensors()`, `compile_gemm_kernel()` |
| `cute_dsl_utils.py` | `@mlir_namedtuple`, `ParamsBase`, Constexpr converter patch |
| `gemm_config.py` | `GemmConfig` dataclass, SM90/SM100 config generation |
| `gemm_interface.py` | `custom_op` registration, `register_fake`, autotuner integration |
| `varlen_utils.py` | `VarlenManager`, `VarlenArguments` |
| `reduction_base.py` | Reduction kernel base class (cluster, tiled copy, mbarrier) |
| `nvmmh_heuristic.py` | nvidia-matmul-heuristics analytical config selection |
| `copy_utils.py` | Memory copy operations (shared/register, async copy, tiled copy) |
| `layout_utils.py` | Layout algebra (transpose, select, expand, permute) |

## Cross-References

- [CuTeDSL Programming Model](cutedsl-programming-model.md) -- CuTeDSL programming model fundamentals, `@jit`/`@kernel` decorators, type system
- [Pipeline Patterns](cutedsl-pipeline-patterns.md) -- CuTeDSL pipeline patterns (WGMMA, TMA, mbarrier), on which QuACK GEMM's pingpong/cooperative patterns are based
- [CUTLASS GEMM Optimization](cutlass-gemm-optimization.md) -- CUTLASS 3.x GEMM optimization strategies, which QuACK's tile scheduling and epilogue fusion reference
- [CUTLASS Tile Scheduling](cutlass-tile-scheduling.md) -- Tile scheduler design, QuACK's `tile_scheduler.py` implements persistent kernel tile allocation


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CuTeDSL Programming Model](cutedsl-programming-model.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS GEMM Optimization Strategy](cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
