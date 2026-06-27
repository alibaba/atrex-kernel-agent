# [FlyDSL Original Documentation] Python API Reference

Applicability: backend: flydsl; hardware: amd; topic: reference

> Source: [FlyDSL](https://github.com/ROCm/FlyDSL) `docs/api/dsl.rst` + `docs/api/compiler.rst` + `docs/api/kernels.rst` | Compiled Version: [FlyDSL Programming Guide](flydsl-programming-guide.md)

> Complete Python API reference for the FlyDSL package, synthesized from the Sphinx API docs.

---

## 1. Compiler API (`flydsl.compiler`)

```python
import flydsl.compiler as flyc
```

| API | Description |
|-----|-------------|
| `@flyc.kernel` | Decorator for GPU kernel functions |
| `@flyc.jit` | Decorator for host-side JIT launch functions |
| `flyc.from_dlpack(tensor)` | Convert DLPack-compatible tensors (PyTorch, etc.) to FlyDSL |
| `JitArgumentRegistry` | Registry for custom argument type adapters |
| `CompilationContext` | Context object available during kernel compilation |

### Compilation Flow

On first call, `@flyc.jit` runs the following pipeline:

1. **AST Rewriting**: Python source → MLIR ops
2. **MLIR Module Construction**: Kernel body traced into `gpu`, `arith`, `scf`, `memref`, `fly` dialect ops
3. **Fly Pass Pipeline**:
   - `gpu-kernel-outlining`
   - `fly-canonicalize`
   - `fly-layout-lowering`
   - `convert-fly-to-rocdl`
   - `canonicalize` + `cse`
   - `gpu.module(convert-gpu-to-rocdl{...})`
   - `rocdl-attach-target{chip=gfxNNN}`
   - `gpu-to-llvm`, `convert-arith/func-to-llvm`
   - `gpu-module-to-binary{format=fatbin}`
4. **Cached Artifact**: Compiled binary cached to `~/.flydsl/cache/`

### Tensor Arguments

```python
tA = flyc.from_dlpack(torch_tensor).mark_layout_dynamic(
    leading_dim=0, divisibility=4
)
```

### fly-opt CLI

```bash
fly-opt --fly-canonicalize input.mlir
fly-opt --fly-layout-lowering input.mlir
fly-opt --help
```

---

## 2. Expression API (`flydsl.expr`)

```python
import flydsl.expr as fx
```

### 2.1 Type Annotations

| Type | Description |
|------|-------------|
| `fx.Tensor` | GPU tensor argument |
| `fx.Constexpr[int]` | Compile-time constant |
| `fx.Int32` | Dynamic int32 argument |
| `fx.Float32`, `fx.Float16`, `fx.BFloat16`, `fx.Float8` | Scalar types |
| `fx.Stream` | GPU stream argument |
| `fx.T` | Type namespace (`T.f32`, `T.f16`, `T.bf16`, `T.i8`, `T.index`, etc.) |

### 2.2 Layout Construction

| API | Description |
|-----|-------------|
| `fx.make_layout(shape, stride)` | Create layout from shape and stride tuples |
| `fx.make_shape(*dims)` | Create shape tuple |
| `fx.make_stride(*strides)` | Create stride tuple |
| `fx.make_coord(*coords)` | Create coordinate tuple |
| `fx.make_ordered_layout(shape, order)` | Layout with explicit mode ordering |
| `fx.make_identity_layout(shape)` | Identity layout (strides = prefix products) |
| `fx.make_identity_tensor(shape)` | Identity coordinate tensor |

### 2.3 Layout Inspection

| API | Description |
|-----|-------------|
| `fx.size(layout)` | Total number of elements |
| `fx.cosize(layout)` | Codomain size |
| `fx.rank(layout)` | Number of modes |
| `fx.depth(layout)` | Nesting depth |
| `fx.get_shape(layout)` | Extract shape tuple |
| `fx.get_stride(layout)` | Extract stride tuple |
| `fx.get_scalar(int_tuple, idx)` | Extract scalar from nested int tuple |
### 2.4 Layout Algebra

| API | Description |
|-----|-------------|
| `fx.composition(a, b)` | Compose two layouts |
| `fx.complement(layout, codomain_size)` | Complementary layout |
| `fx.right_inverse(layout)` | Right inverse |
| `fx.coalesce(layout)` | Coalesce contiguous modes |
| `fx.recast_layout(layout, old_type, new_type)` | Recast layout for type change |

### 2.5 Layout Products & Divides

| API | Description |
|-----|-------------|
| `fx.logical_divide(tensor, tiler)` | Partition tensor by tiler layout |
| `fx.zipped_divide(tensor, tiler)` | Zipped divide variant |
| `fx.tiled_divide(tensor, tiler)` | Tiled divide variant |
| `fx.flat_divide(tensor, tiler)` | Flat divide variant |
| `fx.logical_product(a, b)` | Layout product |
| `fx.zipped_product(a, b)` | Zipped product variant |
| `fx.tiled_product(a, b)` | Tiled product variant |
| `fx.flat_product(a, b)` | Flat product variant |
| `fx.raked_product(thr_layout, val_layout)` | Interleaved (raked) product |
| `fx.block_product(a, b)` | Blocked product |

### 2.6 Coordinate Mapping

| API | Description |
|-----|-------------|
| `fx.crd2idx(coord, shape, stride)` | Coordinate to linear index |
| `fx.idx2crd(idx, shape)` | Linear index to coordinate |
| `fx.slice(tensor, slices)` | Slice tensor by coordinates or `None` |
| `fx.get(layout, idx)` | Access element at index |

### 2.7 Memory Operations

| API | Description |
|-----|-------------|
| `fx.memref_alloca(type, layout)` | Allocate register-file memory |
| `fx.memref_load(memref, indices)` | Scalar load from memref |
| `fx.memref_store(value, memref, indices)` | Scalar store to memref |
| `fx.memref_load_vec(memref)` | Load entire register as a vector |
| `fx.memref_store_vec(vec, memref)` | Store vector to register memref |
| `fx.make_fragment_like(tensor)` | Allocate register fragment with same layout |

### 2.8 Copy & GEMM

| API | Description |
|-----|-------------|
| `fx.make_copy_atom(instr, dtype)` | Create CopyAtom from instruction descriptor |
| `fx.make_mma_atom(instr, dtype)` | Create MmaAtom from MFMA descriptor |
| `fx.make_tile(layouts)` | Build tile from layout list |
| `fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)` | Build TiledCopy |
| `fx.make_tiled_mma(mma_atom, ...)` | Build TiledMma |
| `fx.copy(tiled_copy, src, dst, pred=None)` | Execute tiled copy (optional predicate mask) |
| `fx.gemm(tiled_mma, accum, A, B)` | Execute tiled matrix multiply-accumulate |
| `fx.copy_atom_call(atom, src, dst)` | Invoke single copy atom |
| `fx.mma_atom_call(atom, accum, A, B)` | Invoke single MMA atom |

### 2.9 Derived Tiled Operations (`flydsl.expr.derived`)

| Class/Function | Description |
|----------------|-------------|
| `CopyAtom` | Single hardware copy instruction descriptor |
| `MmaAtom` | Single MMA instruction descriptor (MFMA) |
| `TiledCopy` | Multi-thread tiled copy; `get_slice(tid)` → `ThrCopy` |
| `TiledMma` | Multi-thread tiled MMA; `get_slice(tid)` → `ThrMma` |
| `ThrCopy` | Per-thread copy: `partition_S(src)`, `partition_D(dst)`, `retile(t)` |
| `ThrMma` | Per-thread MMA: `partition_A(a)`, `partition_B(b)`, `partition_C(c)` |
| `make_layout_tv(thr, val)` | Build thread-value layout |
| `make_tiled_copy_A/B/C(copy_atom, tiled_mma)` | TiledCopy matched to MMA operands |

---

## 3. GPU Intrinsics (`flydsl.expr.gpu`)

| API | Description |
|-----|-------------|
| `fx.thread_idx` | Thread index (`Tuple3D` with `.x`, `.y`, `.z`) |
| `fx.block_idx` | Block index |
| `fx.block_dim` | Block dimensions |
| `fx.grid_dim` | Grid dimensions |
| `fx.gpu.barrier()` | Workgroup barrier synchronization |
| `fx.gpu.smem_space()` | Shared memory (LDS) address space attribute |

---

## 4. Arithmetic (`flydsl.expr.arith`)

```python
from flydsl.expr import arith
```

Operator-overloaded arithmetic via `ArithValue`:

| API | Description |
|-----|-------------|
| `arith.constant(value, type=None, index=False)` | Create constant |
| `arith.constant_vector(value, vec_type)` | Create splat vector constant |
| `arith.index_cast(target_type, value)` | Cast to/from index type |
| `arith.select(cond, true_val, false_val)` | Ternary select |
| `arith.cmpi(predicate, lhs, rhs)` | Integer comparison |
| `arith.cmpf(predicate, lhs, rhs)` | Float comparison |
| `arith.sitofp(type, value)` | Signed int to float |
| `arith.trunc_f(type, value)` | Truncate float precision |
| `ArithValue` | Wrapper enabling `+`, `-`, `*`, `/`, `%`, `<<`, `>>` operators |

---

## 5. Vector Operations (`flydsl.expr.vector`)

| API | Description |
|-----|-------------|
| `vector.from_elements(type, elements)` | Construct vector from scalars (auto-unwraps ArithValue) |
| `vector.store(value, memref, indices)` | Store vector to memref |
| `vector.extract(vector, position)` | Extract element from vector |
| `vector.load_op(result_type, memref, indices)` | Load vector from memref |
| `vector.bitcast(result_type, source)` | Bitcast vector type |
| `vector.broadcast` | Broadcast operations (from MLIR upstream) |
| `vector.reduction` | Reduce operations (from MLIR upstream) |

---

## 6. Buffer Operations (`flydsl.expr.buffer_ops`)

AMD CDNA3/CDNA4 buffer load/store with hardware bounds checking:

```python
from flydsl.expr import buffer_ops

rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=T.i32)
buffer_ops.buffer_store(data, rsrc, offset, mask=is_valid)
```

| API | Description |
|-----|-------------|
| `create_buffer_resource(tensor, num_records=None, max_size=False)` | Create buffer descriptor |
| `buffer_load(rsrc, offset, vec_width, dtype, soffset_bytes, mask)` | Vector buffer load |
| `buffer_store(data, rsrc, offset, soffset_bytes, mask)` | Buffer store |
| `BufferResourceDescriptor` | Descriptor dataclass |

---

## 7. ROCDL Operations (`flydsl.expr.rocdl`)

AMD-specific operations for ROCm:

| API | Description |
|-----|-------------|
| `fx.rocdl.make_buffer_tensor(tensor)` | Create buffer resource from tensor |
| `fx.rocdl.BufferCopy32b` / `BufferCopy128b` | Buffer copy instruction atoms |
| `fx.rocdl.MFMA(M, N, K, acc_type)` | MFMA instruction atom constructor |
| `fx.rocdl.sched_mfma(cnt)` | Insert MFMA scheduling barrier |
| `fx.rocdl.sched_vmem(cnt)` | Insert VMEM scheduling barrier |
| `fx.rocdl.sched_dsrd(cnt)` | Insert DS read scheduling barrier |
| `fx.rocdl.sched_dswr(cnt)` | Insert DS write scheduling barrier |
| `fx.rocdl.exp2(type, x)` | Hardware exp2 (single VALU cycle) |
| `fx.rocdl.rcp(type, x)` | Hardware reciprocal (single VALU cycle) |
| `fx.rocdl.ds_bpermute(idx, src)` | Warp shuffle via LDS |
| `mfma_f32_16x16x16f16` | FP16 MFMA intrinsic |
| `mfma_f32_16x16x32_fp8_fp8` | FP8 MFMA intrinsic |
| `mfma_i32_16x16x32_i8` | INT8 MFMA intrinsic |
| `mfma_f32_16x16x16bf16_1k` | BF16 1K MFMA intrinsic |
| `mfma_scale_f32_16x16x128_f8f6f4` | GFX950 scaled MFMA (MXFP4) |

---

## 8. Pre-built Kernels (`kernels/`)

| Module | Description |
|--------|-------------|
| `kernels.preshuffle_gemm` | MFMA GEMM with LDS pipeline and pre-shuffled weights (FP8, INT8, BF16) |
| `kernels.blockscale_preshuffle_gemm` | Block-scale (MXFP4) preshuffle GEMM |
| `kernels.moe_gemm_2stage` | MoE GEMM with 2-stage pipeline |
| `kernels.mixed_moe_gemm_2stage` | Mixed-precision MoE GEMM |
| `kernels.moe_blockscale_2stage` | MoE with block-scale quantization |
| `kernels.moe_reduce` | MoE reduction: `Y[t,d] = sum(X[t,:,d])` |
| `kernels.pa_decode_fp8` | Paged attention decode with FP8 |
| `kernels.flash_attn_func` | Flash Attention |
| `kernels.layernorm_kernel` | Layer normalization |
| `kernels.rmsnorm_kernel` | RMS normalization |
| `kernels.softmax_kernel` | Numerically stable softmax |
| `kernels.reduce` | Warp-level reduction utilities |
| `kernels.mfma_epilogues` | MFMA epilogue patterns |
| `kernels.mfma_preshuffle_pipeline` | Preshuffle data movement helpers |
| `kernels.kernels_common` | Shared constants and helpers |

---

## 9. Autotuner (`flydsl.autotune`)

| API | Description |
|-----|-------------|
| `Config(num_warps=, waves_per_eu=, maxnreg=, **kwargs)` | Single tuning configuration |
| `@autotune(configs=[...], key=[...])` | Decorator: benchmark all configs, cache best |
| `do_bench(fn, warmup=5, rep=25)` | Standalone GPU benchmarking |

---

## 10. Utilities

### SmemAllocator (`flydsl.utils.smem_allocator`)

| API | Description |
|-----|-------------|
| `SmemAllocator(ctx, arch, global_sym_name)` | LDS memory manager |
| `allocator.allocate_array(type, count)` | Allocate typed LDS array |
| `allocator.get_base()` | Get LDS base pointer (inside kernel) |
| `allocator.finalize()` | Emit `memref.global` in GPU module |
| `SmemPtr` | Typed shared memory pointer with `.load()` / `.store()` |

### EnvManager (`flydsl.utils.env`)

Typed environment variable configuration with 30+ variables. See [Architecture Reference](flydsl-ref-architecture.md) §5 for the full list.

### Device Detection (`flydsl.runtime.device`)

| API | Description |
|-----|-------------|
| `get_rocm_arch()` | Detect GPU architecture (priority: `FLYDSL_GPU_ARCH` → `HSA_OVERRIDE_GFX_VERSION` → `rocm_agent_enumerator` → default `gfx942`) |
