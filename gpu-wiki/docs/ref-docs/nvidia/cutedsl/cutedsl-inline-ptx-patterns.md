# CuTeDSL Inline PTX Writing Overview

Source: `tilelang/contrib/cutedsl/` (CuTeDSL backend of TileLang).
Corresponding reference-kernels: `reference-kernels/nvidia/hopper/cutedsl/tilelang/`.

CuTeDSL does not have an equivalent to Triton's `tl.inline_asm`, but PTX can be emitted directly via `cutlass._mlir.dialects.llvm.inline_asm`. This article extracts common patterns from this backend, suitable for embedding PTX instructions not covered by the NVVM dialect (atomic, ldmatrix, mma, lop3, IEEE math, grid sync, etc.) in CuTeDSL.

---

## 1. Minimal Skeleton

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.base_dsl.typing import Float32, Int32, Uint32

@dsl_user_op
def my_op(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
 T.f32, # 1. returnstype (MLIR Type or None)
 [Float32(a).ir_value, # 2. input (MLIR Value column)
             Float32(b).ir_value()],
 "add.rn.f32 $0, $1, $2;", # 3. PTX , $0..$N operand
 "=f,f,f", # 4. (LLVM constraint string)
 has_side_effects=False, # 5. none CSE/DCE
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
```

Key points:
- Must use the `@dsl_user_op` decorator so that CuTeDSL injects `loc`/`ip`, enabling inline asm to be embedded into the JIT-compiled MLIR.
- Inputs must first be converted to the Numeric type of `cutlass` (`Float32`, `Int32`, `Uint32` …), then use `.ir_value()` to obtain the MLIR Value.
- The return type must be an MLIR `Type` (e.g., `T.f32()`, `T.i32()`); do not pass `cutlass.Float32` types.
- The obtained MLIR Value must then be wrapped back into a CuTeDSL Numeric using `Float32(result)`.

---

## 2. Constraint String Conventions

LLVM inline asm constraint syntax is consistent with CUDA inline asm:

| Constraint | Meaning | Corresponding CUDA `asm` |
|------|------|----------------|
| `=r` / `r` | 32-bit integer register | `r` |
| `=h` / `h` | 16-bit integer register | `h` |
| `=l` / `l` | 64-bit integer (including generic/global pointer) | `l` |
| `=f` / `f` | 32-bit float | `f` |
| `=d` / `d` | 64-bit double | `d` |
| `=` prefix | Output | Same |
| `+` prefix | Input-output (read-write) | Same |

The order of the constraint string must strictly match the positions of `$0..$N` in the PTX template:
- Write all output constraints first (with `=`), then all input constraints.
- `$0` corresponds to the first output, `$N_outputs` to the first input.
- Multiple outputs rely on `llvm.StructType.get_literal([T.i32(), T.i32()])` as the return type, then unpacked using `llvm.extractvalue`.

---

## 3. Multiple Outputs (Struct Return)

PTX `ldmatrix.x4` produces 4 `i32` at once:

```python
ret_type = llvm.StructType.get_literal([T.i32()] * 4)
out_struct = llvm.inline_asm(
    ret_type,
    [smem_ptr.llvm_ptr],
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];",
    "=r,=r,=r,=r,l",
    has_side_effects=True,
    ...
)
for i in range(4):
    out_tensor[i] = cute.Int32(
        llvm.extractvalue(T.i32(), out_struct, [i], loc=loc, ip=ip)
    )
```

`vec atom.add.v2.f32` follows the same pattern: return type uses `StructType`, with `{$0,$1}` written in the template.

---

## 4. Getting Pointers / Tensor Elements

CuTeDSL wraps pointers as `cute.Pointer`; inline asm uses the underlying `llvm_ptr`:

```python
nvvm.cp_async_shared_global(dst=dst.llvm_ptr, src=src.llvm_ptr, ...)
llvm.inline_asm(..., [ptr.llvm_ptr, val_ir], "atom.add.f32 ...", ...)
```

If the input comes from a register tensor, you need:

```python
a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
a_tensor = cute.make_tensor(a_base, (n_a,))
a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_a)]
`````recast_ptr`` is key: many PTX instructions (mma, ldmatrix) require fp16/bf16 to be input as ``i32``.

---

## 5. fp16 / bf16 via i16 bitcast

LLVM inline asm does not support ``f16`` directly as an operand, but PTX does. You need to bitcast to ``i16`` first, then use ``=h``/``h`` constraints:

````python
val_ir  = cutlass.Float16(value).ir_value(loc=loc, ip=ip)
val_i16 = llvm.bitcast(T.i16(), val_ir, loc=loc, ip=ip)

res_i16 = llvm.inline_asm(
    T.i16(),
    [ptr.llvm_ptr, val_i16],
    "atom.add.noftz.f16 $0, [$1], $2;",
    "=h,l,h",
    has_side_effects=True,
    ...
)
result = llvm.bitcast(T.f16(), res_i16, loc=loc, ip=ip)
return cutlass.Float16(result)
````

Same for bf16 (``=h,h``), with ``f16x2``/``bf16x2`` typically packed as ``b32`` or ``i32x2``.

---

## 6. Multi-line PTX + temporary registers

PTX allows declaring ``.reg``/``.pred``/labels inside ``{ ... }``. Multi-line strings are passed directly to ``llvm.inline_asm``:

````python
llvm.inline_asm(
    T.f32(),
    [ptr.llvm_ptr, val_ir],
    """
    {
        .reg .pred p;
        .reg .f32 expected, new_val;
        .reg .b32 expected_bits, new_bits, result_bits;
        ld.f32 expected, [$1];
    retry:
        max.f32 new_val, expected, $2;
        mov.b32 expected_bits, expected;
        mov.b32 new_bits, new_val;
        atom.cas.b32 result_bits, [$1], expected_bits, new_bits;
        setp.ne.b32 p, result_bits, expected_bits;
        mov.b32 expected, result_bits;
        @p bra retry;
        mov.f32 $0, expected;
    }
    """,
    "=f,l,f",
    ...
)
````

Suitable for: CAS loop (float atomic max/min), ``mbarrier.try_wait`` spin, grid soft sync.

Note: ``@!p bra LAB``, ``bra`` label syntax is exactly the same as hand-written ``asm volatile``; ``%p`` / ``%r`` naming prefixes (e.g., ``%r_tid``) are also supported.

---

## 7. Templates + factory functions

inline asm templates have many variants (layout=row/col, shape=m16n8k16/m16n8k32 …). ``ptx_mma.py`` uses factories to build a set:

````python
def _make_ptx_mma(ptx_shape, ptx_dtypes, n_a, n_b, n_c, flavor):
    constraints = ",".join(
        [f"={c_con}"] * n_c +
        [ab_con] * (n_a + n_b) +
        [c_con] * n_c
    )
    ptx_template = (
        f"mma.sync.aligned.{ptx_shape}.{{a_layout}}.{{b_layout}}.{ptx_dtypes}"
        f" {{{{{d_regs}}}}}, {{{{{a_regs}}}}}, {{{{{b_regs}}}}}, {{{{{c_regs}}}}};"
    )

    @dsl_user_op
    def mma_op(... a_layout="row", b_layout="col", ...):
        ptx_asm = ptx_template.format(a_layout=a_layout, b_layout=b_layout)
        result = llvm.inline_asm(res_type, a_vals + b_vals + c_vals,
                                 ptx_asm, constraints, ...)
        ...

    return mma_op

ptx_mma_m16n8k16_f16_f16_f32 = _make_ptx_mma("m16n8k16", "f32.f16.f16.f32", 4, 2, 4, "f32")
````

Note: in ``f-string``, ``{{`` ``}}`` is escaped; if you want to keep ``{a_layout}`` placeholders for ``format``, add an extra wrapping layer.

``ieee_math.py`` similarly uses f-string templates to generate 8 variants like ``add.{rounding}.f32`` with rn/rz/rm/rp.

---

## 8. Prefer NVVM dialect, then inline asm

Many instructions already have ops in ``cutlass._mlir.dialects.nvvm`` and should be preferred to avoid maintaining PTX strings:

````python
nvvm.cp_async_shared_global(...)
nvvm.cp_async_bulk_tensor_shared_cluster_global(...)  # TMA load
nvvm.cp_async_bulk_tensor_global_shared_cta(...)      # TMA store
nvvm.ldmatrix(ret_type, smem_ptr.llvm_ptr, num=4, layout=nvvm.MMALayout.row, ...)
nvvm.stmatrix(smem_ptr.llvm_ptr, ir_values, layout=...)
nvvm.atomicrmw(T.f32(), AtomicOpKind.FADD, ptr.llvm_ptr, val, ...)
nvvm.fmin / nvvm.fmax / nvvm.prefetch_tensormap / nvvm.cp_async_bulk_commit_group
````Inline asm is only used for:
- Instructions not covered by NVVM (`activemask`, `atom.add.noftz.f16`, `atom.global.add.v4.f32`, `tanh.approx.f32`, `lop3.b32 + sub.f16x2`, `mma.sync` all dtype combinations, IEEE explicit rounding, `mbarrier.try_wait` spin, CAS loop, `prmt.b32`, `mul.bf16x2`, `fence.sc.gpu` with `ld.relaxed.gpu`, grid sync protocol)
- When you need precise control over a single instruction / combined multi-instruction sequences

---

## 9. How to Choose `has_side_effects`

| Scenario | `has_side_effects` |
|------|-------------------|
| Pure arithmetic (add/sub/mul/fma/rcp/sqrt/tanh/exp, bit pack, prmt, lop3+sub) | `False`, allows CSE / DCE |
| Any atomic (add/max/min/CAS), load/store, TMA, mbarrier, bar.sync, grid sync | `True` |
| `activemask` | `False` (can be treated as constant within the same warp; set `True` if reordering is a concern) |
| ldmatrix | `True` (implies SMEM access) |

If you get it wrong:
- Writing `True` as `False`: the compiler may optimize away, hoist, or merge the instruction.
- Writing `False` as `True`: degraded performance but behavior remains correct.

---

## 10. Common Pitfalls

1. **In PTX templates, `$0` is the output, not the first input**.
   When writing `mma.sync ... {$0..$3}, {$4..$7}, {$8,$9}, {$10..$13}`, you must count operands in D-A-B-C order.
2. **fp16 cannot be used directly as an inline asm operand**; it must be `llvm.bitcast` to `i16` + `=h`/`h`.
3. **Pointer constraint must be written as `l`** (not `r`). For SMEM pointers, it is recommended to add the `.shared::cta` modifier in PTX.
4. **Forgetting `StructType.get_literal` for multiple outputs**. If you only return `T.i32()`, writing multiple `=` outputs in the template will cause a compilation failure.
5. **f-string brace escaping**: When the template needs to keep `{ ... }` PTX blocks while inserting `a_layout` via `format`, you must use `{{...}}` with double braces, then `{single}` for the placeholder.
6. **`@dsl_user_op` cannot be omitted**. Otherwise, `loc`/`ip` injection will fail, and the inline asm won't be embedded into the correct IR scope.
7. **CAS spin uses b32 comparison, not f32 comparison**. NaN will cause `setp.ne.f32` to always be true, resulting in an infinite loop (see `AtomicMax` comments).
8. **`mbarrier.try_wait` must be written as a loop**: breaking out after a single failed try leads to a race condition; you must `@!p bra LAB_WAIT`.
9. **Do not add `.global` to `atom.add.noftz.f16`**; PTX will automatically select the state space when none is explicitly specified. Explicitly adding one may cause errors in certain SMEM scenarios.
10. **`v4.f32` must use `atom.global.add.v4.f32`** to hit the true vectorized atomic on SM90+; otherwise, PTX will split it into 4 separate operations.

---

## 11. Index: Which file demonstrates which pattern

| File | Primary Pattern |
|------|-----------------|
| [`utils.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/utils.py) | `bitcast`, `pack_half2` (minimal inline asm examples) |
| [`ieee_math.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/ieee_math.py) | f-string template + explicit rounding modes (rn/rz/rm/rp) |
| [`math.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/math.py) | `tanh.approx.f32` single instruction |
| [`atomic.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/atomic.py) | `atom.add.noftz.f16/v2.f16/v2.bf16`, CAS spin atomic max/min, `fence.sc.gpu + ld.relaxed.gpu` |
| [`cpasync.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/cpasync.py) | `mbarrier.try_wait` spin; otherwise prefer NVVM dialect |
| [`ptx_mma.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/ptx_mma.py) | `mma.sync.aligned` factory, Dense + Sparse all dtype combinations |
| [`ldsm.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/ldsm.py) | NVVM `ldmatrix`/`stmatrix` wrapper (no PTX) |
| [`quantize.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/quantize.py) | `lop3.b32 + sub.f16x2` quantized decode, `prmt.b32` + `mul.bf16x2` FP4→BF16 twiddling |
| [`reduce.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/reduce.py) | `bar.sync $0, $1` (multiple barrier ids), warp shuffle patterns |
| [`warp.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/warp.py) | `activemask.b32` single instruction |
| [`grid_sync.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/tilelang/grid_sync.py) | Multi-line PTX + `.reg`/`.pred` soft sync protocol |## 12. Relationship with SM120 NVFP4 Inline PTX

 uses the same `llvm.inline_asm` approach described in this article to issue `mma.sync.aligned.kind::mxf4nvf4...` operations. The template strings, constraint rules, `StructType` output decomposition, and `u32` packing of FP4 registers are all identical, with the only differences being:

- The SM120 nvfp4 mma has no NVVM op and requires hand-written PTX (the method described in this article).
- SM120 uses warp MMA instead of SM100 tcgen05/TMEM.
- A/B are nibble-packed into `u32`, and the scale (SF) is also `u32`.

You can use `ptx_mma.py` as a template to add a new `_make_ptx_mma_blockscaled(...)` factory, and by reusing all the rules from this article, you can extend it to SM120 NVFP4 / MXFP8 / MXFP4.
