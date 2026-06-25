# FlyDSL inline_asm Writing Guide (AMD GFX942 / gfx1250)

Source: `FlyDSL/kernels/` (FlyDSL built-in kernel library).
Corresponding reference kernels:
- CDNA: `reference-kernels/amd/cdna/flydsl/FlyDSL/`
- RDNA4 / gfx1250: `reference-kernels/amd/rdna4/flydsl/FlyDSL/`

FlyDSL is an AMD GPU DSL built on top of MLIR Transcript. Most instructions are already wrapped into `flydsl._mlir.dialects.rocdl` ops (such as `rocdl.s_waitcnt`, `rocdl.sched_barrier`, `rocdl.wave_id`) and `flydsl.expr.buffer_ops` (buffer load/store + cache modifier). The scenarios that **truly require inline assembly** are very rareega, mainly falling into two categories:

151. **Hardware control instructions speed_mm with no ROCDL op**: cache invalidation, L2 writeback, HW reg setup, s_barrier_signal/wait, prefetch instructions.
2. **Fine-grained cache modifier control bypassing buffer_ops**: In split-K scenarios, `global_load_dword` / `global_store_dword` with `sc0/sc1` are needed to avoid buffer descriptor overhead.

Below is a categorized summary of **all** inline_asm usages in the FlyDSL kernel library.

---

## 1. Minimum Skeleton (Two Equivalent APIs)

FlyDSL reuses MLIR LLVM dialect's inline asm. Both calling forms are valid:

### 1.1 `llvm.InlineAsmOp` Class-based Form

```python
from flydsl._mlir.dialects import llvm

# nonereturns + none operand
llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)

# returns + operand
data = llvm.InlineAsmOp(
 T.i32, # returnstype
 [counter_ptr_v], # input operand(MLIR Value column)
 "global_load_dword $0, $1, off sc1", # asm
 "=v,v", #
    has_side_effects=True,
).result # result SSA Value
```

### 1.2 `llvm_dialect.inline_asm` Function Form

```python
from flydsl._mlir.dialects import llvm as llvm_dialect

llvm_dialect.inline_asm(
 None, , # void result, none operand
 "s_setreg_imm32_b32 hwreg(26, 4, 1), 1", # asm
English description
    has_side_effects=True,
)
```

`rdna_f16_gemm.py` uses the keyword form:

```python
_llvm.inline_asm(
    res=None,
    operands_=[],
    asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
    constraints="",
    has_side_effects=True,
)
```

All three forms are equivalent. **Convention**: In the project, hgemm_splitk / custom_all_reduce use the `InlineAsmOp` class-based form, while the gfx1250 series uses the `inline_asm` function form.

---

## 2. Constraint Strings (AMDGPU Versions)

Unlike the writing style used by Triton/CUTLASS on NVIDIA, AMDGPU's LLVM backend uses only a few core constraints:

| Constraint | Meaning |
|------|------|
| `v` | VGPR (vector register) — input |
| `=v` | VGPR — output |
| `s` | SGPR (scalar register) |
| `=s` | SGPR — output |
| `+v` / `+s` | read-write |

99% of FlyDSL inline asm uses `v`/`=v`. In `global_load_dword $0, $1, off sc1`, `$0` is the output VGPR, and `$1` is a 64-bit address (pointers also use the `v` constraint, because on AMDGPU a 64-bit address occupies two VGPRs).Unlike PTX: **AMDGPU does not have a dedicated pointer constraint** (not ``l``), addresses go directly through ``v``.

---

## 3. Eight Major Uses of AMDGPU Inline Assembly

### 3.1 Cache Control — ``buffer_inv`` / ``buffer_wbl2``

```python
# L1 scalar cache( GPU signal loop)
llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)

# L2 dirty row HBM( signal previous, GPU )
llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
```

Reference: ``custom_all_reduce_kernel.py:98,106``, ``hgemm_splitk.py:299``.

**When needed**: Multi-GPU signal/flag protocols (Custom AllReduce, Split-K barrier). ``buffer_ops.buffer_load(cache_modifier=_CM_SC1)`` only controls the load itself and will not invalidate old lines already in L1 → must be paired with ``buffer_inv sc1``.

### 3.2 Global Load/Store with Cache Modifiers

```python
# Store: bypass L1+L2, split-K counter
llvm.InlineAsmOp(
    None, [counter_ptr_v, arith.constant(1, type=T.i32)],
    "global_store_dword $0, $1, off sc0 sc1", "v,v",
    has_side_effects=True,
)

# Load: bypass L2, split-K counter
data = llvm.InlineAsmOp(
    T.i32, [counter_ptr_v],
    "global_load_dword $0, $1, off sc1", "=v,v",
    has_side_effects=True,
).result
```

Reference: ``hgemm_splitk.py:300,340``.

**Why not use ``buffer_ops.buffer_store(cache_modifier=...)``**: That path requires constructing a buffer descriptor first (``s_load_dwordx4`` + 4×SGPR), with latency an order of magnitude higher than ``global_store_dword``. On the Split-K critical path, this latency gets amplified by countless loop iterations.

### 3.3 HW Register Configuration — ``s_setreg_imm32_b32``

```python
# WGP mode / wave mode
llvm_dialect.inline_asm(
    None, [],
    "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
    "",
    has_side_effects=True,
)
```

Reference: ``moe_gemm_2stage_wmma_gfx1250.py:124,512``, ``moe_gemm_2stage_mxscale_gfx1250.py:223,1735``.

``hwreg(26, 4, 1)`` explanation: Register ID=26 (RDNA4 mode control bits for the ``HW_REG_FLAT_SCRATCH_LO`` segment), offset=4, size=1. Setting it to 1 is equivalent to enabling a certain HW mode, used on gfx1250 to put the wave into a specific scheduling mode.

**Critical**: Must be placed at the kernel **entry point**, and triggered only by the first wave that executes across the entire grid; placing it inside a hot loop will pollute the wave state.

### 3.4 Instruction Prefetch — ``s_prefetch_inst_pc_rel``

```python
if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
              arith.constant(0, type=T.i32)):
    _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
    for _pg in range_constexpr(10):
        _prefetch_lines.append(
            f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
    llvm_dialect.inline_asm(
        None, [],
        "\n".join(_prefetch_lines),
        "", has_side_effects=True,
    )
```

Reference: ``gemm_fp8fp4_gfx1250.py:373``, ``moe_gemm_2stage_mxscale_gfx1250.py:218,1730``, ``moe_gemm_2stage_common_gfx1250.py``.

**Pattern**: Use Python f-strings to expand multiple PTX lines at compile time, then join them into a single assembly block with ``\n``.join. This avoids multiple ``inline_asm`` calls (each inserts an op).

**Applicable scenario**: When the kernel body is very large (gfx1250 fused MoE GEMM 30+KB instructions), I-cache misses are inevitable on first launch. Prefetching 10 instruction pages of 4KB each can eliminate the first-launch latency.### 3.5 RDNA4 barrier — `s_barrier_signal/wait`

```python
def _barrier():
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
        constraints="",
        has_side_effects=True,
    )
```

Reference: `rdna_f16_gemm.py:264`.

**Why not use `gpu.barrier()`**: FlyDSL's default `gpu.barrier()` on RDNA4 expands into the more conservative `s_barrier`. The actual correct barrier sequence for RDNA4 is to first wait for all LDS/store counters, then do a split-barrier signal+wait. Hand-writing this is more precise than relying on compiler-generated code.

`s_wait_dscnt 0x0` = wait for all LDS operations to complete; `s_wait_storecnt 0x0` = wait for all stores to complete.

---

## 4. Template Rules Cheat Sheet

| Rule | Example |
|------|------|
| Operand references are `$0..$N` | In `global_load_dword $0, $1, off sc1`, `$0` is the output, `$1` is the input |
| Outputs come before inputs | The constraint string `=v,v` means 1 output + 1 input |
| Multi-line asm uses `\n` | `"a\nb\nc"` or `"\n".join([...])` |
| Compile-time f-string concatenation | `f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31"` unrolled in loops |
| No operand: leave empty string | `llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)` |
| Getting return values | `.result` (for `InlineAsmOp`); `inline_asm` returns directly from function |

---

## 5. How to Choose `has_side_effects`

| Scenario | `has_side_effects` |
|------|-------------------|
| Any cache invalidation / writeback (`buffer_inv`, `buffer_wbl2`) | **`True`** — required, otherwise DCE |
| HW register writes (`s_setreg_imm32_b32`) | **`True`** |
| Barrier sequences | **`True`** |
| Load/store with cache modifiers | **`True`** — required for loads to avoid being merged; required for stores |
| Instruction prefetch | **`True`** — otherwise the compiler will consider it unused and delete it |

**All inline_asm calls in the FlyDSL project use `True`**. This is a rule of thumb: inline assembly usually means you're performing side effects that the compiler cannot see; marking things as `False` very easily triggers bugs.

---

## 6. Companion ROCDL Ops (no inline_asm needed)

Before writing inline_asm, first confirm that the ROCDL dialect doesn't already cover it:

```python
from flydsl._mlir.dialects import rocdl

rocdl.s_waitcnt(0)               # s_waitcnt 0
rocdl.sched_barrier(0)           # sched_barrier mask
rocdl.wave_id # current wave ID(i32 SSA)
rocdl.s_setprio(...)             # s_setprio
```

`flydsl.expr.buffer_ops` also covers:

```python
buffer_ops.create_buffer_resource_from_addr(addr_i64)
buffer_ops.create_buffer_resource(tensor, max_size=False, num_records_bytes=...)
buffer_ops.buffer_load(rsrc, off, vec_width=4, dtype=T.i32, cache_modifier=2)
buffer_ops.buffer_store(val, rsrc, off, cache_modifier=4)
```

`cache_modifier` values (gfx942 buffer instruction aux field):
- `0` cached
- `2` SC1 — bypass L2
- `3` SC0+SC1 — bypass L1+L2
- `4` NT — nontemporal

Only fall back to inline_asm when you need `global_*_dword` (generic load/store with cache modifiers) or instructions like `buffer_inv`/`buffer_wbl2` that **have no ROCDL op**.

---

## 7. Common Pitfalls

1. **AMDGPU does not have NVPTX-style pointer constraints like `l`**. Use `v` for addresses, because at the hardware level they are 64-bit values composed of two VGPRs.
2. **The cache_modifier of `buffer_ops` does not automatically invalidate L1**: `load(cache_modifier=2)` only makes this load bypass L2; the next ordinary load will still read the old value in L1 — you must `buffer_inv sc1`.
3. **`buffer_wbl2 sc0 sc1` must precede the store signal**: CDNA3's L2 is write-back, so dirty lines are not visible across GPUs; without writing back, you get a deadlock where GPU0 wrote the signal but GPU1 never sees it.
4. **HW register writes should be placed at the kernel entry**: putting them inside a loop pollutes the wave state grad may cause other waves to also observe the wrong mode.
5. **Do not use `gpu.barrier()` as a substitute for `s_wait_dscnt + s_barrier_signal/wait` on RDNA4**: FlyDSL's default lowering does not necessarily take the fastest path.
6. **f-string expansion for prefetch must use `range_constexpr` instead of `range`**: `range_constexpr` is a compile-time loop (unroll); a regular `range` inside the JIT body triggers a runtime loop, but the inline asm template must be resolved at compile time.7. **Don't forget ``has_side_effects=True``**: especially instructions that only do prefetch / cache control without reading or writing — the compiler will treat them as dead code and delete them.
8. **Multiple `inline_asm` have higher overhead than a single large block**: each ``inline_asm`` is an independent LLVM op with inline asm parsing overhead. Combine them into one block whenever possible (see ``_prefetch_lines``).

---

## 8. Index: Which file demonstrates which type of inline_asm

| File | inline_asm Usage | Learning Focus |
|------|----------------|---------|
| [``custom_all_reduce_kernel.py``](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py) | ``buffer_inv sc1``, ``buffer_wbl2 sc0 sc1`` | Minimal cache control example under cross-GPU signal protocol |
| [``hgemm_splitk.py``](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/hgemm_splitk.py) | ``global_store_dword sc0 sc1``, ``global_load_dword sc1``, ``buffer_wbl2`` | Fine-grained cache control on split-K counter (**full example with operands + output**) |
| [``rdna_f16_gemm.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/rdna_f16_gemm.py) | ``s_wait_dscnt + s_wait_storecnt + s_barrier_signal/wait`` | RDNA4 split barrier sequence |
| [``gemm_fp8fp4_gfx1250.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/gemm_fp8fp4_gfx1250.py) | f-string concatenation of ``s_prefetch_inst_pc_rel`` × 10 | First-launch instruction prefetch for large gfx1250 kernel |
| [``moe_gemm_2stage_mxscale_gfx1250.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/moe_gemm_2stage_mxscale_gfx1250.py) | prefetch + ``s_setreg_imm32_b32 hwreg(26, 4, 1), 1`` | Wave mode control |
| [``moe_gemm_2stage_wmma_gfx1250.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/moe_gemm_2stage_wmma_gfx1250.py) | Same as above | gfx1250 WMMA fused MoE |

---

## 9. Comparison with NVIDIA CuTeDSL inline PTX

| Dimension | FlyDSL (AMDGPU) | CuTeDSL (NVIDIA) |
|------|----------------|------------------|
| API | ``llvm.InlineAsmOp(...)`` or ``llvm_dialect.inline_asm(...)`` | ``cutlass._mlir.dialects.llvm.inline_asm(...)`` |
| Decorator | None (FlyDSL calls directly inside ``@flyc.kernel``) | Must inject ``@dsl_user_op`` / ``loc`` / ``ip`` |
| Output register constraint | ``=v`` (VGPR) / ``=s`` (SGPR) | ``=r`` (i32) / ``=h`` (i16) / ``=l`` (i64) / ``=f`` (f32) / ``=d`` (f64) |
| Pointer constraint | ``v`` (address = two VGPRs) | ``l`` (generic 64-bit pointer) |
| fp16/bf16 | No special handling needed (placed directly in VGPR) | Must ``bitcast`` to i16 + ``=h`` / ``h`` |
| Multiple outputs | ``.result`` returns SSA value or vec/struct | ``llvm.StructType.get_literal([...])`` + ``extractvalue`` |
| Prefer dialect op | ``flydsl._mlir.dialects.rocdl`` + ``flydsl.expr.buffer_ops`` | ``cutlass._mlir.dialects.nvvm`` |
| Primary use cases | cache invalidation, HW reg settings, barrier, prefetch, load/store with cache modifiers | mma.sync, ldmatrix, atomic (including vectorized), IEEE rounding, quantization lop3, CAS spin, grid sync, mbarrier try_wait |

See the companion doc: .

---

## 10. Recommended Learning Path

1. Start with [``custom_all_reduce_kernel.py``](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/custom_all_reduce_kernel.py) — the shortest two-line ``InlineAsmOp`` example to get started.
2. Then study [``hgemm_splitk.py``](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/hgemm_splitk.py) for the full "input + output + constraint string + side effects" usage.
3. Next, examine [``gemm_fp8fp4_gfx1250.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/gemm_fp8fp4_gfx1250.py) to learn how to use f-string templates bell generate multi-line asm.
4. Finally, review [``rdna_f16_gemm.py``](../../../../reference-kernels/amd/rdna4/flydsl/FlyDSL/rdna_f16_gemm.py) to learn the RDNA4 split barrier.
5. Before writing, always grep ``flydsl/_mlir/dialects/rocdl`` and ``flydsl/expr/buffer_ops`` to confirm that no existing op is available before resorting to inline asm.
