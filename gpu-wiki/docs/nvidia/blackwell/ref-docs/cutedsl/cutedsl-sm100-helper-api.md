# Using the SM100 Blackwell CuTeDSL Helper Layer

An index + usage guide for the **high-level `cutlass.utils.sm100` / `blackwell_helpers` helper layer** on Blackwell (SM100/SM103): which helper exposes each hardware capability, the call signature, and the rules for using it correctly. Applies to GEMM, attention, MoE, and other tensor-core kernels.

This doc deliberately stays at the helper-API level and defers mechanism/worked examples to the docs it links:

- Generic CuTeDSL model (`@cute.jit`/`@cute.kernel`, constexpr-vs-dynamic, JIT cache, control flow): [../cutedsl-programming-model.md](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md)
- CUTLASS internals on SM100 (descriptor iterators, `calculate_umma_peer_mask`, CLC, warp roles, dispatch policies): [blackwell-cutedsl-sm100.md](blackwell-cutedsl-sm100.md)
- Pipeline classes/patterns (state machine, `.create()` signatures, role table): [../cutedsl-pipeline-patterns.md](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md)
- Worked GEMM tutorials: [blackwell-tcgen05-gemm-from-scratch.md](blackwell-tcgen05-gemm-from-scratch.md), [blackwell-gemm-tensor-memory.md](blackwell-gemm-tensor-memory.md), [blackwell-gemm-thread-block-cluster.md](blackwell-gemm-thread-block-cluster.md), [blackwell-gemm-low-precision.md](blackwell-gemm-low-precision.md), [cutedsl-nvfp4-gemm-tutorial.md](cutedsl-nvfp4-gemm-tutorial.md)

> API names/signatures are functional facts paraphrased from NVIDIA's official CuTe DSL documentation
> (<https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html>). Code is illustrative, not copied source.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils as utils
import cutlass.utils.sm100 as sm100              # high-level Blackwell helpers
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync, CooperativeGroup, PipelineOp
```

---

## 1. SM100 capability → helper map

The single thing this doc adds over the rest of the corpus: a map from each Blackwell capability to the **high-level helper** you call (not the raw atom / PTX / CUTLASS-C++ form the other docs use).

| SM100 capability | CuTeDSL helper entry point |
|---|---|
| 5th-gen MMA (UMMA / tcgen05) | `sm100.make_trivial_tiled_mma(...)` (wraps `tcgen05.MmaF16BF16Op` / `MmaFP8Op` / `MmaTF32Op` / `MmaI8Op`) |
| Block-scaled MMA (MXFP/NVFP4) | `sm100.make_blockscaled_trivial_tiled_mma(...)`; **B300/sm_103 NVFP4-Ultra: `blackwell_helpers.sm103_make_blockscaled_trivial_tiled_mma(...)`** |
| CTA-pair / 2-SM cooperation | `cta_group=tcgen05.CtaGroup.TWO` + `use_2cta_instrs=True` on the helpers; `cluster=(2,1)` |
| Tensor Memory (TMEM) | `utils.TmemAllocator(...)` (`allocate`/`retrieve_ptr`/`free`); columns via `sm100.get_num_tmem_alloc_cols(...)` |
| TMEM ↔ register movement | `sm100.get_tmem_load_op(...)` → `tcgen05.make_tmem_copy(...)`; store-out via `sm100.get_smem_store_op(...)` |
| SMEM operand layouts (TMA+UMMA-legal) | `sm100.make_smem_layout_a/b/epi(...)` or `tcgen05.make_smem_layout_atom(kind, dtype)` |
| TMA + cluster multicast | `sm100.cluster_shape_to_tma_atom_A/B/SFB(...)` or `cpasync.make_tiled_tma_atom(...)`; mask via `cpasync.create_tma_multicast_mask(...)` |
| Software pipeline | `PipelineTmaUmma` (TMA→UMMA) / `PipelineUmmaAsync` (UMMA→AsyncThread) / `PipelineAsyncUmma` / `PipelineTmaAsync` |
| Cluster / persistent grid | `utils.HardwareInfo().get_max_active_clusters(cluster_size)`; `cute.arch` cluster primitives |
| Epilogue tile sizing | `sm100.compute_epilogue_tile_shape(...)` |

Low-level primitives (`elect_one`, `mbarrier_*`, cluster indexing): [../nvidia-cutedsl-arch-primitives.md](../../../common/ref-docs/cutedsl/nvidia-cutedsl-arch-primitives.md).

---

## 2. Per-helper usage rules

Signatures plus the *correctness rules* specific to each helper. For the hardware mechanism behind each, follow the linked doc.

### 2.1 `make_trivial_tiled_mma` / `make_blockscaled_trivial_tiled_mma`

```python
tiled_mma = sm100.make_trivial_tiled_mma(
    ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
    a_leading_mode=tcgen05.OperandMajorMode.K, b_leading_mode=tcgen05.OperandMajorMode.K,
    cta_group=tcgen05.CtaGroup.ONE, mma_tiler_mn=(128, 256),
    a_source=tcgen05.OperandSource.SMEM)      # A from SMEM (default) or TMEM
```
- Issue with `cute.gemm(tiled_mma, acc, A, B, acc)` — **accumulator-first** (D = A·B + C). Issued by one elected thread; calling from many threads deadlocks.
- Accumulator lives in TMEM (2.2). Use `tcgen05.Field.ACCUMULATE = False` on the first K-tile, then `True`. `Field.NEGATE_A/B`, `Field.SFA/SFB` are runtime-settable.
- All tcgen05 ops in a kernel must share one `cta_group`.
- Block-scaled: `make_blockscaled_trivial_tiled_mma(..., sf_dtype=cutlass.Float8E4M3, sf_vec_size=16)` (NVFP4 = 16, MXFP = 32). Mechanism: [blackwell-cutedsl-sm100.md §1](blackwell-cutedsl-sm100.md), [docs/nvidia/blackwell/kernel-opt/hardware/tcgen05-mma.md](../../kernel-opt/hardware/tcgen05-mma.md).

### 2.2 `TmemAllocator` + TMEM movement helpers

```python
num_cols = sm100.get_num_tmem_alloc_cols(acc_tmem_tensors)
tmem_alloc = utils.TmemAllocator(alloc_result_dst_smem_ptr=..., barrier_for_retrieve=...,
    is_two_cta=use_2cta_instrs, num_allocated_columns=num_cols,
    two_cta_tmem_dealloc_mbar_ptr=dealloc_mbar)        # last arg 2-SM only
tmem_alloc.allocate(...); acc_ptr = tmem_alloc.retrieve_ptr(cutlass.Float32); ...; tmem_alloc.free(...)
```
Rules (violating these hangs/corrupts): always `free` before exit (`allocate` can block → back-off retry); read-back needs a **full warpgroup** (one warp reaches only 1/4 of TMEM lanes) — build it via `sm100.get_tmem_load_op(...)` → `tcgen05.make_tmem_copy(...)`; column count is power-of-two in [32, 512] (use `get_num_tmem_alloc_cols`). Full TMEM model: [docs/nvidia/blackwell/kernel-opt/hardware/tmem.md](../../kernel-opt/hardware/tmem.md), [blackwell-cutedsl-sm100.md §2](blackwell-cutedsl-sm100.md).

### 2.3 CTA-pair (2-SM)

Enable `cta_group=tcgen05.CtaGroup.TWO` consistently across the MMA helper, the TMA atoms, the pipeline, and `TmemAllocator(is_two_cta=True, ...)`; use `cluster=(2,1)`. The even CTA is leader; peer-mbarrier / `calculate_umma_peer_mask` are handled for you — don't hand-roll. When/why and internals: [docs/nvidia/blackwell/kernel-opt/hardware/2sm-cooperative.md](../../kernel-opt/hardware/2sm-cooperative.md), [blackwell-cutedsl-sm100.md §3](blackwell-cutedsl-sm100.md), [blackwell-gemm-thread-block-cluster.md](blackwell-gemm-thread-block-cluster.md).

### 2.4 TMA atoms + multicast

```python
atom_A = sm100.cluster_shape_to_tma_atom_A(cluster_shape_mnk, atom_thr_id)   # auto-multicast iff cluster dim > 1
op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.TWO)
tma_atom, gA = cpasync.make_tiled_tma_atom(op, gmem_A, smem_layout_A, cta_tiler, num_multicast=cluster_m)
mask = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode=...)
```
Also `cluster_shape_to_tma_atom_B/SFB`. Sync model and 16-byte alignment: [docs/nvidia/blackwell/kernel-opt/hardware/tma.md](../../kernel-opt/hardware/tma.md) — let the pipeline class drive the mbarriers.

### 2.5 Pipelines

Pick by producer→consumer role: **`PipelineTmaUmma`** (TMA→UMMA, mainloop), `PipelineUmmaAsync` (UMMA→AsyncThread, accumulator drain), `PipelineAsyncUmma` (input fusion), `PipelineTmaAsync` (Hopper-style). Full role table, `.create()` signatures, and state machine: [../cutedsl-pipeline-patterns.md](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md).

---

## 3. Performance & autotuning

B200/B300 default starting point (from NVIDIA's autotuning guidance):

```python
use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = True, (256, 256), (2, 1)
max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0]*cluster_shape_mn[1])
# sizes the persistent grid; the CLC tile scheduler then distributes tiles dynamically
```
Sweep `use_2cta_instrs`, `mma_tiler_m/n`, `cluster_shape_m/n`, `use_tma_store`, `num_stages`. The JIT cache is keyed on function + arg types/layouts + CuTeDSL env vars, so each config compiles once and is reused. **Mark dynamic dims with `cute.mark_layout_dynamic`** or autotuning recompiles per shape (see [../cutedsl-programming-model.md](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md)).

---

## 4. Debugging knobs

The full `CUTE_DSL_*` env-var list is in [cutedsl-api-reference-guide.md](../../../common/ref-docs/cutedsl/cutedsl-api-reference-guide.md) §10. SM100-relevant additions:
- Inspect generated code programmatically: compiled-kernel attrs `__ptx__`, `__cubin__`, `__mlir__`.
- `cute.printf("...", x)` prints a value at runtime; Python `print(tensor.layout)` prints a layout at compile time.
- When dropping to CUDA/PTX, the `--g-tensor-memory-access-check` build flag catches uninitialized / out-of-bounds TMEM access (silent otherwise). In DSL, keep TMEM copies built from `make_tmem_copy` so the warpgroup/lane mapping is correct by construction.

---

## 5. Common pitfalls (SM100 DSL-level)

- `cute.gemm` / tcgen05 ops issued from more than one thread → deadlock.
- Wrong `cute.gemm` operand order — it is **accumulator-first**: `cute.gemm(mma, acc, A, B, acc)`.
- Mixing `cta_group=ONE` and `TWO` in one kernel → undefined.
- TMEM epilogue from a single warp (needs a full warpgroup) → wrong / partial results.
- Forgetting `tmem_alloc.free` → later blocks' `allocate` blocks forever.
- Block scaling without scale factors in TMEM / `Field.SFA/SFB` unset → wrong results.
- Not marking layouts dynamic → per-shape recompilation during autotuning.

---

## Related

- [blackwell-cutedsl-sm100.md](blackwell-cutedsl-sm100.md) — CUTLASS-internals panorama
- [../cutedsl-pipeline-patterns.md](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md) · [../cutedsl-programming-model.md](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md) · [../nvidia-cutedsl-arch-primitives.md](../../../common/ref-docs/cutedsl/nvidia-cutedsl-arch-primitives.md) · [../cutedsl-api-reference-guide.md](../../../common/ref-docs/cutedsl/cutedsl-api-reference-guide.md)
- Worked tutorials: [blackwell-tcgen05-gemm-from-scratch.md](blackwell-tcgen05-gemm-from-scratch.md), [cutedsl-nvfp4-gemm-tutorial.md](cutedsl-nvfp4-gemm-tutorial.md)
- [blackwell hardware mechanism docs](../../kernel-opt/hardware/README.md) — TMEM / tcgen05 / TMA / 2sm / CLC
- Official API: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html>
