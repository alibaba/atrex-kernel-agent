# Triton → Gluon cheat sheet — NVIDIA Blackwell data-center (sm_100 B200 / sm_103 B300)

Single-file conversion reference for a **convert-only** session. Read this once; open the cited
Triton 3.7.1 source **only** for the construct you're converting. Do not paraphrase from memory — the
source is ground truth.

> Covers **data-center Blackwell: sm_100 (B200) and sm_103 (B300)** — they share the tcgen05/TMEM path
> (sm_103 ≈ sm_100). This is **NOT** for sm_120 (Blackwell GeForce), which uses different Gluon
> primitives — do not use this sheet there.

Source root: `reference-projects/triton` @ `v3.7.1`
- API: `python/triton/experimental/gluon/language/nvidia/blackwell/{__init__.py,tma.py,float2.py}`
- Reference kernel (**read this first — it's our op class**): `python/examples/gluon/01-attention-forward.py`
  (tcgen05 + TMEM + TMA + mbarrier, warp-specialized flash-attention forward)
- Patterns: `python/tutorials/gluon/{06-tcgen05,11-tcgen05-mma-scaled,10-tcgen05-copy,04-tma,09-tma-gather-scatter,08-warp-specialization,07-persistence,02-layouts}.py`

## Job (convert only, no optimization)
Lower the current Triton `kernel.py` to Gluon **preserving algorithm and tiling**. Blackwell levers
(TMEM residency, tcgen05 scheduling, warp specialization, persistence) are for the *following* gluon
sessions. Preserve numerics and per-shape tolerances. Gluon ships inside the `triton` package
(`triton.experimental.gluon`), so only the kernel body changes — not the framework declared to the harness.

## Workflow
1. **Dump TTGIR FIRST, before drafting any Gluon**: `python tools/extract_ttgir.py <driver>.py -o k.ttgir` (driver must launch the kernel).
2. Confirm `ttg.target` is `cuda:100` (B200) or `cuda:103` (B300). Use the real `#blocked/#shared/#tmem` layouts **from THIS kernel's TTGIR** — never fabricate, and never copy the reference example's layouts/shapes (the example is for code structure only).
3. Map with the table below; open source for anything unmapped.
4. Rewrite loads→TMA+mbarrier, matmuls→tcgen05+TMEM; reproduce the original `num_stages` (nothing more).
5. Compile-check the module, then verify outputs match the reference within tolerance for every shape.

## API map (Triton → Gluon Blackwell; deltas from Hopper noted)
| Triton | Gluon (Blackwell) | vs Hopper |
|--------|-------------------|-----------|
| `tl.dot(a,b,acc)` | `tcgen05_mma(a,b,acc_tmem, use_acc=True, mbarriers=[bar])` → `tcgen05_commit(bar)` → `mbarrier.wait(bar,ph)` | replaces `warpgroup_mma`+`warpgroup_mma_wait`; **acc in TMEM, not registers** |
| scaled `tl.dot` (mxfp8/fp4) | `tcgen05_mma_scaled(a,b,acc,a_scale,b_scale,a_type,b_type,mbarriers=[bar])` | new; scales in `TensorMemoryScalesLayout` |
| accumulator storage | `allocate_tensor_memory(dtype,[M,N],TensorMemoryLayout([M,N],col_stride=1))`; `acc=tmem.load(get_tmem_reg_layout(dtype,(M,N),layout,num_warps))`; `tmem.store(acc)` | registers on Hopper |
| bulk `tl.load` | `mbarrier.expect(bar,desc.block_type.nbytes)`→`tma.async_copy_global_to_shared(desc,[x,y],bar,smem)`→`mbarrier.wait(bar,ph)` | Hopper used `async_copy`(cp.async); Blackwell uses TMA |
| bulk `tl.store` | `tma.async_copy_shared_to_global(desc,[x,y],smem)`→`tma.store_wait(0)` | |
| gather/scatter load/store | `tma.async_gather(desc,x_off,y_off,bar,res)` / `tma.async_scatter(desc,x_off,y_off,src)` | Blackwell-specific |
| small/masked load/store | `gl.load(base+gl.cast(off,gl.int32),mask=…)` / `gl.store(...)` | same |
| barriers | `bar=gl.allocate_shared_memory(gl.int64,[n,1],mbarrier.MBarrierLayout())`; `mbarrier.{init,arrive,wait,expect,invalidate}` | same family |
| smem-write fence before MMA | `fence_async_shared()` (from `...gluon.nvidia.hopper`) | same |

Imports: `from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout, tma, mbarrier, tcgen05_mma, tcgen05_commit)`

## Canonical MMA (from 06-tcgen05.py)
```python
acc_tmem = allocate_tensor_memory(gl.float32, [M,N], TensorMemoryLayout([M,N], col_stride=1))
fence_async_shared()
tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True, mbarriers=[mma_bar])   # B in SMEM(NVMMASharedLayout); A in SMEM or TMEM
tcgen05_commit(mma_bar); mbarrier.wait(mma_bar, phase)
acc = acc_tmem.load(get_tmem_reg_layout(gl.float32,(M,N),acc_tmem.layout,num_warps))
```
Warp-specialized producer/consumer pipeline and TMEM-buffer borrowing: copy the structure in
`01-attention-forward.py` (`ready`/`empty` mbarriers; `_borrow_s_as_p`/`_borrow_s_as_alpha`).

## Pitfalls (Blackwell-specific)
1. `tcgen05`, **not** `wgmma`; accumulator in **TMEM**, RHS must be SMEM `NVMMASharedLayout`.
2. **Never reuse one mbarrier for TMA and tcgen05_mma** → UB; use separate bars or `invalidate`+`init`.
3. Don't replace TMA with `gl.load`+`smem.store` (loses the TMA path).
4. Warp size 32; a full warpgroup (4 warps/128 threads) is needed for 128 TMEM rows (1 warp = 32 rows).
5. Layouts from real TTGIR only; confirm `cuda:100`/`cuda:103`. Don't invent; `get_tmem_reg_layout` derives reg layouts.
6. No `gl.amd.*` (unregistered-dialect crash).

## Common failures (symptom → cause → fix)
| Symptom | Cause → Fix |
|---------|-------------|
| `LLVM ERROR: ... unregistered dialect` | `gl.amd.*` used on NVIDIA → use the Blackwell APIs above |
| wrong results / GPU hang | one mbarrier reused for TMA **and** `tcgen05_mma` (UB) → separate barriers, or `mbarrier.invalidate`+`init` between phases |
| wrong results | RHS not in SMEM `NVMMASharedLayout`, or accumulator not in TMEM → B in SMEM(NVMMAShared), acc in TMEM |
| >5% slower than Triton | `gl.load`+`smem.store` instead of `tma.async_copy_global_to_shared`, or accumulator round-tripping registers instead of staying in TMEM across the K-loop → use TMA + keep acc TMEM-resident |
| wrong rows / partial output | TMEM accessed by too few warps (1 warp = 32 rows) → use a full warpgroup (4 warps / 128 threads) for 128 rows |
| tensor op layout error | missing layout on `gl.arange`/`gl.zeros`/… → pass an explicit layout (from TTGIR or `get_tmem_reg_layout`) |
