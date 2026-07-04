# Triton → Gluon cheat sheet — NVIDIA Hopper (sm_90, H20/H100/H200)

Single-file conversion reference for a **convert-only** session. Read this once; open the cited
Triton source only for the construct you're converting.

Source root: `reference-projects/triton` (gluon @ `python/triton/experimental/gluon/language/nvidia/hopper/`)
- Patterns: `python/tutorials/gluon/{03-async-copy,04-tma,05-wgmma,02-layouts}.py`

## Job (convert only, no optimization)
Lower the current Triton kernel to Gluon **preserving algorithm and tiling** — same numerics and
per-shape tolerances. Gluon ships inside the `triton` package, so only the kernel body changes.

## Workflow
1. `python tools/extract_ttgir.py <driver>.py -o k.ttgir` (driver must launch the kernel).
2. Confirm `ttg.target = "cuda:90"`. Reuse the real `#blocked/#shared` layouts — never fabricate.
3. Map with the table below; open source for anything unmapped.
4. If original `num_stages > 1`, implement the pipeline manually with `async_copy` (see below).
5. Compile-check the module, then verify outputs match the reference within tolerance for every shape.

## API map (Triton → Gluon Hopper)
| Triton | Gluon (Hopper) | Notes |
|--------|----------------|-------|
| `tl.program_id/num_programs` | `gl.program_id/num_programs` | identical |
| `tl.arange/zeros/full/zeros_like` | `gl.arange/zeros/full/zeros_like(..., layout=…)` | **must pass a layout** |
| `tl.load(ptr,mask,other)` | `gl.load(base_ptr + gl.cast(offs, gl.int32), mask, other)` | pointer arithmetic |
| `tl.store(ptr,val,mask)` | `gl.store(base_ptr + gl.cast(offs, gl.int32), val, mask)` | pointer arithmetic |
| `tl.make_block_ptr` | **prohibited** — compute offsets manually | |
| `tl.dot(a,b,acc)` | `fence_async_shared()` → `acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)` → `acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))` | operands in SMEM; **accumulator in registers** |
| pipelined `tl.load` (num_stages>1) | `async_copy.async_copy_global_to_shared(smem.index(slot), base_ptr + gl.cast(offs,gl.int32), mask=mask)` → `async_copy.commit_group()` → `async_copy.wait_group(0)` | CP_ASYNC DMA; **50%+ faster than `gl.load`+`smem.store`** |
| shared memory | `gl.allocate_shared_memory(dtype, [depth,…], layout)`; `smem.index(i).store(x)`; `smem.index(i).load(layout=…)` | |

Imports: `from triton.experimental.gluon.language.nvidia.hopper import async_copy, fence_async_shared, warpgroup_mma, warpgroup_mma_wait`

## Pitfalls (Hopper-specific)
1. **No `gl.amd.*`** (`buffer_load/buffer_store/mfma`, `AMDMFMALayout`) → `LLVM ERROR: unregistered dialect`. Use the Hopper APIs above.
2. **Pipeline data into SMEM with `async_copy`, not `gl.load`+`smem.store`** (the two-step transit is 50%+ slower). Data that goes straight to registers (not SMEM) still uses `gl.load`.
3. **`fence_async_shared()` before every wgmma**, and `warpgroup_mma_wait` after — omitting either is UB.
4. Warp size = 32 (not 64): every `BlockedLayout` `threads_per_warp` product must equal 32.
5. Layouts from real TTGIR only; confirm `cuda:90` (if `cuda:100`, use `blackwell.md`). `make_block_ptr` is unavailable — compute offsets.
