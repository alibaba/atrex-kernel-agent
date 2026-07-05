# Triton → Gluon cheat sheet — AMD CDNA3 (MI300, gfx942)

Single-file conversion reference for a **convert-only** session. Read once; open source for the
construct you convert.

Source root: `reference-projects/triton` (gluon @ `python/triton/experimental/gluon/language/amd/cdna3/`)

## Job (convert only, no optimization)
Lower the current Triton kernel to Gluon **preserving algorithm and tiling** — same numerics and
per-shape tolerances. Only the kernel body changes.

## Workflow
1. `python tools/extract_ttgir.py <driver>.py -o k.ttgir`; confirm `ttg.target` is `hip:gfx942`.
2. Reuse the real layouts (map TTGIR `threadsPerWarp/warpsPerCTA` → gluon `threads_per_warp/warps_per_cta`). Never fabricate.
3. Map with the table; open source for anything unmapped.
4. If original `num_stages > 1`, implement **software** pipelining (CDNA3 has no hardware DMA — see below).
5. Compile-check the module, then verify outputs match the reference within tolerance for every shape.

## API map (Triton → Gluon CDNA3)
| Triton | Gluon (CDNA3) | Notes |
|--------|---------------|-------|
| `tl.load(ptr,mask,other)` | `gl.amd.cdna3.buffer_load(ptr=base, offsets=offs, mask=mask)` | 2D block access |
| `tl.store(ptr,val,mask)` | `gl.amd.cdna3.buffer_store(stored_value=val, ptr=base, offsets=offs)` | |
| `tl.make_block_ptr` | **prohibited** — compute offsets manually | |
| `tl.dot(a,b,acc)` | `acc = gl.amd.cdna3.mfma(a, b, acc)` | CDNA3 MFMA |
| MMA layout | `gl.amd.AMDMFMALayout(...)` (+ `gl.DotOperandLayout(...)` for operands) | from TTGIR |
| shared memory | `gl.allocate_shared_memory(dtype,[depth,…],layout)`; `smem.index(i).store/.load` | |

## Software pipeline (num_stages>1; CDNA3 has no hardware async)
Manual prefetch: prologue loads iter 0 to smem slot 0 (and register-resident operands via
`buffer_load`); main loop computes `i` while `buffer_load`-prefetching `i+1`; epilogue handles the
last iter. `mfma` accumulates across iters. (See the CDNA3 guide's pipeline example for the exact
prologue/main/epilogue shape.)

## Pitfalls (CDNA3-specific)
1. **Warp size = 64** (not NVIDIA's 32): every `BlockedLayout` `threads_per_warp` product must equal 64.
2. Do **not** use NVIDIA APIs (`warpgroup_mma`, `async_copy.async_copy_global_to_shared`, `NVMMA*`) — wrong dialect.
3. `in_thread_transpose` (gfx942 auto-pass in Triton) isn't auto-run in Gluon — express it as
   `buffer_load → convert_layout → allocate_shared_memory + smem.load`; the compiler inserts the equivalent.
4. Layouts from real TTGIR only (`hip:gfx942`); map TTGIR camelCase param names to gluon snake_case.
5. `make_block_ptr` unavailable — compute offsets.

## Common failures (symptom → cause → fix)
| Symptom | Cause → Fix |
|---------|-------------|
| `ValueError: Unsupported ptr type for buffer_load` | wrong ptr arg → pass base ptr + `offsets=` per the `buffer_load` signature |
| `TypeError: arange() missing required argument: 'layout'` | tensor op without a layout → `gl.arange(..., layout=BlockedLayout(...))` (all tensor creation needs a layout) |
| `TypeError: mfma() inputs must be float16 or bfloat16` | wrong operand dtype → cast A/B to fp16/bf16 before `mfma` |
| "mask cannot be block type" | block-typed mask → use the scalar/allowed mask form for `buffer_load` |
| wrong results (numerical) | accumulation dtype/precision → accumulate in fp32 |
| smem OOM | CDNA3 LDS **64 KB**; duplicate live buffers → pre-allocate `[depth, …]` and reuse `smem.index(i)` |
| `smem.load()` returns wrong data | missing DotOperandLayout on the load → `smem.index(i).load(layout=dot_op)` |
| compile > 5 min | block/layout too complex → simpler layouts, smaller blocks |
