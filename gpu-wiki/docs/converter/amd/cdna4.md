# Triton → Gluon cheat sheet — AMD CDNA4 (MI355X, gfx950)

Single-file conversion reference for a **convert-only** session. CDNA4 **inherits all CDNA3 APIs**
(`from ..cdna3 import *` — `buffer_load`, `buffer_store`, `mfma` are the same objects), so start from
[cdna3.md](cdna3.md) and apply the CDNA4 deltas below. Source: `reference-projects/triton` (gluon @
`python/triton/experimental/gluon/language/amd/cdna4/`).

## Job / workflow / commit gate
Same as CDNA3 (convert only, preserve numerics and per-shape tolerances). `extract_ttgir.py` target
should read `hip:gfx950`.

## CDNA4 deltas vs CDNA3
| Concern | CDNA3 | CDNA4 |
|--------|-------|-------|
| pipeline (num_stages>1) | software prefetch (`buffer_load`→smem) | **hardware DMA**: `async_copy.buffer_load_to_shared(smem.index(slot), ptr, offsets, mask)` → `async_copy.commit_group()` → `async_copy.wait_group(n)`; read via `async_copy.load_shared_relaxed(smem.index(slot), layout=dot_op)` |
| scaled MMA | — | `gl.amd.cdna4.mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)` (OCP microscaling fp8/fp4); scale layout via `get_mfma_scale_layout` |
| atomics | `buffer_atomic_*` | `buffer_atomic_*` adds bf16 `fadd` |

Import for the DMA path: `from triton.experimental.gluon.language.amd.cdna4 import async_copy`.

## CDNA4 pipeline (ping-pong, async_copy only)
Double-buffer (`depth=2` ⇒ num_stages=3): prologue asynchronously prefetches the first two iters
(`buffer_load_to_shared` + `commit_group` each); main loop `wait_group(num_outstanding=1)`,
`load_shared_relaxed` slot `i%2`, `mfma`, then prefetch into the freed slot; epilogue drains the last
two. **Using `buffer_load` + `smem.store` instead of `async_copy` is 40–60% slower.**

## Pitfalls
1. Warp size = 64. Do not use NVIDIA APIs.
2. For pipelines use the **hardware `async_copy`** path, not the CDNA3 software prefetch (perf).
3. Everything in [cdna3.md](cdna3.md) pitfalls also applies (layouts from TTGIR `hip:gfx950`, no `make_block_ptr`, `in_thread_transpose` handling).
