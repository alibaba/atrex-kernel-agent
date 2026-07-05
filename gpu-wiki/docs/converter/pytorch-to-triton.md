# PyTorch → Triton cheat sheet

Single-file reference for the **first framework migration** (V0 is a PyTorch reference wrapper; the
first optimization iteration lowers it to Triton). Read once. This is the one transition where the
target DSL (Triton) is well-known, so this sheet is the map + the non-obvious gotchas; no external
source pointers are needed.

## Rules
1. **Full conversion** — every op moves to Triton, *including* matmul (`nn.Linear`/`torch.matmul` →
   tiled `tl.dot`) and reductions (`tl.sum`/`tl.max`). Nothing stays in PyTorch on the compute path
   (PyTorch is allowed only for glue: allocation, reshape/view, launch).
2. **Fuse — that's the value.** Collapse chains into one kernel: `matmul + bias + activation` → fused
   epilogue after the K-loop; `scale + add + clamp + reduction` → one pass. Don't emit a kernel per op.
3. **Preserve the `run(...)` signature (inputs then outputs) and per-workload correctness** — outputs
   must match the reference within each shape's own tolerance.

## Non-obvious gotchas
- **`nn.Linear` weight is `[out, in]` = `[N, K]`.** `tl.dot` needs `[BLOCK_M,BLOCK_K] × [BLOCK_K,BLOCK_N]`,
  so load the weight tile as `[offs_n, offs_k]` and **transpose** it before the dot.
- **`tl.dot` is not IEEE-FP32 by default** — pass `input_precision="ieee"` (or `allow_tf32=False`) when
  the tolerance needs it; otherwise TF32 rounding can exceed atol.
- **Missing builtins**: `tl.math.tanh`, `tl.math.log1p` don't exist — implement via `sigmoid` / `log(1+x)`.
- **Reductions**: reduce along the correct axis with `tl.sum`/`tl.max`, masking out-of-range lanes
  (`other=0.0` for sum, `-inf` for max) so tail blocks are correct.
- **Numerical stability**: upcast to fp32 for softmax/norm/reduction accumulation, cast back at the end.

## Structure
`@triton.jit` kernel (grid over output tiles / rows) + a thin wrapper that computes the grid and
launches. Autotune/`num_warps`/`num_stages` are optimization concerns for later iterations — get a
correct, fused Triton kernel first.
