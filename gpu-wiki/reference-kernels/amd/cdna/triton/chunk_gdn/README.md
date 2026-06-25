# Chunk-GDN Triton Back-Half Baseline

This directory contains the RTP-tuned Triton implementations used as the
baseline for the MI308 FlyDSL Chunk-GDN megakernel.

The baseline starts after the front-half pipeline.  Callers pass precomputed
`a` and `g_cumsum`, then run:

```text
recompute_w_u_fwd -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o
```

These three stages are exactly the stages fused by the FlyDSL megakernel under
`reference-kernels/amd/cdna3/flydsl/FlyDSL/`.

| File | Source | Role |
|---|---|---|
| `wy_fast.py` | RTP `fla/wy_fast.py` | current tuned `recompute_w_u_fwd` |
| `chunk_delta_h.py` | RTP `fla/chunk_delta_h.py` | current tuned Triton `fwd_h` baseline; CDNA4 Gluon dispatch removed for standalone reference |
| `chunk_o.py` | RTP `fla/chunk_o.py` | current tuned `fwd_o` |
| `chunk_gdn_triton_baseline.py` | gpu-wiki wrapper | runs the three-stage back-half baseline from precomputed `a/g_cumsum` |
| `index.py`, `op.py`, `utils.py` | RTP support code | local support utilities, with RTP package imports removed |

Performance conclusions for this operator family should use `rocprofv3`, not
`do_bench`, manual wall-clock timing, or `torch.cuda.Event`.

When summing the standalone back-half baseline from `rocprofv3 --kernel-trace`,
include the `zeros_like` fill dispatch inside `chunk_fwd_o`; it is part of the
current Triton `fwd_o` implementation and is therefore part of the fair
back-half comparison against the fused FlyDSL megakernel.
