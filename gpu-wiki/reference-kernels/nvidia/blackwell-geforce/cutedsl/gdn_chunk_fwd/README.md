# SM120 Gated DeltaNet Chunk Forward (CuTeDSL)

CuteDSL implementation of `chunk_gated_delta_rule` forward (GDN chunk-level forward)
for NVIDIA RTX PRO 5000 Blackwell client GPU (sm_120).

| File | What it is |
|------|------------|
| [`sm120_gdn_chunk_fwd_3k.py`](sm120_gdn_chunk_fwd_3k.py) | **Production V113**. No-cache 3-kernel pipeline (K0 preprocess + K_inv Neumann + K1 fused chunk_h+o+final_state). cp.async 128-bit staging, LDSM-fed MMA, R2S/STSM stores, direct output copy, scaled-vnew state update, reuse-B LDSM for K*S/Q*S and final-state fragments, plus Bx2 non-tail launch. **P50=0.531-0.533ms at T=6144 directional `output_final_state=True`, 1.51× faster than same-process FLA (0.804ms).** |
| [`sm120_gdn_chunk_fwd_3k.py`](sm120_gdn_chunk_fwd_3k.py) | Archived V31 production snapshot. scaled-vnew + LDSM/R2S path, **P50=0.615ms at T=6144, 1.42× faster than FLA varlen**. Kept for historical comparison before the V113 no-cache final-state pass. |
| [`sm120_gdn_chunk_fwd_3k.py`](sm120_gdn_chunk_fwd_3k.py) | Archived pre-V31 production snapshot. cp.async val_layout fix + K2 fusion version, **P50=1.18ms at T=6144**. Kept only for historical comparison. |

## Workload contract

```
q    [B, T, H,  K=128]    bf16
k    [B, T, H,  K=128]    bf16
v    [B, T, HV, V=128]    bf16    (HV >= H, GVA when HV>H)
g    [B, T, HV]            bf16    (log-decay scalar gate, logsigmoid)
beta [B, T, HV]            bf16    (delta-rule mixing weight, sigmoid)
→
o    [B, T, HV, V]         bf16
```

Production shape: B=1, T=6144, H=16, HV=32, K=128, V=128, BT=32, BV=16.

## Architecture

3-kernel pipeline with precomputed Neumann inverse:

```
K0 (preprocess)    : L2-norm Q,K + kk = K_norm @ K_norm^T   [Grid: NT x B*H]
K_inv (Neumann)    : M = I - A + A^2 - A^3 (parallel)       [Grid: NT x B*HV]
K1 (fused chunk_h) : sequential state update + inline chunk_o + final_state [Grid: V/BV x B*HV]
```

Key design choices:
- **Precomputed Neumann inverse**: moves iterative solve out of sequential loop
- **K2 fusion**: chunk_o inlined into K1, eliminating ~250MB GMEM intermediates
- **t-outer/v-inner state update**: reduces register hoisting from 64 regs to 2
- **cp.async 128-bit for K/Q loads**: non-blocking GMEM->SMEM with latency hiding
- **LDSM-fed HMMA**: shared MMA operands loaded through `LdMatrix8x8x16bOp`
- **R2S/STSM cleanup**: removes scalar `STS.U16` accumulator-to-shared stores
- **Scaled-vnew state update**: stores `v_new * exp_decay` and reuses transposed `sK`,
  avoiding K-decay scratch materialization
- **Reuse-B LDSM (V113)**: reuses the `sS` B fragment for K*S/Q*S and the `sNK_A`
  B fragment for four final-state fragments; non-tail T=6144 uses a Bx2 launch,
  while tail / `STATE_SPLIT=True` keeps the B4 safe launch

## Why NOT the GDN decode kernel

The sister directory [`../gdn_decode/`](../gdn_decode/) handles T=1 decode with fp32
state and per-element GEMV. This kernel handles chunk-level forward (T >> 1) with
MMA-based chunk computations, fundamentally different algorithm and memory access pattern.

## Related docs

- **Optimization journey** — baseline to V113 final, including null results:
  [`docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md`](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md)
- **Pitfalls** — 26 traps with trap -> symptom -> why -> lesson:
  [`docs/nvidia/blackwell-geforce/pitfalls/cutedsl/gdn-chunk-fwd-pitfalls.md`](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/gdn-chunk-fwd-pitfalls.md)

## Verification

```python
from sm120_gdn_chunk_fwd_3k import run_3kernel

# rel_err < 0.01 on all shapes (T=128,256,512,1024,6144)
# tail directional regression passed; T=6144 final-state strict gate is 1.51x FLA
```
