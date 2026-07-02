# SM120 Gated DeltaNet Decode (CuTeDSL)

CuteDSL implementations of `fused_recurrent_gated_delta_rule_fwd` (GDN decode) for
NVIDIA RTX PRO 5000 / 4000 Blackwell client GPUs (sm_120).

| File | What it is |
|------|------------|
| [`sm120_gdn_fwd_T1_v13.py`](sm120_gdn_fwd_T1_v13.py) | **Production version (V13/18 iterations)**. cp.async + `LoadCacheMode.GLOBAL` + `assumed_align=16`. Matches FLA Triton wall-clock at B≥32 (246us vs 247us @ B=64). Memory throughput 1.04 TB/s = 100.8% memcpy ceiling. |
| [`reference.py`](reference.py) | PyTorch fp32 reference (oracle for correctness; mirrors FLA semantics: scalar gate g, scalar beta, USE_QK_L2NORM_IN_KERNEL, fp32 state, bf16 q/k/v/o). |

## Workload contract

```
q [B, T=1, H,  K=128]   bf16
k [B, T=1, H,  K=128]   bf16
v [B, T=1, HV, V=256]   bf16   (HV >= H, GVA when HV>H)
g [B, T=1, HV]          fp32   (log-decay scalar gate)
beta [B, T=1, HV]       fp32
h0 [B, HV, K, V]        fp32   (initial state)
→
o  [B, T=1, HV, V]      bf16
ht [B, HV, K, V]        fp32   (final state)
```

Compute per (n, hv) per t:
```
h = h0 * exp(g)                              # apply scalar gate
hk = h^T @ k                                 # GEMV
v_new = beta * (v - hk)                      # delta-rule remove
h += k ⊗ v_new                               # rank-1 update
o = h^T @ q                                  # output GEMV
ht = h
```

## Why a separate sm_120 file (not extending Hopper variant)

The sister directory [`../../../hopper/cutedsl/flashinfer/gdn_decode_*.py`](../../../hopper/cutedsl/flashinfer/) is FlashInfer's
sm_90 variant with bf16 state. **It does not transfer to sm_120**:

| Dimension | Hopper FlashInfer | sm_120 V13 (this file) |
|-----------|-------------------|------------------------|
| State dtype | bf16 (2× less BW) | fp32 (FLA contract) |
| MMA path | tcgen05 / wgmma | warp ALU + warp shuffle (Ampere style) |
| Memory path | TMA bulk + descriptor | cp.async vec=128b GLOBAL cache mode |
| State size in regs/CTA | smaller (bf16) | ~32 fp32 / thread |

## Related docs (read together)

- **Optimization journey** — what each of 18 iterations changed and why:
  [`docs/ref-docs/nvidia/cutedsl/sm120/sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](../../../../../docs/nvidia/blackwell-geforce/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- **Pitfalls** — 9+ traps with trap → symptom → why → lesson:
  [`docs/pitfalls/nvidia/cutedsl/gdn-decode-pitfalls.md`](../../../../../docs/nvidia/blackwell-geforce/cutedsl/pitfalls/gdn-decode-pitfalls.md)

## Verification

```python
from src.reference import gdn_recurrent_ref
from src.cute_v13 import run_gdn_fwd_T1_v13

# rel_err(o)  ≤ 8.4e-5  on bf16 inputs
# rel_err(ht) ≤ 5.3e-8  on fp32 state
```
