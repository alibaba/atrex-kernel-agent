"""PyTorch reference for fused_recurrent_gated_delta_rule (Qwen3-Next decode focus).

Mirrors fla/ops/gated_delta_rule/fused_recurrent.py semantics with:
  * scalar gate g (per (B,T,HV))
  * scalar beta (per (B,T,HV))
  * use_qk_l2norm_in_kernel
  * GVA: HV >= H, HV % H == 0
  * Default state layout [K, V] (transpose_state_layout=False)
"""
from __future__ import annotations
import torch


@torch.no_grad()
def gdn_recurrent_ref(
    q: torch.Tensor,        # [B, T, H,  K]  bf16
    k: torch.Tensor,        # [B, T, H,  K]  bf16
    v: torch.Tensor,        # [B, T, HV, V]  bf16
    g: torch.Tensor | None,        # [B, T, HV]    fp32 (log decay)
    beta: torch.Tensor,            # [B, T, HV]    fp32
    h0: torch.Tensor | None,       # [B, HV, K, V] fp32
    scale: float,
    use_qk_l2norm: bool = True,
):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    assert HV % H == 0
    G = HV // H

    h = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    if h0 is not None:
        h = h + h0.float()

    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    for t in range(T):
        qt = q[:, t].float()                       # [B, H,  K]
        kt = k[:, t].float()                       # [B, H,  K]
        vt = v[:, t].float()                       # [B, HV, V]
        bt = beta[:, t].float()                    # [B, HV]
        if use_qk_l2norm:
            qt = qt / (qt.pow(2).sum(-1, keepdim=True).clamp_min(1e-6).sqrt())
            kt = kt / (kt.pow(2).sum(-1, keepdim=True).clamp_min(1e-6).sqrt())
        qt = qt * scale

        # broadcast q,k from H to HV via grouping
        qt_e = qt.repeat_interleave(G, dim=1)      # [B, HV, K]
        kt_e = kt.repeat_interleave(G, dim=1)      # [B, HV, K]

        if g is not None:
            gt = g[:, t].float()                   # [B, HV]
            h = h * gt[..., None, None].exp()

        # delta rule: v_new = beta * (v - h^T @ k) , h += k outer v_new
        hk = torch.einsum('bhkv,bhk->bhv', h, kt_e)                  # [B, HV, V]
        v_new = bt[..., None] * (vt - hk)                            # [B, HV, V]
        h = h + torch.einsum('bhk,bhv->bhkv', kt_e, v_new)           # [B, HV, K, V]

        # output: o = h^T @ q
        ot = torch.einsum('bhkv,bhk->bhv', h, qt_e)                  # [B, HV, V]
        o[:, t] = ot.to(v.dtype)

    return o, h


def make_qwen3_next_decode_inputs(
    B: int = 64,
    H: int = 2,
    HV: int = 16,
    K: int = 128,
    V: int = 256,
    T: int = 1,
    dtype=torch.bfloat16,
    device='cuda',
    seed: int = 0,
):
    """Default Qwen3-Next decode shape; tweak to match real prod values."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H,  K,  dtype=dtype, device=device, generator=g) * 0.5
    k = torch.randn(B, T, H,  K,  dtype=dtype, device=device, generator=g) * 0.5
    v = torch.randn(B, T, HV, V,  dtype=dtype, device=device, generator=g) * 0.5
    gg = torch.nn.functional.logsigmoid(
        torch.randn(B, T, HV, dtype=torch.float32, device=device, generator=g)
    )
    beta = torch.sigmoid(
        torch.randn(B, T, HV, dtype=torch.float32, device=device, generator=g)
    )
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=device, generator=g) * 0.1
    scale = K ** -0.5
    return dict(q=q, k=k, v=v, g=gg, beta=beta, h0=h0, scale=scale)
