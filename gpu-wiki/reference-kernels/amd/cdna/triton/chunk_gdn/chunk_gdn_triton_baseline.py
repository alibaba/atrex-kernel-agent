#!/usr/bin/env python3
"""Standalone Triton Chunk-GDN back-half baseline from RTP tuned kernels.

This module intentionally starts after the front-half pipeline.  Callers pass
precomputed:

    a:        [B, T, H, 64] bf16 KKT-solve/A_inv output
    g_cumsum: [B, T, H] fp32 cumsum in log2 domain on AMD

The baseline then runs the current RTP tuned Triton back half:

    recompute_w_u_fwd -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o

It is useful as the apples-to-apples baseline for the FlyDSL megakernel, which
fuses the same three stages.
"""

import sys
from pathlib import Path
from typing import Optional

import torch

_THIS_DIR = Path(__file__).resolve().parent
_THIS_DIR_STR = str(_THIS_DIR)
if _THIS_DIR_STR not in sys.path:
    sys.path.insert(0, _THIS_DIR_STR)

from chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from chunk_o import chunk_fwd_o  # noqa: E402
from wy_fast import recompute_w_u_fwd  # noqa: E402


def chunk_gdn_triton_backhalf(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    state_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run RTP-tuned Triton recompute_w_u_fwd + fwd_h + fwd_o.

    Args:
        q/k: [B, T, Hg, K] bf16.
        v: [B, T, H, V] bf16.
        a: [B, T, H, 64] bf16 KKT-solve/A_inv output.
        g_cumsum: [B, T, H] fp32 cumsum. On AMD this is log2-domain.
        beta: [B, T, H] bf16.
        initial_state: [N, H, K, V] optional state.
        state_dtype: optional per-chunk h buffer dtype for fwd_h.

    Returns:
        (o, final_state, w, h, v_new).
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and cu_seqlens.dtype != torch.long:
        cu_seqlens = cu_seqlens.to(torch.long)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state, w, h, v_new


__all__ = ["chunk_gdn_triton_backhalf"]
