#!/usr/bin/env python3
"""Standalone FlyDSL Chunk-GDN megakernel operator for MI308X.

This reference entry point has no RTP-LLM runtime dependency.  It runs only
the FlyDSL megakernel that fuses:

    recompute_w_u + fwd_h + fwd_o

Callers must pass the precomputed inputs produced by the front half of the
Chunk-GDN pipeline:

    g_cumsum: [B, T, H] fp32 in log2 domain
    a:        [B, T, H, 64] bf16 KKT-solve/A_inv output

The wrapper handles shape validation, varlen chunk offset construction, and
optional direct SSM-state store arguments before launching the FlyDSL kernel.
"""

import sys
from pathlib import Path
from typing import Optional

import torch

_THIS_DIR = Path(__file__).resolve().parent
_THIS_DIR_STR = str(_THIS_DIR)
if _THIS_DIR_STR not in sys.path:
    sys.path.insert(0, _THIS_DIR_STR)

CHUNK_SIZE = 64

SUPPORTED_CHUNK_GDN_SHAPES = frozenset(
    {
        (16, 16, 128, 128),
        (8, 8, 128, 128),
        (16, 32, 128, 128),
        (8, 16, 128, 128),
        (16, 48, 128, 128),
        (8, 24, 128, 128),
        (16, 64, 128, 128),
        (8, 32, 128, 128),
        (4, 16, 128, 128),
        (2, 8, 128, 128),
    }
)


def _get_megakernel_fwd():
    from fused_fwd_mi308x_v2 import megakernel_fwd

    return megakernel_fwd


def _shape(q: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int]:
    _, _, hg, k_dim = q.shape
    _, _, h, v_dim = v.shape
    return hg, h, k_dim, v_dim


def is_supported_shape(q: torch.Tensor, v: torch.Tensor) -> bool:
    return _shape(q, v) in SUPPORTED_CHUNK_GDN_SHAPES


def make_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """Build per-sequence chunk offsets for varlen megakernel launches."""
    if cu_seqlens.dtype != torch.long:
        cu_seqlens = cu_seqlens.to(torch.long)
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks = torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")
    return torch.cat([cu_seqlens.new_zeros(1), chunks]).cumsum(0)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    chunk_offsets: Optional[torch.Tensor],
    prefix_lengths: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    ssm_states: Optional[torch.Tensor],
    seq_size_per_block: Optional[int],
) -> None:
    errors = []
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        errors.append(f"q/k/v must be bf16, got {q.dtype}/{k.dtype}/{v.dtype}")
    if a.dtype != torch.bfloat16:
        errors.append(f"a must be bf16, got {a.dtype}")
    if g_cumsum.dtype != torch.float32:
        errors.append(f"g_cumsum must be fp32 log2-domain values, got {g_cumsum.dtype}")
    if beta.dtype != torch.bfloat16:
        errors.append(f"beta must be bf16, got {beta.dtype}")
    if q.shape != k.shape:
        errors.append(f"q and k must have identical shape, got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.shape[:2] != v.shape[:2]:
        errors.append(f"q/k and v must share [B,T], got {tuple(q.shape[:2])} vs {tuple(v.shape[:2])}")
    if q.shape[:2] != g_cumsum.shape[:2] or v.shape[2] != g_cumsum.shape[2]:
        errors.append(
            "g_cumsum must have shape [B,T,H], got "
            f"{tuple(g_cumsum.shape)} for q={tuple(q.shape)} v={tuple(v.shape)}"
        )
    if q.shape[:2] != beta.shape[:2] or v.shape[2] != beta.shape[2]:
        errors.append(
            "beta must have shape [B,T,H], got "
            f"{tuple(beta.shape)} for q={tuple(q.shape)} v={tuple(v.shape)}"
        )
    if a.shape != (q.shape[0], q.shape[1], v.shape[2], CHUNK_SIZE):
        errors.append(
            "a must have shape [B,T,H,64], got "
            f"{tuple(a.shape)} for q={tuple(q.shape)} v={tuple(v.shape)}"
        )
    if not is_supported_shape(q, v):
        errors.append(f"unsupported Hg/H/K/V={_shape(q, v)}")
    if initial_state is not None:
        n_state = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]
        expected_h0 = (n_state, v.shape[2], q.shape[3], v.shape[3])
        if initial_state.shape != expected_h0:
            errors.append(f"initial_state must have shape {expected_h0}, got {tuple(initial_state.shape)}")
        if initial_state.dtype not in (torch.bfloat16, torch.float32):
            errors.append(f"initial_state dtype must be bf16 or fp32, got {initial_state.dtype}")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            errors.append("q.shape[0] must be 1 when cu_seqlens is provided")
        if cu_seqlens.dtype != torch.long:
            errors.append(f"cu_seqlens must be int64/torch.long, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            errors.append("cu_seqlens must be a 1D tensor with at least two elements")
        elif int(cu_seqlens[-1].item()) != q.shape[1]:
            errors.append(f"cu_seqlens[-1] must equal T={q.shape[1]}, got {int(cu_seqlens[-1].item())}")
        if chunk_offsets is not None and chunk_offsets.dtype != torch.long:
            errors.append(f"chunk_offsets must be int64/torch.long, got {chunk_offsets.dtype}")
        if (
            chunk_offsets is not None
            and cu_seqlens.ndim == 1
            and chunk_offsets.numel() != cu_seqlens.numel()
        ):
            errors.append(
                "chunk_offsets must have the same length as cu_seqlens, got "
                f"{chunk_offsets.numel()} vs {cu_seqlens.numel()}"
            )
    elif chunk_offsets is not None:
        errors.append("chunk_offsets is only valid when cu_seqlens is provided")
    if ssm_states is not None:
        if prefix_lengths is None or block_map is None or seq_size_per_block is None:
            errors.append(
                "prefix_lengths, block_map, and seq_size_per_block are required "
                "when writing ssm_states directly"
            )
        if ssm_states.dtype not in (torch.bfloat16, torch.float32):
            errors.append(f"ssm_states dtype must be bf16 or fp32, got {ssm_states.dtype}")
        _, _, h, v_dim = v.shape
        k_dim = k.shape[-1]
        if ssm_states.shape[1:] != (h, v_dim, k_dim):
            errors.append(
                "ssm_states must have shape [num_blocks, H, V, K], got "
                f"{tuple(ssm_states.shape)}"
            )
        if (
            ssm_states.ndim == 4
            and (
                ssm_states.stride(1) != k_dim * v_dim
                or ssm_states.stride(2) != k_dim
                or ssm_states.stride(3) != 1
            )
        ):
            errors.append("ssm_states must be contiguous per head in [H, V, K] layout")
    if errors:
        raise ValueError("FlyDSL Chunk-GDN megakernel input validation failed: " + "; ".join(errors))


@torch.compiler.disable
def chunk_gdn_flydsl_fwd(
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
    chunk_offsets: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    ssm_states: Optional[torch.Tensor] = None,
    seq_size_per_block: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the standalone FlyDSL Chunk-GDN megakernel.

    Args:
        q/k: [B, T, Hg, K] bf16.
        v: [B, T, H, V] bf16.
        a: [B, T, H, 64] bf16, precomputed KKT-solve/A_inv output.
        g_cumsum: [B, T, H] fp32, precomputed cumsum in log2 domain.
        beta: [B, T, H] bf16.
        initial_state: [N, H, K, V] fp32/bf16 state, where N=B for dense mode
            and N=len(cu_seqlens)-1 for varlen mode.
        ssm_states: optional [num_blocks, H, V, K] bf16/fp32 cache state buffer.

    Returns:
        (o, final_state), where o is [B, T, H, V] bf16 and final_state is
        [N, H, K, V] fp32 when requested.
    """
    _validate_inputs(
        q=q,
        k=k,
        v=v,
        a=a,
        g_cumsum=g_cumsum,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        prefix_lengths=prefix_lengths,
        block_map=block_map,
        ssm_states=ssm_states,
        seq_size_per_block=seq_size_per_block,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    g_cumsum = g_cumsum.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = (
            initial_state.float()
            if initial_state.dtype != torch.float32
            else initial_state.contiguous()
        )
    if prefix_lengths is not None and prefix_lengths.dtype != torch.int32:
        prefix_lengths = prefix_lengths.to(torch.int32)
    if block_map is not None and block_map.dtype != torch.int32:
        block_map = block_map.to(torch.int32)
    if prefix_lengths is not None:
        prefix_lengths = prefix_lengths.contiguous()
    if block_map is not None:
        block_map = block_map.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()
        if chunk_offsets is None:
            chunk_offsets = make_chunk_offsets(cu_seqlens)
        else:
            chunk_offsets = chunk_offsets.contiguous()

    o, final_state = _get_megakernel_fwd()(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        prefix_lengths=prefix_lengths,
        block_map=block_map,
        ssm_states=ssm_states,
        seq_size_per_block=seq_size_per_block,
    )
    return o.to(q.dtype), final_state


chunk_gdn_flydsl_megakernel_fwd = chunk_gdn_flydsl_fwd


__all__ = [
    "CHUNK_SIZE",
    "SUPPORTED_CHUNK_GDN_SHAPES",
    "chunk_gdn_flydsl_fwd",
    "chunk_gdn_flydsl_megakernel_fwd",
    "is_supported_shape",
    "make_chunk_offsets",
]
