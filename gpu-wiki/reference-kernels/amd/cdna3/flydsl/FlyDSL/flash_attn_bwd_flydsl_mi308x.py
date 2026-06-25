# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
⚠ TUNED FOR AMD MI308X (CDNA3, gfx942) ⚠

FlyDSL-based Flash Attention Backward API wrapper.

Wraps two separate FlyDSL backward kernels (dQ and dK+dV) into a unified
backward pass interface. The split-kernel design avoids atomic_add and
reduces VGPR pressure compared to a single fused backward kernel.

Supports arbitrary additive masks via bit-packed u32 bitmask format.
The mask is converted from float32 additive form to bit-packed u32 on the host,
and per-workgroup loop bounds are precomputed from mask sparsity.

Architecture decisions:
  - dQ kernel: Grid partitioned by Q-tiles, inner loop over KV-tiles.
  - dK+dV kernel: Grid partitioned by KV-tiles, inner loop over Q-tiles.
  - Both kernels require precomputed LSE and Delta from the forward pass.
  - Both kernels accept bit-packed u32 mask and precomputed loop bounds.

Tuning vs generic CDNA version (reference-kernels/amd/cdna/flydsl/FlyDSL/):
  - block_m=32, block_n=32 for dQ (vs generic 64/128)
  - block_m=64, block_n=32 for dK/dV
  - OOB guards via arith.select to avoid F.pad overhead
  - Loop bounds precomputed on host from mask sparsity

Hardware: Tuned for AMD MI308X (CDNA3, gfx942).

Optimization report: docs/ref-docs/amd/flydsl/gfx942/cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md
Pitfalls: docs/pitfalls/amd/flydsl/flash-attn-bwd-mask-integration-pitfalls.md
"""
import functools
import math

import torch
import torch.nn.functional as F


def _compute_lse_and_delta(query, key, value, mask_f32, grad_output, scale):
    """Compute LSE and Delta for backward kernel inputs.

    Args:
        query: (B, H, S, D) bf16/fp16
        key: (B, H, S, D) bf16/fp16
        value: (B, H, S, D) bf16/fp16
        mask_f32: (B, 1, S, S) float32 additive mask (0 for valid, -1e6 for masked)
        grad_output: (B, H, S, D) bf16/fp16
        scale: softmax scale factor

    Returns:
        lse: (B, H, S) float32 — log-sum-exp of scaled attention scores
        delta: (B, H, S) float32 — row-wise dot product of grad_output and output
    """
    query_f32 = query.float()
    key_f32 = key.float()
    value_f32 = value.float()
    scores = torch.matmul(query_f32, key_f32.transpose(-1, -2)) * scale + mask_f32
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, value_f32)
    delta = (grad_output.float() * output).sum(dim=-1)
    return lse, delta


def pack_mask_to_u32(mask_f32, S_pad):
    """Pack additive f32 mask (B, 1, S, S) to u32 bitmask (B, 1, S_pad, S_pad//32).

    bit=1 means attend (mask >= 0), bit=0 means masked (mask < 0).
    Padding bits beyond S are set to 0 (masked).
    """
    B_m = mask_f32.shape[0]
    S_rows = mask_f32.shape[-2]
    S_cols = mask_f32.shape[-1]
    device = mask_f32.device
    words_per_row = S_pad // 32

    attend = (mask_f32 >= -0.5)  # (B, 1, S_rows, S_cols) bool

    if S_pad == S_cols and S_pad == S_rows:
        attend_pad = attend
    else:
        attend_pad = torch.zeros(B_m, 1, S_pad, S_pad, dtype=torch.bool, device=device)
        attend_pad[:, :, :S_rows, :S_cols] = attend

    attend_pad = attend_pad.reshape(B_m, 1, S_pad, words_per_row, 32)
    powers = (1 << torch.arange(32, device=device, dtype=torch.int32))
    packed = (attend_pad.int() * powers).sum(dim=-1).to(torch.int32)
    return packed.contiguous()


def compute_dq_loop_bounds(mask_packed, B, H, S_pad, BLOCK_N=32, BLOCK_M=32):
    """Compute per-workgroup loop bounds for dQ kernel (vectorized).

    Returns: int32 tensor of shape [grid_x * 2] where grid_x = B * num_q_tiles * H.
    """
    num_kv_tiles = S_pad // BLOCK_N
    num_q_tiles = S_pad // BLOCK_M
    device = mask_packed.device

    # mask_packed: (B, 1, S_pad, num_kv_tiles)
    # Reshape to (B, num_q_tiles, BLOCK_M, num_kv_tiles)
    m = mask_packed[:, 0, :, :].reshape(B, num_q_tiles, BLOCK_M, num_kv_tiles)
    # active[b, qt, kvt] = any non-zero mask word in BLOCK_M rows
    active = (m != 0).any(dim=2)  # (B, num_q_tiles, num_kv_tiles)
    any_active = active.any(dim=2)  # (B, num_q_tiles)

    first_kv = active.int().argmax(dim=2)  # (B, num_q_tiles)
    last_kv = (num_kv_tiles - 1
               - active.flip(2).int().argmax(dim=2))  # (B, num_q_tiles)

    start = torch.where(any_active, first_kv * BLOCK_N, 0).to(torch.int32)
    end = torch.where(any_active, (last_kv + 1) * BLOCK_N, 0).to(torch.int32)

    # Broadcast over heads: bounds[b, qt, h, 0/1]
    # wg_id = (b * num_q_tiles + qt) * H + h
    bounds = torch.zeros(B, num_q_tiles, H, 2, dtype=torch.int32, device=device)
    bounds[:, :, :, 0] = start.unsqueeze(2).expand(B, num_q_tiles, H)
    bounds[:, :, :, 1] = end.unsqueeze(2).expand(B, num_q_tiles, H)

    return bounds.reshape(-1).contiguous()


def compute_dkdv_loop_bounds(mask_packed, B, H, S_pad, BLOCK_N=32, BLOCK_M=32):
    """Compute per-workgroup loop bounds for dK+dV kernel (vectorized).

    Returns: int32 tensor of shape [grid_x * 2] where grid_x = B * num_kv_tiles * H.
    """
    num_kv_tiles = S_pad // BLOCK_N
    num_q_tiles = S_pad // BLOCK_M
    device = mask_packed.device

    # mask_packed: (B, 1, S_pad, num_kv_tiles)
    # For each kv_tile, check which q_tiles have any active mask bits.
    # Reshape to (B, num_q_tiles, BLOCK_M, num_kv_tiles) then permute
    m = mask_packed[:, 0, :, :].reshape(B, num_q_tiles, BLOCK_M, num_kv_tiles)
    # active[b, kvt, qt] = any non-zero mask word
    active = (m != 0).any(dim=2).permute(0, 2, 1)  # (B, num_kv_tiles, num_q_tiles)
    any_active = active.any(dim=2)  # (B, num_kv_tiles)

    first_qt = active.int().argmax(dim=2)  # (B, num_kv_tiles)
    last_qt = (num_q_tiles - 1
               - active.flip(2).int().argmax(dim=2))  # (B, num_kv_tiles)

    start = torch.where(any_active, first_qt * BLOCK_M, 0).to(torch.int32)
    end = torch.where(any_active, (last_qt + 1) * BLOCK_M, 0).to(torch.int32)

    # Broadcast over heads: bounds[b, kvt, h, 0/1]
    # wg_id = (b * num_kv_tiles + kvt) * H + h
    bounds = torch.zeros(B, num_kv_tiles, H, 2, dtype=torch.int32, device=device)
    bounds[:, :, :, 0] = start.unsqueeze(2).expand(B, num_kv_tiles, H)
    bounds[:, :, :, 1] = end.unsqueeze(2).expand(B, num_kv_tiles, H)

    return bounds.reshape(-1).contiguous()


@functools.lru_cache(maxsize=8)
def _compile_dq_kernel(num_heads, head_dim, dtype_str, sm_scale):
    """Compile and cache the dQ backward kernel."""
    from atrex.src.flydsl.flash_attn.bwd.kernel_bwd_dq import build_dq_module
    return build_dq_module(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        block_m=32,
        block_n=32,
    )


@functools.lru_cache(maxsize=8)
def _compile_dkdv_kernel(num_heads, head_dim, dtype_str, sm_scale):
    """Compile and cache the dK+dV backward kernel."""
    from atrex.src.flydsl.flash_attn.bwd.kernel_bwd_dkdv import build_dkdv_module
    return build_dkdv_module(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        block_m=64,
        block_n=32,
        flat_work_group_size=64,
    )


def flash_attn_bwd_precompute_mask(mask_f32, batch_size, num_heads, seq_len,
                                    block_n=32):
    """Precompute mask-related data for repeated backward calls.

    Call once per mask pattern, then pass the result as mask_ctx to
    flash_attn_bwd() to skip mask packing and bounds computation on each call.

    Args:
        mask_f32: (B, 1, S, S) float32 additive mask.
        batch_size: Batch size B.
        num_heads: Number of attention heads H.
        seq_len: Sequence length S.
        block_n: KV-tile size (default 32).

    Returns:
        dict with packed mask, loop bounds, strides, and padded mask.
    """
    seq_len_padded = ((seq_len + block_n - 1) // block_n) * block_n
    pad_size = seq_len_padded - seq_len

    if pad_size > 0:
        mask_pad = F.pad(mask_f32, [0, pad_size, 0, pad_size], value=-1e6)
    else:
        mask_pad = mask_f32.contiguous()

    mask_packed = pack_mask_to_u32(mask_pad, seq_len_padded)
    dq_bounds = compute_dq_loop_bounds(mask_packed, batch_size, num_heads, seq_len_padded)
    dkdv_bounds = compute_dkdv_loop_bounds(mask_packed, batch_size, num_heads, seq_len_padded)

    mask_words_per_row = seq_len_padded // 32
    return {
        "mask_packed": mask_packed,
        "dq_bounds": dq_bounds,
        "dkdv_bounds": dkdv_bounds,
        "mask_stride_s": mask_words_per_row,
        "mask_stride_b": seq_len_padded * mask_words_per_row,
        "seq_len_padded": seq_len_padded,
    }


def flash_attn_bwd_build(num_heads, head_dim, dtype_str="bf16", sm_scale=None,
                          block_m=32, block_n=32, is_causal=True):
    """Pre-compile both backward kernels and return a context dict.

    Args:
        num_heads: Number of attention heads.
        head_dim: Head dimension (e.g. 64, 128).
        dtype_str: "bf16" or "fp16".
        sm_scale: Softmax scale. Defaults to 1/sqrt(head_dim).
        block_m: Q-tile size (used for dQ kernel; dK+dV always uses 64).
        block_n: KV-tile size for MFMA tiling.
        is_causal: Whether to auto-generate causal mask when mask=None.

    Returns:
        dict with compiled kernel launchers and config.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    dq_launcher = _compile_dq_kernel(num_heads, head_dim, dtype_str, sm_scale)
    dkdv_launcher = _compile_dkdv_kernel(num_heads, head_dim, dtype_str, sm_scale)

    return {
        "dq_launcher": dq_launcher,
        "dkdv_launcher": dkdv_launcher,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype_str": dtype_str,
        "sm_scale": sm_scale,
        "block_m": block_m,
        "block_n": block_n,
        "is_causal": is_causal,
    }


def flash_attn_bwd(ctx, query, key, value, grad_output,
                    mask=None, mask_ctx=None,
                    lse=None, delta=None,
                    output=None):
    """Run the backward pass using pre-compiled kernels.

    Data layout: (B, H, S, D) — head-major.

    Args:
        ctx: Context dict from flash_attn_bwd_build().
        query: (B, H, S, D) bf16/fp16.
        key: (B, H, S, D) bf16/fp16.
        value: (B, H, S, D) bf16/fp16.
        grad_output: (B, H, S, D) bf16/fp16 — upstream gradient dO.
        mask: Optional (B, 1, S, S) float32 additive mask. If is_causal and mask
              is None, a causal mask is auto-generated. If not is_causal and mask
              is None, a full-attend mask (all zeros) is used.
        mask_ctx: Optional precomputed mask context from
              flash_attn_bwd_precompute_mask(). Skips mask packing and bounds
              computation when provided.
        lse: Optional (B, H, S) float32 precomputed log-sum-exp.
        delta: Optional (B, H, S) float32 precomputed Delta = sum(dO * O, dim=-1).
        output: Optional (B, H, S, D) forward output O.

    Returns:
        grad_query: (B, H, S, D) same dtype as input.
        grad_key: (B, H, S, D) same dtype as input.
        grad_value: (B, H, S, D) same dtype as input.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    device = query.device
    sm_scale = ctx["sm_scale"]
    block_n = ctx["block_n"]
    is_causal = ctx["is_causal"]

    # Auto-generate mask if needed (only when mask_ctx not provided)
    if mask is None and mask_ctx is None:
        if is_causal:
            mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32, device=device)
            causal_indices = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask[:, :, causal_indices] = -1e6
        else:
            mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32, device=device)

    # Compute LSE and Delta if not provided
    if delta is None and output is not None:
        delta = (grad_output.float() * output.float()).sum(dim=-1)
    if lse is None and delta is not None:
        query_f32 = query.float()
        key_f32 = key.float()
        scores = torch.matmul(query_f32, key_f32.transpose(-1, -2)) * sm_scale
        if mask is not None:
            scores = scores + mask
        lse = torch.logsumexp(scores, dim=-1)
    if lse is None or delta is None:
        lse, delta = _compute_lse_and_delta(query, key, value, mask, grad_output, sm_scale)

    # Compute padded seq_len for mask/bounds (kernel handles OOB internally)
    seq_len_padded = ((seq_len + block_n - 1) // block_n) * block_n

    # Ensure inputs are contiguous (no padding needed — kernel has OOB guards)
    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    grad_output_c = grad_output.contiguous()
    lse_c = lse.contiguous()
    delta_c = delta.contiguous()

    # Use precomputed mask context or compute on the fly
    if mask_ctx is not None:
        mask_packed = mask_ctx["mask_packed"]
        dq_bounds = mask_ctx["dq_bounds"]
        dkdv_bounds = mask_ctx["dkdv_bounds"]
        mask_stride_s = mask_ctx["mask_stride_s"]
        mask_stride_b = mask_ctx["mask_stride_b"]
    else:
        pad_size = seq_len_padded - seq_len
        if pad_size > 0:
            mask_pad = F.pad(mask, [0, pad_size, 0, pad_size], value=-1e6)
        else:
            mask_pad = mask.contiguous()
        mask_packed = pack_mask_to_u32(mask_pad, seq_len_padded)
        dq_bounds = compute_dq_loop_bounds(mask_packed, batch_size, num_heads, seq_len_padded)
        dkdv_bounds = compute_dkdv_loop_bounds(mask_packed, batch_size, num_heads, seq_len_padded)
        mask_words_per_row = seq_len_padded // 32
        mask_stride_s = mask_words_per_row
        mask_stride_b = seq_len_padded * mask_words_per_row

    # Allocate output tensors at actual size (stores are bounds-guarded in kernel)
    grad_query = torch.zeros_like(query_c)
    grad_key = torch.zeros_like(key_c)
    grad_value = torch.zeros_like(value_c)

    # Shared dummy tensor for unused kernel outputs
    dummy = torch.zeros_like(query_c)

    # Run dQ kernel — pass actual seq_len (kernel uses it for memory stride and OOB guards)
    # Grid tiles are computed inside the kernel as ceil(seq_len / BLOCK), which matches
    # the loop bounds computed with seq_len_padded since ceil(316/32) == ceil(320/32) == 10.
    dq_launcher = ctx["dq_launcher"]
    dq_launcher(
        query_c, key_c, value_c,
        dummy, dummy, grad_query,
        mask_packed, lse_c, delta_c, grad_output_c,
        dq_bounds, mask_stride_b, mask_stride_s,
        batch_size, seq_len,
    )

    # Run dK+dV kernel
    dkdv_launcher = ctx["dkdv_launcher"]
    dkdv_launcher(
        query_c, key_c, value_c,
        grad_value, grad_key, dummy,
        mask_packed, lse_c, delta_c, grad_output_c,
        dkdv_bounds, mask_stride_b, mask_stride_s,
        batch_size, seq_len,
    )

    return grad_query, grad_key, grad_value


def flash_attn_bwd_fast(ctx, query_pad, key_pad, value_pad, grad_output_pad,
                        mask_ctx, lse_pad, delta_pad,
                        grad_query, grad_key, grad_value, dummy,
                        batch_size, seq_len_padded):
    """Zero-overhead backward pass with pre-padded inputs and pre-allocated outputs.

    All tensors must already be padded to seq_len_padded (a multiple of block_n).
    Output tensors (grad_query, grad_key, grad_value) are zeroed and overwritten in-place.
    """
    grad_query.zero_()
    grad_key.zero_()
    grad_value.zero_()

    mask_packed = mask_ctx["mask_packed"]
    dq_bounds = mask_ctx["dq_bounds"]
    dkdv_bounds = mask_ctx["dkdv_bounds"]
    mask_stride_b = mask_ctx["mask_stride_b"]
    mask_stride_s = mask_ctx["mask_stride_s"]

    ctx["dq_launcher"](
        query_pad, key_pad, value_pad,
        dummy, dummy, grad_query,
        mask_packed, lse_pad, delta_pad, grad_output_pad,
        dq_bounds, mask_stride_b, mask_stride_s,
        batch_size, seq_len_padded,
    )

    ctx["dkdv_launcher"](
        query_pad, key_pad, value_pad,
        grad_value, grad_key, dummy,
        mask_packed, lse_pad, delta_pad, grad_output_pad,
        dkdv_bounds, mask_stride_b, mask_stride_s,
        batch_size, seq_len_padded,
    )
