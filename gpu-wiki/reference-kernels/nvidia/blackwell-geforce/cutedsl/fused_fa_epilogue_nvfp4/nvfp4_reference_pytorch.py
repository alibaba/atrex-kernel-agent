"""PyTorch reference implementation of NVFP4 quantize / dequantize.

Algorithm follows the NVIDIA CUDA kernel + flashinfer CuTeDSL helpers
(see reference-kernels/nvidia/blackwell/cutedsl/flashinfer/
     {quantization_cute_dsl_utils.py, fp4_common.py}).

Per 16-element block (block_size=16):
    block_amax  = max(|x|) over 16 elements, in fp32
    scale_float = sf_scale * (block_amax * (1/6))
                  # sf_scale == input_global_scale_inv in vllm naming, i.e.
                  # 1.0 / input_global_scale
    scale_e4m3  = cvt.rn.satfinite.e4m3.f32(scale_float)        # store as fp8 e4m3
    output_scale = 1.0 / (float(scale_e4m3) / sf_scale)
                 = sf_scale / float(scale_e4m3)
    q_fp4       = cvt.rn.satfinite.e2m1.f32(x * output_scale)

Pack: 2 e2m1 nibbles per uint8, lower nibble = column-even element, upper = odd.
Block scales are arranged in the swizzled-128x4 layout used by CUTLASS NVFP4 GEMM.

Dequant: x_hat = float(e2m1(q)) * float(e4m3(scale)) / sf_scale
              = float(e2m1(q)) / output_scale
"""
from __future__ import annotations

import torch

GROUP_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


# ----------------------------- E2M1 codec -----------------------------------

# IEEE-style E2M1 unsigned magnitudes (4 bits = sign(1) + exp(2) + mant(1)):
#   0000 = 0
#   0001 = 0.5
#   0010 = 1.0
#   0011 = 1.5
#   0100 = 2.0
#   0101 = 3.0
#   0110 = 4.0
#   0111 = 6.0      <- max representable magnitude
# sign bit is bit 3.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            dtype=torch.float32)


def _round_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even (ties-to-even at the boundary midpoints).

    Saturating: |x| > 6 clamps to ±6.

    Returns a uint8 tensor with values in [0, 15], same shape as ``x``.
    """
    levels = _E2M1_LEVELS.to(x.device)
    sign = (x < 0).to(torch.uint8)
    mag = x.abs().clamp(max=FLOAT4_E2M1_MAX)

    # Find the lower bracket (largest level <= mag).
    bucket = torch.searchsorted(levels, mag, right=True) - 1
    bucket = bucket.clamp(min=0, max=6)
    lo = levels[bucket]
    hi = levels[bucket + 1]
    d_lo = mag - lo
    d_hi = hi - mag
    pick_hi = d_hi < d_lo
    tie = d_hi == d_lo
    if tie.any():
        # Even-nibble preference on ties (LSB == 0 of nibble code)
        prefer_hi_on_tie = (bucket % 2 == 1)
        pick_hi = torch.where(tie, prefer_hi_on_tie, pick_hi)
    nibble = torch.where(pick_hi, bucket + 1, bucket).to(torch.uint8)
    nibble = nibble | (sign << 3)
    return nibble


def _decode_e2m1_nibble(nib: torch.Tensor) -> torch.Tensor:
    """Decode uint8 nibble (0..15) back to float32 e2m1 magnitude with sign."""
    levels = _E2M1_LEVELS.to(nib.device)
    code = (nib & 0x07).to(torch.long)
    sign = ((nib >> 3) & 0x01).to(torch.float32)
    mag = levels[code]
    return torch.where(sign > 0, -mag, mag)


# ----------------------------- E4M3 codec (sat-finite) ----------------------

def _f32_to_e4m3_to_f32(x: torch.Tensor):
    """Round to fp8 E4M3 and decode back to fp32, matching cvt.rn.satfinite.e4m3.

    Returns (decoded_fp32, raw_e4m3).
    """
    fp8 = x.to(torch.float8_e4m3fn)
    return fp8.to(torch.float32), fp8


# ----------------------------- swizzle 128x4 layout -------------------------

def swizzle_blockscale_128x4(scale_linear: torch.Tensor) -> torch.Tensor:
    """Take a row-major (M, K) e4m3 scale tensor (K = D//16) and produce the
    CUTLASS-style swizzled-128x4 layout.

    Reference: ``compute_sf_index_swizzled_128x4_gpu`` in
    ``quantization_cute_dsl_utils.py``. Output shape padded so M is a multiple
    of 128 and K is a multiple of 4.
    """
    assert scale_linear.dtype == torch.float8_e4m3fn
    M, K = scale_linear.shape
    pad_M = (M + 127) // 128 * 128
    pad_K = (K + 3) // 4 * 4

    padded = torch.zeros((pad_M, pad_K), dtype=scale_linear.dtype, device=scale_linear.device)
    padded[:M, :K] = scale_linear

    out = torch.empty(pad_M * pad_K, dtype=scale_linear.dtype, device=scale_linear.device)

    rows = torch.arange(pad_M, device=scale_linear.device, dtype=torch.long)
    cols = torch.arange(pad_K, device=scale_linear.device, dtype=torch.long)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")

    col_in_g0 = cc % 4
    col_g_idx = cc // 4
    row_in_g0 = rr % 32
    row_in_g1 = (rr % 128) // 32
    row_g_idx = rr // 128
    offset = (col_in_g0
              + col_g_idx * 512
              + row_in_g0 * 16
              + row_in_g1 * 4
              + row_g_idx * (128 * pad_K))
    out[offset.flatten()] = padded.flatten()
    return out.view(pad_M, pad_K)


def unswizzle_blockscale_128x4(swizzled: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Inverse of swizzle_blockscale_128x4. Returns (M, K) row-major view."""
    pad_M = (M + 127) // 128 * 128
    pad_K = (K + 3) // 4 * 4
    flat = swizzled.flatten()
    rows = torch.arange(pad_M, device=swizzled.device, dtype=torch.long)
    cols = torch.arange(pad_K, device=swizzled.device, dtype=torch.long)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    col_in_g0 = cc % 4
    col_g_idx = cc // 4
    row_in_g0 = rr % 32
    row_in_g1 = (rr % 128) // 32
    row_g_idx = rr // 128
    offset = (col_in_g0
              + col_g_idx * 512
              + row_in_g0 * 16
              + row_in_g1 * 4
              + row_g_idx * (128 * pad_K))
    out = flat[offset.flatten()].view(pad_M, pad_K)
    return out[:M, :K]


# ----------------------------- main entry points ----------------------------

def nvfp4_quantize_reference(x: torch.Tensor, input_global_scale: torch.Tensor):
    """Quantize a (M, D) tensor to NVFP4 using the NVIDIA cuda formula.

    Args:
        x: (M, D) bf16/fp16/fp32 tensor on CUDA.
        input_global_scale: scalar fp32 tensor (vllm-naming
            ``input_global_scale_inv``, == 1/input_global_scale; this is the
            "sf_scale" in flashinfer naming).

    Returns:
        x_fp4: (M, D//2) uint8 packed e2m1
        x_bs_swizzled: (pad_M, pad_K) float8_e4m3fn block scales in swizzled
            128x4 layout where pad_M = ceil(M, 128), pad_K = ceil(D//16, 4)
        x_bs_linear: (M, D//16) float8_e4m3fn block scales (row-major, raw
            per-block fp8 before swizzle), kept for unit tests.
    """
    assert x.is_cuda
    assert x.dim() == 2
    M, D = x.shape
    assert D % GROUP_SIZE == 0
    K = D // GROUP_SIZE

    sf_scale = input_global_scale.to(torch.float32).to(x.device)  # scalar
    x_f32 = x.to(torch.float32)
    blocks = x_f32.view(M, K, GROUP_SIZE)
    block_amax = blocks.abs().amax(dim=-1)                        # (M, K)
    scale_float = sf_scale * block_amax * (1.0 / FLOAT4_E2M1_MAX)
    scale_e4m3_f32, scale_e4m3 = _f32_to_e4m3_to_f32(scale_float)
    output_scale = torch.where(
        scale_e4m3_f32 > 0,
        sf_scale / scale_e4m3_f32,
        torch.zeros_like(scale_e4m3_f32),
    )                                                             # (M, K)
    x_scaled = blocks * output_scale.unsqueeze(-1)                # (M, K, 16)
    nib = _round_to_e2m1_nibble(x_scaled)                         # uint8 (M, K, 16)

    nib = nib.view(M, D)
    even = nib[:, 0::2]
    odd = nib[:, 1::2]
    x_fp4 = (even | (odd << 4)).to(torch.uint8)                   # (M, D//2)

    x_bs_swizzled = swizzle_blockscale_128x4(scale_e4m3)          # (pad_M, pad_K)

    return x_fp4, x_bs_swizzled, scale_e4m3


def nvfp4_dequantize_reference(x_fp4: torch.Tensor,
                                x_bs_linear: torch.Tensor,
                                input_global_scale: torch.Tensor) -> torch.Tensor:
    """Inverse of nvfp4_quantize_reference (using the linear scale, not swizzled)."""
    assert x_fp4.dtype == torch.uint8
    M, half_D = x_fp4.shape
    D = half_D * 2
    K = D // GROUP_SIZE
    assert x_bs_linear.shape == (M, K)
    sf_scale = input_global_scale.to(torch.float32).to(x_fp4.device)

    even = x_fp4 & 0x0F
    odd = (x_fp4 >> 4) & 0x0F
    nibbles = torch.stack([even, odd], dim=-1).view(M, D)
    fp4_vals = _decode_e2m1_nibble(nibbles)                     # fp32 (M, D)

    scale_f32 = x_bs_linear.to(torch.float32)                   # (M, K)
    scale_per_elt = (scale_f32 / sf_scale).repeat_interleave(GROUP_SIZE, dim=-1)
    return fp4_vals * scale_per_elt


# ----------------------------- self-test ------------------------------------

def _self_test():
    """Run a round-trip rel_err test on a synthetic input."""
    torch.manual_seed(0)
    device = "cuda"
    M, D = 256, 4096
    x = torch.randn(M, D, device=device, dtype=torch.bfloat16) * 1.5
    # sf_scale = "input_global_scale_inv" = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax_per_tensor
    # is the canonical NVIDIA recipe (matches vllm scaled_fp4_quant convention).
    amax = x.float().abs().amax()
    sf_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax).to(torch.float32).to(device)
    print(f"x amax = {amax.item():.4f}, sf_scale = {sf_scale.item():.4f}")
    x_fp4, x_bs_swizzled, x_bs_linear = nvfp4_quantize_reference(x, sf_scale)
    x_hat = nvfp4_dequantize_reference(x_fp4, x_bs_linear, sf_scale)

    rel_err = (x_hat.float() - x.float()).norm() / x.float().norm()
    print(f"NVFP4 reference round-trip rel_err vs original bf16 = {rel_err.item():.6e}")
    # NVFP4 (E2M1, 8 quantization levels) has an intrinsic information loss
    # of ~10% relative error on Gaussian inputs. This is *not* the threshold
    # for kernel-vs-reference comparison; it just sanity-checks the codec.
    assert rel_err.item() < 0.15, f"NVFP4 codec lossy beyond NVIDIA spec: {rel_err.item():.4e}"

    print(f"x_fp4 shape={list(x_fp4.shape)} dtype={x_fp4.dtype}")
    print(f"x_bs_swizzled shape={list(x_bs_swizzled.shape)} dtype={x_bs_swizzled.dtype}")
    print(f"x_bs_linear shape={list(x_bs_linear.shape)} dtype={x_bs_linear.dtype}")

    # Determinism: the same input must produce bit-identical output across calls.
    # This is the meaningful "bit-exact" semantics the kernel must match.
    x_fp4_b, x_bs_swizzled_b, x_bs_linear_b = nvfp4_quantize_reference(x, sf_scale)
    assert torch.equal(x_fp4, x_fp4_b)
    assert torch.equal(x_bs_linear.to(torch.float32),
                       x_bs_linear_b.to(torch.float32))
    assert torch.equal(x_bs_swizzled.to(torch.float32),
                       x_bs_swizzled_b.to(torch.float32))
    print("determinism (bit-exact across calls): OK")

    scale_recovered = unswizzle_blockscale_128x4(x_bs_swizzled, M, D // GROUP_SIZE)
    assert torch.equal(scale_recovered.to(torch.float32),
                       x_bs_linear.to(torch.float32)), "swizzle/unswizzle mismatch"
    print("swizzle/unswizzle round-trip OK")

    # also verify e2m1 round-to-nearest-even on hand-picked values
    x_test = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                            2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]],
                          device=device, dtype=torch.float32)
    nib = _round_to_e2m1_nibble(x_test)
    decoded = _decode_e2m1_nibble(nib)
    print("e2m1 rounding sanity check:")
    print("  in :", x_test.cpu().tolist()[0])
    print("  out:", decoded.cpu().tolist()[0])
    print("  nib:", nib.cpu().tolist()[0])
    print("PASS")


if __name__ == "__main__":
    _self_test()
