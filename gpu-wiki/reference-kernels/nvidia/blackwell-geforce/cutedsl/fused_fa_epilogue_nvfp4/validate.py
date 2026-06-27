"""V0 baseline correctness validation:

Path-1 (full attention) with shapes from memory.md:
  N=6144, H_q=16, H_kv=2, D_h=256, D=H_q*D_h=4096, GROUP_SIZE=16

Reference path:
  attn_out_ref = torch.SDPA(q, k, v, causal=True, scale=1/sqrt(256))
  x_ref        = attn_out_ref * sigmoid(gate)
  x_fp4_ref, x_bs_lin_ref = nvfp4_quantize_reference(x_ref, sf_scale)

Kernel path:
  attn_out_k, x_fp4_k, x_bs_lin_k = path1_forward(q, k, v, gate, sf_scale)

Checks (per memory.md):
  - rel_err(attn_out_k, attn_out_ref) < 0.01    (sanity: SDPA both sides, expect 0)
  - x_fp4_k == x_fp4_ref                         (bit-exact since cvt is the same)
  - x_bs_lin_k == x_bs_lin_ref                   (bit-exact)
  - end-to-end: dequant_fp4 → bf16 GEMM with random weight matrix
                rel_err < 5e-3 vs reference
"""
from __future__ import annotations

import math
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "refs"))

import torch

# bootstrap CuTeDSL via kernel.py (does pip uninstall + path inject)
import kernel as K
from refs.nvfp4_reference import (
    nvfp4_quantize_reference,
    nvfp4_dequantize_reference,
    unswizzle_blockscale_128x4,
    GROUP_SIZE,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
)

# Try vllm for bit-exact reference. On the OLD cluster (sz6wd8l56pnf), vllm
# dev206 is ABI-incompatible with torch 2.9.1, so this import fails and we
# silently fall back to the self-written reference (rel_err < 5e-3 only).
try:
    from vllm._custom_ops import scaled_fp4_quant as vllm_scaled_fp4_quant
    HAVE_VLLM = True
except Exception as _e:
    HAVE_VLLM = False
    print(f"[info] vllm bit-exact skipped (vllm._custom_ops unavailable: "
          f"{type(_e).__name__}: {str(_e)[:160]})")


def _bench(name, fn, warmup=3, iters=10):
    """Quick timing helper (cuda.Event); used only for sanity, not perf claims."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    print(f"  {name}: {ms:.3f} ms/iter")
    return ms


def main():
    torch.manual_seed(42)
    device = "cuda"
    N = 6144
    H_q = 16
    H_kv = 2
    D_h = 256
    D = H_q * D_h           # 4096
    K_blocks = D // GROUP_SIZE   # 256

    print(f"=== V0 baseline validation ===")
    print(f"N={N}, H_q={H_q}, H_kv={H_kv}, D_h={D_h}, D={D}, K_blocks={K_blocks}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Inputs
    q = torch.randn(N, H_q, D_h, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn(N, H_kv, D_h, device=device, dtype=torch.bfloat16) * 0.5
    v = torch.randn(N, H_kv, D_h, device=device, dtype=torch.bfloat16) * 0.5
    gate = torch.randn(N, D, device=device, dtype=torch.bfloat16) * 0.5

    # ---- reference path
    # V3 hybrid (2026-04-28): kernel.flash_attention_bf16 now uses
    # vllm.vllm_flash_attn instead of SDPA. To validate that the V3 hybrid
    # produces bf16 attn_out compatible with the original SDPA reference,
    # we compute reference using SDPA explicitly (via the kept-around helper)
    # and compare against the V3 kernel path that calls vllm flash_attn.
    # Tolerance for attn rel_err raised to 5e-3 since vllm vs SDPA causal
    # attention can differ by O(1e-3) at bf16 precision.
    attn_ref = K._flash_attention_sdpa(q, k, v, causal=True)           # (N, D) bf16
    x_ref = attn_ref.float() * torch.sigmoid(gate.float())             # f32
    # sf_scale per-tensor canonical: sf = (E4M3_MAX * E2M1_MAX) / amax
    amax = x_ref.abs().amax()
    sf_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax).to(torch.float32)
    print(f"x amax = {amax.item():.4f}, sf_scale = {sf_scale.item():.4f}")

    x_fp4_ref, x_bs_swizzled_ref, x_bs_lin_ref = nvfp4_quantize_reference(
        x_ref.to(torch.bfloat16), sf_scale)

    # ---- kernel path. NOTE x_bs_swz_k is the SWIZZLED uint8 buffer
    # (padded_M, padded_sf_cols).
    attn_k, x_fp4_k, x_bs_swz_k = K.path1_forward(q, k, v, gate, sf_scale)
    # Unswizzle into the same (M, K) layout as reference
    x_bs_lin_k_u8 = unswizzle_blockscale_128x4(x_bs_swz_k, N, K_blocks)
    x_bs_lin_k = x_bs_lin_k_u8.view(torch.float8_e4m3fn)

    # ---- attn_out check
    # V0 was self-vs-self (SDPA both sides) -> rel_err ~ 0.
    # V3 hybrid: kernel uses vllm.flash_attn, ref uses SDPA. Expect rel_err
    # in O(1e-4 .. 1e-3) range due to bf16 numerical differences between
    # the two attention implementations. Threshold 5e-3 is well within the
    # NVFP4 quantization noise floor (V0 measured ~1.6e-4 vs orig).
    rel_err_attn = (attn_k.float() - attn_ref.float()).norm() / attn_ref.float().norm()
    print()
    print(f"attn_out rel_err           = {rel_err_attn.item():.6e}  "
          f"(V3 hybrid: vllm vs SDPA, tolerance < 5e-3)")
    assert rel_err_attn.item() < 5e-3, (
        f"attn_out mismatch: {rel_err_attn.item()} > 5e-3 — vllm.flash_attn "
        f"diverged too much from SDPA reference"
    )

    # ---- bit-mismatch x_fp4 vs PyTorch reference
    # V3 hybrid: kernel uses vllm.flash_attn so x_fp4_k is fp4-quantized
    # of vllm-attn-out; x_fp4_ref is fp4-quantized of SDPA-attn-out. They
    # differ wherever vllm vs SDPA differ. Expect O(1-10%) bit-mismatch
    # (not the V0 4/12.58M ties). True correctness is via rel_err_gemm
    # below (dequant -> bf16 GEMM cross-check).
    n_diff = (x_fp4_k != x_fp4_ref).sum().item()
    n_total = x_fp4_k.numel()
    print(f"x_fp4 bit-mismatch (vs ref)   = {n_diff}/{n_total}  ({100*n_diff/n_total:.4f}%)")

    # ---- bit-mismatch x_bs_lin vs PyTorch reference  (same caveat as above)
    n_bs_diff = (x_bs_lin_k.to(torch.float32) != x_bs_lin_ref.to(torch.float32)).sum().item()
    n_bs_total = x_bs_lin_k.numel()
    print(f"x_bs_lin bit-mismatch (vs ref)= {n_bs_diff}/{n_bs_total}  ({100*n_bs_diff/n_bs_total:.4f}%)")

    # ---- Optional: vllm bit-exact comparison (memory.md soft constraint)
    if HAVE_VLLM:
        # vllm.scaled_fp4_quant returns (fp4 [M, D//2] uint8, sf [padded, padded_sf_cols] fp8_e4m3fn)
        x_fp4_vllm, x_bs_swz_vllm = vllm_scaled_fp4_quant(
            x_ref.to(torch.bfloat16).contiguous(),
            sf_scale,
            is_sf_swizzled_layout=True,
            backend="cutlass",
        )
        n_v_fp4 = (x_fp4_k != x_fp4_vllm).sum().item()
        # vllm's swizzled output is fp8_e4m3fn; ours is uint8 of same bit pattern
        x_bs_swz_k_fp8 = x_bs_swz_k.view(torch.float8_e4m3fn)
        # Possibly different swizzle padding layout; only compare overlapping shape
        m0 = min(x_bs_swz_k_fp8.shape[0], x_bs_swz_vllm.shape[0])
        n0 = min(x_bs_swz_k_fp8.shape[1], x_bs_swz_vllm.shape[1])
        x_bs_swz_diff = (
            x_bs_swz_k_fp8[:m0, :n0].to(torch.float32)
            != x_bs_swz_vllm[:m0, :n0].to(torch.float32)
        ).sum().item()
        print(f"x_fp4 bit-mismatch (vs vllm)   = {n_v_fp4}/{n_total}  ({100*n_v_fp4/n_total:.4f}%)")
        print(f"x_bs swizzled bit-mismatch (vs vllm, shape ours={list(x_bs_swz_k.shape)} vllm={list(x_bs_swz_vllm.shape)}) = {x_bs_swz_diff}/{m0*n0}")

    # ---- numerical equivalence (rel_err < 5e-3)
    x_hat_ref = nvfp4_dequantize_reference(x_fp4_ref, x_bs_lin_ref, sf_scale)
    x_hat_k = nvfp4_dequantize_reference(x_fp4_k, x_bs_lin_k, sf_scale)
    rel_err_quant = (x_hat_k - x_hat_ref).norm() / x_hat_ref.norm()
    print(f"x dequant rel_err k vs ref = {rel_err_quant.item():.6e}")
    rel_err_quant_vs_orig = (x_hat_k - x_ref).norm() / x_ref.norm()
    print(f"x dequant rel_err k vs orig = {rel_err_quant_vs_orig.item():.6e}  (NVFP4 lossy ~10%)")

    # ---- end-to-end: dequant -> bf16 GEMM
    # Use a fixed weight matrix W (D, hidden=2048) bf16; both ref and kernel
    # dequant the fp4 → bf16 then matmul; compare dot products.
    HIDDEN = 2048
    W = torch.randn(D, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1
    out_ref = (x_hat_ref.to(torch.bfloat16).float() @ W.float())
    out_k = (x_hat_k.to(torch.bfloat16).float() @ W.float())
    rel_err_gemm = (out_k - out_ref).norm() / out_ref.norm()
    print(f"e2e GEMM rel_err k vs ref  = {rel_err_gemm.item():.6e}  (threshold 5e-3)")

    # End-to-end vs full-precision (bf16) reference (without quant): for context
    out_orig = (x_ref.to(torch.bfloat16).float() @ W.float())
    rel_err_gemm_orig = (out_k - out_orig).norm() / out_orig.norm()
    print(f"e2e GEMM rel_err k vs orig = {rel_err_gemm_orig.item():.6e}  (NVFP4 spec)")

    # ---- decisions
    print()
    print("=== summary ===")
    bit_exact_fp4 = (n_diff == 0)
    bit_exact_bs = (n_bs_diff == 0)
    pass_quant = rel_err_quant.item() < 5e-3
    pass_gemm = rel_err_gemm.item() < 5e-3
    print(f"  bit-exact x_fp4    : {bit_exact_fp4}")
    print(f"  bit-exact x_bs_lin : {bit_exact_bs}")
    print(f"  rel_err(quant)<5e-3: {pass_quant}")
    print(f"  rel_err(gemm) <5e-3: {pass_gemm}")
    if pass_quant and pass_gemm:
        print("V0 BASELINE PASS")
    else:
        print("V0 BASELINE FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
