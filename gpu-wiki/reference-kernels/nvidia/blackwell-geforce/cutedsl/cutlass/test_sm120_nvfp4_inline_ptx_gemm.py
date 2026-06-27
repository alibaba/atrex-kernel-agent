"""Regression test for the SM120 NVFP4 inline PTX GEMM atom demo."""

import importlib.util
from pathlib import Path

import torch


def _load_demo_module():
    path = Path(__file__).with_name("sm120_nvfp4_inline_ptx_gemm.py")
    spec = importlib.util.spec_from_file_location("sm120_nvfp4_inline_ptx_gemm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = _load_demo_module()
    worst_rel = 0.0
    worst_abs = 0.0
    worst_seed = None

    for seed in range(10):
        torch.manual_seed(seed)
        a = torch.randn((mod.TILE_M, mod.TILE_K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((mod.TILE_N, mod.TILE_K), device="cuda", dtype=torch.bfloat16)
        a_q, a_sf, b_q, b_sf, a_g, b_g, alpha = mod.quantize_problem(a, b)

        out = torch.empty((mod.TILE_M, mod.TILE_N), device="cuda", dtype=torch.float32)
        stream = mod.cutlass_torch.default_stream()
        mod._launch(
            mod.from_dlpack(a_q, assumed_align=16),
            mod.from_dlpack(b_q, assumed_align=16),
            mod.from_dlpack(a_sf, assumed_align=16),
            mod.from_dlpack(b_sf, assumed_align=16),
            mod.from_dlpack(alpha, assumed_align=4),
            mod.from_dlpack(out, assumed_align=16),
            stream,
        )
        torch.cuda.synchronize()

        a_deq = mod.e2m1_and_ufp8sf_scale_to_float(
            a_q,
            a_sf,
            1.0 / a_g,
            sf_vec_size=mod.SF_VEC_SIZE,
            ufp8_type=1,
            is_sf_swizzled_layout=True,
        ).to(device="cuda")
        b_deq = mod.e2m1_and_ufp8sf_scale_to_float(
            b_q,
            b_sf,
            1.0 / b_g,
            sf_vec_size=mod.SF_VEC_SIZE,
            ufp8_type=1,
            is_sf_swizzled_layout=True,
        ).to(device="cuda")
        ref = a_deq.float() @ b_deq.float().T

        rel = ((out - ref).norm() / ref.norm().clamp_min(1e-8)).item()
        max_abs = (out - ref).abs().max().item()
        if rel > worst_rel:
            worst_rel = rel
            worst_seed = seed
        worst_abs = max(worst_abs, max_abs)

    print(f"worst_seed={worst_seed}, worst_rel={worst_rel:.9e}, worst_abs={worst_abs:.9e}")
    assert worst_rel < 1e-6
    assert worst_abs < 1e-5


if __name__ == "__main__":
    main()
