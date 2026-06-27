"""Triton fused RMSNormGated kernel for vLLM GDN (Gated DeltaNet) post-processing.

⚠ TUNED FOR sm_120 (RTX PRO 5000 / 4000 Blackwell-GeForce).
   Key tuning vs a naive [BLOCK_M=8] config: BLOCK_M=2, num_warps=4, num_stages=3.
   See diff vs naive at the bottom of this docstring.

Replaces the eager (RMSNormGated + SiLU(z) gating) sequence in vLLM
`_deltanet_post`, leaving the downstream `scaled_fp4_quant` + cutlass NVFP4 mm
unchanged.

Shape contract (canonical workload, attn.py default):
    core_out : bf16 [N, H_V, D]    e.g. [6144, 32, 128]
    z        : bf16 [N, H_V, D]    or [N, H_V*D] flat
    norm_w   : bf16 [D]            e.g. [128]
    out      : bf16 [N, H_V*D]     e.g. [6144, 4096]

Algorithm (matches attn.py::_deltanet_post lines 75-92):
    x        = core_out.float()
    var      = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * rsqrt(var + eps) * norm_w.float()
    silu_z   = z.float() * sigmoid(z.float())
    out      = (x_normed * silu_z).to(bf16).reshape(N, H_V*D)

Each Triton program handles BLOCK_M *rows* (one row = H_V * D = 4096 elems
@ canonical shape). Per-head RMSNorm is independent; we vectorize over
(BLOCK_M * H_V) heads inside one program by indexing a 3D tile [BM, H_V, D].

Final perf @ canonical [6144, 32, 128] bf16 on RTX PRO 5000 (sm_120, 110 SMs):
    eager  norm+gate (8 ops)             1404.66 us
    triton norm+gate (this kernel)        107.96 us   13.01x speedup
    eager  norm+gate + scaled_fp4_quant  1411.62 us
    triton norm+gate + scaled_fp4_quant   129.02 us   10.94x speedup
    end-to-end deltanet_forward          2402.17 us  ->  1112.36 us  (2.16x)

Working set & bandwidth (this kernel only):
    read 100 MB (core_out 50 + z 50 + norm_w 256B) + write 48 MB = 148 MB
    measured time 108 us -> 1370 GB/s achieved (122% of D2D memcpy ceiling
    1110 GB/s, because the ceiling assumes balanced 1:1 R:W; real R:W is
    100:48 ≈ 2:1, which is more bandwidth-efficient on sm_120).

Diff vs naive [BLOCK_M=8] config — reasoning recorded for posterity:
  | Knob        | Naive | This | Why                                              |
  |-------------|-------|------|--------------------------------------------------|
  | BLOCK_M     | 8     | 2    | Naive packs 65 LDG.E.128 per program; SM stays  |
  |             |       |      | busy on a single program → on-flight programs   |
  |             |       |      | per SM collapses; grid becomes 768 progs vs 110 |
  |             |       |      | SMs (waves≈7.03, partial-wave tail). BM=2 makes  |
  |             |       |      | grid=3072, 4× more concurrency, 4× speedup.      |
  | num_warps   | 4     | 4    | Sweep showed 2 also tied; 4 is safe + matches   |
  |             |       |      | block-shape align.                               |
  | num_stages  | 2     | 3    | Marginal (108.6 → 108.5 us). cp.async pipeline  |
  |             |       |      | is not the bottleneck at BM=2.                   |
  | cache_mod   | none  | .cg  | LANDED in PTX (ld.global.cg.v4.b32) but ZERO    |
  |             |       |      | perf gain on its own — see pitfalls.            |
  | multiple_of | none  | 8    | Forces LDG.E.128. Same caveat as cache_mod.     |

Related docs:
  Optimization journey (V1 → V2 → V3, with measurements + decision log):
    docs/ref-docs/nvidia/triton/sm120/sm120-fused-rmsnorm-gated-bf16-optimization.md
  Pitfalls discovered along the way:
    docs/pitfalls/nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md

Cross-architecture relatives (for the *real* fusion, not done here):
  CuTeDSL real-fusion attempt (blocked on cluster cutlass 4.4.2):
    docs/ref-docs/nvidia/cutedsl/sm120/v3-fa-fusion-deferred-plan.md
  CuTeDSL standalone fused-quant epilogue (3 architectures all hit memcpy wall):
    docs/ref-docs/nvidia/cutedsl/sm120/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rmsnorm_gated_kernel(
    core_out_ptr,    # bf16 [N, H_V, D]
    z_ptr,           # bf16 [N, H_V, D]
    norm_w_ptr,      # bf16 [D]
    out_ptr,         # bf16 [N, H_V * D]
    co_sn, co_sh, co_sd,
    z_sn, z_sh, z_sd,
    out_sn, out_sd,
    N: tl.constexpr,
    H_V: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < N

    h_off = tl.arange(0, H_V)
    d_off = tl.arange(0, D)

    core_ptrs = (
        core_out_ptr
        + row_offsets[:, None, None] * co_sn
        + h_off[None, :, None] * co_sh
        + d_off[None, None, :] * co_sd
    )
    z_ptrs = (
        z_ptr
        + row_offsets[:, None, None] * z_sn
        + h_off[None, :, None] * z_sh
        + d_off[None, None, :] * z_sd
    )

    # Vectorization hints for 16B loads (8 bf16 per LDG.E.128).
    # NOTE: hints alone gave 0% perf gain on V2 (cf. pitfalls #1) — they pay off
    # only after BLOCK_M was reduced to 2 in V3.
    d_off = tl.multiple_of(d_off, 8)
    d_off = tl.max_contiguous(d_off, 128)

    mask3d = row_mask[:, None, None] & tl.full([1, H_V, D], 1, tl.int1)

    x = tl.load(core_ptrs, mask=mask3d, other=0.0, cache_modifier=".cg").to(tl.float32)
    z = tl.load(z_ptrs,    mask=mask3d, other=0.0, cache_modifier=".cg").to(tl.float32)
    w = tl.load(norm_w_ptr + d_off, cache_modifier=".ca").to(tl.float32)

    var = tl.sum(x * x, axis=2) / D                                      # [M, H]
    inv_rms = tl.rsqrt(var + EPS)                                        # [M, H]
    x_normed = x * inv_rms[:, :, None]
    x_normed = x_normed * w[None, None, :]

    silu_z = z * tl.sigmoid(z)
    out = x_normed * silu_z
    out_bf16 = out.to(tl.bfloat16)

    flat_d = h_off[None, :, None] * D + d_off[None, None, :]
    out_ptrs = (
        out_ptr
        + row_offsets[:, None, None] * out_sn
        + flat_d * out_sd
    )
    tl.store(out_ptrs, out_bf16, mask=mask3d)


def fused_rmsnorm_gated(
    core_out: torch.Tensor,
    z: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float = 1e-6,
    BLOCK_M: int = 2,        # sm_120 sweep showed BM=1/2/4 are 4x faster than BM=8
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """RMSNorm + SiLU(z) gating, returns bf16 [N, H_V*D] flat for downstream quant.
    Drop-in replacement for the eager block in vLLM `_deltanet_post`.
    """
    assert core_out.dtype == torch.bfloat16
    assert z.dtype == torch.bfloat16
    assert norm_w.dtype == torch.bfloat16
    assert core_out.is_contiguous()

    N, H_V, D = core_out.shape
    PROJ_IN = H_V * D
    assert norm_w.shape == (D,)

    if z.dim() == 2:
        assert z.shape == (N, PROJ_IN)
        z = z.view(N, H_V, D)
    z = z.contiguous()

    out = torch.empty((N, PROJ_IN), dtype=torch.bfloat16, device=core_out.device)

    grid = (triton.cdiv(N, BLOCK_M),)
    fused_rmsnorm_gated_kernel[grid](
        core_out, z, norm_w, out,
        core_out.stride(0), core_out.stride(1), core_out.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        out.stride(0), out.stride(1),
        N=N, H_V=H_V, D=D, EPS=eps,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def fused_rmsnorm_gated_then_quant(
    core_out: torch.Tensor,
    z: torch.Tensor,
    norm_w: torch.Tensor,
    input_gs_inv: torch.Tensor,
    eps: float = 1e-6,
    BLOCK_M: int = 2,
    num_warps: int = 4,
    num_stages: int = 3,
):
    """Drop-in replacement for the (RMSNormGated + scaled_fp4_quant) block of
    vLLM `_deltanet_post`. Calls vLLM's scaled_fp4_quant on the bf16 result.
    Returns (x_fp4, x_bs) compatible with cutlass_scaled_fp4_mm.
    """
    from vllm._custom_ops import scaled_fp4_quant
    x_bf16 = fused_rmsnorm_gated(
        core_out, z, norm_w, eps=eps,
        BLOCK_M=BLOCK_M, num_warps=num_warps, num_stages=num_stages,
    )
    x_fp4, x_bs = scaled_fp4_quant(
        x_bf16, input_gs_inv,
        is_sf_swizzled_layout=True, backend="cutlass",
    )
    return x_fp4, x_bs
