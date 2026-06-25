# Copyright (c) 2026, Tri Dao.

from functools import partial
from typing import Callable, Optional, Type, Tuple

import torch

import cutlass
import cutlass.cute as cute

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import div_for_dtype, make_scheduler_args
from quack.mx_utils import (
    to_mx_compiled,
    to_mxfp4_compiled,
    to_nvfp4_compiled,
)
from quack.varlen_utils import VarlenArguments


TORCH_DTYPE_MAP = {
    cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
    cutlass.Float32: torch.float32,
    cutlass.Float8E4M3FN: torch.float8_e4m3fn,
    cutlass.Float8E5M2: torch.float8_e5m2,
    cutlass.Float8E8M0FNU: torch.float8_e8m0fnu,
}

FLOAT8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
}


FP4_E2M1FN_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_dtype_for_cutlass(dtype: Type[cutlass.Numeric]) -> torch.dtype:
    if dtype not in TORCH_DTYPE_MAP:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return TORCH_DTYPE_MAP[dtype]


def _make_fake_tensor_like(tensor: torch.Tensor, dtype: Type[cutlass.Numeric]) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        tensor.shape,
        stride=tensor.stride(),
        assumed_align=16,
    )


def _leading_dim_from_stride(tensor: torch.Tensor) -> int:
    for i, stride in enumerate(tensor.stride()):
        if stride == 1:
            return i
    raise ValueError(
        f"Tensor has no unit stride dimension: shape={tensor.shape}, stride={tensor.stride()}"
    )


def _make_compile_tensor_like(
    tensor: torch.Tensor, dtype: Type[cutlass.Numeric], dynamic_layout: bool = False
) -> cute.Tensor:
    compile_tensor = cute.runtime.from_dlpack(tensor)
    compile_tensor.element_type = dtype
    if dynamic_layout:
        marked = compile_tensor.mark_layout_dynamic(leading_dim=_leading_dim_from_stride(tensor))
        if marked is not None:
            compile_tensor = marked
    return compile_tensor


def _make_fake_compact_tensor(
    shape: Tuple[int, ...], dtype: Type[cutlass.Numeric], leading_dim: int
) -> cute.Tensor:
    logical_shape = list(shape)
    if dtype == cutlass.Float4E2M1FN:
        logical_shape[leading_dim] *= 2
    return fake_tensor(
        dtype,
        tuple(logical_shape),
        leading_dim=leading_dim,
        divisibility=div_for_dtype(dtype),
    )


def _fp4_e2m1fn_value_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(FP4_E2M1FN_VALUES, dtype=torch.float32, device=device)


def _pack_fp4_e2m1fn_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical FP4 codes into torch.float4_e2m1fn_x2 storage."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 FP4 codes, got {codes.dtype}")
    packed_shape = (codes.shape[0], ceil_div(codes.shape[1], 2), codes.shape[2])
    packed = torch.empty(packed_shape, dtype=torch.float4_e2m1fn_x2, device=codes.device)
    packed_u8 = packed.view(torch.uint8)
    low = codes[:, 0::2, :]
    high = torch.zeros_like(low)
    high[:, : codes[:, 1::2, :].shape[1], :] = codes[:, 1::2, :]
    packed_u8.copy_(low | (high << 4))
    return packed


def _create_fp4_operand_tensor(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    *,
    init: str,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if is_mode0_major:
        raise ValueError("Float4E2M1FN blockscaled operands must be K-major")
    tensor = torch.empty(
        (mode0, ceil_div(mode1, 2), l), dtype=torch.float4_e2m1fn_x2, device="cuda"
    )
    tensor.view(torch.uint8).zero_()
    if init == "empty":
        return None, tensor
    if init != "normal":
        raise ValueError(f"Unsupported init: {init}")

    magnitudes = torch.randint(0, 8, (mode0, mode1, l), device="cuda", dtype=torch.uint8)
    signs = torch.randint(0, 2, (mode0, mode1, l), device="cuda", dtype=torch.uint8)
    signs = torch.where(magnitudes == 0, torch.zeros_like(signs), signs << 3)
    codes = magnitudes | signs
    tensor.copy_(_pack_fp4_e2m1fn_codes(codes))
    ref = _fp4_e2m1fn_value_table(tensor.device)[codes.long()]
    return ref, tensor


def create_blockscaled_operand_tensor(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: Type[cutlass.Numeric],
    *,
    init: str = "normal",
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if dtype == cutlass.Float4E2M1FN:
        return _create_fp4_operand_tensor(l, mode0, mode1, is_mode0_major, init=init)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_dtype = torch_dtype_for_cutlass(dtype)
    gen_dtype = torch.bfloat16 if torch_dtype in FLOAT8_DTYPES else torch_dtype
    tensor = torch.empty(shape, dtype=gen_dtype, device="cuda")
    if init == "normal":
        tensor.normal_(std=mode1 ** (-0.5))
    elif init != "empty":
        raise ValueError(f"Unsupported init: {init}")
    # Do NOT .contiguous() after .permute() — that would re-materialize with wrong
    # strides (L innermost) and break K-majorness / N-majorness for l > 1.
    # The original (l, mode0/1, mode1/0) is contiguous, and the permuted view has
    # the correct per-mode strides: stride=1 on the intended contiguous dim.
    tensor = tensor.to(torch_dtype).permute(permute_order)
    ref = tensor.float() if init != "empty" else None
    return ref, tensor


def _pack_blockscaled_scales(ref_blocks: torch.Tensor) -> torch.Tensor:
    mn, sf_k, l = ref_blocks.shape
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    packed = torch.zeros((l, rm, rk, 32, 4, 4), dtype=torch.float32, device=ref_blocks.device)
    packed = packed.permute(3, 4, 1, 5, 2, 0)
    m_idx = torch.arange(mn, device=ref_blocks.device)
    k_idx = torch.arange(sf_k, device=ref_blocks.device)
    l_idx = torch.arange(l, device=ref_blocks.device)
    packed[
        m_idx[:, None, None] % 32,
        (m_idx[:, None, None] // 32) % 4,
        m_idx[:, None, None] // 128,
        k_idx[None, :, None] % 4,
        k_idx[None, :, None] // 4,
        l_idx[None, None, :],
    ] = ref_blocks
    return packed


def create_blockscaled_scale_tensor(
    l: int,
    mn: int,
    k: int,
    sf_vec_size: int,
    dtype: Type[cutlass.Numeric],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sf_k = ceil_div(k, sf_vec_size)
    if dtype == cutlass.Float8E8M0FNU:
        exponents = torch.randint(0, 2, (mn, sf_k, l), device="cuda", dtype=torch.int32)
        ref_blocks = torch.pow(2.0, exponents.float())
    else:
        ref_blocks = torch.randint(1, 4, (mn, sf_k, l), device="cuda", dtype=torch.int32).float()

    packed_f32 = _pack_blockscaled_scales(ref_blocks)
    packed = torch.empty_like(packed_f32, dtype=torch_dtype_for_cutlass(dtype))
    packed.copy_(packed_f32)
    ref = (
        ref_blocks.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )[:, :k, :]
    return ref, packed


def pack_scale_2d_to_blocked_contig(scale_2d: torch.Tensor) -> torch.Tensor:
    """Rearrange a (l, mn, sf_k) or (mn, sf_k) e8m0 scale tensor into the
    contiguous (l, rm, rk, 32, 4, 4) layout shared by the quack kernel and
    cuBLAS's block-scaling. Pads `mn` to a multiple of 128 and `sf_k` to a
    multiple of 4 with zeros."""
    if scale_2d.dim() == 2:
        scale_2d = scale_2d.unsqueeze(0)
    assert scale_2d.dim() == 3, f"expected (l, mn, sf_k), got shape {tuple(scale_2d.shape)}"
    orig_dtype = scale_2d.dtype
    l, mn, sf_k = scale_2d.shape
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    u8 = scale_2d.contiguous().view(torch.uint8)
    if mn_pad != mn or sf_k_pad != sf_k:
        padded = torch.zeros(l, mn_pad, sf_k_pad, device=scale_2d.device, dtype=torch.uint8)
        padded[:, :mn, :sf_k] = u8
    else:
        padded = u8
    # (l, mn_pad, sf_k_pad) -> (l, rm, 128, rk, 4) -> (l, rm, rk, 128, 4)
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    # split 128 into (4 outer, 32 inner), then swap to (32, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    return blocks.view(orig_dtype)


def scale_view_for_kernel(scale_contig: torch.Tensor, mn: int, sf_k: int, l: int) -> torch.Tensor:
    """Wrap a contiguous (l, rm, rk, 32, 4, 4) scale tensor in the
    (32, 4, rm, 4, rk, l) permuted view expected by the quack SM100 kernel.
    Works for both E8M0 (MX) and E4M3 (NVFP4) scale dtypes; the underlying
    byte layout is identical to cuBLAS's `to_blocked` output."""
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    assert scale_contig.shape == (l, rm, rk, 32, 4, 4), (
        f"expected (l, rm, rk, 32, 4, 4) = ({l}, {rm}, {rk}, 32, 4, 4), got {tuple(scale_contig.shape)}"
    )
    assert scale_contig.is_contiguous(), "scale_contig must be contiguous"
    return scale_contig.permute(3, 4, 1, 5, 2, 0)


def scale_blocked_for_cublas(
    scale_contig: torch.Tensor, mn: int, sf_k: int, l_idx: int = 0
) -> torch.Tensor:
    """Flatten a (l, rm, rk, 32, 4, 4) e8m0 scale tensor to the 1D swizzled
    layout that torch._scaled_mm expects. Uses a single l slice."""
    assert scale_contig.is_contiguous() and scale_contig.dim() == 6
    return scale_contig[l_idx].reshape(-1)


_FP4_E2M1_CODE_TO_VALUE = torch.tensor(FP4_E2M1FN_VALUES, dtype=torch.float32)


def _fp4_unpacked_to_value(codes_u8: torch.Tensor) -> torch.Tensor:
    """Convert FP4 E2M1 codes in [0,16) to signed float values via table lookup.
    Code layout: bit 3 = sign, bits 0-2 = magnitude index into {0,.5,1,1.5,2,3,4,6}."""
    table = _FP4_E2M1_CODE_TO_VALUE.to(codes_u8.device)
    return table[codes_u8.long()]


def _blockscaled_format_of(ab_dtype, sf_dtype, sf_vec_size) -> str:
    """Identify which blockscaled format the (ab, sf, vec) tuple corresponds to."""
    if ab_dtype == cutlass.Float8E4M3FN and sf_dtype == cutlass.Float8E8M0FNU and sf_vec_size == 32:
        return "mxfp8"
    if ab_dtype == cutlass.Float4E2M1FN and sf_dtype == cutlass.Float8E8M0FNU and sf_vec_size == 32:
        return "mxfp4"
    if ab_dtype == cutlass.Float4E2M1FN and sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 16:
        return "nvfp4"
    raise ValueError(
        f"init=quant does not support (ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size}). "
        f"Supported: MXFP8 (e4m3+e8m0+32), MXFP4 (e2m1+e8m0+32), NVFP4 (e2m1+e4m3+16)."
    )


def create_blockscaled_operand_quantized(
    l: int,
    mn: int,
    k: int,
    is_mn_major: bool,
    sf_vec_size: int = 32,
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
    *,
    randn_std: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate bf16 randn, quantize to MXFP8/MXFP4/NVFP4 and produce:
    ref:   (mn, k, l) float32 dequantized reference
    q_mkl: (mn, k, l) operand tensor in the layout the quack kernel consumes
           (float8_e4m3fn for fp8 formats; int8 with packed nibbles for fp4)
    scale_contig: (l, rm, rk, 32, 4, 4) contiguous scale storage with byte
           layout matching cuBLAS `to_blocked`. Use `scale_view_for_kernel` to
           get the quack-kernel view, or `scale_blocked_for_cublas` for cuBLAS.
    """
    if is_mn_major:
        raise NotImplementedError("is_mn_major=True not yet supported for init=quant")
    fmt = _blockscaled_format_of(ab_dtype, sf_dtype, sf_vec_size)
    assert k % sf_vec_size == 0, f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size})"
    sf_k = k // sf_vec_size
    std = randn_std if randn_std is not None else k**-0.5

    x_hp = (torch.randn(l, mn, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    x_flat = x_hp.view(l * mn, k)

    if fmt == "mxfp8":
        q_flat, scale_2d = to_mx_compiled(x_flat, sf_vec_size)  # (l*mn, k), (l*mn, sf_k)
        # Operand: (mn, k, l) K-major VIEW of contiguous (l, mn, k).
        # Do NOT call .contiguous() here — that would materialize as (mn, k, l) row-major,
        # making L the innermost stride=1 dim and BREAKING K-majorness for l > 1.
        q_mkl = q_flat.view(l, mn, k).contiguous().permute(1, 2, 0)  # strides (k, 1, mn*k)
        q_vals = q_flat.float().view(l, mn, k)
        scale_vals = scale_2d.float().view(l, mn, sf_k).repeat_interleave(sf_vec_size, dim=-1)
        ref_mkl = (q_vals * scale_vals).permute(1, 2, 0).contiguous()
        scale_2d = scale_2d.view(l, mn, sf_k)
    elif fmt in ("mxfp4", "nvfp4"):
        if fmt == "mxfp4":
            q_packed, scale_2d = to_mxfp4_compiled(x_flat, sf_vec_size)  # (l*mn, k/2), (l*mn, sf_k)
        else:
            q_packed, scale_2d, _pts = to_nvfp4_compiled(x_flat, sf_vec_size, None)
        # q_packed is uint8, two 4-bit codes per byte (low nibble=even K, high=odd K).
        # Decode for ref: code -> {0,.5,1,1.5,2,3,4,6,-0,-.5,...} via lookup.
        codes_lo = (q_packed & 0x0F).view(l, mn, k // 2)
        codes_hi = ((q_packed >> 4) & 0x0F).view(l, mn, k // 2)
        vals_lo = _fp4_unpacked_to_value(codes_lo)  # (l, mn, k/2)
        vals_hi = _fp4_unpacked_to_value(codes_hi)
        q_values = torch.stack([vals_lo, vals_hi], dim=-1).reshape(l, mn, k)  # interleave back
        scale_vals = scale_2d.float().view(l, mn, sf_k).repeat_interleave(sf_vec_size, dim=-1)
        ref_mkl = (q_values * scale_vals).permute(1, 2, 0).contiguous()
        # Kernel operand: (mn, k/2, l) K-major view (no post-contiguous!)
        q_mkl = (
            q_packed.view(l, mn, k // 2).contiguous().permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        )
        scale_2d = scale_2d.view(l, mn, sf_k)

    scale_contig = pack_scale_2d_to_blocked_contig(scale_2d)
    return ref_mkl, q_mkl, scale_contig


def create_blockscaled_varlen_m_operands(
    num_experts: int,
    m_per: int,
    n: int,
    k: int,
    sf_vec_size: int,
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
    *,
    randn_std: Optional[float] = None,
):
    """Generate bf16 randn + quantize for a varlen_m blockscaled GEMM.

    Returns (a_ref, b_ref, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_m) where:
      a_ref: (total_m, k) fp32 dequantized
      b_ref: (num_experts, n, k) fp32 dequantized
      qa:   (total_m, k) 2D K-major quantized operand (fp8) or (total_m, k/2) (fp4)
      qb:   (n, k, num_experts) 3D K-major quantized operand (fp8) or (n, k/2, num_experts) (fp4)
      scales: (l, rm, rk, 32, 4, 4) contiguous in the shared byte layout
      cu_seqlens_m: (num_experts+1,) int32
    """
    assert m_per % 128 == 0, "varlen_m requires m_per % 128 == 0 for scale alignment"
    assert k % sf_vec_size == 0
    total_m = num_experts * m_per
    std = randn_std if randn_std is not None else k**-0.5
    sf_k = k // sf_vec_size

    if ab_dtype == cutlass.Float8E4M3FN and sf_dtype == cutlass.Float8E8M0FNU and sf_vec_size == 32:
        from quack.mx_utils import to_mx_compiled

        to_fn = to_mx_compiled
    else:
        raise NotImplementedError(
            f"varlen_m currently only supports MXFP8 (got ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size}). "
            "FP4 support pending."
        )

    # Quantize A: (total_m, k) bf16 -> (total_m, k) fp8 K-major
    a_hp = (torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    qa, sa_2d = to_fn(a_hp, sf_vec_size)  # (total_m, k), (total_m, sf_k)
    a_sc_contig = pack_scale_2d_to_blocked_contig(sa_2d.view(1, total_m, sf_k))
    a_ref = qa.float() * sa_2d.float().repeat_interleave(sf_vec_size, dim=-1)

    # Quantize B: (num_experts, n, k) bf16 -> (n, k, num_experts) K-major
    # Create as (l, n, k) contiguous then permute (no final .contiguous!) to get K-major.
    b_hp = (torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    qb_flat, sb_2d = to_fn(b_hp.view(num_experts * n, k), sf_vec_size)
    qb = qb_flat.view(num_experts, n, k).contiguous().permute(1, 2, 0)  # strides (k, 1, n*k)
    sb_2d = sb_2d.view(num_experts, n, sf_k)
    b_sc_contig = pack_scale_2d_to_blocked_contig(sb_2d)
    b_ref = qb_flat.float().view(num_experts, n, k) * sb_2d.float().repeat_interleave(
        sf_vec_size, dim=-1
    )

    cu_seqlens_m = torch.arange(0, num_experts + 1, dtype=torch.int32, device="cuda") * m_per
    return a_ref, b_ref, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_m


def compile_blockscaled_gemm_tvm_ffi(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    d_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    mA: torch.Tensor,
    mB: torch.Tensor,
    mD: torch.Tensor,
    mSFA: torch.Tensor,
    mSFB: torch.Tensor,
    *,
    use_clc_persistence: bool = True,
    varlen_m: bool = False,
    varlen_k: bool = False,
) -> Callable:
    """Compile the SM100 blockscaled GEMM.

    When varlen_m: mA is (total_m, k) K-major, mD is (total_m, n) N-major,
    mB is (n, k, l); run(...) takes an extra cu_seqlens_m tensor.
    When varlen_k: mA is (m, total_k), mB is (n, total_k), mD is (m, n, l);
    run(...) takes an extra cu_seqlens_k tensor.
    """
    device_capacity = get_device_capacity(mA.device)
    if device_capacity[0] not in (10, 11):
        raise RuntimeError("Blockscaled SM100 GEMM requires SM100/SM110")
    assert not (varlen_m and varlen_k), "Only one of varlen_m / varlen_k"

    gemm = partial(
        GemmDefaultSm100,
        sf_vec_size=sf_vec_size,
        use_clc_persistence=use_clc_persistence,
    )(cutlass.Float32, ab_dtype, mma_tiler_mn, (*cluster_shape_mn, 1))
    compile_epi_args = gemm.EpilogueArguments()
    scheduler_args = make_scheduler_args(
        get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
        max_swizzle_size=8,
        tile_count_semaphore=None,
        batch_idx_permute=None,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    from quack.gemm_tvm_ffi_utils import make_fake_varlen_args

    varlen_args_fake = make_fake_varlen_args(varlen_m, varlen_k, False, None) or VarlenArguments()

    # Fake operand tensors with sym_ints (varlen-aware shapes).
    if varlen_m:
        total_m_sym = cute.sym_int()
        n_sym, k_sym, l_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
        fake_mA = fake_tensor(
            ab_dtype, (total_m_sym, k_sym), leading_dim=1, divisibility=div_for_dtype(ab_dtype)
        )
        fake_mB = fake_tensor(
            ab_dtype, (n_sym, k_sym, l_sym), leading_dim=1, divisibility=div_for_dtype(ab_dtype)
        )
        fake_mD = fake_tensor(
            d_dtype, (total_m_sym, n_sym), leading_dim=1, divisibility=div_for_dtype(d_dtype)
        )
    elif varlen_k:
        total_k_sym = cute.sym_int()
        m_sym, n_sym, l_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
        fake_mA = fake_tensor(
            ab_dtype, (m_sym, total_k_sym), leading_dim=1, divisibility=div_for_dtype(ab_dtype)
        )
        fake_mB = fake_tensor(
            ab_dtype, (n_sym, total_k_sym), leading_dim=1, divisibility=div_for_dtype(ab_dtype)
        )
        fake_mD = fake_tensor(
            d_dtype, (m_sym, n_sym, l_sym), leading_dim=1, divisibility=div_for_dtype(d_dtype)
        )
    else:
        fake_mA = _make_fake_compact_tensor(mA.shape, ab_dtype, leading_dim=1)
        fake_mB = _make_fake_compact_tensor(mB.shape, ab_dtype, leading_dim=1)
        fake_mD = _make_fake_compact_tensor(mD.shape, d_dtype, leading_dim=1)

    @cute.jit
    def runner(
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        varlen_args,
        stream,
    ):
        gemm(a, b, d, None, compile_epi_args, scheduler_args, varlen_args, stream, sfa, sfb, None)

    compiled = cute.compile(
        runner,
        fake_mA,
        fake_mB,
        fake_mD,
        _make_compile_tensor_like(mSFA, sf_dtype, dynamic_layout=True),
        _make_compile_tensor_like(mSFB, sf_dtype, dynamic_layout=True),
        varlen_args_fake,
        stream,
        options="--enable-tvm-ffi",
    )

    if varlen_m or varlen_k:

        def run(a, b, d, sfa, sfb, cu_seqlens):
            varlen_args = VarlenArguments(
                mCuSeqlensM=cu_seqlens if varlen_m else None,
                mCuSeqlensK=cu_seqlens if varlen_k else None,
            )
            compiled(a, b, d, sfa, sfb, varlen_args)
    else:

        def run(a, b, d, sfa, sfb):
            compiled(a, b, d, sfa, sfb, VarlenArguments())

    return run


def blockscaled_gemm_reference(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum(
        "mkl,nkl->mnl",
        torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref),
        torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref),
    )
