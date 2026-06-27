"""Task39-local runner for the SM120 b12x CuTe fork.

gpu-wiki archive note:
    Diagnostic runner for dense_blockscaled_gemm_sm120_task39_diagnostic.py.
    It mirrors FlashInfer's internal b12x FP4 runner contract so task39 could
    compare shallow SF-layout variants on identical vLLM-layout tensors.

The API mirrors the internal FlashInfer b12x FP4 runner input contract so that
the benchmark can compare the fork and upstream b12x on identical vLLM-layout
NVFP4 tensors.
"""

from __future__ import annotations

from typing import List

import torch

from flashinfer.gemm import gemm_base

from .dense_blockscaled_gemm_sm120_task39_diagnostic import (
    Task39Sm120GateUpDenseGemmKernel,
    Task39Sm120GateUpAlphaOneEpilogueDenseGemmKernel,
    Task39Sm120GateUpAlphaOneFixedSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfPad1DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfPad4DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfPad16DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyFirstDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU64HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU128HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32Bits64HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32Bits128HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfaV4HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfabV4HoistSfaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm02461357DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04152637DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04261537DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEvenDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven048c26aeDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven082a4c6eDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaUnfilteredDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfUnfilteredDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPackedSfTmaDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaBankMajorSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneHoistSfaBankMajorSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFragMajorSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32WaitLateDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32PreloadKDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32PreloadKOverlapDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPreloadSfDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfPretryDenseGemmKernel,
    Task39Sm120GateUpAlphaOneFixedSfCopyU32DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfSwizzle32DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfSwizzle64DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfSwizzle128DenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfTvSwapDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfaTvSwapDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfbTvSwapDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfbTvSwapAtomDenseGemmKernel,
    Task39Sm120GateUpAlphaOneSfbTvSwapGroupDenseGemmKernel,
    Task39Sm120GateUpDirectGridDenseGemmKernel,
    Task39Sm120GateUpFixedSfTileDenseGemmKernel,
    Task39Sm120GateUpOccupancy2DenseGemmKernel,
    Task39Sm120GateUpRasterNDenseGemmKernel,
    Task39Sm120GateUpUnfilteredSfDenseGemmKernel,
)


_TASK39_B12X_FORK_KERNEL_CACHE: dict[tuple, tuple] = {}
_TASK39_B12X_FORK_FIXED_KERNEL_CACHE: dict[tuple, tuple] = {}


def pack_task39_scale_to_packed_blocks(
    scale: torch.Tensor,
    rows: int,
    real_k: int,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Return a flat view of vLLM/FlashInfer scales already in packed-block order."""

    scale_cols = real_k // sf_vec_size
    if rows % 128 != 0:
        raise ValueError(f"packed SF layout requires rows divisible by 128, got {rows}")
    if scale_cols % 4 != 0:
        raise ValueError(
            f"packed SF layout requires scale columns divisible by 4, got {scale_cols}"
        )
    if scale.numel() != rows * scale_cols:
        raise ValueError(
            f"scale tensor has {scale.numel()} elements, expected {rows * scale_cols}"
        )

    # `scaled_fp4_quant(..., is_sf_swizzled_layout=True)` and
    # `swizzle_blockscale()` already write the physical layout documented as
    # [MN/128, SF_K/4, 32, 4, 4].  The kernel only needs a stable data pointer.
    return scale.contiguous().reshape(-1)


def _compile_block_scaled_gemm_fixed_shape(
    cache,
    cache_key,
    make_gemm_kernel,
    *,
    m: int,
    n: int,
    k_packed: int,
    sf_m: int,
    sf_n: int,
    sf_k: int,
    batch_size: int,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align: int,
    cluster_shape_mn: tuple[int, int],
    swap_ab: bool,
):
    """Compile a task39 fixed-shape CuTe GEMM.

    This intentionally mirrors FlashInfer's `_compile_block_scaled_gemm`, but
    uses concrete fake tensor extents for M/N/K instead of symbolic extents.
    """

    if cache_key in cache:
        return cache[cache_key]

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    gemm = make_gemm_kernel()

    a_fake = cute.runtime.make_fake_compact_tensor(
        ab_cutlass_dtype,
        (m, k_packed),
        stride_order=(1, 0),
        assumed_align=ab_assumed_align,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        ab_cutlass_dtype,
        (n, k_packed),
        stride_order=(1, 0),
        assumed_align=ab_assumed_align,
    )
    if swap_ab:
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (n, m),
            stride_order=(0, 1),
            assumed_align=16,
        )
    else:
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (m, n),
            stride_order=(1, 0),
            assumed_align=16,
        )

    a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_gemm = cute.compile(
        gemm.wrapper,
        a_fake,
        b_fake,
        c_fake,
        sf_m,
        sf_n,
        sf_k,
        batch_size,
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        max_active_clusters,
        stream_fake,
        swap_ab,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    result = (compiled_gemm, max_active_clusters)
    cache[cache_key] = result
    return result


def get_task39_b12x_fork_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    use_nvfp4: bool,
    variant: str = "noop",
):
    """Create a b12x-compatible runner backed by the task39 local CuTe fork."""

    if (sm_major, sm_minor) not in ((12, 0), (12, 1)):
        raise ValueError(f"Task39 b12x fork requires SM120/121, got sm_{sm_major}{sm_minor}")
    if not use_nvfp4:
        raise ValueError("Task39 b12x fork only supports NVFP4.")

    import cutlass

    cutlass_dtype_attr = gemm_base._TORCH_TO_CUTLASS_DTYPE_ATTR.get(out_dtype)
    c_cutlass_dtype = (
        getattr(cutlass, cutlass_dtype_attr) if cutlass_dtype_attr is not None else None
    )
    if c_cutlass_dtype is None:
        raise ValueError(f"Unsupported output dtype for task39 b12x fork: {out_dtype}")

    kernel_cls_by_variant = {
        "noop": Task39Sm120GateUpDenseGemmKernel,
        "alpha1_epilogue": Task39Sm120GateUpAlphaOneEpilogueDenseGemmKernel,
        "alpha1_fixed_sf_tile": Task39Sm120GateUpAlphaOneFixedSfDenseGemmKernel,
        "alpha1_sfpad1": Task39Sm120GateUpAlphaOneSfPad1DenseGemmKernel,
        "alpha1_sfpad4": Task39Sm120GateUpAlphaOneSfPad4DenseGemmKernel,
        "alpha1_sfpad16": Task39Sm120GateUpAlphaOneSfPad16DenseGemmKernel,
        "alpha1_sfcopy_first": Task39Sm120GateUpAlphaOneSfCopyFirstDenseGemmKernel,
        "alpha1_sfcopy_u32": Task39Sm120GateUpAlphaOneSfCopyU32DenseGemmKernel,
        "alpha1_sfcopy_u32_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u64_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU64HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u128_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU128HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_bits64_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32Bits64HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_bits128_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32Bits128HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfa_v4_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfaV4HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfab_v4_hoistsfa": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfabV4HoistSfaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_02461357": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm02461357DenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_04152637": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04152637DenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_04261537": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04261537DenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_even": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEvenDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_even_048c26ae": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven048c26aeDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_manualsfb_v4_fixed_even_082a4c6e": (
            Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven082a4c6eDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_unfiltered": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaUnfilteredDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_fixedsf": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_fixedsf_unfiltered": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfUnfilteredDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_packed_sf_tma": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPackedSfTmaDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_bankmajor_sf": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaBankMajorSfDenseGemmKernel
        ),
        "alpha1_hoistsfa_bankmajor_sf": (
            Task39Sm120GateUpAlphaOneHoistSfaBankMajorSfDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_fragmajor_sf": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFragMajorSfDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_waitlate": (
            Task39Sm120GateUpAlphaOneSfCopyU32WaitLateDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_preloadk": (
            Task39Sm120GateUpAlphaOneSfCopyU32PreloadKDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_preloadk_overlap": (
            Task39Sm120GateUpAlphaOneSfCopyU32PreloadKOverlapDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_preloadsf": (
            Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_hoistsfa_preloadsf": (
            Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPreloadSfDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_preloadsf_pretry": (
            Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfPretryDenseGemmKernel
        ),
        "alpha1_sfcopy_u32_shape_specialized": (
            Task39Sm120GateUpAlphaOneSfCopyU32DenseGemmKernel
        ),
        "alpha1_fixed_sfcopy_u32": Task39Sm120GateUpAlphaOneFixedSfCopyU32DenseGemmKernel,
        "alpha1_sfswizzle32": Task39Sm120GateUpAlphaOneSfSwizzle32DenseGemmKernel,
        "alpha1_sfswizzle64": Task39Sm120GateUpAlphaOneSfSwizzle64DenseGemmKernel,
        "alpha1_sfswizzle128": Task39Sm120GateUpAlphaOneSfSwizzle128DenseGemmKernel,
        "alpha1_sf_tvswap": Task39Sm120GateUpAlphaOneSfTvSwapDenseGemmKernel,
        "alpha1_sfa_tvswap": Task39Sm120GateUpAlphaOneSfaTvSwapDenseGemmKernel,
        "alpha1_sfb_tvswap": Task39Sm120GateUpAlphaOneSfbTvSwapDenseGemmKernel,
        "alpha1_sfb_tvswap_atom": Task39Sm120GateUpAlphaOneSfbTvSwapAtomDenseGemmKernel,
        "alpha1_sfb_tvswap_group": Task39Sm120GateUpAlphaOneSfbTvSwapGroupDenseGemmKernel,
        "shape_specialized": Task39Sm120GateUpDenseGemmKernel,
        "alpha1_shape_specialized": Task39Sm120GateUpAlphaOneEpilogueDenseGemmKernel,
        "direct_grid": Task39Sm120GateUpDirectGridDenseGemmKernel,
        "occupancy2": Task39Sm120GateUpOccupancy2DenseGemmKernel,
        "raster_n": Task39Sm120GateUpRasterNDenseGemmKernel,
        "sf_unfiltered": Task39Sm120GateUpUnfilteredSfDenseGemmKernel,
        "fixed_sf_tile": Task39Sm120GateUpFixedSfTileDenseGemmKernel,
    }
    if variant not in kernel_cls_by_variant:
        raise ValueError(f"Unknown task39 b12x fork variant {variant!r}")
    kernel_cls = kernel_cls_by_variant[variant]

    class Task39B12xForkRunner:
        """Minimal runner with the same input list layout as FlashInfer b12x."""

        def get_valid_tactics(self, inputs: List[torch.Tensor]) -> list:
            a, b, _, _, _, _, _, _, _, _ = inputs
            m = a.shape[0]
            n = b.shape[1]
            k_packed = a.shape[1]
            real_k = k_packed * 2

            tactics = []
            for mma_tiler_mn in ((64, 64), (64, 128), (128, 64), (128, 128)):
                if not kernel_cls.can_implement(
                    cutlass.Float4E2M1FN,
                    cutlass.Float8E4M3FN,
                    16,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    (1, 1),
                    m,
                    n,
                    real_k,
                    1,
                    "k",
                    "k",
                    "n",
                ):
                    continue
                for use_prefetch in (False, True):
                    tactics.append((mma_tiler_mn, (1, 1), False, use_prefetch, "task39_sm120", None))
            return tactics

        def forward(self, inputs: List[torch.Tensor], tactic=None, **_) -> torch.Tensor:
            a, b, a_descale, b_descale, alpha_tensor, _, out, _, _, _ = inputs
            m = a.shape[0]
            n = b.shape[1]
            k_packed = a.shape[1]
            real_k = k_packed * 2
            sf_vec_size = 16
            batch_size = 1
            sf_dtype = cutlass.Float8E4M3FN

            if tactic is None or tactic == -1:
                sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
                tactic = (
                    gemm_base._select_default_sm120_mma_tiler(m, n, sm_count),
                    (1, 1),
                    False,
                    False,
                    "task39_sm120",
                    None,
                )

            mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, _, use_tma_store = tactic
            if use_tma_store is not None or swap_ab:
                raise ValueError("Task39 b12x fork does not support swap_ab or explicit TMA-store tactics.")

            kernel_a, kernel_b = a, b.T
            kernel_a_sf = a_descale
            kernel_b_sf = b_descale if b_descale.dim() == 1 else b_descale.T

            sf_m = (m + 127) // 128
            sf_n = (n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                "task39_b12x_fork_v1",
                variant,
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                use_prefetch,
                enable_pdl,
                out_dtype,
            )

            def make_kernel():
                return kernel_cls(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_prefetch,
                    enable_pdl,
                )

            if variant in (
                "shape_specialized",
                "alpha1_shape_specialized",
                "alpha1_sfcopy_u32_shape_specialized",
            ):
                fixed_cache_key = cache_key + (m, n, k_packed, sf_m, sf_n, sf_k)
                compiled_gemm, _ = _compile_block_scaled_gemm_fixed_shape(
                    _TASK39_B12X_FORK_FIXED_KERNEL_CACHE,
                    fixed_cache_key,
                    make_kernel,
                    m=m,
                    n=n,
                    k_packed=k_packed,
                    sf_m=sf_m,
                    sf_n=sf_n,
                    sf_k=sf_k,
                    batch_size=batch_size,
                    ab_cutlass_dtype=cutlass.Uint8,
                    sf_dtype=sf_dtype,
                    c_cutlass_dtype=c_cutlass_dtype,
                    ab_assumed_align=32,
                    cluster_shape_mn=cluster_shape_mn,
                    swap_ab=False,
                )
            else:
                compiled_gemm, _ = gemm_base._compile_block_scaled_gemm(
                    _TASK39_B12X_FORK_KERNEL_CACHE,
                    cache_key,
                    make_kernel,
                    ab_cutlass_dtype=cutlass.Uint8,
                    sf_dtype=sf_dtype,
                    c_cutlass_dtype=c_cutlass_dtype,
                    ab_assumed_align=32,
                    cluster_shape_mn=cluster_shape_mn,
                    swap_ab=False,
                    sf_m=sf_m,
                    sf_n=sf_n,
                    sf_k=sf_k,
                    batch_size=batch_size,
                )

            alpha_for_launch = gemm_base._prepare_alpha_for_launch(alpha_tensor, a.device)
            compiled_gemm(
                kernel_a,
                kernel_b,
                out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return Task39B12xForkRunner()
