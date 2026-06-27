# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# gpu-wiki archive note:
# TUNED FOR RTX PRO 5000 / SM120 small-M NVFP4 decode dispatch. This wrapper
# routes allowlisted M<=16 C2-like shapes to the archived CTA-3D TMA CUDA
# kernel and leaves all other shapes on FlashInfer b12x. It is a production
# integration reference for task38, not a standalone kernel.
#
# Task38 served-E2E harness.
# Route measured-fast small-M NVFP4 GEMM shapes to the custom CTA-3D TMA
# kernel and keep every other shape on FlashInfer b12x. The default allowlist
# started from the 2026-05-13 race-fixed same-input M=1 operator sweep:
#   CTA-3D TMA faster than b12x:
#     N,K = 34816,5120; 5120,17408; 14336,5120; 16384,5120;
#           5120,6144; 96,5120
#   b12x faster:
#     N,K = 152064,5120

import ctypes
import logging
import os
from pathlib import Path

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


logger = logging.getLogger(__name__)

_FLASHINFER_MM_BACKEND = os.environ.get("VLLM_FLASHINFER_NVFP4_MM_BACKEND", "b12x")
_SPLITK_S = int(os.environ.get("VLLM_NVFP4_SPLITK_S", "8"))
_SPLITK_TILE_N = int(os.environ.get("VLLM_NVFP4_SPLITK_TILE_N", "8"))
_CTA3D_MAX_M = int(os.environ.get("VLLM_NVFP4_CTA3D_MAX_M", "16"))
_DEFAULT_SPLITK_SO = str(
    Path(__file__).with_name(
        "gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.so"
    )
)
_DEFAULT_ALLOWED_SHAPES = (
    "34816:5120,5120:17408,14336:5120,16384:5120,5120:6144,96:5120"
)

_flashinfer_mm_backend_logged = False
_sk_lib = None
_sk_fn = None
_sk_loaded = False
_sk_workspaces: dict[tuple[torch.device, int, int], torch.Tensor] = {}
_sk_shape_hits: dict[tuple[int, int, int], int] = {}


def _parse_allowed_shapes() -> frozenset[tuple[int, int]]:
    raw = os.environ.get("VLLM_NVFP4_CTA3D_ALLOWED_SHAPES", _DEFAULT_ALLOWED_SHAPES)
    shapes: set[tuple[int, int]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            n_raw, k_raw = item.split(":", 1)
            shapes.add((int(n_raw), int(k_raw)))
        except ValueError:
            logger.warning("Ignoring invalid CTA-3D shape entry: %s", item)
    return frozenset(shapes)


def _load_splitk():
    global _sk_lib, _sk_fn, _sk_loaded
    if _sk_loaded:
        return _sk_fn
    _sk_loaded = True
    so_path = os.environ.get("VLLM_NVFP4_SPLITK_SO", _DEFAULT_SPLITK_SO)
    if not os.path.exists(so_path):
        logger.info("NVFP4 CTA-3D TMA kernel .so not found at %s", so_path)
        return None
    try:
        _sk_lib = ctypes.CDLL(so_path)
        _sk_fn = _sk_lib.kernel_v3_splitk
        _sk_fn.restype = None
        _sk_fn.argtypes = (
            [ctypes.c_int] * 5
            + [ctypes.c_void_p] * 6
            + [ctypes.c_void_p]
            + [ctypes.c_ulonglong]
        )
    except OSError:
        logger.warning(
            "NVFP4 CTA-3D TMA kernel .so failed to load from %s",
            so_path,
            exc_info=True,
        )
        return None
    logger.info("NVFP4 CTA-3D TMA kernel loaded from %s", so_path)
    return _sk_fn


def _shape_allows_cta3d(m: int, n: int, k: int) -> bool:
    if os.environ.get("VLLM_DISABLE_SPLITK", "") == "1":
        return False
    if os.environ.get("VLLM_DISABLE_CTA3D_TMA", "") == "1":
        return False
    if m < 1 or m > min(_CTA3D_MAX_M, 16):
        return False
    if (n, k) not in _parse_allowed_shapes():
        return False
    if _SPLITK_TILE_N != 8 or _SPLITK_S != 8:
        return False
    if n % _SPLITK_TILE_N != 0:
        return False
    if (k // _SPLITK_S) % 128 != 0:
        return False
    return True


def _layer_may_use_cta3d(n: int, k: int) -> bool:
    if os.environ.get("VLLM_DISABLE_SPLITK", "") == "1":
        return False
    if os.environ.get("VLLM_DISABLE_CTA3D_TMA", "") == "1":
        return False
    if (n, k) not in _parse_allowed_shapes():
        return False
    if _SPLITK_TILE_N != 8 or _SPLITK_S != 8:
        return False
    if n % _SPLITK_TILE_N != 0:
        return False
    if (k // _SPLITK_S) % 128 != 0:
        return False
    return True


def _use_cta3d(m: int, n: int, k: int) -> bool:
    if not _shape_allows_cta3d(m, n, k):
        return False
    if _load_splitk() is None:
        return False
    return True


def _get_splitk_workspace(m: int, n: int, device: torch.device) -> torch.Tensor:
    key = (device, m, n)
    if key not in _sk_workspaces:
        _sk_workspaces[key] = torch.zeros(
            m, _SPLITK_S, n, dtype=torch.float32, device=device
        )
        logger.info(
            "Allocated CTA-3D workspace for M=%d N=%d (S=%d, %.1f KB)",
            m,
            n,
            _SPLITK_S,
            m * _SPLITK_S * n * 4 / 1024,
        )
    return _sk_workspaces[key]


@torch.library.custom_op(
    "vllm::nvfp4_flashinfer_b12x_cta3d_m1_dispatch",
    mutates_args=[],
    device_types="cuda",
)
def _nvfp4_flashinfer_b12x_cta3d_m1_dispatch(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x_fp4.shape[0]
    if _use_cta3d(m, n, k):
        shape = (m, n, k)
        _sk_shape_hits[shape] = _sk_shape_hits.get(shape, 0) + 1
        if _sk_shape_hits[shape] <= 3:
            logger.info(
                "DISPATCH CTA3D TMA HIT: M=%d N=%d K=%d tile_n=%d S=%d "
                "(shape_count=%d)",
                m,
                n,
                k,
                _SPLITK_TILE_N,
                _SPLITK_S,
                _sk_shape_hits[shape],
            )
        out = torch.empty(m, n, dtype=out_dtype, device=x_fp4.device)
        workspace = _get_splitk_workspace(m, n, x_fp4.device)
        _sk_fn(
            m,
            n,
            k,
            _SPLITK_TILE_N,
            _SPLITK_S,
            x_fp4.data_ptr(),
            weight.data_ptr(),
            x_blockscale.view(torch.uint8).data_ptr(),
            weight_scale.view(torch.uint8).reshape(-1).data_ptr(),
            alpha.data_ptr(),
            out.data_ptr(),
            workspace.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    return flashinfer_scaled_fp4_mm(
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        out_dtype,
        backend=_FLASHINFER_MM_BACKEND,
    )


@torch.library.register_fake("vllm::nvfp4_flashinfer_b12x_cta3d_m1_dispatch")
def _nvfp4_flashinfer_b12x_cta3d_m1_dispatch_fake(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(x_fp4.shape[0], n, dtype=out_dtype, device=x_fp4.device)


class FlashInferCutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer b12x, with CTA-3D TMA for measured small-M shapes."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            cutlass_fp4_supported,
        )

        if (
            cutlass_fp4_supported()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _flashinfer_mm_backend_logged
        if not _flashinfer_mm_backend_logged:
            logger.info(
                "DISPATCH FlashInfer NVFP4 mm backend=%s with CTA-3D small-M "
                "max_m=%d allowlist=%s tile_n=%d S=%d",
                _FLASHINFER_MM_BACKEND,
                min(_CTA3D_MAX_M, 16),
                sorted(_parse_allowed_shapes()),
                _SPLITK_TILE_N,
                _SPLITK_S,
            )
            _flashinfer_mm_backend_logged = True
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
        )

        padding_cols = getattr(layer, "weights_padding_cols", 0)
        n = layer.weight.shape[0]
        k = x_fp4.shape[1] * 2

        if padding_cols == 0 and _layer_may_use_cta3d(n, k):
            out = torch.ops.vllm.nvfp4_flashinfer_b12x_cta3d_m1_dispatch(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                n,
                k,
                output_dtype,
            )
        elif padding_cols == 0:
            out = flashinfer_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
                backend=_FLASHINFER_MM_BACKEND,
            )
        else:
            x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, padding_cols)
            out = flashinfer_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
                backend=_FLASHINFER_MM_BACKEND,
            )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferTrtllmNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        epilogue_tile_m = 128

        layer.weight = torch.nn.Parameter(
            shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale.shape)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-trtllm",
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="trtllm",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCudnnNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's cuDNN wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cudnn",
        )

        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, getattr(layer, "weights_padding_cols", 0)
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="cudnn",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
