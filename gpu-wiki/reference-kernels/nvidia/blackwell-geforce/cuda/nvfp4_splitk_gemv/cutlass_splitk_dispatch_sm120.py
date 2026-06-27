# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# TUNED FOR NVIDIA RTX PRO 5000 Blackwell-GeForce / sm_120.
#
# This file archives the vLLM CUTLASS NVFP4 dispatch pattern used with
# gemm_v3_splitk_sm120.cu. It is not a generic vLLM replacement:
# - C2-like M=1, small-N, aligned-K shapes call the CUDA Split-K kernel.
# - C1 / large-N and prefill shapes stay on stock CUTLASS.
# - Workspace is cached by N to keep allocation out of the hot path.
# - SF_B is expected to already be in CUTLASS-swizzled layout.
#
# Related docs:
# - docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-split-k-gemv-bf16-optimization.md
# - docs/pitfalls/nvidia/cuda/nvfp4-split-k-gemv-pitfalls.md
#
# Split-K dispatch for the current vLLM CUTLASS NVFP4 backend.
# C1 / large-N shapes remain on CUTLASS; C2 decode shapes route to split-K.

import ctypes
import logging
import os

import torch

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    scaled_fp4_quant,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


logger = logging.getLogger(__name__)

_sk_lib = None
_sk_fn = None
_sk_loaded = False
_sk_workspaces: dict[int, torch.Tensor] = {}

_SPLITK_S = 4
_SPLITK_N_THRESH = 8192
_TILE_N = 32


def _load_splitk():
    global _sk_lib, _sk_fn, _sk_loaded
    if _sk_loaded:
        return _sk_fn
    _sk_loaded = True
    so_path = os.environ.get("VLLM_NVFP4_SPLITK_SO")
    if not so_path:
        logger.info("Set VLLM_NVFP4_SPLITK_SO to enable the archived split-K kernel.")
        return None
    if not os.path.exists(so_path):
        logger.info("NVFP4 split-K kernel .so not found at %s", so_path)
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
            "NVFP4 split-K kernel .so failed to load from %s",
            so_path,
            exc_info=True,
        )
        return None
    logger.info("NVFP4 split-K kernel loaded from %s", so_path)
    return _sk_fn


_load_splitk()


def _get_splitk_workspace(N: int, device: torch.device) -> torch.Tensor:
    if N not in _sk_workspaces:
        _sk_workspaces[N] = torch.zeros(_SPLITK_S, N, dtype=torch.float32, device=device)
        logger.info(
            "Allocated split-K workspace for N=%d (S=%d, %.1f KB)",
            N,
            _SPLITK_S,
            _SPLITK_S * N * 4 / 1024,
        )
    return _sk_workspaces[N]


def _use_splitk(N: int, K: int) -> bool:
    if os.environ.get("VLLM_DISABLE_SPLITK", "") == "1":
        return False
    if _sk_fn is None:
        return False
    if N > _SPLITK_N_THRESH:
        return False
    if (K // _SPLITK_S) % 128 != 0:
        return False
    return True


@torch.library.custom_op(
    "vllm::nvfp4_cutlass_splitk_dispatch",
    mutates_args=[],
    device_types="cuda",
)
def _nvfp4_cutlass_splitk_dispatch(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = x_fp4.shape[0]
    if M == 1 and _use_splitk(N, K):
        _nvfp4_cutlass_splitk_dispatch._sk_hit = getattr(
            _nvfp4_cutlass_splitk_dispatch, "_sk_hit", 0
        ) + 1
        if _nvfp4_cutlass_splitk_dispatch._sk_hit <= 5:
            logger.info(
                "DISPATCH split-K HIT: M=%d N=%d K=%d S=%d (count=%d)",
                M,
                N,
                K,
                _SPLITK_S,
                _nvfp4_cutlass_splitk_dispatch._sk_hit,
            )
        out = torch.empty(M, N, dtype=out_dtype, device=x_fp4.device)
        workspace = _get_splitk_workspace(N, x_fp4.device)
        workspace.zero_()
        _sk_fn(
            M,
            N,
            K,
            _TILE_N,
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
    return cutlass_scaled_fp4_mm(
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        out_dtype,
    )


@torch.library.register_fake("vllm::nvfp4_cutlass_splitk_dispatch")
def _nvfp4_cutlass_splitk_dispatch_fake(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(x_fp4.shape[0], N, dtype=out_dtype, device=x_fp4.device)


class CutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via CUTLASS, with C2 split-K dispatch for decode."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if cutlass_fp4_supported():
            return True, None
        return False, "CUTLASS FP4 kernels not available"

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
            backend="cutlass",
        )

        padding_cols = getattr(layer, "weights_padding_cols", 0)
        N = layer.weight.shape[0]
        K = x_fp4.shape[1] * 2

        if padding_cols == 0:
            out = torch.ops.vllm.nvfp4_cutlass_splitk_dispatch(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                N,
                K,
                output_dtype,
            )
        else:
            x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, padding_cols)
            out = cutlass_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
            )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
