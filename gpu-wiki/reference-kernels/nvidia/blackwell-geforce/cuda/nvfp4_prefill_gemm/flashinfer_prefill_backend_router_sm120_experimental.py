# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# gpu-wiki archive note:
# Experimental omoExplore task39 vLLM router for SM120 NVFP4 prefill GEMM.
# It is archived to document shape guards, backend selection, and custom .so
# integration boundaries. The related knowledge conclusion is baseline-first
# and fusion-boundary-first; do not promote this router without fresh served
# TTFT and nsys evidence.

import logging
import os
import ctypes
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
_FLASHINFER_MM_BACKEND = os.environ.get("VLLM_FLASHINFER_NVFP4_MM_BACKEND", "cutlass")
_FLASHINFER_PREFILL_MM_BACKEND = os.environ.get(
    "VLLM_FLASHINFER_NVFP4_PREFILL_MM_BACKEND", ""
)
_FLASHINFER_PREFILL_M_MIN = int(os.environ.get("VLLM_NVFP4_PREFILL_M_MIN", "128"))
_FLASHINFER_PREFILL_LOG_SHAPES = (
    os.environ.get("VLLM_NVFP4_PREFILL_LOG_SHAPES", "0") == "1"
)
_FLASHINFER_PREFILL_LOG_LIMIT = int(
    os.environ.get("VLLM_NVFP4_PREFILL_LOG_LIMIT", "32")
)
_TASK39_PREFILL_CUSTOM_ENABLED = (
    os.environ.get("VLLM_NVFP4_PREFILL_CUSTOM", "0") == "1"
)
_TASK39_PREFILL_PREPACK = os.environ.get("VLLM_TASK39_PREFILL_PREPACK", "0") == "1"
_TASK39_PREFILL_LAZY_REPACK = (
    os.environ.get("VLLM_TASK39_PREFILL_LAZY_REPACK", "0") == "1"
)
_TASK39_PREFILL_DIRECT_LAYOUT = (
    os.environ.get("VLLM_TASK39_PREFILL_DIRECT_LAYOUT", "0") == "1"
)
_TASK39_PREFILL_DIRECT_A = (
    os.environ.get("VLLM_TASK39_PREFILL_DIRECT_A", "0") == "1"
)
_TASK39_PREFILL_FUSED_QUANT_A = (
    os.environ.get("VLLM_TASK39_PREFILL_FUSED_QUANT_A", "0") == "1"
)
_TASK39_PREFILL_FUSED_A_REPACK = (
    os.environ.get("VLLM_TASK39_PREFILL_FUSED_A_REPACK", "0") == "1"
)
_TASK39_PREFILL_COMPILED = (
    os.environ.get("VLLM_TASK39_PREFILL_COMPILED", "0") == "1"
)
_TASK39_PREFILL_PREPACK_MAX_LAYERS = int(
    os.environ.get("VLLM_TASK39_PREFILL_PREPACK_MAX_LAYERS", "-1")
)
_TASK39_PREFILL_PREPACK_MAX_MB = float(
    os.environ.get("VLLM_TASK39_PREFILL_PREPACK_MAX_MB", "-1")
)
_TASK39_PREFILL_PREPACK_MAX_BYTES = (
    int(_TASK39_PREFILL_PREPACK_MAX_MB * 1024 * 1024)
    if _TASK39_PREFILL_PREPACK_MAX_MB >= 0
    else -1
)
_TASK39_PREFILL_PREPACK_MIN_FREE_MB = float(
    os.environ.get("VLLM_TASK39_PREFILL_PREPACK_MIN_FREE_MB", "128")
)
_TASK39_PREFILL_PREPACK_MIN_FREE_BYTES = int(
    _TASK39_PREFILL_PREPACK_MIN_FREE_MB * 1024 * 1024
)
_TASK39_PREFILL_CUSTOM_SO = os.environ.get(
    "VLLM_NVFP4_PREFILL_SO",
    str(Path(__file__).with_name("prefill_mma_padded_sm120_experimental.so")),
)
_flashinfer_mm_backend_logged = False
_flashinfer_shape_log_counts: dict[tuple[int, int, int, str], int] = {}

_task39_prefill_lib = None
_task39_prefill_gemm_fn = None
_task39_prefill_gemm_direct_fn = None
_task39_prefill_gemm_direct_a_fn = None
_task39_prefill_gemm_fused_a_fn = None
_task39_prefill_gemm_quant_bf16_fn = None
_task39_prefill_repack_fn = None
_task39_prefill_supported_fn = None
_task39_prefill_loaded = False
_task39_prefill_b_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_task39_prefill_workspaces: dict[tuple[torch.device, str], torch.Tensor] = {}
_task39_prefill_shape_hits: dict[tuple[int, int, int], int] = {}
_task39_prefill_prepack_count = 0
_task39_prefill_prepack_bytes = 0
_task39_prefill_missing_repack_warned: set[int] = set()
_TASK39_PREFILL_B_REP_ATTR = "_task39_prefill_b_rep"
_TASK39_PREFILL_B_SF_REP_ATTR = "_task39_prefill_b_sf_rep"


def _parse_prefill_shape_allowlist(
    env_name: str = "VLLM_NVFP4_PREFILL_SHAPES",
) -> tuple[
    set[tuple[int, int, int]], set[tuple[int, int]]
]:
    """Parse MxNxK or NxK shape guards from an environment variable."""
    raw = os.environ.get(env_name, "")
    mnk_shapes: set[tuple[int, int, int]] = set()
    nk_shapes: set[tuple[int, int]] = set()
    for item in raw.replace(";", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        dims = [int(part) for part in item.replace("x", ":").split(":") if part]
        if len(dims) == 2:
            nk_shapes.add((dims[0], dims[1]))
        elif len(dims) == 3:
            mnk_shapes.add((dims[0], dims[1], dims[2]))
        else:
            raise ValueError(
                f"{env_name} entries must be NxK or MxNxK; got {item!r}"
            )
    return mnk_shapes, nk_shapes


_FLASHINFER_PREFILL_MNK_SHAPES, _FLASHINFER_PREFILL_NK_SHAPES = (
    _parse_prefill_shape_allowlist()
)
_TASK39_PREFILL_MNK_SHAPES, _TASK39_PREFILL_NK_SHAPES = (
    _parse_prefill_shape_allowlist("VLLM_TASK39_PREFILL_SHAPES")
)
_TASK39_PREFILL_DEFAULT_NK_SHAPES: set[tuple[int, int]] = {
    # Conservative default: only the down projection is enabled unless an
    # explicit VLLM_TASK39_PREFILL_SHAPES allowlist is supplied.
    (5120, 17408),
}


def _task39_prefill_shape_allowed(m: int, n: int, k: int) -> bool:
    if _TASK39_PREFILL_MNK_SHAPES or _TASK39_PREFILL_NK_SHAPES:
        return (m, n, k) in _TASK39_PREFILL_MNK_SHAPES or (
            n,
            k,
        ) in _TASK39_PREFILL_NK_SHAPES
    return (n, k) in _TASK39_PREFILL_DEFAULT_NK_SHAPES


def _task39_prefill_layer_allowed(n: int, k: int) -> bool:
    if _TASK39_PREFILL_MNK_SHAPES or _TASK39_PREFILL_NK_SHAPES:
        return (n, k) in _TASK39_PREFILL_NK_SHAPES or any(
            shape_n == n and shape_k == k
            for _, shape_n, shape_k in _TASK39_PREFILL_MNK_SHAPES
        )
    return (n, k) in _TASK39_PREFILL_DEFAULT_NK_SHAPES


def _load_task39_prefill_custom():
    global _task39_prefill_lib
    global _task39_prefill_gemm_fn
    global _task39_prefill_gemm_direct_fn
    global _task39_prefill_gemm_direct_a_fn
    global _task39_prefill_gemm_fused_a_fn
    global _task39_prefill_gemm_quant_bf16_fn
    global _task39_prefill_repack_fn
    global _task39_prefill_supported_fn
    global _task39_prefill_loaded

    if _task39_prefill_loaded:
        return _task39_prefill_gemm_fn
    _task39_prefill_loaded = True

    if not _TASK39_PREFILL_CUSTOM_ENABLED:
        return None
    if os.environ.get("VLLM_DISABLE_TASK39_PREFILL_CUSTOM", "0") == "1":
        logger.info("Task39 prefill custom route disabled by env")
        return None
    if not os.path.exists(_TASK39_PREFILL_CUSTOM_SO):
        logger.info("Task39 prefill custom .so not found at %s", _TASK39_PREFILL_CUSTOM_SO)
        return None

    try:
        _task39_prefill_lib = ctypes.CDLL(_TASK39_PREFILL_CUSTOM_SO)
        _task39_prefill_gemm_fn = _task39_prefill_lib.prefill_mma_padded_gemm
        _task39_prefill_gemm_fn.restype = None
        _task39_prefill_gemm_fn.argtypes = (
            [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8 + [ctypes.c_ulonglong]
        )
        _task39_prefill_gemm_direct_fn = getattr(
            _task39_prefill_lib, "prefill_mma_padded_gemm_direct_layout", None
        )
        if _task39_prefill_gemm_direct_fn is not None:
            _task39_prefill_gemm_direct_fn.restype = None
            _task39_prefill_gemm_direct_fn.argtypes = (
                [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8 + [ctypes.c_ulonglong]
            )
        _task39_prefill_gemm_fused_a_fn = getattr(
            _task39_prefill_lib,
            "prefill_mma_padded_gemm_direct_layout_fused_a_repack",
            None,
        )
        if _task39_prefill_gemm_fused_a_fn is not None:
            _task39_prefill_gemm_fused_a_fn.restype = None
            _task39_prefill_gemm_fused_a_fn.argtypes = (
                [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8 + [ctypes.c_ulonglong]
            )
        _task39_prefill_gemm_direct_a_fn = getattr(
            _task39_prefill_lib, "prefill_mma_padded_gemm_direct_a_layout", None
        )
        if _task39_prefill_gemm_direct_a_fn is not None:
            _task39_prefill_gemm_direct_a_fn.restype = None
            _task39_prefill_gemm_direct_a_fn.argtypes = (
                [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8 + [ctypes.c_ulonglong]
            )
        _task39_prefill_gemm_quant_bf16_fn = getattr(
            _task39_prefill_lib, "prefill_mma_padded_quant_bf16_gemm", None
        )
        if _task39_prefill_gemm_quant_bf16_fn is not None:
            _task39_prefill_gemm_quant_bf16_fn.restype = None
            _task39_prefill_gemm_quant_bf16_fn.argtypes = (
                [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8 + [ctypes.c_ulonglong]
            )

        _task39_prefill_repack_fn = (
            _task39_prefill_lib.prefill_mma_padded_repack_weight
        )
        _task39_prefill_repack_fn.restype = None
        _task39_prefill_repack_fn.argtypes = (
            [ctypes.c_int] * 2 + [ctypes.c_void_p] * 4 + [ctypes.c_ulonglong]
        )

        _task39_prefill_supported_fn = (
            _task39_prefill_lib.prefill_mma_padded_is_supported
        )
        _task39_prefill_supported_fn.restype = ctypes.c_int
        _task39_prefill_supported_fn.argtypes = [ctypes.c_int] * 3
    except (AttributeError, OSError):
        logger.warning(
            "Task39 prefill custom .so failed to load from %s",
            _TASK39_PREFILL_CUSTOM_SO,
            exc_info=True,
        )
        _task39_prefill_gemm_fn = None
        _task39_prefill_gemm_direct_fn = None
        _task39_prefill_gemm_direct_a_fn = None
        _task39_prefill_gemm_fused_a_fn = None
        _task39_prefill_gemm_quant_bf16_fn = None
        _task39_prefill_repack_fn = None
        _task39_prefill_supported_fn = None
        return None

    logger.info("Task39 prefill custom kernel loaded from %s", _TASK39_PREFILL_CUSTOM_SO)
    return _task39_prefill_gemm_fn


def _task39_prefill_round_m(m: int) -> int:
    return (m + 63) & ~63


def _task39_prefill_compiled_available() -> bool:
    return hasattr(
        torch.ops._C, "task39_prefill_gemm_direct_layout_fused_a_repack"
    )


def _task39_prefill_compiled_supported(m: int, n: int, k: int) -> bool:
    # Keep in sync with the narrow accepted C++/CUDA support predicate.
    return m > 256 and m <= 320 and n == 5120 and k == 17408


def _get_task39_prefill_workspace(
    device: torch.device,
    name: str,
    nbytes: int,
) -> torch.Tensor:
    key = (device, name)
    if (
        key not in _task39_prefill_workspaces
        or _task39_prefill_workspaces[key].nbytes < nbytes
    ):
        _task39_prefill_workspaces[key] = torch.empty(
            nbytes, dtype=torch.uint8, device=device
        )
    return _task39_prefill_workspaces[key]


def _task39_prefill_supported(m: int, n: int, k: int) -> bool:
    if not _task39_prefill_shape_allowed(m, n, k):
        return False
    if _TASK39_PREFILL_COMPILED:
        return (
            _task39_prefill_compiled_available()
            and _task39_prefill_compiled_supported(m, n, k)
        )
    if _load_task39_prefill_custom() is None:
        return False
    if _task39_prefill_supported_fn is None:
        return False
    return bool(_task39_prefill_supported_fn(m, n, k))


def _task39_prefill_supported_for_python_branch(m: int, n: int, k: int) -> bool:
    # In vLLM compile mode, branching on the symbolic M dimension can leave
    # size nodes in the FX graph and break the vLLM backend. Keep the old
    # custom_op-internal fallback behavior while Dynamo is tracing.
    if torch.compiler.is_compiling():
        return _layer_may_use_task39_prefill(n, k)
    return _task39_prefill_supported(m, n, k)


def _layer_may_use_task39_prefill(n: int, k: int) -> bool:
    if not _TASK39_PREFILL_CUSTOM_ENABLED:
        return False
    if os.environ.get("VLLM_DISABLE_TASK39_PREFILL_CUSTOM", "0") == "1":
        return False
    return _task39_prefill_layer_allowed(n, k)


def _layer_ready_for_task39_prefill(layer: torch.nn.Module, n: int, k: int) -> bool:
    if not _layer_may_use_task39_prefill(n, k):
        return False
    if _TASK39_PREFILL_DIRECT_LAYOUT or _TASK39_PREFILL_LAZY_REPACK:
        return True
    if _get_layer_task39_prefill_repacked_weight(layer) is not None:
        return True
    return layer.weight.data_ptr() in _task39_prefill_b_cache


def _set_layer_task39_prefill_repacked_weight(
    layer: torch.nn.Module,
    b_rep: torch.Tensor,
    b_sf_rep: torch.Tensor,
) -> None:
    for name, tensor in (
        (_TASK39_PREFILL_B_REP_ATTR, b_rep),
        (_TASK39_PREFILL_B_SF_REP_ATTR, b_sf_rep),
    ):
        if name in layer._buffers:
            setattr(layer, name, tensor)
            continue
        if hasattr(layer, name):
            delattr(layer, name)
        layer.register_buffer(name, tensor, persistent=False)


def _get_layer_task39_prefill_repacked_weight(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    b_rep = getattr(layer, _TASK39_PREFILL_B_REP_ATTR, None)
    b_sf_rep = getattr(layer, _TASK39_PREFILL_B_SF_REP_ATTR, None)
    if isinstance(b_rep, torch.Tensor) and isinstance(b_sf_rep, torch.Tensor):
        return b_rep, b_sf_rep
    return None


def _task39_prefill_repack_nbytes(n: int, k: int) -> int:
    kh = k // 2
    ksf = k // 16
    n_pad = (n + 127) & ~127
    return n_pad * kh + n_pad * ksf


def _task39_cuda_mem_get_info(device: torch.device) -> tuple[int, int]:
    with torch.cuda.device(device):
        return torch.cuda.mem_get_info()


def _prepare_task39_prefill_weight(layer: torch.nn.Module) -> None:
    global _task39_prefill_prepack_count
    global _task39_prefill_prepack_bytes

    if not _TASK39_PREFILL_CUSTOM_ENABLED:
        return
    if not _TASK39_PREFILL_PREPACK:
        return
    if _TASK39_PREFILL_DIRECT_LAYOUT:
        return
    if _load_task39_prefill_custom() is None:
        return

    n = layer.weight.shape[0]
    k = layer.weight.shape[1] * 2
    if not _layer_may_use_task39_prefill(n, k):
        return

    bptr = layer.weight.data_ptr()
    if bptr in _task39_prefill_b_cache:
        return

    kh = k // 2
    ksf = k // 16
    n_pad = (n + 127) & ~127
    required_bytes = _task39_prefill_repack_nbytes(n, k)
    if (
        _TASK39_PREFILL_PREPACK_MAX_LAYERS >= 0
        and _task39_prefill_prepack_count >= _TASK39_PREFILL_PREPACK_MAX_LAYERS
    ):
        logger.info(
            "Skip task39 prefill weight repack: max_layers reached "
            "N=%d K=%d count=%d max_layers=%d",
            n,
            k,
            _task39_prefill_prepack_count,
            _TASK39_PREFILL_PREPACK_MAX_LAYERS,
        )
        return
    if (
        _TASK39_PREFILL_PREPACK_MAX_BYTES >= 0
        and _task39_prefill_prepack_bytes + required_bytes
        > _TASK39_PREFILL_PREPACK_MAX_BYTES
    ):
        logger.info(
            "Skip task39 prefill weight repack: max_mb reached "
            "N=%d K=%d prepared=%.2f MiB required=%.2f MiB max=%.2f MiB",
            n,
            k,
            _task39_prefill_prepack_bytes / (1024 * 1024),
            required_bytes / (1024 * 1024),
            _TASK39_PREFILL_PREPACK_MAX_BYTES / (1024 * 1024),
        )
        return

    free_before = -1
    try:
        free_before, _ = _task39_cuda_mem_get_info(layer.weight.device)
    except RuntimeError:
        logger.debug("Unable to query CUDA free memory before task39 prepack", exc_info=True)
    if (
        free_before >= 0
        and free_before < required_bytes + _TASK39_PREFILL_PREPACK_MIN_FREE_BYTES
    ):
        logger.info(
            "Skip task39 prefill weight repack: free-memory guard "
            "N=%d K=%d free=%.2f MiB required=%.2f MiB reserve=%.2f MiB",
            n,
            k,
            free_before / (1024 * 1024),
            required_bytes / (1024 * 1024),
            _TASK39_PREFILL_PREPACK_MIN_FREE_BYTES / (1024 * 1024),
        )
        return

    stream = torch.cuda.current_stream().cuda_stream
    b_rep = None
    b_sf_rep = None
    try:
        b_rep = torch.empty(n_pad * kh, dtype=torch.uint8, device=layer.weight.device)
        b_sf_rep = torch.empty(
            n_pad * ksf, dtype=torch.uint8, device=layer.weight_scale.device
        )
        _task39_prefill_repack_fn(
            n,
            k,
            layer.weight.data_ptr(),
            b_rep.data_ptr(),
            layer.weight_scale.view(torch.uint8).data_ptr(),
            b_sf_rep.data_ptr(),
            stream,
        )
    except torch.OutOfMemoryError:
        del b_rep
        del b_sf_rep
        torch.cuda.empty_cache()
        logger.warning(
            "Skip task39 prefill weight repack after CUDA OOM: "
            "N=%d K=%d required=%.2f MiB prepared_count=%d prepared=%.2f MiB",
            n,
            k,
            required_bytes / (1024 * 1024),
            _task39_prefill_prepack_count,
            _task39_prefill_prepack_bytes / (1024 * 1024),
            exc_info=True,
        )
        return

    _task39_prefill_b_cache[bptr] = (b_rep, b_sf_rep)
    _set_layer_task39_prefill_repacked_weight(layer, b_rep, b_sf_rep)
    _task39_prefill_prepack_count += 1
    _task39_prefill_prepack_bytes += required_bytes
    logger.info(
        "Prepared task39 prefill weight repack at load time: "
        "N=%d K=%d count=%d bytes=%.2f MiB free_before=%.2f MiB",
        n,
        k,
        _task39_prefill_prepack_count,
        _task39_prefill_prepack_bytes / (1024 * 1024),
        free_before / (1024 * 1024) if free_before >= 0 else -1.0,
    )


def _run_task39_prefill_gemm(
    m: int,
    n: int,
    k: int,
    x_fp4: torch.Tensor,
    b_arg: torch.Tensor,
    x_blockscale: torch.Tensor,
    b_sf_arg: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    gemm_fn,
    needs_a_workspace: bool = True,
) -> torch.Tensor:
    stream = torch.cuda.current_stream().cuda_stream
    kh = k // 2
    ksf = k // 16
    m_pad = _task39_prefill_round_m(m)

    if needs_a_workspace:
        a_rep_ptr = _get_task39_prefill_workspace(
            x_fp4.device, "a_rep", m_pad * kh
        ).data_ptr()
        a_sf_rep_ptr = _get_task39_prefill_workspace(
            x_fp4.device, "a_sf_rep", m_pad * ksf
        ).data_ptr()
    else:
        a_rep_ptr = None
        a_sf_rep_ptr = None
    out = torch.empty(m, n, dtype=out_dtype, device=x_fp4.device)

    gemm_fn(
        m,
        n,
        k,
        x_fp4.data_ptr(),
        b_arg.data_ptr(),
        x_blockscale.view(torch.uint8).data_ptr(),
        b_sf_arg.data_ptr(),
        alpha.data_ptr(),
        out.data_ptr(),
        a_rep_ptr,
        a_sf_rep_ptr,
        stream,
    )
    return out


def _run_task39_prefill_compiled_fused_arepack_gemm(
    m: int,
    n: int,
    k: int,
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    kh = k // 2
    ksf = k // 16
    m_pad = _task39_prefill_round_m(m)
    a_rep = _get_task39_prefill_workspace(x_fp4.device, "a_rep", m_pad * kh)
    a_sf_rep = _get_task39_prefill_workspace(
        x_fp4.device, "a_sf_rep", m_pad * ksf
    )
    out = torch.empty(m, n, dtype=out_dtype, device=x_fp4.device)
    torch.ops._C.task39_prefill_gemm_direct_layout_fused_a_repack(
        out,
        x_fp4,
        weight,
        x_blockscale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        alpha,
        a_rep,
        a_sf_rep,
    )
    return out


def _run_task39_prefill_quant_bf16_gemm(
    m: int,
    n: int,
    k: int,
    x: torch.Tensor,
    b_rep: torch.Tensor,
    b_sf_rep: torch.Tensor,
    input_global_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    stream = torch.cuda.current_stream().cuda_stream
    kh = k // 2
    ksf = k // 16
    m_pad = _task39_prefill_round_m(m)

    a_rep = _get_task39_prefill_workspace(x.device, "a_rep", m_pad * kh)
    a_sf_rep = _get_task39_prefill_workspace(
        x.device, "a_sf_rep", m_pad * ksf
    )
    out = torch.empty(m, n, dtype=out_dtype, device=x.device)

    _task39_prefill_gemm_quant_bf16_fn(
        m,
        n,
        k,
        x.data_ptr(),
        b_rep.data_ptr(),
        input_global_scale.data_ptr(),
        b_sf_rep.data_ptr(),
        alpha.data_ptr(),
        out.data_ptr(),
        a_rep.data_ptr(),
        a_sf_rep.data_ptr(),
        stream,
    )
    return out


@torch.library.custom_op(
    "vllm::nvfp4_task39_prefill_dispatch",
    mutates_args=[],
    device_types="cuda",
)
def _nvfp4_task39_prefill_dispatch(
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

    if out_dtype is torch.bfloat16 and _task39_prefill_supported(m, n, k):
        shape = (m, n, k)
        _task39_prefill_shape_hits[shape] = (
            _task39_prefill_shape_hits.get(shape, 0) + 1
        )
        if _task39_prefill_shape_hits[shape] <= 5:
            logger.info(
                "DISPATCH task39 prefill custom HIT: M=%d N=%d K=%d "
                "prepack=%s lazy_repack=%s direct_layout=%s "
                "fused_a_repack=%s compiled=%s count=%d",
                m,
                n,
                k,
                _TASK39_PREFILL_PREPACK,
                _TASK39_PREFILL_LAZY_REPACK,
                _TASK39_PREFILL_DIRECT_LAYOUT,
                _TASK39_PREFILL_FUSED_A_REPACK,
                _TASK39_PREFILL_COMPILED,
                _task39_prefill_shape_hits[shape],
            )

        stream = torch.cuda.current_stream().cuda_stream
        kh = k // 2
        ksf = k // 16
        m_pad = _task39_prefill_round_m(m)
        n_pad = (n + 127) & ~127

        if _TASK39_PREFILL_DIRECT_LAYOUT:
            if _TASK39_PREFILL_COMPILED and _TASK39_PREFILL_FUSED_A_REPACK:
                return _run_task39_prefill_compiled_fused_arepack_gemm(
                    m,
                    n,
                    k,
                    x_fp4,
                    weight,
                    x_blockscale,
                    weight_scale,
                    alpha,
                    out_dtype,
                )
            gemm_fn = (
                _task39_prefill_gemm_fused_a_fn
                if _TASK39_PREFILL_FUSED_A_REPACK
                else _task39_prefill_gemm_direct_fn
            )
            if gemm_fn is None:
                logger.warning(
                    "Task39 direct-layout GEMM entry point missing "
                    "(fused_a_repack=%s); falling back to FlashInfer for "
                    "M=%d N=%d K=%d",
                    _TASK39_PREFILL_FUSED_A_REPACK,
                    m,
                    n,
                    k,
                )
                backend = _select_flashinfer_backend(m, n, k)
                return flashinfer_scaled_fp4_mm(
                    x_fp4,
                    weight,
                    x_blockscale,
                    weight_scale,
                    alpha,
                    out_dtype,
                    backend=backend,
                )
            b_arg = weight
            b_sf_arg = weight_scale.view(torch.uint8)
            needs_a_workspace = True
        else:
            bptr = weight.data_ptr()
            if bptr not in _task39_prefill_b_cache:
                if not _TASK39_PREFILL_LAZY_REPACK:
                    if bptr not in _task39_prefill_missing_repack_warned:
                        _task39_prefill_missing_repack_warned.add(bptr)
                        logger.warning(
                            "Task39 prefill weight repack missing at runtime; "
                            "falling back to FlashInfer for M=%d N=%d K=%d",
                            m,
                            n,
                            k,
                        )
                    backend = _select_flashinfer_backend(m, n, k)
                    return flashinfer_scaled_fp4_mm(
                        x_fp4,
                        weight,
                        x_blockscale,
                        weight_scale,
                        alpha,
                        out_dtype,
                        backend=backend,
                    )
                b_rep = _get_task39_prefill_workspace(
                    weight.device, "b_rep_lazy", n_pad * kh
                )
                b_sf_rep = _get_task39_prefill_workspace(
                    weight.device, "b_sf_rep_lazy", n_pad * ksf
                )
                _task39_prefill_repack_fn(
                    n,
                    k,
                    weight.data_ptr(),
                    b_rep.data_ptr(),
                    weight_scale.view(torch.uint8).data_ptr(),
                    b_sf_rep.data_ptr(),
                    stream,
                )
                if _task39_prefill_shape_hits[shape] <= 5:
                    logger.info("Lazy task39 prefill weight repack: N=%d K=%d", n, k)
            else:
                b_rep, b_sf_rep = _task39_prefill_b_cache[bptr]
            b_arg = b_rep
            b_sf_arg = b_sf_rep
            gemm_fn = _task39_prefill_gemm_fn
            needs_a_workspace = True

        return _run_task39_prefill_gemm(
            m,
            n,
            k,
            x_fp4,
            b_arg,
            x_blockscale,
            b_sf_arg,
            alpha,
            out_dtype,
            gemm_fn,
            needs_a_workspace=needs_a_workspace,
        )

    backend = _select_flashinfer_backend(m, n, k)
    return flashinfer_scaled_fp4_mm(
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        out_dtype,
        backend=backend,
    )


@torch.library.register_fake("vllm::nvfp4_task39_prefill_dispatch")
def _nvfp4_task39_prefill_dispatch_fake(
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


@torch.library.custom_op(
    "vllm::nvfp4_task39_prefill_prepacked_dispatch",
    mutates_args=[],
    device_types="cuda",
)
def _nvfp4_task39_prefill_prepacked_dispatch(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    b_rep: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    b_sf_rep: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x_fp4.shape[0]

    if out_dtype is torch.bfloat16 and _task39_prefill_supported(m, n, k):
        shape = (m, n, k)
        _task39_prefill_shape_hits[shape] = (
            _task39_prefill_shape_hits.get(shape, 0) + 1
        )
        if _task39_prefill_shape_hits[shape] <= 5:
            logger.info(
                "DISPATCH task39 prefill prepacked custom HIT: "
                "M=%d N=%d K=%d direct_a=%s count=%d",
                m,
                n,
                k,
                _TASK39_PREFILL_DIRECT_A,
                _task39_prefill_shape_hits[shape],
            )

        gemm_fn = _task39_prefill_gemm_fn
        if _TASK39_PREFILL_DIRECT_A:
            if _task39_prefill_gemm_direct_a_fn is None:
                logger.warning(
                    "Task39 direct-A GEMM entry point missing; falling back to "
                    "the standard repacked-A custom path for M=%d N=%d K=%d",
                    m,
                    n,
                    k,
                )
            else:
                gemm_fn = _task39_prefill_gemm_direct_a_fn

        return _run_task39_prefill_gemm(
            m,
            n,
            k,
            x_fp4,
            b_rep,
            x_blockscale,
            b_sf_rep,
            alpha,
            out_dtype,
            gemm_fn,
        )

    backend = _select_flashinfer_backend(m, n, k)
    return flashinfer_scaled_fp4_mm(
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        out_dtype,
        backend=backend,
    )


@torch.library.register_fake("vllm::nvfp4_task39_prefill_prepacked_dispatch")
def _nvfp4_task39_prefill_prepacked_dispatch_fake(
    x_fp4: torch.Tensor,
    weight: torch.Tensor,
    b_rep: torch.Tensor,
    x_blockscale: torch.Tensor,
    weight_scale: torch.Tensor,
    b_sf_rep: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(x_fp4.shape[0], n, dtype=out_dtype, device=x_fp4.device)


@torch.library.custom_op(
    "vllm::nvfp4_task39_prefill_fused_quant_a_prepacked_dispatch",
    mutates_args=[],
    device_types="cuda",
)
def _nvfp4_task39_prefill_fused_quant_a_prepacked_dispatch(
    x: torch.Tensor,
    weight: torch.Tensor,
    b_rep: torch.Tensor,
    weight_scale: torch.Tensor,
    b_sf_rep: torch.Tensor,
    input_global_scale: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x.shape[0]

    if (
        out_dtype is torch.bfloat16
        and x.dtype is torch.bfloat16
        and _task39_prefill_supported(m, n, k)
        and _task39_prefill_gemm_quant_bf16_fn is not None
    ):
        shape = (m, n, k)
        _task39_prefill_shape_hits[shape] = (
            _task39_prefill_shape_hits.get(shape, 0) + 1
        )
        if _task39_prefill_shape_hits[shape] <= 5:
            logger.info(
                "DISPATCH task39 prefill fused-quant-A custom HIT: "
                "M=%d N=%d K=%d count=%d",
                m,
                n,
                k,
                _task39_prefill_shape_hits[shape],
            )

        return _run_task39_prefill_quant_bf16_gemm(
            m,
            n,
            k,
            x,
            b_rep,
            b_sf_rep,
            input_global_scale,
            alpha,
            out_dtype,
        )

    x_fp4, x_blockscale = scaled_fp4_quant(
        x,
        input_global_scale,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )
    backend = _select_flashinfer_backend(m, n, k)
    return flashinfer_scaled_fp4_mm(
        x_fp4,
        weight,
        x_blockscale,
        weight_scale,
        alpha,
        out_dtype,
        backend=backend,
    )


@torch.library.register_fake(
    "vllm::nvfp4_task39_prefill_fused_quant_a_prepacked_dispatch"
)
def _nvfp4_task39_prefill_fused_quant_a_prepacked_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    b_rep: torch.Tensor,
    weight_scale: torch.Tensor,
    b_sf_rep: torch.Tensor,
    input_global_scale: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(x.shape[0], n, dtype=out_dtype, device=x.device)


def _select_flashinfer_backend(m: int, n: int, k: int) -> str:
    if not _FLASHINFER_PREFILL_MM_BACKEND:
        return _FLASHINFER_MM_BACKEND
    if torch.compiler.is_compiling():
        # Avoid branching on symbolic M in the compiled vLLM graph. In compile
        # mode the allowlist is interpreted as an N/K route.
        if _FLASHINFER_PREFILL_MNK_SHAPES:
            for _shape_m, shape_n, shape_k in _FLASHINFER_PREFILL_MNK_SHAPES:
                if n == shape_n and k == shape_k:
                    return _FLASHINFER_PREFILL_MM_BACKEND
        if _FLASHINFER_PREFILL_NK_SHAPES:
            if (n, k) in _FLASHINFER_PREFILL_NK_SHAPES:
                return _FLASHINFER_PREFILL_MM_BACKEND
            return _FLASHINFER_MM_BACKEND
        return _FLASHINFER_PREFILL_MM_BACKEND
    if m < _FLASHINFER_PREFILL_M_MIN:
        return _FLASHINFER_MM_BACKEND
    if _FLASHINFER_PREFILL_MNK_SHAPES or _FLASHINFER_PREFILL_NK_SHAPES:
        if (m, n, k) not in _FLASHINFER_PREFILL_MNK_SHAPES and (
            n,
            k,
        ) not in _FLASHINFER_PREFILL_NK_SHAPES:
            return _FLASHINFER_MM_BACKEND
    return _FLASHINFER_PREFILL_MM_BACKEND


def _maybe_log_flashinfer_shape(m: int, n: int, k: int, backend: str) -> None:
    if not _FLASHINFER_PREFILL_LOG_SHAPES:
        return
    key = (m, n, k, backend)
    count = _flashinfer_shape_log_counts.get(key, 0) + 1
    _flashinfer_shape_log_counts[key] = count
    if count <= _FLASHINFER_PREFILL_LOG_LIMIT:
        phase = "prefill" if m >= _FLASHINFER_PREFILL_M_MIN else "decode"
        logger.info(
            "DISPATCH FlashInfer NVFP4 GEMM shape: phase=%s M=%d N=%d K=%d "
            "backend=%s count=%d",
            phase,
            m,
            n,
            k,
            backend,
            count,
        )


class FlashInferCutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's CUTLASS wrapper."""

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
                "DISPATCH FlashInfer NVFP4 mm backend=%s prefill_backend=%s "
                "prefill_m_min=%d prefill_shapes_mnk=%s prefill_shapes_nk=%s "
                "task39_prefill_custom=%s task39_prefill_so=%s "
                "task39_prefill_prepack=%s task39_prefill_lazy_repack=%s "
                "task39_prefill_direct_layout=%s task39_prefill_direct_a=%s "
                "task39_prefill_fused_quant_a=%s "
                "task39_prefill_fused_a_repack=%s "
                "task39_prefill_compiled=%s "
                "task39_prefill_prepack_max_layers=%d "
                "task39_prefill_prepack_max_mb=%.2f "
                "task39_prefill_prepack_min_free_mb=%.2f",
                _FLASHINFER_MM_BACKEND,
                _FLASHINFER_PREFILL_MM_BACKEND or "<default>",
                _FLASHINFER_PREFILL_M_MIN,
                sorted(_FLASHINFER_PREFILL_MNK_SHAPES),
                sorted(_FLASHINFER_PREFILL_NK_SHAPES),
                _TASK39_PREFILL_CUSTOM_ENABLED,
                _TASK39_PREFILL_CUSTOM_SO,
                _TASK39_PREFILL_PREPACK,
                _TASK39_PREFILL_LAZY_REPACK,
                _TASK39_PREFILL_DIRECT_LAYOUT,
                _TASK39_PREFILL_DIRECT_A,
                _TASK39_PREFILL_FUSED_QUANT_A,
                _TASK39_PREFILL_FUSED_A_REPACK,
                _TASK39_PREFILL_COMPILED,
                _TASK39_PREFILL_PREPACK_MAX_LAYERS,
                _TASK39_PREFILL_PREPACK_MAX_MB,
                _TASK39_PREFILL_PREPACK_MIN_FREE_MB,
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
        _prepare_task39_prefill_weight(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weight_padding_cols = getattr(layer, "weights_padding_cols", 0)
        n = layer.weight.shape[0]
        k = layer.weight.shape[1] * 2
        m = x.numel() // x.shape[-1]

        prepacked_weight = _get_layer_task39_prefill_repacked_weight(layer)
        if (
            _TASK39_PREFILL_FUSED_QUANT_A
            and output_dtype is torch.bfloat16
            and x.dtype is torch.bfloat16
            and weight_padding_cols == 0
            and prepacked_weight is not None
            and _layer_may_use_task39_prefill(n, k)
            and _task39_prefill_supported_for_python_branch(m, n, k)
            and _task39_prefill_gemm_quant_bf16_fn is not None
        ):
            b_rep, b_sf_rep = prepacked_weight
            x_2d = x.reshape(m, x.shape[-1])
            out = torch.ops.vllm.nvfp4_task39_prefill_fused_quant_a_prepacked_dispatch(
                x_2d,
                layer.weight,
                b_rep,
                layer.weight_scale,
                b_sf_rep,
                layer.input_global_scale_inv,
                layer.alpha,
                n,
                k,
                output_dtype,
            )
            out = slice_nvfp4_output(out, output_size)

            if bias is not None:
                out = out + bias
            return out.view(*output_shape)

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
        )

        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, weight_padding_cols
        )
        m = x_fp4.shape[0]
        k = x_fp4.shape[1] * 2
        backend = _select_flashinfer_backend(m, n, k)
        _maybe_log_flashinfer_shape(m, n, k, backend)
        task39_prefill_shape_supported = _task39_prefill_supported_for_python_branch(
            m, n, k
        )

        if (
            weight_padding_cols == 0
            and prepacked_weight is not None
            and task39_prefill_shape_supported
        ):
            b_rep, b_sf_rep = prepacked_weight
            out = torch.ops.vllm.nvfp4_task39_prefill_prepacked_dispatch(
                x_fp4,
                layer.weight,
                b_rep,
                x_blockscale,
                layer.weight_scale,
                b_sf_rep,
                layer.alpha,
                n,
                k,
                output_dtype,
            )
        elif (
            weight_padding_cols == 0
            and _layer_ready_for_task39_prefill(layer, n, k)
            and task39_prefill_shape_supported
        ):
            out = torch.ops.vllm.nvfp4_task39_prefill_dispatch(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                n,
                k,
                output_dtype,
            )
        else:
            out = flashinfer_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
                backend=backend,
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
        # cuDNN uses the same swizzled + padded layout as CUTLASS
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
