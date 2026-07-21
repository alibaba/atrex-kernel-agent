"""gpu-wiki archive note.

TUNED FOR RTX PRO 5000 / SM120 diagnostic runs of linear_qkvz
M=1..16,N=16384,K=5120. This is the omoExplore task41 ATREX JIT wrapper for
the env-gated A-staging / CTA-3D experiments. The task did not meet the 16/16
acceptance gate, so keep this as structural-ceiling evidence, not a production
route.

Related docs:
  docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-decode-gemm-production-lessons.md
"""

import os
import sys
import importlib
import logging
from pathlib import Path
from functools import lru_cache

import torch
from torch import Tensor

_archive_dir = Path(__file__).resolve().parent
_package_root = _archive_dir
_repo_root = _archive_dir
_src_dir_candidates = [
    _archive_dir,
    _package_root / "src" / "cuda" / "nvfp4gemm",
    _repo_root / "src" / "cuda" / "nvfp4gemm",
]
_src_dir = next((path for path in _src_dir_candidates if path.exists()),
                _src_dir_candidates[0])
_MTILE_N_GROUPS = int(os.environ.get("ATREX_NVFP4_MTILE_N_GROUPS", "3"))
_MTILE_CONSUMER_WAIT_BARRIER = int(
    os.environ.get("ATREX_NVFP4_MTILE_CONSUMER_WAIT_BARRIER", "1"))
_MTILE_TMA_STAGES = int(os.environ.get("ATREX_NVFP4_MTILE_TMA_STAGES", "3"))
_MTILE_POST_K_SYNC = int(os.environ.get("ATREX_NVFP4_MTILE_POST_K_SYNC", "1"))
_MTILE_NWARP_N_GROUPS = int(os.environ.get("ATREX_NVFP4_MTILE_NWARP_N_GROUPS", "2"))
_MTILE_NWARP_MIN_M = int(os.environ.get("ATREX_NVFP4_MTILE_NWARP_MIN_M", "17"))
_MTILE_HYBRID_SPLITS = int(os.environ.get("ATREX_NVFP4_MTILE_HYBRID_SPLITS", "2"))
_MTILE_HYBRID_MIN_M = int(os.environ.get("ATREX_NVFP4_MTILE_HYBRID_MIN_M", "1"))
_MTILE_HYBRID_MIN_N = int(os.environ.get("ATREX_NVFP4_MTILE_HYBRID_MIN_N", "20000"))
_MTILE_HYBRID_WARPS_N = int(os.environ.get("ATREX_NVFP4_MTILE_HYBRID_WARPS_N", "0"))
_MTILE_SFA_PAIR_BCAST = int(os.environ.get("ATREX_NVFP4_MTILE_SFA_PAIR_BCAST", "0"))
_MTILE_HYBRID_TMA_STAGES = int(os.environ.get("ATREX_NVFP4_MTILE_HYBRID_TMA_STAGES", "2"))
_MTILE_CROSS_TILE_PIPE = int(os.environ.get("ATREX_NVFP4_MTILE_CROSS_TILE_PIPE", "0"))
_MAXRREGCOUNT = int(os.environ.get("ATREX_NVFP4_MAXRREGCOUNT", "0"))
_build_dir = (
    _package_root
    / "build"
    / "nvfp4gemm"
    / (
        f"build_ng{_MTILE_N_GROUPS}_cw{_MTILE_CONSUMER_WAIT_BARRIER}"
        f"_st{_MTILE_TMA_STAGES}_ps{_MTILE_POST_K_SYNC}"
        f"_nw{_MTILE_NWARP_N_GROUPS}_nwm{_MTILE_NWARP_MIN_M}"
        f"_hs{_MTILE_HYBRID_SPLITS}_hm{_MTILE_HYBRID_MIN_M}"
        f"_hn{_MTILE_HYBRID_MIN_N}_hwn{_MTILE_HYBRID_WARPS_N}"
        f"_sfab{_MTILE_SFA_PAIR_BCAST}_hst{_MTILE_HYBRID_TMA_STAGES}"
        f"_pc{_MTILE_CROSS_TILE_PIPE}_rr{_MAXRREGCOUNT}"
    )
)

CTA3D_TMA_SPLIT_K = int(os.environ.get("ATREX_NVFP4_CTA3D_SPLIT_K", "8"))
CTA3D_TMA_TILE_N = 8
_CTA3D_MAX_M = int(os.environ.get("ATREX_NVFP4_CTA3D_MAX_M", "16"))


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _shape_allows_cta3d(m: int, n: int, k: int) -> bool:
    return (
        1 <= m <= min(_CTA3D_MAX_M, 16)
        and n > 0
        and k > 0
        and n % 8 == 0
        and k % (CTA3D_TMA_SPLIT_K * 128) == 0
    )


def _sf_numel_required(rows: int, k: int) -> int:
    return _round_up(rows, 128) * _round_up(k // 16, 4)


def _workspace_can_fit(workspace: Tensor, m: int, split_k: int, n: int) -> bool:
    return workspace is None or workspace.numel() >= m * split_k * n


def _workspace_can_launch(workspace: Tensor) -> bool:
    return workspace is None or (
        workspace.is_cuda
        and workspace.is_contiguous()
        and workspace.dtype == torch.float32
    )


def _decode_tensors_can_launch(
    A_packed: Tensor,
    B_packed: Tensor,
    SF_A: Tensor,
    SF_B: Tensor,
    alpha: Tensor,
    output: Tensor,
    m: int,
    n: int,
    k: int,
) -> bool:
    if not (
        A_packed.is_cuda
        and B_packed.is_cuda
        and SF_A.is_cuda
        and SF_B.is_cuda
        and alpha.is_cuda
    ):
        return False
    if output is not None and not output.is_cuda:
        return False
    if not (
        A_packed.is_contiguous()
        and B_packed.is_contiguous()
        and SF_A.is_contiguous()
        and SF_B.is_contiguous()
    ):
        return False
    if output is not None and not output.is_contiguous():
        return False
    if (
        A_packed.dtype != torch.uint8
        or B_packed.dtype != torch.uint8
        or SF_A.dtype != torch.uint8
        or SF_B.dtype != torch.uint8
        or alpha.dtype != torch.float32
    ):
        return False
    if output is not None and output.dtype != torch.bfloat16:
        return False
    if B_packed.dim() != 2 or B_packed.size(0) != n or B_packed.size(1) * 2 != k:
        return False
    if A_packed.dim() == 1:
        if m != 1 or A_packed.numel() != k // 2:
            return False
    elif A_packed.dim() == 2:
        if A_packed.size(0) != m or A_packed.size(1) * 2 != k:
            return False
    else:
        return False
    if output is not None:
        if m == 1:
            if output.shape not in ((n,), (1, n)):
                return False
        elif output.shape != (m, n):
            return False
    if SF_A.numel() < _sf_numel_required(m, k):
        return False
    if SF_B.numel() < _sf_numel_required(n, k):
        return False
    return True


def _find_cutlass_include_dir() -> Path:
    candidate_roots = [
        os.environ.get("ATREX_CUTLASS_DIR"),
        os.environ.get("CUTLASS_DIR"),
        str(_package_root / "build" / "cutlass"),
        str(_package_root / "third_party" / "cutlass"),
        str(_repo_root / "third_party" / "cutlass"),
        str(_repo_root / "src" / "cuda" / "cutlass"),
        str(_repo_root / ".deps" / "cutlass-src"),
    ]
    for root in candidate_roots:
        if not root:
            continue
        root_path = Path(root)
        include_dir = root_path if root_path.name == "include" else root_path / "include"
        if (include_dir / "cute" / "arch" / "copy_sm90_tma.hpp").exists():
            return include_dir
    raise RuntimeError(
        "CUTLASS/CuTe headers are required for the NVFP4 CTA-3D TMA fast path. "
        "Set ATREX_CUTLASS_DIR or CUTLASS_DIR to a CUTLASS checkout."
    )


@lru_cache(maxsize=1)
def _build_nvfp4gemm():
    """JIT compile and load nvfp4gemm kernel (SM120a only)."""
    _build_dir.mkdir(parents=True, exist_ok=True)
    so_path = _build_dir / "nvfp4gemm.so"
    srcs = [
        str(
            _src_dir
            / "nvfp4gemm_splitk_linear_qkvz_atrex_sm120_shape_m1_16_n16384_k5120_diagnostic.cu"
        ),
    ]
    pybind_src = _src_dir / "nvfp4gemm_pybind.cu"
    if pybind_src.exists():
        srcs.append(str(pybind_src))

    should_build = not so_path.exists()
    if not should_build:
        so_mtime = so_path.stat().st_mtime
        should_build = any(Path(src).stat().st_mtime > so_mtime for src in srcs)

    if should_build:
        from atrex.utils.compile_utils_cu import jit_compile_cu

        cutlass_include = _find_cutlass_include_dir()
        cutlass_root = cutlass_include.parent
        flags_cuda = [
            "-gencode=arch=compute_120a,code=sm_120a",
            "-DUSE_TMA_B=1",
            "-DTMA_B_CTA_3D=1",
            "-DB_SWIZZLE_MODE=0",
            "-DZERO_A3_ONLY=1",
            f"-DMTILE_N_GROUPS={_MTILE_N_GROUPS}",
            f"-DMTILE_CONSUMER_WAIT_BARRIER={_MTILE_CONSUMER_WAIT_BARRIER}",
            f"-DMTILE_TMA_STAGES={_MTILE_TMA_STAGES}",
            f"-DMTILE_POST_K_SYNC={_MTILE_POST_K_SYNC}",
            f"-DMTILE_NWARP_N_GROUPS={_MTILE_NWARP_N_GROUPS}",
            f"-DMTILE_NWARP_MIN_M={_MTILE_NWARP_MIN_M}",
            f"-DMTILE_HYBRID_SPLITS={_MTILE_HYBRID_SPLITS}",
            f"-DMTILE_HYBRID_MIN_M={_MTILE_HYBRID_MIN_M}",
            f"-DMTILE_HYBRID_MIN_N={_MTILE_HYBRID_MIN_N}",
            f"-DMTILE_HYBRID_WARPS_N={_MTILE_HYBRID_WARPS_N}",
            f"-DMTILE_SFA_PAIR_BCAST={_MTILE_SFA_PAIR_BCAST}",
            f"-DMTILE_HYBRID_TMA_STAGES={_MTILE_HYBRID_TMA_STAGES}",
            f"-DMTILE_CROSS_TILE_PIPE={_MTILE_CROSS_TILE_PIPE}",
        ]
        if _MAXRREGCOUNT > 0:
            flags_cuda.append(f"-maxrregcount={_MAXRREGCOUNT}")
        extra_include = [str(_src_dir / "include"), str(cutlass_include)]
        cutlass_util_include = cutlass_root / "tools" / "util" / "include"
        if cutlass_util_include.exists():
            extra_include.append(str(cutlass_util_include))

        logging.info("JIT compiling nvfp4gemm (splitK + small-M CTA-3D TMA) for SM120a ...")
        jit_compile_cu(
            "nvfp4gemm",
            srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=flags_cuda,
            extra_ldflags=["-lcuda"],
            extra_include_paths=extra_include,
            build_directory=str(_build_dir),
            verbose=True,
            with_cuda=True,
            is_python_module=True,
            is_standalone=False,
        )

    spec = importlib.util.spec_from_file_location("nvfp4gemm", str(so_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["nvfp4gemm"] = module
    spec.loader.exec_module(module)
    return module


def nvfp4_gemv_splitk(
    A_packed: Tensor,
    B_packed: Tensor,
    SF_A: Tensor,
    SF_B: Tensor,
    alpha: Tensor,
    output: Tensor,
    workspace: Tensor = None,
    tile_n: int = 32,
    split_k: int = 4,
) -> Tensor:
    """Split-K NVFP4 decode GEMM/GEMV on SM120a.

    Recommended for CG production on small-N shapes. M=1..16 calls with
    tile_n=8 and split_k=8 use the task_38 CTA-3D TMA fast path when N is
    8-aligned and K is split-aligned. Other supported shapes fall back to the
    two-stage split-K implementation from task_33.

    Args:
        A_packed:  [M, K/2] or [K/2] uint8.
        B_packed:  [N, K/2] uint8.
        SF_A:      uint8 tensor.
        SF_B:      uint8 tensor.
        alpha:     float32 scalar tensor.
        output:    [M, N] or [N] bfloat16, pre-allocated output.
        workspace: [M, split_k, N] or [split_k, N] float32. Auto-allocated if None.
        tile_n:    tile size (default 32).
        split_k:   number of K splits (default 4).

    Returns:
        The workspace tensor (for reuse across calls).
    """
    if A_packed.dim() not in (1, 2) or B_packed.dim() != 2:
        raise ValueError("A_packed must be [M,K/2] or [K/2], B_packed must be [N,K/2]")
    M = 1 if A_packed.dim() == 1 else A_packed.size(0)
    N = B_packed.size(0)
    K = B_packed.size(1) * 2
    if not _decode_tensors_can_launch(
        A_packed, B_packed, SF_A, SF_B, alpha, output, M, N, K
    ):
        raise ValueError("invalid NVFP4 decode tensor shapes, dtypes, or storage")
    if (_shape_allows_cta3d(M, N, K)
            and tile_n in (-1, 0, 32, CTA3D_TMA_TILE_N)
            and split_k in (4, CTA3D_TMA_SPLIT_K)):
        tile_n = CTA3D_TMA_TILE_N
        split_k = CTA3D_TMA_SPLIT_K
    uses_cta3d = (
        _shape_allows_cta3d(M, N, K)
        and tile_n == CTA3D_TMA_TILE_N
        and split_k == CTA3D_TMA_SPLIT_K
    )
    if workspace is None:
        workspace_shape = (split_k, N) if M == 1 else (M, split_k, N)
        workspace = torch.zeros(*workspace_shape, dtype=torch.float32,
                                device=output.device)
    elif not (
        workspace.is_cuda
        and workspace.is_contiguous()
        and workspace.dtype == torch.float32
    ):
        raise ValueError("workspace must be contiguous CUDA float32")
    elif not uses_cta3d and workspace.numel() < M * split_k * N:
        raise ValueError(
            f"workspace has {workspace.numel()} elements, but split_k={split_k}, "
            f"M={M}, N={N} require at least {M * split_k * N}"
        )
    module = _build_nvfp4gemm()
    module.nvfp4_gemv_splitk(A_packed, B_packed, SF_A, SF_B, alpha,
                             output, workspace, tile_n, split_k)
    return workspace


def nvfp4_gemv(
    A_packed: Tensor,
    B_packed: Tensor,
    SF_A: Tensor,
    SF_B: Tensor,
    alpha: Tensor,
    output: Tensor,
    tile_n: int = -1,
) -> None:
    """Compatibility entry point for legacy atrex.nvfp4_gemv callers.

    Production decode dispatch is handled by nvfp4_decode(), so structurally
    compatible M=1..16 decode shapes use the CTA-3D TMA fast path. Other
    small-M shapes fall back to split-K with auto-selected S.
    """
    handled, _ = nvfp4_decode(
        A_packed, B_packed, SF_A, SF_B, alpha, output, tile_n=tile_n)
    if handled:
        return None

    M = 1 if A_packed.dim() == 1 else A_packed.size(0)
    N = B_packed.size(0)
    K = B_packed.size(1) * 2
    S, auto_tile_n = _select_splitk_params(M, N, K)
    if S == 0:
        S = 1
    fallback_tile_n = tile_n if tile_n > 0 else auto_tile_n
    if fallback_tile_n <= 0:
        fallback_tile_n = 32 if N >= 2048 else 16
    nvfp4_gemv_splitk(
        A_packed, B_packed, SF_A, SF_B, alpha, output,
        tile_n=fallback_tile_n, split_k=S)
    return None


DECODE_MAX_M = 16


def _select_splitk_params(m: int, n: int, k: int) -> tuple:
    """Select (split_k, tile_n) for the two-stage split-K path.

    Returns (0, 0) when the shape cannot be handled.
    """
    if n <= 0 or k <= 0:
        return 0, 0
    for s in (8, 4, 2, 1):
        if k % (s * 128) == 0:
            tile_n = 32 if n >= 2048 else 16
            return s, tile_n
    return 0, 0


def _can_use_nvfp4_decode_shape(
    m: int,
    n: int,
    k: int,
    workspace: Tensor = None,
    *,
    max_m: int = DECODE_MAX_M,
    tile_n: int = -1,
) -> bool:
    if m < 1 or n <= 0 or k <= 0:
        return False
    if max_m < 1 or m > min(max_m, DECODE_MAX_M):
        return False
    if not _workspace_can_launch(workspace):
        return False

    if _shape_allows_cta3d(m, n, k):
        # nvfp4_decode() calls nvfp4_gemv_splitk() with split_k=8.  The
        # split-K wrapper uses the CTA-3D path for auto/32/8 tile_n.  Other
        # explicit tile_n values are still handled by the two-stage kernel and
        # therefore need ordinary workspace capacity.
        if tile_n in (-1, 0, 32, CTA3D_TMA_TILE_N):
            return True
        return _workspace_can_fit(workspace, m, CTA3D_TMA_SPLIT_K, n)

    split_k, _ = _select_splitk_params(m, n, k)
    if split_k == 0:
        return False
    return _workspace_can_fit(workspace, m, split_k, n)


def can_use_nvfp4_decode(
    A_packed: Tensor,
    B_packed: Tensor,
    SF_A: Tensor,
    SF_B: Tensor,
    alpha: Tensor,
    output: Tensor = None,
    workspace: Tensor = None,
    *,
    max_m: int = DECODE_MAX_M,
    tile_n: int = -1,
) -> bool:
    """Return whether nvfp4_decode() should handle this call.

    This is the public pre-dispatch check used by vLLM.  It mirrors the
    handled/unsupported boundary of nvfp4_decode() without launching a kernel.
    """
    if A_packed.dim() not in (1, 2) or B_packed.dim() != 2:
        return False
    m = 1 if A_packed.dim() == 1 else A_packed.size(0)
    n = B_packed.size(0)
    k = B_packed.size(1) * 2
    if not _decode_tensors_can_launch(
        A_packed, B_packed, SF_A, SF_B, alpha, output, m, n, k
    ):
        return False
    return _can_use_nvfp4_decode_shape(
        m, n, k, workspace, max_m=max_m, tile_n=tile_n)


def nvfp4_decode(
    A_packed: Tensor,
    B_packed: Tensor,
    SF_A: Tensor,
    SF_B: Tensor,
    alpha: Tensor,
    output: Tensor,
    workspace: Tensor = None,
    tile_n: int = -1,
):
    """Recommended entry point for CG production NVFP4 decode GEMM.

    Handles small-M decode batches (M <= DECODE_MAX_M) via the split-K
    kernel. Structurally compatible M=1..16 calls use the CTA-3D TMA fast
    path; other eligible shapes use the two-stage split-K kernel with
    automatically selected S and tile_n.

    Returns (True, workspace) on success, (False, None) when the shape
    is not eligible (e.g. M too large for decode, K not split-aligned).

    Args:
        A_packed:  [M, K/2] or [K/2] uint8.
        B_packed:  [N, K/2] uint8.
        SF_A:      uint8 tensor.
        SF_B:      uint8 tensor.
        alpha:     float32 scalar tensor.
        output:    [M, N] or [N] bfloat16, pre-allocated output.
        workspace: [M, S, N] or [S, N] float32. Auto-allocated if None.
        tile_n:    tile size (-1=auto).

    Returns:
        (handled: bool, workspace: Tensor or None)
    """
    if A_packed.dim() not in (1, 2) or B_packed.dim() != 2:
        return False, None
    M = 1 if A_packed.dim() == 1 else A_packed.size(0)
    N = B_packed.size(0)
    K = B_packed.size(1) * 2
    if A_packed.dim() == 1 and A_packed.numel() != B_packed.size(1):
        return False, None
    if not _decode_tensors_can_launch(
        A_packed, B_packed, SF_A, SF_B, alpha, output, M, N, K
    ):
        return False, None

    # Path 1: M=1..16 CTA-3D TMA fast path.
    if _shape_allows_cta3d(M, N, K):
        S = CTA3D_TMA_SPLIT_K
        if tile_n <= 0:
            tile_n = CTA3D_TMA_TILE_N
        ws = nvfp4_gemv_splitk(A_packed, B_packed, SF_A, SF_B, alpha,
                               output, workspace, tile_n, S)
        return True, ws

    # Path 2: small-M two-stage split-K.
    if M > DECODE_MAX_M:
        return False, None
    S, auto_tile_n = _select_splitk_params(M, N, K)
    if S == 0:
        return False, None
    if tile_n <= 0:
        tile_n = auto_tile_n
    if not _workspace_can_fit(workspace, M, S, N):
        return False, None
    ws = nvfp4_gemv_splitk(A_packed, B_packed, SF_A, SF_B, alpha,
                           output, workspace, tile_n, S)
    return True, ws
