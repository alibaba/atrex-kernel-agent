# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone atrex-open FlyDSL v2 FP8 PTPC fused_moe path for MI308X.

This wrapper is adapted from atrex-open
``src/triton/fused_moe/fused_moe_flydsl_fp8.py`` at
``c917d1e12f7a8eaf49e3e6f0453dc025173a5239`` plus its task11 dirty comment
update. It preserves the v2 full-pipeline dispatch/profile behavior while using
the local gpu-wiki kernel package.

TUNED FOR: AMD MI308X (CDNA3 / gfx942), task16
``E=512, topk=10, model_dim=4096, inter_dim=256`` and
``M=1/16/32/64/128/256/512``.

Related docs:
- docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md
- docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md
- docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md
"""

import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch

from aiter import ActivationType
from aiter.fused_moe import moe_sorting

from . import moe_kernels as flydsl_v2_kernels


_FORMAL_TOKENS = {1, 16, 32, 64, 128, 256, 512}
_SUPPORTED_SHAPES_BY_VERSION = {
    "v2": {
        "task16": {
            "experts": 512,
            "topk": 10,
            "model_dim": 4096,
            "inter_dim": 256,
        },
    },
}
_BACKEND_ENV = "ATREX_FUSED_MOE_FP8_BACKEND"
_VERSION_ENV = "ATREX_FUSED_MOE_FP8_VERSION"
_DISABLED_BACKENDS = ("0", "false", "off", "disable", "disabled", "triton")
_FORCED_FLYDSL_BACKENDS = (
    "1",
    "true",
    "on",
    "enable",
    "enabled",
    "flydsl",
    "flydsl_v2",
    "v2",
)
_FINAL_INIT_GROUP_CACHE_MAX = 32
_FINAL_INIT_GROUP_CACHE: "OrderedDict[tuple[object, ...], Dict[str, object]]" = (
    OrderedDict()
)
_HYBRID16SORT_GROUP_CACHE_MAX = 32
_HYBRID16SORT_GROUP_CACHE: "OrderedDict[tuple[object, ...], Dict[str, object]]" = (
    OrderedDict()
)
_VALID_BLOCKS_CACHE_MAX = 128
_VALID_BLOCKS_CACHE: "OrderedDict[tuple[object, ...], int]" = OrderedDict()
_STREAM_FENCE_EVENT_CACHE_MAX = 64
_STREAM_FENCE_EVENTS: list[torch.cuda.Event] = []


@contextmanager
def _scoped_env(updates: Dict[str, Optional[str]]):
    old_values = {}
    try:
        for key, value in updates.items():
            old_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _valid_blocks(num_valid_ids: torch.Tensor, block_m: int) -> int:
    return (int(num_valid_ids[0].item()) + block_m - 1) // block_m


def _tensor_identity_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        int(getattr(tensor, "_version", 0)),
    )


def _valid_blocks_cached(
    topk_ids: torch.Tensor, num_valid_ids: torch.Tensor, block_m: int
) -> int:
    key = ("valid_blocks", _tensor_identity_key(topk_ids), int(block_m))
    cached = _VALID_BLOCKS_CACHE.get(key)
    if cached is not None:
        _VALID_BLOCKS_CACHE.move_to_end(key)
        return cached

    valid_blocks = _valid_blocks(num_valid_ids, block_m)
    _VALID_BLOCKS_CACHE[key] = valid_blocks
    while len(_VALID_BLOCKS_CACHE) > _VALID_BLOCKS_CACHE_MAX:
        _VALID_BLOCKS_CACHE.popitem(last=False)
    return valid_blocks


def _launch_block_upper_bound(
    topk_ids: torch.Tensor, sorted_expert_ids: torch.Tensor
) -> int:
    return max(1, min(int(sorted_expert_ids.numel()), int(topk_ids.numel())))


def _record_stream_fence_event(device: torch.device) -> None:
    """Insert a stream packet boundary without host sync or visible CUDA kernels."""
    event = torch.cuda.Event(enable_timing=False)
    event.record(torch.cuda.current_stream(device))
    _STREAM_FENCE_EVENTS.append(event)
    del _STREAM_FENCE_EVENTS[:-_STREAM_FENCE_EVENT_CACHE_MAX]


def _match_supported_shape(
    *,
    token: int,
    experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    activation,
    version: str,
) -> Optional[str]:
    if (
        token not in _FORMAL_TOKENS
        or dtype != torch.bfloat16
        or int(activation) != int(ActivationType.Silu)
    ):
        return None

    for name, spec in _SUPPORTED_SHAPES_BY_VERSION[version].items():
        if (
            experts == spec["experts"]
            and topk == spec["topk"]
            and model_dim == spec["model_dim"]
            and inter_dim == spec["inter_dim"]
        ):
            return name
    return None


def _requested_version(backend: str) -> Optional[str]:
    if backend in ("flydsl_v2", "v2"):
        return "v2"

    version = os.environ.get(_VERSION_ENV, "auto").strip().lower()
    if version in ("", "auto", "default"):
        return None
    if version in ("2", "v2"):
        return "v2"
    raise ValueError(
        f"{_VERSION_ENV} must be 'v2' or 'auto' when set, got {version!r}"
    )


def _unsupported_message() -> str:
    return (
        "FP8 PTPC FlyDSL supports formal tokens in "
        "{1,16,32,64,128,256,512}, dtype=torch.bfloat16, activation=Silu. "
        "This gpu-wiki package archives atrex-open v2 task16 only: "
        "E=512, TOPK=10, model_dim=4096, inter_dim=256."
    )


def _select_flydsl_version(
    *,
    token: int,
    experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    activation,
) -> Optional[tuple[str, str]]:
    backend = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if backend in _DISABLED_BACKENDS:
        return None

    requested = _requested_version(backend)
    strict = backend in _FORCED_FLYDSL_BACKENDS or requested is not None

    match = _match_supported_shape(
        token=token,
        experts=experts,
        topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
        activation=activation,
        version="v2",
    )

    if requested is not None:
        if requested != "v2":
            raise ValueError(f"only FlyDSL v2 is archived here, got {requested!r}")
        if match is not None:
            return "v2", match
        raise ValueError(
            "requested FP8 PTPC FlyDSL v2 does not support "
            f"token={token}, E={experts}, topk={topk}, "
            f"model_dim={model_dim}, inter_dim={inter_dim}. "
            f"{_unsupported_message()}"
        )

    if match is not None:
        return "v2", match

    if strict:
        raise ValueError(
            "ATREX_FUSED_MOE_FP8_BACKEND=flydsl_v2 does not support "
            f"token={token}, E={experts}, topk={topk}, "
            f"model_dim={model_dim}, inter_dim={inter_dim}. "
            f"{_unsupported_message()}"
        )
    return None


def _kernels_for_version(version: str):
    if version != "v2":
        raise ValueError(f"only FlyDSL v2 is archived here, got {version!r}")
    return flydsl_v2_kernels


def should_use_flydsl_fp8_ptpc(
    *,
    token: int,
    experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    activation,
) -> bool:
    """Return whether the FP8 PTPC FlyDSL path should handle this call."""
    return _select_flydsl_version(
        token=token,
        experts=experts,
        topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
        activation=activation,
    ) is not None


def _stage1_config(shape_name: str, token: int, version: str) -> Dict[str, object]:
    if version == "v3":
        if shape_name != "task16":
            raise ValueError(f"FlyDSL v3 does not support shape {shape_name!r}")
        if token == 1:
            return {
                "block_m": 16,
                "tile_n": 32,
                "tile_k": 256,
                "b_nt": 0,
                "waves_per_eu": 0,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_THREADS": "128",
                    "FLYDSL_CK_LDS128": "0",
                    "AITER_FLYDSL_MOE1_DSWR_ADVANCE": "0",
                    "AITER_FLYDSL_MOE1_FAST_BARRIER": "1",
                    "AITER_FLYDSL_MOE1_TINY_ROW0_X": "1",
                    "AITER_FLYDSL_MOE1_PREFETCH_EPI_TID": "1",
                    "AITER_FLYDSL_MOE1_SCHED": "nosched",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "1",
                },
            }
        if token == 16:
            return {
                "block_m": 16,
                "tile_n": 64,
                "tile_k": 256,
                "b_nt": 2,
                "waves_per_eu": 0,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_TID_LDS": "1",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "3",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "0",
                    "AITER_FLYDSL_MOE1_OUT_NT": "0",
                    "AITER_FLYDSL_MOE1_FAST_BARRIER": "0",
                },
            }
        if token == 32:
            return {
                "block_m": 16,
                "tile_n": 32,
                "tile_k": 256,
                "b_nt": 3,
                "waves_per_eu": 0,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_THREADS": "128",
                    "AITER_FLYDSL_MOE1_TID_LDS": "1",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "7",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "0",
                    "AITER_FLYDSL_MOE1_PRIO": "1",
                },
            }
        if token == 64:
            return {
                "block_m": 16,
                "tile_n": 32,
                "tile_k": 256,
                "b_nt": 2,
                "waves_per_eu": 4,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_THREADS": "128",
                    "AITER_FLYDSL_MOE1_TID_LDS": "1",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "8",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "0",
                    "AITER_FLYDSL_MOE1_FAST_BARRIER": "1",
                },
            }
        if token == 128:
            return {
                "block_m": 16,
                "tile_n": 64,
                "tile_k": 256,
                "b_nt": 2,
                "waves_per_eu": 4,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_FAST_BARRIER": "1",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "9",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "0",
                    "AITER_FLYDSL_MOE1_OUT_NT": "4",
                    "AITER_FLYDSL_MOE1_PREFETCH_A": "1",
                },
            }
        if token == 512:
            return {
                "block_m": 16,
                "tile_n": 128,
                "tile_k": 128,
                "b_nt": 2,
                "waves_per_eu": 0,
                "use_cshuffle_epilog": False,
                "hybrid16sort": True,
                "env": {},
            }
        return {
            "block_m": 32,
            "tile_n": 64,
            "tile_k": 128,
            "b_nt": 2,
            "waves_per_eu": 0,
            "use_cshuffle_epilog": False,
            "hybrid16sort": False,
            "env": {
                "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "16",
                "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "16",
            },
        }

    if version == "v2":
        if shape_name != "task16":
            raise ValueError(f"FlyDSL v2 does not support shape {shape_name!r}")
        if token == 1:
            return {
                "block_m": 16,
                "tile_n": 32,
                "tile_k": 256,
                "b_nt": 0,
                "use_cshuffle_epilog": False,
                "hybrid16sort": False,
                "env": {
                    "AITER_FLYDSL_MOE1_THREADS": "128",
                    "FLYDSL_CK_LDS128": "0",
                    "AITER_FLYDSL_MOE1_DSWR_ADVANCE": "0",
                    "AITER_FLYDSL_MOE1_FAST_BARRIER": "1",
                    "AITER_FLYDSL_MOE1_TINY_ROW0_X": "1",
                    "AITER_FLYDSL_MOE1_PREFETCH_EPI_TID": "1",
                    "AITER_FLYDSL_MOE1_SCHED": "nosched",
                    "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "1",
                },
            }
        if token == 512:
            return {
                "block_m": 16,
                "tile_n": 128,
                "tile_k": 128,
                "b_nt": 2,
                "use_cshuffle_epilog": False,
                "hybrid16sort": True,
                "env": {},
            }
        return {
            "block_m": 16,
            "tile_n": 64,
            "tile_k": 128,
            "b_nt": 0 if token == 16 else 2,
            "use_cshuffle_epilog": False,
            "hybrid16sort": False,
            "env": {
                "AITER_FLYDSL_MOE1_ROW_LIMIT_X": "16",
                "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "16",
            },
        }

    if token == 1:
        return {
            "block_m": 16,
            "tile_n": 32,
            "tile_k": 256,
            "b_nt": 0,
            "use_cshuffle_epilog": False,
            "hybrid16sort": False,
            "env": {
                "AITER_FLYDSL_MOE1_THREADS": "128",
                "FLYDSL_CK_LDS128": "0",
                "AITER_FLYDSL_MOE1_DSWR_ADVANCE": "0",
                "AITER_FLYDSL_MOE1_FAST_BARRIER": "1",
                "AITER_FLYDSL_MOE1_TINY_ROW0_X": "1",
                "AITER_FLYDSL_MOE1_PREFETCH_EPI_TID": "1",
                "AITER_FLYDSL_MOE1_SCHED": "nosched",
                "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG": "1",
            },
        }
    return {
        "block_m": 32,
        "tile_n": 128,
        "tile_k": 256,
        "b_nt": 2,
        "use_cshuffle_epilog": False,
        "hybrid16sort": False,
        "env": {},
    }


def _stage2_config(shape_name: str, token: int, version: str) -> Dict[str, object]:
    if version == "v3":
        if shape_name != "task16":
            raise ValueError(f"FlyDSL v3 does not support shape {shape_name!r}")
        return {
            "block_m": 16,
            "tile_n": 256,
            "tile_k": 128,
            "b_nt": 0 if token in (1, 16, 32) else 2,
            "mode": "multiinit" if token == 512 else "initatomic",
            "init_slots": (0, 1) if token == 512 else (0,),
            "hybrid16sort": False,
            "env": {},
        }

    if version == "v2":
        if shape_name != "task16":
            raise ValueError(f"FlyDSL v2 does not support shape {shape_name!r}")
        env: Dict[str, str] = {}
        if token in (16, 32):
            env = {
                "AITER_FLYDSL_MOE2_SKIP_EVEN_MASK": "1",
                "AITER_FLYDSL_MOE2_SCHED": "1",
                "AITER_FLYDSL_MOE2_SCHED_VMEM": "0",
                "AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM": "0",
            }
        elif token == 64:
            env = {
                "AITER_FLYDSL_MOE2_SCHED": "1",
                "AITER_FLYDSL_MOE2_SCHED_VMEM": "0",
                "AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM": "0",
                "AITER_FLYDSL_MOE2_DSWR_ADVANCE": "2",
                "AITER_FLYDSL_MOE2_SKIP_EVEN_MASK": "1",
            }
        elif token == 128:
            env = {
                "AITER_FLYDSL_MOE2_SCHED": "1",
                "AITER_FLYDSL_MOE2_SCHED_VMEM": "0",
                "AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM": "0",
                "AITER_FLYDSL_MOE2_SKIP_EVEN_MASK": "1",
                "AITER_FLYDSL_MOE2_CSHUFFLE_NLANE": "64",
            }
        elif token in (256, 512):
            env = {
                "AITER_FLYDSL_MOE2_ROWCTX_BASE": "1",
                "AITER_FLYDSL_MOE2_ROWCTX_BCAST": "1",
            }
        return {
            "block_m": 16,
            "tile_n": 256,
            "tile_k": 128,
            "b_nt": 0 if token in (1, 16, 32) else 2,
            "mode": "atomic",
            "hybrid16sort": False,
            "env": env,
        }

    if shape_name == "task16":
        if token == 1:
            return {
                "block_m": 16,
                "tile_n": 256,
                "tile_k": 128,
                "b_nt": 0,
                "mode": "atomic",
                "hybrid16sort": False,
                "env": {},
            }
        return {
            "block_m": 16,
            "tile_n": 256,
            "tile_k": 64,
            "b_nt": 2,
            "mode": "atomic",
            "hybrid16sort": False,
            "env": {},
        }

    if token == 1:
        return {
            "block_m": 16,
            "tile_n": 256,
            "tile_k": 128,
            "b_nt": 0,
            "mode": "atomic",
            "hybrid16sort": False,
            "env": {},
        }
    if token == 512:
        return {
            "block_m": 16,
            "tile_n": 128,
            "tile_k": 128,
            "b_nt": 0,
            "mode": "atomic",
            "hybrid16sort": False,
            "env": {},
        }
    return {
        "block_m": 32,
        "tile_n": 128,
        "tile_k": 128,
        "b_nt": 2,
        "mode": "atomic",
        "hybrid16sort": False,
        "env": {},
    }


def _sort_moe(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    experts: int,
    model_dim: int,
    dtype: torch.dtype,
    block_m: int,
    *,
    exact_valid_blocks: bool = False,
    cache_valid_blocks: bool = False,
    use_aiter_opus: bool = False,
):
    if use_aiter_opus:
        import aiter

        device = topk_ids.device
        token, topk = topk_ids.shape
        max_num_tokens_padded = int(topk_ids.numel() + experts * block_m - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_m - 1) // block_m)
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=device
        )
        sorted_weights = torch.empty(
            (max_num_tokens_padded,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
        moe_buf = torch.empty((token, model_dim), dtype=dtype, device=device)
        workspace_size = aiter.moe_sorting_opus_get_workspace_size(
            token, experts, topk, 0
        )
        workspace = (
            torch.empty(workspace_size, dtype=torch.uint8, device=device)
            if workspace_size > 0
            else None
        )
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weight,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            experts,
            int(block_m),
            None,
            None,
            workspace,
            0,
        )
    else:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids,
            topk_weight,
            experts,
            model_dim,
            dtype,
            block_m,
        )
    if exact_valid_blocks:
        valid_blocks = (
            _valid_blocks_cached(topk_ids, num_valid_ids, block_m)
            if cache_valid_blocks
            else _valid_blocks(num_valid_ids, block_m)
        )
    else:
        valid_blocks = _launch_block_upper_bound(topk_ids, sorted_expert_ids)

    return {
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "valid_blocks": valid_blocks,
    }


def _build_hybrid16sort_groups(sort_data: Dict[str, object], block_m: int):
    """Pair adjacent 16-row sorted blocks from the same expert into m32 groups."""
    if block_m != 16:
        raise ValueError("hybrid16sort requires block_m=16")

    sorted_ids = sort_data["sorted_ids"]
    sorted_weights = sort_data["sorted_weights"]
    sorted_expert_ids = sort_data["sorted_expert_ids"]
    valid_blocks = int(sort_data["valid_blocks"])
    dev = sorted_ids.device

    eids = sorted_expert_ids[:valid_blocks].detach().cpu().tolist()
    ids32 = []
    weights32 = []
    eids32 = []
    ids16 = []
    weights16 = []
    eids16 = []

    i = 0
    while i < valid_blocks:
        expert = int(eids[i])
        j = i + 1
        while j < valid_blocks and int(eids[j]) == expert:
            j += 1

        k = i
        while k + 1 < j:
            ids32.append(sorted_ids[k * block_m : (k + 2) * block_m])
            if sorted_weights is not None:
                weights32.append(sorted_weights[k * block_m : (k + 2) * block_m])
            eids32.append(expert)
            k += 2
        if k < j:
            ids16.append(sorted_ids[k * block_m : (k + 1) * block_m])
            if sorted_weights is not None:
                weights16.append(sorted_weights[k * block_m : (k + 1) * block_m])
            eids16.append(expert)
        i = j

    def make_group(
        ids_parts: Iterable[torch.Tensor],
        weight_parts: Iterable[torch.Tensor],
        expert_parts,
        group_m: int,
    ):
        ids_parts = list(ids_parts)
        weight_parts = list(weight_parts)
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = (
                torch.cat(weight_parts).contiguous()
                if sorted_weights is not None
                else None
            )
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = (
                torch.empty(0, dtype=sorted_weights.dtype, device=dev)
                if sorted_weights is not None
                else None
            )
            group_eids = torch.empty(0, dtype=torch.int32, device=dev)
        group_blocks = int(group_eids.numel())
        group_num_valid = torch.tensor(
            [group_blocks * group_m], dtype=torch.int32, device=dev
        )
        return {
            "sorted_ids": group_ids,
            "sorted_weights": group_weights,
            "sorted_expert_ids": group_eids,
            "num_valid_ids": group_num_valid,
            "valid_blocks": group_blocks,
        }

    return {
        "m32": make_group(ids32, weights32, eids32, 32),
        "m16": make_group(ids16, weights16, eids16, 16),
    }


def _build_hybrid16sort_groups_cached(
    sort_data: Dict[str, object],
    block_m: int,
    *,
    topk_ids: torch.Tensor,
):
    """Cache stage1 hybrid16sort groups for repeated calls over unchanged topk_ids."""
    if sort_data["sorted_weights"] is not None:
        return _build_hybrid16sort_groups(sort_data, block_m)

    cache_key = (
        "hybrid16sort",
        _tensor_cache_key(topk_ids),
        int(block_m),
        str(sort_data["sorted_ids"].dtype),
        str(sort_data["sorted_ids"].device),
    )
    cached = _HYBRID16SORT_GROUP_CACHE.get(cache_key)
    if cached is not None:
        _HYBRID16SORT_GROUP_CACHE.move_to_end(cache_key)
        return cached

    groups = _build_hybrid16sort_groups(sort_data, block_m)
    groups["_cache_refs"] = (topk_ids,)
    _HYBRID16SORT_GROUP_CACHE[cache_key] = groups
    _HYBRID16SORT_GROUP_CACHE.move_to_end(cache_key)
    while len(_HYBRID16SORT_GROUP_CACHE) > _HYBRID16SORT_GROUP_CACHE_MAX:
        _HYBRID16SORT_GROUP_CACHE.popitem(last=False)
    return groups


def _build_stage2_final_init_groups(
    *,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    sort_data: Dict[str, object],
    experts: int,
    topk: int,
    token: int,
    block_m: int,
    init_slots: Iterable[int],
):
    """Split stage2 into ordered final-output slot groups and atomic rest."""
    init_slots = tuple(int(slot) for slot in init_slots)
    if not init_slots:
        raise ValueError("stage2 final-init path requires at least one init slot")
    if len(set(init_slots)) != len(init_slots):
        raise ValueError(f"duplicate stage2 init slots: {init_slots!r}")
    for slot in init_slots:
        if slot < 0 or slot >= topk:
            raise ValueError(f"stage2 init slot must be in [0, {topk}), got {slot}")

    topk_ids_cpu = topk_ids.detach().cpu()
    topk_weight_cpu = topk_weight.detach().cpu()
    init_slot_set = set(init_slots)
    dev = sort_data["sorted_ids"].device
    id_dtype = sort_data["sorted_ids"].dtype
    weight_dtype = topk_weight.dtype
    sentinel = (topk << 24) | token

    selected_ids = {slot: [[] for _ in range(experts)] for slot in init_slots}
    selected_weights = {slot: [[] for _ in range(experts)] for slot in init_slots}
    rest_ids = [[] for _ in range(experts)]
    rest_weights = [[] for _ in range(experts)]

    for t in range(token):
        for s in range(topk):
            expert = int(topk_ids_cpu[t, s])
            fused = (s << 24) | t
            weight = float(topk_weight_cpu[t, s])
            if s in init_slot_set:
                selected_ids[s][expert].append(fused)
                selected_weights[s][expert].append(weight)
            else:
                rest_ids[expert].append(fused)
                rest_weights[expert].append(weight)

    def make_group(ids_by_expert, weights_by_expert):
        flat_ids = []
        flat_weights = []
        expert_parts = []
        for expert in range(experts):
            ids = ids_by_expert[expert]
            weights = weights_by_expert[expert]
            for start in range(0, len(ids), block_m):
                chunk_ids = list(ids[start : start + block_m])
                chunk_weights = list(weights[start : start + block_m])
                pad = block_m - len(chunk_ids)
                if pad:
                    chunk_ids.extend([sentinel] * pad)
                    chunk_weights.extend([0.0] * pad)
                flat_ids.extend(chunk_ids)
                flat_weights.extend(chunk_weights)
                expert_parts.append(expert)

        if expert_parts:
            group_ids = torch.tensor(flat_ids, dtype=id_dtype, device=dev)
            group_weights = torch.tensor(flat_weights, dtype=weight_dtype, device=dev)
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=id_dtype, device=dev)
            group_weights = torch.empty(0, dtype=weight_dtype, device=dev)
            group_eids = torch.empty(0, dtype=torch.int32, device=dev)
        group_blocks = int(group_eids.numel())
        group_num_valid = torch.tensor(
            [group_blocks * block_m], dtype=torch.int32, device=dev
        )
        return {
            "sorted_ids": group_ids,
            "sorted_weights": group_weights,
            "sorted_expert_ids": group_eids,
            "num_valid_ids": group_num_valid,
            "valid_blocks": group_blocks,
        }

    store_slot = init_slots[0]
    return {
        "store": make_group(selected_ids[store_slot], selected_weights[store_slot]),
        "adds": [
            (slot, make_group(selected_ids[slot], selected_weights[slot]))
            for slot in init_slots[1:]
        ],
        "rest": make_group(rest_ids, rest_weights),
    }


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        int(getattr(tensor, "_version", 0)),
    )


def _build_stage2_final_init_groups_cached(
    *,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    sort_data: Dict[str, object],
    experts: int,
    topk: int,
    token: int,
    block_m: int,
    init_slots: Iterable[int],
):
    """Cache CPU-built final-init groups for repeated calls over unchanged tensors."""
    init_slots = tuple(int(slot) for slot in init_slots)
    cache_key = (
        _tensor_cache_key(topk_ids),
        _tensor_cache_key(topk_weight),
        int(experts),
        int(topk),
        int(token),
        int(block_m),
        init_slots,
    )
    cached = _FINAL_INIT_GROUP_CACHE.get(cache_key)
    if cached is not None:
        _FINAL_INIT_GROUP_CACHE.move_to_end(cache_key)
        return cached

    groups = _build_stage2_final_init_groups(
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        sort_data=sort_data,
        experts=experts,
        topk=topk,
        token=token,
        block_m=block_m,
        init_slots=init_slots,
    )
    groups["_cache_refs"] = (topk_ids, topk_weight)
    _FINAL_INIT_GROUP_CACHE[cache_key] = groups
    _FINAL_INIT_GROUP_CACHE.move_to_end(cache_key)
    while len(_FINAL_INIT_GROUP_CACHE) > _FINAL_INIT_GROUP_CACHE_MAX:
        _FINAL_INIT_GROUP_CACHE.popitem(last=False)
    return groups


def fused_moe_flydsl_fp8_ptpc(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    token = hidden_states.shape[0]
    experts = w1.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w2.shape[2]
    model_dim = hidden_states.shape[1]
    selected = _select_flydsl_version(
        token=token,
        experts=experts,
        topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=hidden_states.dtype,
        activation=ActivationType.Silu,
    )
    if selected is None:
        raise ValueError(
            "unsupported FP8 PTPC FlyDSL call: "
            f"token={token}, E={experts}, topk={topk}, "
            f"model_dim={model_dim}, inter_dim={inter_dim}"
        )
    version, shape_name = selected
    kernels = _kernels_for_version(version)
    use_v2_profile_sort = version == "v2" and shape_name == "task16"
    use_v2_opus_sort = use_v2_profile_sort and token != 1
    use_v2_m1_stream_fence = use_v2_profile_sort and token == 1

    import aiter

    a1_qt = torch.empty(
        hidden_states.shape,
        dtype=torch.float8_e4m3fnuz,
        device=hidden_states.device,
    )
    a1_scale = torch.empty([token, 1], dtype=torch.float32, device=hidden_states.device)
    aiter.dynamic_per_token_scaled_quant(a1_qt, hidden_states, a1_scale)

    s1_cfg = _stage1_config(shape_name, token, version)
    s1_sort = _sort_moe(
        topk_ids,
        topk_weight,
        experts,
        model_dim,
        hidden_states.dtype,
        int(s1_cfg["block_m"]),
        exact_valid_blocks=(
            version in ("v2", "v3") and not use_v2_m1_stream_fence
        ),
        cache_valid_blocks=(
            use_v2_profile_sort and not use_v2_m1_stream_fence
        ),
        use_aiter_opus=use_v2_opus_sort,
    )
    gemm1_out = torch.empty(
        [token, topk, inter_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    with _scoped_env(s1_cfg["env"]):
        if bool(s1_cfg["hybrid16sort"]):
            groups = _build_hybrid16sort_groups_cached(
                {**s1_sort, "sorted_weights": None},
                int(s1_cfg["block_m"]),
                topk_ids=topk_ids,
            )
            for group_m in (32, 16):
                group = groups[f"m{group_m}"]
                if int(group["valid_blocks"]) == 0:
                    continue
                kernels.flydsl_moe_stage1(
                    a=a1_qt,
                    w1=w1,
                    sorted_token_ids=group["sorted_ids"],
                    sorted_expert_ids=group["sorted_expert_ids"],
                    num_valid_ids=group["num_valid_ids"],
                    out=gemm1_out,
                    topk=topk,
                    tile_m=group_m,
                    tile_n=int(s1_cfg["tile_n"]),
                    tile_k=int(s1_cfg["tile_k"]),
                    a_dtype="fp8",
                    b_dtype="fp8",
                    out_dtype="bf16",
                    w1_scale=w1_scale,
                    a1_scale=a1_scale,
                    sorted_weights=None,
                    k_batch=1,
                    waves_per_eu=int(s1_cfg.get("waves_per_eu", 0)),
                    b_nt=int(s1_cfg["b_nt"]),
                    xcd_swizzle=0,
                    grid_y_override=int(group["valid_blocks"]),
                    use_cshuffle_epilog=False,
                    assume_valid_grid=True,
                )
        else:
            if use_v2_m1_stream_fence:
                # M=1 stage1 is a tiny launch and is bimodal without a stream
                # packet boundary here. This preserves the old .item() boundary
                # effect without host sync or profiler-visible memcpy.
                _record_stream_fence_event(hidden_states.device)
            kernels.flydsl_moe_stage1(
                a=a1_qt,
                w1=w1,
                sorted_token_ids=s1_sort["sorted_ids"],
                sorted_expert_ids=s1_sort["sorted_expert_ids"],
                num_valid_ids=s1_sort["num_valid_ids"],
                out=gemm1_out,
                topk=topk,
                tile_m=int(s1_cfg["block_m"]),
                tile_n=int(s1_cfg["tile_n"]),
                tile_k=int(s1_cfg["tile_k"]),
                a_dtype="fp8",
                b_dtype="fp8",
                out_dtype="bf16",
                w1_scale=w1_scale,
                a1_scale=a1_scale,
                sorted_weights=None,
                k_batch=1,
                waves_per_eu=int(s1_cfg.get("waves_per_eu", 0)),
                b_nt=int(s1_cfg["b_nt"]),
                xcd_swizzle=0,
                grid_y_override=int(s1_sort["valid_blocks"]),
                use_cshuffle_epilog=bool(s1_cfg["use_cshuffle_epilog"]),
                assume_valid_grid=(version in ("v2", "v3") and token == 1),
            )

    a2_qt = torch.empty(
        gemm1_out.shape,
        dtype=torch.float8_e4m3fnuz,
        device=gemm1_out.device,
    )
    a2_scale = torch.empty([token, topk, 1], dtype=torch.float32, device=gemm1_out.device)
    aiter.dynamic_per_token_scaled_quant(a2_qt, gemm1_out, a2_scale)

    s2_cfg = _stage2_config(shape_name, token, version)
    reuse_stage1_sort = (
        use_v2_profile_sort and int(s1_cfg["block_m"]) == int(s2_cfg["block_m"])
    )
    if reuse_stage1_sort:
        s2_sort = s1_sort
    else:
        s2_sort = _sort_moe(
            topk_ids,
            topk_weight,
            experts,
            model_dim,
            hidden_states.dtype,
            int(s2_cfg["block_m"]),
            exact_valid_blocks=(
                version in ("v2", "v3") and not use_v2_m1_stream_fence
            ),
            cache_valid_blocks=(
                use_v2_profile_sort and not use_v2_m1_stream_fence
            ),
            use_aiter_opus=use_v2_opus_sort,
        )
    out = torch.empty(
        (token, model_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    with _scoped_env(s2_cfg["env"]):
        if str(s2_cfg["mode"]) in ("initatomic", "multiinit"):
            out.fill_(0)
            groups = _build_stage2_final_init_groups_cached(
                topk_ids=topk_ids,
                topk_weight=topk_weight,
                sort_data=s2_sort,
                experts=experts,
                topk=topk,
                token=token,
                block_m=int(s2_cfg["block_m"]),
                init_slots=s2_cfg.get("init_slots", (0,)),
            )

            def run_final_group(group: Dict[str, object], env_key: str):
                if int(group["valid_blocks"]) == 0:
                    return
                with _scoped_env({env_key: "1"}):
                    kernels.flydsl_moe_stage2(
                        inter_states=a2_qt,
                        w2=w2,
                        sorted_token_ids=group["sorted_ids"],
                        sorted_expert_ids=group["sorted_expert_ids"],
                        num_valid_ids=group["num_valid_ids"],
                        out=out,
                        topk=topk,
                        tile_m=int(s2_cfg["block_m"]),
                        tile_n=int(s2_cfg["tile_n"]),
                        tile_k=int(s2_cfg["tile_k"]),
                        a_dtype="fp8",
                        b_dtype="fp8",
                        out_dtype="bf16",
                        mode="reduce",
                        w2_scale=w2_scale,
                        a2_scale=a2_scale,
                        sorted_weights=group["sorted_weights"],
                        sort_block_m=int(s2_cfg["block_m"]),
                        b_nt=int(s2_cfg["b_nt"]),
                        xcd_swizzle=0,
                        grid_y_override=int(group["valid_blocks"]),
                        zero_output=False,
                    )

            def run_atomic_group(group: Dict[str, object]):
                if int(group["valid_blocks"]) == 0:
                    return
                kernels.flydsl_moe_stage2(
                    inter_states=a2_qt,
                    w2=w2,
                    sorted_token_ids=group["sorted_ids"],
                    sorted_expert_ids=group["sorted_expert_ids"],
                    num_valid_ids=group["num_valid_ids"],
                    out=out,
                    topk=topk,
                    tile_m=int(s2_cfg["block_m"]),
                    tile_n=int(s2_cfg["tile_n"]),
                    tile_k=int(s2_cfg["tile_k"]),
                    a_dtype="fp8",
                    b_dtype="fp8",
                    out_dtype="bf16",
                    mode="atomic",
                    w2_scale=w2_scale,
                    a2_scale=a2_scale,
                    sorted_weights=group["sorted_weights"],
                    sort_block_m=int(s2_cfg["block_m"]),
                    b_nt=int(s2_cfg["b_nt"]),
                    xcd_swizzle=0,
                    grid_y_override=int(group["valid_blocks"]),
                    zero_output=False,
                )

            run_final_group(groups["store"], "AITER_FLYDSL_MOE2_REDUCE_STORE_FINAL")
            for _, add_group in groups["adds"]:
                run_final_group(add_group, "AITER_FLYDSL_MOE2_REDUCE_ADD_FINAL")
            run_atomic_group(groups["rest"])
        elif bool(s2_cfg["hybrid16sort"]):
            out.fill_(0)
            groups = _build_hybrid16sort_groups(s2_sort, int(s2_cfg["block_m"]))
            for group_m, tile_n, tile_k, b_nt, env in (
                (32, 256, 128, 0, {"AITER_FLYDSL_MOE2_CSHUFFLE_NLANE": "16"}),
                (
                    16,
                    256,
                    64,
                    0,
                    {
                        "AITER_FLYDSL_MOE2_CSHUFFLE_NLANE": "16",
                        "AITER_FLYDSL_MOE2_SCHED": "1",
                        "AITER_FLYDSL_MOE2_SCHED_VMEM": "0",
                    },
                ),
            ):
                group = groups[f"m{group_m}"]
                if int(group["valid_blocks"]) == 0:
                    continue
                with _scoped_env(env):
                    kernels.flydsl_moe_stage2(
                        inter_states=a2_qt,
                        w2=w2,
                        sorted_token_ids=group["sorted_ids"],
                        sorted_expert_ids=group["sorted_expert_ids"],
                        num_valid_ids=group["num_valid_ids"],
                        out=out,
                        topk=topk,
                        tile_m=group_m,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        a_dtype="fp8",
                        b_dtype="fp8",
                        out_dtype="bf16",
                        mode="atomic",
                        w2_scale=w2_scale,
                        a2_scale=a2_scale,
                        sorted_weights=group["sorted_weights"],
                        sort_block_m=group_m,
                        b_nt=b_nt,
                        xcd_swizzle=0,
                        grid_y_override=int(group["valid_blocks"]),
                        zero_output=False,
                    )
        else:
            kernels.flydsl_moe_stage2(
                inter_states=a2_qt,
                w2=w2,
                sorted_token_ids=s2_sort["sorted_ids"],
                sorted_expert_ids=s2_sort["sorted_expert_ids"],
                num_valid_ids=s2_sort["num_valid_ids"],
                out=out,
                topk=topk,
                tile_m=int(s2_cfg["block_m"]),
                tile_n=int(s2_cfg["tile_n"]),
                tile_k=int(s2_cfg["tile_k"]),
                a_dtype="fp8",
                b_dtype="fp8",
                out_dtype="bf16",
                mode=str(s2_cfg["mode"]),
                w2_scale=w2_scale,
                a2_scale=a2_scale,
                sorted_weights=s2_sort["sorted_weights"],
                sort_block_m=int(s2_cfg["block_m"]),
                b_nt=int(s2_cfg["b_nt"]),
                xcd_swizzle=0,
                grid_y_override=int(s2_sort["valid_blocks"]),
            )

    return out
