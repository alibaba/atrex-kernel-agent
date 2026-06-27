# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing harness for FP8 PTPC FlyDSL MoE stage1/stage2.

Example:
    python aiter/ops/flydsl/test_flydsl_moe_fp8_ptpc.py \
        --tokens 16 --model-dim 4096 --inter-dim 256 \
        --experts 512 --topk 10 --stage stage1 --block-m 32
"""

import argparse
import csv
import os
import statistics
import sys
import traceback
from pathlib import Path


def _require_aiter_base() -> Path:
    value = os.environ.get("AITER_BASE")
    if not value:
        raise RuntimeError(
            "AITER_BASE must point to an aiter checkout. "
            "Example: export AITER_BASE=/path/to/aiter"
        )
    return Path(value).expanduser()


def _bootstrap_project_aiter_overlay():
    """Run this task-local overlay on top of a fixed AITER checkout."""
    task_root = Path(__file__).resolve().parents[3]
    aiter_base = _require_aiter_base()
    if not (aiter_base / "aiter" / "__init__.py").exists():
        raise RuntimeError(
            "AITER_BASE must point to a fixed AITER checkout "
            f"(missing {aiter_base / 'aiter' / '__init__.py'})"
        )

    base_path = str(aiter_base.resolve())
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    import importlib

    def _prepend_package_path(module_name: str, overlay_path: Path):
        module = importlib.import_module(module_name)
        paths = list(getattr(module, "__path__", []))
        overlay = str(overlay_path.resolve())
        if overlay not in paths:
            module.__path__ = [overlay] + paths

    _prepend_package_path("aiter", task_root / "aiter")
    _prepend_package_path("aiter.ops", task_root / "aiter" / "ops")
    aiter_ops = importlib.import_module("aiter.ops")
    if hasattr(aiter_ops, "flydsl"):
        delattr(aiter_ops, "flydsl")
    for module_name in list(sys.modules):
        if module_name == "aiter.ops.flydsl" or module_name.startswith(
            "aiter.ops.flydsl."
        ):
            del sys.modules[module_name]
    _prepend_package_path(
        "aiter.ops.flydsl", task_root / "aiter" / "ops" / "flydsl"
    )
    _prepend_package_path(
        "aiter.ops.flydsl.kernels",
        task_root / "aiter" / "ops" / "flydsl" / "kernels",
    )


def _maybe_add_rtp_tools() -> None:
    value = os.environ.get("RTP_TOOLS_PYTHON")
    if not value:
        return
    tools = Path(value).expanduser()
    path = str(tools)
    if tools.exists() and path not in sys.path:
        sys.path.insert(0, path)


_bootstrap_project_aiter_overlay()

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.quant import get_torch_quant
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")


def _generate_fp8_ptpc_stage1_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    *,
    dtype=torch.bfloat16,
    check: bool = True,
    doweight_stage1: bool = False,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    torch_quant = get_torch_quant(QuantType.per_Token)
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=dtypes.fp8)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp8)

    ref1 = None
    if check:
        w2_stub = torch.empty(1, dtype=dtype, device=inp.device).as_strided(
            (E, model_dim, inter_dim), (0, 0, 0)
        )
        ref1 = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_stub,
            topk_weights,
            topk_ids,
            dtype=dtype,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            doweight=doweight_stage1,
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    return {
        "inp": inp,
        "a1_qt": a1_qt,
        "a1_scale": a1_scale,
        "w1": w1,
        "w1_qt": w1_qt,
        "w1_scale": w1_scale,
        "w1_shuf": shuffle_weight(w1_qt, (16, 16), use_int4=False),
        "ref_stage1": ref1,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights if doweight_stage1 else None,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "valid_blocks": (int(num_valid_ids[0].item()) + block_m - 1) // block_m,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "dtype": dtype,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "topk": topk,
    }


def _generate_fp8_ptpc_stage2_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    *,
    dtype=torch.bfloat16,
    check: bool = True,
    doweight_stage2: bool = True,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    inter_states = torch.randn((token, topk, inter_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    torch_quant = get_torch_quant(QuantType.per_Token)
    a2_qt, a2_scale = torch_quant(inter_states, quant_dtype=dtypes.fp8)
    a2_qt = a2_qt.view(token, topk, inter_dim)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp8)

    ref2 = None
    if check:
        w1_stub = torch.empty(1, dtype=dtype, device=inp.device).as_strided(
            (E, inter_dim * 2, model_dim), (0, 0, 0)
        )
        ref2 = torch_moe_stage2(
            a2_qt,
            w1_stub,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype=dtype,
            quant_type=QuantType.per_Token,
            w2_scale=w2_scale,
            a2_scale=a2_scale,
            doweight=doweight_stage2,
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    return {
        "inter_states": inter_states,
        "a2_qt": a2_qt,
        "a2_scale": a2_scale,
        "w2": w2,
        "w2_qt": w2_qt,
        "w2_scale": w2_scale,
        "w2_shuf": shuffle_weight(w2_qt, (16, 16), use_int4=False),
        "ref_stage2": ref2,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights if doweight_stage2 else None,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "valid_blocks": (int(num_valid_ids[0].item()) + block_m - 1) // block_m,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "dtype": dtype,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "topk": topk,
    }


def _build_stage2_hybrid_groups(data: dict[str, object], block_m: int):
    """Split sorted 32-row expert blocks into paired-64 and leftover-32 groups."""
    if block_m != 32:
        raise ValueError("stage2 hybrid mode currently requires block_m=32")

    sorted_ids = data["sorted_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    valid_blocks = int(data["valid_blocks"])
    dev = sorted_ids.device

    eids = sorted_expert_ids[:valid_blocks].detach().cpu().tolist()
    ids64 = []
    weights64 = []
    eids64 = []
    ids32 = []
    weights32 = []
    eids32 = []

    i = 0
    while i < valid_blocks:
        expert = int(eids[i])
        j = i + 1
        while j < valid_blocks and int(eids[j]) == expert:
            j += 1

        k = i
        while k + 1 < j:
            ids64.append(sorted_ids[k * block_m : (k + 2) * block_m])
            weights64.append(sorted_weights[k * block_m : (k + 2) * block_m])
            eids64.append(expert)
            k += 2
        if k < j:
            ids32.append(sorted_ids[k * block_m : (k + 1) * block_m])
            weights32.append(sorted_weights[k * block_m : (k + 1) * block_m])
            eids32.append(expert)
        i = j

    def make_group(ids_parts, weight_parts, expert_parts, group_m: int):
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = torch.cat(weight_parts).contiguous()
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = torch.empty(0, dtype=sorted_weights.dtype, device=dev)
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
        "m64": make_group(ids64, weights64, eids64, 64),
        "m32": make_group(ids32, weights32, eids32, 32),
    }


def _build_stage2_hybrid16_groups(data: dict[str, object], block_m: int):
    """Split sorted 32-row blocks into m64 pairs, real m32 blocks, and padded m16 blocks."""
    if block_m != 32:
        raise ValueError("stage2 hybrid16 mode currently requires block_m=32")

    sorted_ids = data["sorted_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    valid_blocks = int(data["valid_blocks"])
    token = int(data["token"])
    topk = int(data["topk"])
    dev = sorted_ids.device

    eids = sorted_expert_ids[:valid_blocks].detach().cpu().tolist()
    ids_cpu = sorted_ids[: valid_blocks * block_m].detach().cpu().tolist()
    ids64 = []
    weights64 = []
    eids64 = []
    ids32 = []
    weights32 = []
    eids32 = []
    ids16 = []
    weights16 = []
    eids16 = []

    def row_is_valid(row: int) -> bool:
        fused = int(ids_cpu[row])
        tok = fused & 0xFFFFFF
        slot = fused >> 24
        return tok < token and slot < topk

    i = 0
    while i < valid_blocks:
        expert = int(eids[i])
        j = i + 1
        while j < valid_blocks and int(eids[j]) == expert:
            j += 1

        k = i
        while k + 1 < j:
            ids64.append(sorted_ids[k * block_m : (k + 2) * block_m])
            weights64.append(sorted_weights[k * block_m : (k + 2) * block_m])
            eids64.append(expert)
            k += 2
        if k < j:
            row16 = k * block_m + 16
            if row_is_valid(row16):
                ids32.append(sorted_ids[k * block_m : (k + 1) * block_m])
                weights32.append(sorted_weights[k * block_m : (k + 1) * block_m])
                eids32.append(expert)
            else:
                ids16.append(sorted_ids[k * block_m : k * block_m + 16])
                weights16.append(sorted_weights[k * block_m : k * block_m + 16])
                eids16.append(expert)
        i = j

    def make_group(ids_parts, weight_parts, expert_parts, group_m: int):
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = torch.cat(weight_parts).contiguous()
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = torch.empty(0, dtype=sorted_weights.dtype, device=dev)
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
        "m64": make_group(ids64, weights64, eids64, 64),
        "m32": make_group(ids32, weights32, eids32, 32),
        "m16": make_group(ids16, weights16, eids16, 16),
    }


def _build_stage2_hybrid16sort_groups(data: dict[str, object], block_m: int):
    """Pair adjacent 16-row sorted blocks from the same expert into m32 groups."""
    if block_m != 16:
        raise ValueError("stage2 hybrid16sort mode requires block_m=16")

    sorted_ids = data["sorted_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    valid_blocks = int(data["valid_blocks"])
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
            weights32.append(sorted_weights[k * block_m : (k + 2) * block_m])
            eids32.append(expert)
            k += 2
        if k < j:
            ids16.append(sorted_ids[k * block_m : (k + 1) * block_m])
            weights16.append(sorted_weights[k * block_m : (k + 1) * block_m])
            eids16.append(expert)
        i = j

    def make_group(ids_parts, weight_parts, expert_parts, group_m: int):
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = torch.cat(weight_parts).contiguous()
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = torch.empty(0, dtype=sorted_weights.dtype, device=dev)
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


def _build_stage1_hybrid16sort_groups(data: dict[str, object], block_m: int):
    """Pair adjacent 16-row stage1 sorted blocks from the same expert into m32 groups."""
    if block_m != 16:
        raise ValueError("stage1 hybrid16sort mode requires block_m=16")

    sorted_ids = data["sorted_ids"]
    sorted_expert_ids = data["sorted_expert_ids"]
    sorted_weights = data["sorted_weights"]
    valid_blocks = int(data["valid_blocks"])
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

    def make_group(ids_parts, weight_parts, expert_parts, group_m: int):
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


def _build_stage2_hybrid48sort_groups(data: dict[str, object], block_m: int):
    """Coalesce 16-row sorted blocks into m48/m32/m16 groups per expert."""
    if block_m != 16:
        raise ValueError("stage2 hybrid48sort mode requires block_m=16")

    sorted_ids = data["sorted_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    valid_blocks = int(data["valid_blocks"])
    dev = sorted_ids.device

    eids = sorted_expert_ids[:valid_blocks].detach().cpu().tolist()
    ids_by_m = {48: [], 32: [], 16: []}
    weights_by_m = {48: [], 32: [], 16: []}
    eids_by_m = {48: [], 32: [], 16: []}

    i = 0
    while i < valid_blocks:
        expert = int(eids[i])
        j = i + 1
        while j < valid_blocks and int(eids[j]) == expert:
            j += 1

        k = i
        while k + 2 < j:
            ids_by_m[48].append(sorted_ids[k * block_m : (k + 3) * block_m])
            weights_by_m[48].append(sorted_weights[k * block_m : (k + 3) * block_m])
            eids_by_m[48].append(expert)
            k += 3
        if k + 1 < j:
            ids_by_m[32].append(sorted_ids[k * block_m : (k + 2) * block_m])
            weights_by_m[32].append(sorted_weights[k * block_m : (k + 2) * block_m])
            eids_by_m[32].append(expert)
        elif k < j:
            ids_by_m[16].append(sorted_ids[k * block_m : (k + 1) * block_m])
            weights_by_m[16].append(sorted_weights[k * block_m : (k + 1) * block_m])
            eids_by_m[16].append(expert)
        i = j

    def make_group(group_m: int):
        ids_parts = ids_by_m[group_m]
        weight_parts = weights_by_m[group_m]
        expert_parts = eids_by_m[group_m]
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = torch.cat(weight_parts).contiguous()
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = torch.empty(0, dtype=sorted_weights.dtype, device=dev)
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
        "m48": make_group(48),
        "m32": make_group(32),
        "m16": make_group(16),
    }


def _build_stage2_hybrid48_groups(data: dict[str, object], block_m: int):
    """Split sorted 32-row expert blocks into m48/m32/m16 chunks.

    This keeps one B load per expert chunk while avoiding the m64 padding used by
    `hybrid16` for experts with 33-48 routed rows.
    """
    if block_m != 32:
        raise ValueError("stage2 hybrid48 mode currently requires block_m=32")

    sorted_ids = data["sorted_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    valid_blocks = int(data["valid_blocks"])
    token = int(data["token"])
    topk = int(data["topk"])
    dev = sorted_ids.device

    eids = sorted_expert_ids[:valid_blocks].detach().cpu().tolist()
    ids_cpu = sorted_ids[: valid_blocks * block_m].detach().cpu().tolist()
    ids_by_m = {48: [], 32: [], 16: []}
    weights_by_m = {48: [], 32: [], 16: []}
    eids_by_m = {48: [], 32: [], 16: []}
    min_m48 = int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID48_MIN_M48", "33"))
    if min_m48 < 33 or min_m48 > 49:
        raise ValueError(
            "AITER_FLYDSL_MOE2_HYBRID48_MIN_M48 must be in [33, 49], "
            f"got {min_m48}"
        )

    def row_is_valid(row: int) -> bool:
        fused = int(ids_cpu[row])
        tok = fused & 0xFFFFFF
        slot = fused >> 24
        return tok < token and slot < topk

    i = 0
    while i < valid_blocks:
        expert = int(eids[i])
        j = i + 1
        while j < valid_blocks and int(eids[j]) == expert:
            j += 1

        start = i * block_m
        end = j * block_m
        valid_count = 0
        for row in range(start, end):
            if row_is_valid(row):
                valid_count += 1

        row = start
        remaining = valid_count
        while remaining > 0:
            if remaining >= min_m48:
                group_m = 48
            elif remaining > 16:
                group_m = 32
            else:
                group_m = 16
            ids_by_m[group_m].append(sorted_ids[row : row + group_m])
            weights_by_m[group_m].append(sorted_weights[row : row + group_m])
            eids_by_m[group_m].append(expert)
            row += group_m
            remaining -= min(remaining, group_m)

        i = j

    def make_group(group_m: int):
        ids_parts = ids_by_m[group_m]
        weight_parts = weights_by_m[group_m]
        expert_parts = eids_by_m[group_m]
        if ids_parts:
            group_ids = torch.cat(ids_parts).contiguous()
            group_weights = torch.cat(weight_parts).contiguous()
            group_eids = torch.tensor(expert_parts, dtype=torch.int32, device=dev)
        else:
            group_ids = torch.empty(0, dtype=sorted_ids.dtype, device=dev)
            group_weights = torch.empty(0, dtype=sorted_weights.dtype, device=dev)
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
        "m48": make_group(48),
        "m32": make_group(32),
        "m16": make_group(16),
    }


def _check_result(ref_out, test_out, *, atol: float, rtol: float, pass_pct: float):
    if ref_out is None:
        return None, None, None
    ref_f = ref_out.float()
    test_f = test_out.float()
    max_delta = (ref_f - test_f).abs().max().item()
    close_mask = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100.0
    passed = pct_close >= pass_pct
    print(
        f"  correctness: max_delta={max_delta:.6f}, close={pct_close:.2f}% "
        f"(atol={atol}, rtol={rtol}) -> {'PASS' if passed else 'FAIL'}"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    return passed, max_delta, pct_close


def _time_cuda(fn, *, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0)
    return statistics.mean(latencies_us), statistics.median(latencies_us)


def _profile_ordered(fn, *, warmup: int, iters: int):
    _maybe_add_rtp_tools()
    from tensor_tools import profile_cuda_kernels

    _, table = profile_cuda_kernels(fn, enable_print=False, warmup=warmup, iters=iters)
    return table


def run_stage1_case(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    tile_n: int,
    tile_k: int,
    mode: str,
    b_nt: int,
    k_batch: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    use_cshuffle_epilog: bool | None,
    check: bool,
    profile: bool,
    warmup: int,
    iters: int,
    atol: float,
    rtol: float,
    pass_pct: float,
    trim_grid: bool,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    if mode not in ("atomic", "hybrid16sort"):
        raise ValueError(f"stage1 mode must be atomic or hybrid16sort, got {mode!r}")
    if xcd_swizzle < 0:
        xcd_swizzle = 0
    auto_t512_cshuffle_anchor = (
        use_cshuffle_epilog is not False
        and token == 512
        and block_m == 16
        and tile_n == 256
        and tile_k == 128
        and b_nt == 2
    )
    auto_t1_direct_anchor = (
        use_cshuffle_epilog is False
        and token == 1
        and block_m == 16
        and tile_n == 32
        and tile_k == 256
        and b_nt == 0
    )
    auto_t32_direct_anchor = (
        use_cshuffle_epilog is False
        and token == 32
        and block_m == 16
        and tile_n in (32, 64)
        and tile_k == 128
        and b_nt == 2
    )
    auto_t32_thr128_direct_anchor = auto_t32_direct_anchor and tile_n == 32
    auto_t16_direct_anchor = (
        use_cshuffle_epilog is False
        and token == 16
        and block_m == 16
        and tile_n == 64
        and tile_k == 256
        and b_nt == 2
    )
    auto_t64_direct_anchor = (
        use_cshuffle_epilog is False
        and token == 64
        and block_m == 16
        and tile_n == 64
        and tile_k == 128
        and b_nt == 2
    )
    if waves_per_eu == 0 and auto_t512_cshuffle_anchor:
        waves_per_eu = 3
    if waves_per_eu == 0 and auto_t32_thr128_direct_anchor:
        waves_per_eu = 3

    auto_t1_fast_anchor = auto_t1_direct_anchor
    old_threads = os.environ.get("AITER_FLYDSL_MOE1_THREADS")
    if auto_t1_fast_anchor and old_threads is None:
        os.environ["AITER_FLYDSL_MOE1_THREADS"] = "128"
    threads_effective = os.environ.get("AITER_FLYDSL_MOE1_THREADS", "auto")
    old_lds128 = os.environ.get("FLYDSL_CK_LDS128")
    if auto_t1_fast_anchor and old_lds128 is None:
        os.environ["FLYDSL_CK_LDS128"] = "0"
    lds128_effective = os.environ.get("FLYDSL_CK_LDS128", "1")
    old_dswr_advance = os.environ.get("AITER_FLYDSL_MOE1_DSWR_ADVANCE")
    if auto_t1_fast_anchor and old_dswr_advance is None:
        os.environ["AITER_FLYDSL_MOE1_DSWR_ADVANCE"] = "0"
    dswr_advance_effective = os.environ.get("AITER_FLYDSL_MOE1_DSWR_ADVANCE", "2")

    print(
        f"\n[FP8 PTPC stage1] token={token} dim=({model_dim},{inter_dim}) "
        f"E={E} topk={topk} bm={block_m} tn={tile_n} tk={tile_k} "
        f"mode={mode} b_nt={b_nt} kb={k_batch} wpe={waves_per_eu} xcd={xcd_swizzle}"
    )

    data = _generate_fp8_ptpc_stage1_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        check=check,
    )
    out = torch.empty((token, topk, inter_dim), dtype=dtypes.bf16, device="cuda")
    if mode == "hybrid16sort":
        hybrid_groups = _build_stage1_hybrid16sort_groups(data, block_m)
    else:
        hybrid_groups = None
    force_assume_valid = os.environ.get(
        "AITER_FLYDSL_MOE1_ASSUME_VALID_GRID", "0"
    ).strip() in ("1", "true", "True", "YES", "yes")
    auto_assume_valid = token <= 1 or (
        use_cshuffle_epilog is not False
        and token >= 512
        and block_m == 16
        and tile_n == 256
        and tile_k == 128
    ) or (
        use_cshuffle_epilog is not False
        and token == 64
        and block_m == 16
        and tile_n == 64
        and tile_k == 256
    ) or (
        use_cshuffle_epilog is False
        and token <= 32
        and block_m == 16
        and tile_n in (32, 64)
        and tile_k in (128, 256)
    )
    assume_valid_grid = trim_grid and (
        force_assume_valid or auto_assume_valid
    )
    force_fast_barrier = os.environ.get(
        "AITER_FLYDSL_MOE1_FAST_BARRIER", "0"
    ).strip() in ("1", "true", "True", "YES", "yes")
    auto_fast_barrier = (
        use_cshuffle_epilog is False
        and block_m == 16
        and b_nt == 2
        and token == 64
        and tile_n == 64
        and tile_k == 128
    ) or auto_t1_direct_anchor
    fast_barrier_enabled = force_fast_barrier or auto_fast_barrier
    old_fast_barrier = os.environ.get("AITER_FLYDSL_MOE1_FAST_BARRIER")
    if auto_fast_barrier and old_fast_barrier is None:
        os.environ["AITER_FLYDSL_MOE1_FAST_BARRIER"] = "1"
    force_tid_lds = os.environ.get(
        "AITER_FLYDSL_MOE1_TID_LDS", "0"
    ).strip() in ("1", "true", "True", "YES", "yes")
    auto_tid_lds = (
        use_cshuffle_epilog is False
        and token == 16
        and block_m == 16
        and tile_n == 64
        and tile_k == 256
        and b_nt == 2
    ) or auto_t32_thr128_direct_anchor
    tid_lds_enabled = force_tid_lds or auto_tid_lds
    old_tid_lds = os.environ.get("AITER_FLYDSL_MOE1_TID_LDS")
    if auto_tid_lds and old_tid_lds is None:
        os.environ["AITER_FLYDSL_MOE1_TID_LDS"] = "1"
    force_tiny_row0_x = os.environ.get(
        "AITER_FLYDSL_MOE1_TINY_ROW0_X", "0"
    ).strip() in ("1", "true", "True", "YES", "yes")
    auto_tiny_row0_x = auto_t1_direct_anchor
    tiny_row0_x_enabled = force_tiny_row0_x or auto_tiny_row0_x
    old_tiny_row0_x = os.environ.get("AITER_FLYDSL_MOE1_TINY_ROW0_X")
    if auto_tiny_row0_x and old_tiny_row0_x is None:
        os.environ["AITER_FLYDSL_MOE1_TINY_ROW0_X"] = "1"
    auto_prefetch_epi_tid = auto_t1_direct_anchor
    old_prefetch_epi_tid = os.environ.get("AITER_FLYDSL_MOE1_PREFETCH_EPI_TID")
    if auto_prefetch_epi_tid and old_prefetch_epi_tid is None:
        os.environ["AITER_FLYDSL_MOE1_PREFETCH_EPI_TID"] = "1"
    # With scoped output-store NT enabled, the original split scheduler is more
    # reproducible than split3 on the token512 cshuffle anchor.
    auto_sched_t1_nosched = auto_t1_fast_anchor
    auto_sched_split3 = False
    old_sched = os.environ.get("AITER_FLYDSL_MOE1_SCHED")
    if old_sched is None:
        if auto_sched_t1_nosched:
            os.environ["AITER_FLYDSL_MOE1_SCHED"] = "nosched"
        elif auto_sched_split3:
            os.environ["AITER_FLYDSL_MOE1_SCHED"] = "split3"
    sched_effective = os.environ.get("AITER_FLYDSL_MOE1_SCHED", "auto")
    auto_out_nt = auto_t512_cshuffle_anchor
    old_out_nt = os.environ.get("AITER_FLYDSL_MOE1_OUT_NT")
    if auto_out_nt and old_out_nt is None:
        os.environ["AITER_FLYDSL_MOE1_OUT_NT"] = "3"
    out_nt_effective = os.environ.get("AITER_FLYDSL_MOE1_OUT_NT", "0")
    auto_row_limit_x_value = (
        3
        if auto_t16_direct_anchor
        else (4 if auto_t32_direct_anchor else (6 if auto_t64_direct_anchor else 0))
    )
    auto_row_limit_x = auto_row_limit_x_value > 0
    old_row_limit_x = os.environ.get("AITER_FLYDSL_MOE1_ROW_LIMIT_X")
    if auto_row_limit_x and old_row_limit_x is None:
        os.environ["AITER_FLYDSL_MOE1_ROW_LIMIT_X"] = str(auto_row_limit_x_value)
    row_limit_x_effective = os.environ.get("AITER_FLYDSL_MOE1_ROW_LIMIT_X", "0")
    auto_row_limit_epilog_value = (
        1 if auto_t1_direct_anchor else (4 if auto_t32_thr128_direct_anchor else 0)
    )
    auto_row_limit_epilog = auto_row_limit_epilog_value > 0
    old_row_limit_epilog = os.environ.get("AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG")
    if auto_row_limit_epilog and old_row_limit_epilog is None:
        os.environ["AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG"] = str(
            auto_row_limit_epilog_value
        )
    row_limit_epilog_effective = os.environ.get(
        "AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG", "0"
    )

    def run_once():
        if hybrid_groups is not None:
            for group_m in (32, 16):
                group = hybrid_groups[f"m{group_m}"]
                if group["valid_blocks"] == 0:
                    continue
                flydsl_moe_stage1(
                    a=data["a1_qt"],
                    w1=data["w1_shuf"],
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
                    w1_scale=data["w1_scale"],
                    a1_scale=data["a1_scale"],
                    sorted_weights=group["sorted_weights"],
                    k_batch=k_batch,
                    waves_per_eu=waves_per_eu,
                    b_nt=b_nt,
                    xcd_swizzle=xcd_swizzle,
                    grid_y_override=group["valid_blocks"],
                    use_cshuffle_epilog=False,
                    assume_valid_grid=True,
                )
            return out
        return flydsl_moe_stage1(
            a=data["a1_qt"],
            w1=data["w1_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            out=out,
            topk=topk,
            tile_m=block_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype="bf16",
            w1_scale=data["w1_scale"],
            a1_scale=data["a1_scale"],
            sorted_weights=data["sorted_weights"],
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            xcd_swizzle=xcd_swizzle,
            grid_y_override=data["valid_blocks"] if trim_grid else None,
            use_cshuffle_epilog=use_cshuffle_epilog,
            assume_valid_grid=assume_valid_grid,
        )

    try:
        run_once()
        torch.cuda.synchronize()
        passed, max_delta, pct_close = _check_result(
            data["ref_stage1"],
            out,
            atol=atol,
            rtol=rtol,
            pass_pct=pass_pct,
        )

        mean_us, median_us = _time_cuda(run_once, warmup=warmup, iters=iters)
        print(f"  timing: mean={mean_us:.3f}us median={median_us:.3f}us")

        ordered = {}
        gemm1_ms = None
        if profile:
            ordered = _profile_ordered(run_once, warmup=warmup, iters=iters)
            for name, us in sorted(
                ordered.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"  profile: {name} {us * 1000.0:.3f}us")
            gemm1_ms = sum(us for name, us in ordered.items() if "moe_gemm1" in name)
            if gemm1_ms > 0:
                print(f"  profile_total: moe_gemm1_all {gemm1_ms * 1000.0:.3f}us")
    finally:
        if auto_t1_fast_anchor and old_threads is None:
            os.environ.pop("AITER_FLYDSL_MOE1_THREADS", None)
        if auto_t1_fast_anchor and old_lds128 is None:
            os.environ.pop("FLYDSL_CK_LDS128", None)
        if auto_t1_fast_anchor and old_dswr_advance is None:
            os.environ.pop("AITER_FLYDSL_MOE1_DSWR_ADVANCE", None)
        if auto_fast_barrier and old_fast_barrier is None:
            os.environ.pop("AITER_FLYDSL_MOE1_FAST_BARRIER", None)
        if auto_tid_lds and old_tid_lds is None:
            os.environ.pop("AITER_FLYDSL_MOE1_TID_LDS", None)
        if auto_tiny_row0_x and old_tiny_row0_x is None:
            os.environ.pop("AITER_FLYDSL_MOE1_TINY_ROW0_X", None)
        if auto_prefetch_epi_tid and old_prefetch_epi_tid is None:
            os.environ.pop("AITER_FLYDSL_MOE1_PREFETCH_EPI_TID", None)
        if (auto_sched_t1_nosched or auto_sched_split3) and old_sched is None:
            os.environ.pop("AITER_FLYDSL_MOE1_SCHED", None)
        if auto_out_nt and old_out_nt is None:
            os.environ.pop("AITER_FLYDSL_MOE1_OUT_NT", None)
        if auto_row_limit_x and old_row_limit_x is None:
            os.environ.pop("AITER_FLYDSL_MOE1_ROW_LIMIT_X", None)
        if auto_row_limit_epilog and old_row_limit_epilog is None:
            os.environ.pop("AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG", None)

    profile_top_name = ""
    profile_top_ms = None
    if ordered:
        profile_top_name, profile_top_ms = max(ordered.items(), key=lambda item: item[1])
    if mode == "hybrid16sort" and gemm1_ms is not None and gemm1_ms > 0:
        profile_top_name, profile_top_ms = "moe_gemm1_all", gemm1_ms

    return {
        "stage": "stage1",
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": E,
        "topk": topk,
        "mode": mode,
        "block_m": block_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "b_nt": b_nt,
        "k_batch": k_batch,
        "waves_per_eu": waves_per_eu,
        "xcd_swizzle": xcd_swizzle,
        "threads": threads_effective,
        "lds128": lds128_effective,
        "dswr_advance": dswr_advance_effective,
        "mean_us": f"{mean_us:.3f}",
        "median_us": f"{median_us:.3f}",
        "check": int(check),
        "passed": "" if passed is None else int(passed),
        "max_delta": "" if max_delta is None else f"{max_delta:.6f}",
        "pct_close": "" if pct_close is None else f"{pct_close:.3f}",
        "profile_top": profile_top_name,
        "profile_top_ms": "" if profile_top_ms is None else f"{profile_top_ms:.6f}",
        "profile_top_us": ""
        if profile_top_ms is None
        else f"{profile_top_ms * 1000.0:.3f}",
        "trim_grid": int(trim_grid),
        "assume_valid_grid": int(assume_valid_grid),
        "fast_barrier": int(fast_barrier_enabled),
        "tid_lds": int(tid_lds_enabled),
        "tiny_row0_x": int(tiny_row0_x_enabled),
        "sched": sched_effective,
        "out_nt": out_nt_effective,
        "row_limit_x": row_limit_x_effective,
        "row_limit_epilog": row_limit_epilog_effective,
        "cshuffle": "auto"
        if use_cshuffle_epilog is None
        else int(use_cshuffle_epilog),
        "valid_blocks": data["valid_blocks"],
        "expert_blocks": data["sorted_expert_ids"].shape[0],
        "hybrid_m32_blocks": ""
        if hybrid_groups is None
        else hybrid_groups["m32"]["valid_blocks"],
        "hybrid_m16_blocks": ""
        if hybrid_groups is None
        else hybrid_groups["m16"]["valid_blocks"],
    }


def run_stage2_case(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    tile_n: int,
    tile_k: int,
    mode: str,
    out_dtype: str,
    b_nt: int,
    xcd_swizzle: int,
    check: bool,
    profile: bool,
    warmup: int,
    iters: int,
    atol: float,
    rtol: float,
    pass_pct: float,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    if xcd_swizzle < 0:
        xcd_swizzle = 0

    print(
        f"\n[FP8 PTPC stage2] token={token} dim=({model_dim},{inter_dim}) "
        f"E={E} topk={topk} bm={block_m} tn={tile_n} tk={tile_k} "
            f"mode={mode} out={out_dtype} b_nt={b_nt} xcd={xcd_swizzle}"
    )

    data = _generate_fp8_ptpc_stage2_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        check=check,
    )
    torch_out_dtype = dtypes.bf16 if out_dtype == "bf16" else torch.float16
    out = torch.empty((token, model_dim), dtype=torch_out_dtype, device="cuda")
    if mode == "hybrid":
        hybrid_groups = _build_stage2_hybrid_groups(data, block_m)
    elif mode == "hybrid16":
        hybrid_groups = _build_stage2_hybrid16_groups(data, block_m)
    elif mode == "hybrid16sort":
        hybrid_groups = _build_stage2_hybrid16sort_groups(data, block_m)
    elif mode == "hybrid48sort":
        hybrid_groups = _build_stage2_hybrid48sort_groups(data, block_m)
    elif mode == "hybrid48":
        hybrid_groups = _build_stage2_hybrid48_groups(data, block_m)
    else:
        hybrid_groups = None
    hybrid48_min_m48 = (
        os.environ.get("AITER_FLYDSL_MOE2_HYBRID48_MIN_M48", "33")
        if mode == "hybrid48"
        else ""
    )
    auto_env_restore = []

    def _set_auto_env(key: str, value: str):
        old_value = os.environ.get(key)
        if old_value is None:
            os.environ[key] = value
            auto_env_restore.append((key, old_value))

    auto_t512_hybrid16sort = (
        mode == "hybrid16sort"
        and token == 512
        and block_m == 16
        and tile_n == 128
        and tile_k == 128
    )
    if auto_t512_hybrid16sort:
        _set_auto_env("AITER_FLYDSL_MOE2_SCHED", "1")
        _set_auto_env("AITER_FLYDSL_MOE2_SCHED_VMEM", "1")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_TN_M32", "256")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_TK_M32", "128")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_BNT_M32", "0")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_CSN_M32", "16")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_TN_M16", "256")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_TK_M16", "64")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_BNT_M16", "0")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_CSN_M16", "16")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_SCHED_M16", "1")
        _set_auto_env("AITER_FLYDSL_MOE2_HYBRID_SVM_M16", "0")
        _set_auto_env("AITER_FLYDSL_MOE2_ROWCTX_BASE", "1")
        _set_auto_env("AITER_FLYDSL_MOE2_ROWCTX_BCAST", "1")

    hybrid_b_nt = {
        64: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_BNT_M64", str(b_nt))),
        48: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_BNT_M48", str(b_nt))),
        32: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_BNT_M32", str(b_nt))),
        16: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_BNT_M16", str(b_nt))),
    }
    hybrid_tile_n = {
        64: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TN_M64", str(tile_n))),
        48: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TN_M48", str(tile_n))),
        32: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TN_M32", str(tile_n))),
        16: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TN_M16", str(tile_n))),
    }
    hybrid_tile_k = {
        64: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TK_M64", str(tile_k))),
        48: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TK_M48", str(tile_k))),
        32: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TK_M32", str(tile_k))),
        16: int(os.environ.get("AITER_FLYDSL_MOE2_HYBRID_TK_M16", str(tile_k))),
    }
    hybrid_env = {}
    for group_m in (64, 48, 32, 16):
        suffix = f"M{group_m}"
        hybrid_env[group_m] = {
            "AITER_FLYDSL_MOE2_SCHED": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_SCHED_{suffix}"
            ),
            "AITER_FLYDSL_MOE2_SCHED_VMEM": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_SVM_{suffix}"
            ),
            "AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_SEVM_{suffix}"
            ),
            "AITER_FLYDSL_MOE2_DSWR_ADVANCE": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_DSWA_{suffix}"
            ),
            "AITER_FLYDSL_MOE2_CSHUFFLE_NLANE": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_CSN_{suffix}"
            ),
            "AITER_FLYDSL_MOE2_THREADS": os.environ.get(
                f"AITER_FLYDSL_MOE2_HYBRID_THREADS_{suffix}"
            ),
        }

    def _with_group_env(group_m: int, fn):
        old_env = {}
        for key, value in hybrid_env[group_m].items():
            if value is None or value == "":
                continue
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            return fn()
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def run_once():
        if hybrid_groups is not None:
            out.fill_(0)
            for group_m in (64, 48, 32, 16):
                if f"m{group_m}" not in hybrid_groups:
                    continue
                group = hybrid_groups[f"m{group_m}"]
                if group["valid_blocks"] == 0:
                    continue
                _with_group_env(
                    group_m,
                    lambda group_m=group_m, group=group: flydsl_moe_stage2(
                        inter_states=data["a2_qt"],
                        w2=data["w2_shuf"],
                        sorted_token_ids=group["sorted_ids"],
                        sorted_expert_ids=group["sorted_expert_ids"],
                        num_valid_ids=group["num_valid_ids"],
                        out=out,
                        topk=topk,
                        tile_m=group_m,
                        tile_n=hybrid_tile_n[group_m],
                        tile_k=hybrid_tile_k[group_m],
                        a_dtype="fp8",
                        b_dtype="fp8",
                        out_dtype=out_dtype,
                        w2_scale=data["w2_scale"],
                        a2_scale=data["a2_scale"],
                        mode="atomic",
                        sorted_weights=group["sorted_weights"],
                        sort_block_m=group_m,
                        b_nt=hybrid_b_nt[group_m],
                        xcd_swizzle=xcd_swizzle,
                        grid_y_override=group["valid_blocks"],
                        zero_output=False,
                    ),
                )
            return out
        return flydsl_moe_stage2(
            inter_states=data["a2_qt"],
            w2=data["w2_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            out=out,
            topk=topk,
            tile_m=block_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype=out_dtype,
            w2_scale=data["w2_scale"],
            a2_scale=data["a2_scale"],
            mode=mode,
            sorted_weights=data["sorted_weights"],
            sort_block_m=block_m,
            b_nt=b_nt,
            xcd_swizzle=xcd_swizzle,
            grid_y_override=data["valid_blocks"],
        )

    ordered = {}
    gemm2_ms = None
    try:
        run_once()
        torch.cuda.synchronize()
        passed, max_delta, pct_close = _check_result(
            data["ref_stage2"],
            out,
            atol=atol,
            rtol=rtol,
            pass_pct=pass_pct,
        )

        mean_us, median_us = _time_cuda(run_once, warmup=warmup, iters=iters)
        print(f"  timing: mean={mean_us:.3f}us median={median_us:.3f}us")

        if profile:
            ordered = _profile_ordered(run_once, warmup=warmup, iters=iters)
            for name, us in sorted(
                ordered.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"  profile: {name} {us * 1000.0:.3f}us")
            gemm2_ms = sum(
                us
                for name, us in ordered.items()
                if "moe_gemm2" in name or "moe_gemm2_m" in name
            )
            if gemm2_ms > 0:
                print(f"  profile_total: moe_gemm2_all {gemm2_ms * 1000.0:.3f}us")
    finally:
        for key, old_value in reversed(auto_env_restore):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    profile_top_name = ""
    profile_top_ms = None
    if ordered:
        profile_top_name, profile_top_ms = max(ordered.items(), key=lambda item: item[1])

    return {
        "stage": "stage2",
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": E,
        "topk": topk,
        "block_m": block_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "mode": mode,
        "out_dtype": out_dtype,
        "b_nt": b_nt,
        "hybrid_b_nt_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else hybrid_b_nt[64],
        "hybrid_b_nt_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else hybrid_b_nt[48],
        "hybrid_b_nt_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else hybrid_b_nt[32],
        "hybrid_b_nt_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else hybrid_b_nt[16],
        "hybrid_tn_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else hybrid_tile_n[64],
        "hybrid_tn_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else hybrid_tile_n[48],
        "hybrid_tn_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else hybrid_tile_n[32],
        "hybrid_tn_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else hybrid_tile_n[16],
        "hybrid_tk_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else hybrid_tile_k[64],
        "hybrid_tk_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else hybrid_tile_k[48],
        "hybrid_tk_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else hybrid_tile_k[32],
        "hybrid_tk_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else hybrid_tile_k[16],
        "hybrid_svm_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_SCHED_VMEM"] or ""),
        "hybrid_svm_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_SCHED_VMEM"] or ""),
        "hybrid_svm_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_SCHED_VMEM"] or ""),
        "hybrid_svm_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_SCHED_VMEM"] or ""),
        "hybrid_csn_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_CSHUFFLE_NLANE"] or ""),
        "hybrid_csn_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_CSHUFFLE_NLANE"] or ""),
        "hybrid_csn_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_CSHUFFLE_NLANE"] or ""),
        "hybrid_csn_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_CSHUFFLE_NLANE"] or ""),
        "hybrid_sched_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_SCHED"] or ""),
        "hybrid_sched_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_SCHED"] or ""),
        "hybrid_sched_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_SCHED"] or ""),
        "hybrid_sched_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_SCHED"] or ""),
        "hybrid_sevm_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM"] or ""),
        "hybrid_sevm_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM"] or ""),
        "hybrid_sevm_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM"] or ""),
        "hybrid_sevm_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM"] or ""),
        "hybrid_dswa_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_DSWR_ADVANCE"] or ""),
        "hybrid_dswa_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_DSWR_ADVANCE"] or ""),
        "hybrid_dswa_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_DSWR_ADVANCE"] or ""),
        "hybrid_dswa_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_DSWR_ADVANCE"] or ""),
        "hybrid_threads_m64": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else (hybrid_env[64]["AITER_FLYDSL_MOE2_THREADS"] or ""),
        "hybrid_threads_m48": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else (hybrid_env[48]["AITER_FLYDSL_MOE2_THREADS"] or ""),
        "hybrid_threads_m32": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else (hybrid_env[32]["AITER_FLYDSL_MOE2_THREADS"] or ""),
        "hybrid_threads_m16": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else (hybrid_env[16]["AITER_FLYDSL_MOE2_THREADS"] or ""),
        "hybrid48_min_m48": hybrid48_min_m48,
        "xcd_swizzle": xcd_swizzle,
        "mean_us": f"{mean_us:.3f}",
        "median_us": f"{median_us:.3f}",
        "check": int(check),
        "passed": "" if passed is None else int(passed),
        "max_delta": "" if max_delta is None else f"{max_delta:.6f}",
        "pct_close": "" if pct_close is None else f"{pct_close:.3f}",
        "profile_top": profile_top_name,
        "profile_top_ms": "" if profile_top_ms is None else f"{profile_top_ms:.6f}",
        "profile_top_us": ""
        if profile_top_ms is None
        else f"{profile_top_ms * 1000.0:.3f}",
        "profile_gemm2_ms": "" if gemm2_ms is None else f"{gemm2_ms:.6f}",
        "profile_gemm2_us": ""
        if gemm2_ms is None
        else f"{gemm2_ms * 1000.0:.3f}",
        "valid_blocks": data["valid_blocks"],
        "expert_blocks": data["sorted_expert_ids"].shape[0],
        "hybrid_m64_blocks": ""
        if hybrid_groups is None or "m64" not in hybrid_groups
        else hybrid_groups["m64"]["valid_blocks"],
        "hybrid_m32_blocks": ""
        if hybrid_groups is None or "m32" not in hybrid_groups
        else hybrid_groups["m32"]["valid_blocks"],
        "hybrid_m48_blocks": ""
        if hybrid_groups is None or "m48" not in hybrid_groups
        else hybrid_groups["m48"]["valid_blocks"],
        "hybrid_m16_blocks": ""
        if hybrid_groups is None or "m16" not in hybrid_groups
        else hybrid_groups["m16"]["valid_blocks"],
    }


def write_csv(path: str, rows: list[dict[str, object]]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="FlyDSL MoE FP8 PTPC harness")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 16, 32, 64, 128, 256, 512, 4000, 7000],
    )
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=512)
    parser.add_argument("-k", "--topk", type=int, default=10)
    parser.add_argument("--block-m", type=int, nargs="+", default=[32])
    parser.add_argument("--tile-n", type=int, nargs="+", default=[128])
    parser.add_argument("--tile-k", type=int, nargs="+", default=[256])
    parser.add_argument("--b-nt", type=int, nargs="+", default=[2])
    parser.add_argument("--k-batch", type=int, nargs="+", default=[1])
    parser.add_argument("--waves-per-eu", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=["atomic"],
        choices=[
            "atomic",
            "reduce",
            "hybrid",
            "hybrid16",
            "hybrid16sort",
            "hybrid48",
            "hybrid48sort",
        ],
    )
    parser.add_argument("--out-dtype", choices=["bf16", "f16"], default="bf16")
    parser.add_argument(
        "--xcd-swizzle",
        type=int,
        nargs="+",
        default=[-1],
        help="XCD swizzle; -1 uses the current FP8 PTPC default (0).",
    )
    parser.add_argument(
        "--cshuffle",
        choices=["auto", "on", "off"],
        default="off",
        help="Stage1 epilogue mode: off keeps the direct scalar store baseline.",
    )
    parser.add_argument(
        "--stage", type=str, nargs="+", default=["stage1"], choices=["stage1", "stage2"]
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--check-token-limit", type=int, default=7000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-trim-grid", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--pass-pct", type=float, default=99.0)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()
    if args.cshuffle == "auto":
        use_cshuffle_epilog = None
    elif args.cshuffle == "on":
        use_cshuffle_epilog = True
    else:
        use_cshuffle_epilog = False

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL is not available. Install flydsl package first.")
        return 0

    rows = []
    ok = True
    for stage in args.stage:
        for token in args.tokens:
            for bm in args.block_m:
                check = (not args.skip_check) and token <= args.check_token_limit
                for tn in args.tile_n:
                    if (stage == "stage1" and args.inter_dim % tn != 0) or (
                        stage == "stage2" and args.model_dim % tn != 0
                    ):
                        print(
                            f"[SKIP] stage={stage} output dim is not divisible by tn={tn}"
                        )
                        continue
                    for tk in args.tile_k:
                        if (stage == "stage1" and args.model_dim % tk != 0) or (
                            stage == "stage2" and args.inter_dim % tk != 0
                        ):
                            print(
                                f"[SKIP] stage={stage} K dim is not divisible by tk={tk}"
                            )
                            continue
                        for bnt in args.b_nt:
                            for xcd in args.xcd_swizzle:
                                if stage == "stage1":
                                    for mode in args.mode:
                                        if mode not in ("atomic", "hybrid16sort"):
                                            print(
                                                f"[SKIP] stage1 mode={mode} is not supported"
                                            )
                                            continue
                                        for kb in args.k_batch:
                                            for wpe in args.waves_per_eu:
                                                try:
                                                    row = run_stage1_case(
                                                        token=token,
                                                        model_dim=args.model_dim,
                                                        inter_dim=args.inter_dim,
                                                        E=args.experts,
                                                        topk=args.topk,
                                                        block_m=bm,
                                                        tile_n=tn,
                                                        tile_k=tk,
                                                        mode=mode,
                                                        b_nt=bnt,
                                                        k_batch=kb,
                                                        waves_per_eu=wpe,
                                                        xcd_swizzle=xcd,
                                                        use_cshuffle_epilog=use_cshuffle_epilog,
                                                        check=check,
                                                        profile=args.profile,
                                                        warmup=args.warmup,
                                                        iters=args.iters,
                                                        atol=args.atol,
                                                        rtol=args.rtol,
                                                        pass_pct=args.pass_pct,
                                                        trim_grid=not args.no_trim_grid,
                                                    )
                                                except Exception:
                                                    traceback.print_exc()
                                                    row = {
                                                        "stage": stage,
                                                        "token": token,
                                                        "model_dim": args.model_dim,
                                                        "inter_dim": args.inter_dim,
                                                        "experts": args.experts,
                                                        "topk": args.topk,
                                                        "block_m": bm,
                                                        "tile_n": tn,
                                                        "tile_k": tk,
                                                        "mode": mode,
                                                        "b_nt": bnt,
                                                        "k_batch": kb,
                                                        "waves_per_eu": wpe,
                                                        "xcd_swizzle": xcd,
                                                        "error": "1",
                                                    }
                                                    ok = False
                                                rows.append(row)
                                                if row.get("passed") == 0:
                                                    ok = False
                                else:
                                    for mode in args.mode:
                                        try:
                                            row = run_stage2_case(
                                                token=token,
                                                model_dim=args.model_dim,
                                                inter_dim=args.inter_dim,
                                                E=args.experts,
                                                topk=args.topk,
                                                block_m=bm,
                                                tile_n=tn,
                                                tile_k=tk,
                                                mode=mode,
                                                out_dtype=args.out_dtype,
                                                b_nt=bnt,
                                                xcd_swizzle=xcd,
                                                check=check,
                                                profile=args.profile,
                                                warmup=args.warmup,
                                                iters=args.iters,
                                                atol=args.atol,
                                                rtol=args.rtol,
                                                pass_pct=args.pass_pct,
                                            )
                                        except Exception:
                                            traceback.print_exc()
                                            row = {
                                                "stage": stage,
                                                "token": token,
                                                "model_dim": args.model_dim,
                                                "inter_dim": args.inter_dim,
                                                "experts": args.experts,
                                                "topk": args.topk,
                                                "block_m": bm,
                                                "tile_n": tn,
                                                "tile_k": tk,
                                                "mode": mode,
                                                "out_dtype": args.out_dtype,
                                                "b_nt": bnt,
                                                "xcd_swizzle": xcd,
                                                "error": "1",
                                            }
                                            ok = False
                                        rows.append(row)
                                        if row.get("passed") == 0:
                                            ok = False

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote CSV: {args.csv}")

    print("\nSummary")
    for row in rows:
        status = "ERROR" if row.get("error") else row.get("passed", "")
        print(
            f"  t={row['token']} bm={row['block_m']} tn={row['tile_n']} "
            f"tk={row['tile_k']} bnt={row['b_nt']} mean={row.get('mean_us', '')}us "
            f"median={row.get('median_us', '')}us check={status}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
