#!/usr/bin/env python3
"""Reproduce the task66 FP8 PTPC bandwidth target_us reference.

The task66 acceptance target is not a theoretical peak and not a torch memcpy
number. It is a FlyDSL load-only kernel timed through an inlined
profile_cuda_kernels-compatible torch.profiler path, using measured HBM bytes and
cache_modifier=2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


try:
    import torch
    from torch.profiler import ProfilerActivity, profile
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, buffer_ops, range_constexpr, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf as _scf_dialect
except Exception as exc:  # pragma: no cover - environment guard.
    raise RuntimeError(
        "bandwidth_reference.py requires torch and FlyDSL to be importable from "
        "the active Python environment. Install them in the environment or set "
        "PYTHONPATH before invoking the script; the script intentionally does "
        "not inject host-specific paths."
    ) from exc


BLOCK_THREADS = 256
VEC_DWORDS = 4
DEFAULT_TILES_PER_THREAD = 4
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50
DEFAULT_CACHE_MODIFIER = 2
_COMPILED_CACHE: dict[tuple[int, int], Any] = {}


def flush_cache(
    size_mb: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.int32,
    rounds: int = 2,
) -> torch.Tensor:
    """Minimal inlined tensor_tools.flush_cache equivalent."""
    n = (size_mb * 1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
    buf = torch.empty(n, device=device, dtype=dtype)
    for _ in range(rounds):
        buf.add_(1)
    torch.cuda.synchronize()
    return buf


def profile_cuda_kernels(
    fn,
    enable_print: bool = False,
    warmup: int = 5,
    iters: int = 20,
    topk: int = 50,
):
    """Minimal inlined tensor_tools.profile_cuda_kernels equivalent."""
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            torch.cuda.synchronize()
            out = fn()
            torch.cuda.synchronize()

    table_dict = {}
    for evt in prof.key_averages():
        table_dict[evt.key] = evt.device_time / 1e3
        if enable_print:
            print(f"perf result : name={evt.key} cost={evt.device_time / 1e3}")

    prof.key_averages().table(sort_by="cuda_time_total", row_limit=topk)
    return out, table_dict


TASK66_ROWS: tuple[dict[str, Any], ...] = (
    {
        "stage": "stage1",
        "token": 1,
        "measured_hbm_bytes": 21051392,
        "task66_ref_us": 8.164,
        "task66_target_us": 8.572,
    },
    {
        "stage": "stage1",
        "token": 16,
        "measured_hbm_bytes": 288119040,
        "task66_ref_us": 65.368,
        "task66_target_us": 68.636,
    },
    {
        "stage": "stage1",
        "token": 32,
        "measured_hbm_bytes": 513188365,
        "task66_ref_us": 114.223,
        "task66_target_us": 119.934,
    },
    {
        "stage": "stage1",
        "token": 64,
        "measured_hbm_bytes": 776474938,
        "task66_ref_us": 171.382,
        "task66_target_us": 179.951,
    },
    {
        "stage": "stage1",
        "token": 128,
        "measured_hbm_bytes": 996526080,
        "task66_ref_us": 219.546,
        "task66_target_us": 230.523,
    },
    {
        "stage": "stage1",
        "token": 256,
        "measured_hbm_bytes": 1069248493,
        "task66_ref_us": 235.358,
        "task66_target_us": 247.126,
    },
    {
        "stage": "stage1",
        "token": 512,
        "measured_hbm_bytes": 1090042579,
        "task66_ref_us": 245.740,
        "task66_target_us": 258.027,
    },
    {
        "stage": "stage2",
        "token": 1,
        "measured_hbm_bytes": 10866176,
        "task66_ref_us": 5.611,
        "task66_target_us": 5.892,
    },
    {
        "stage": "stage2",
        "token": 16,
        "measured_hbm_bytes": 150928525,
        "task66_ref_us": 35.613,
        "task66_target_us": 37.394,
    },
    {
        "stage": "stage2",
        "token": 32,
        "measured_hbm_bytes": 259199168,
        "task66_ref_us": 59.409,
        "task66_target_us": 62.379,
    },
    {
        "stage": "stage2",
        "token": 64,
        "measured_hbm_bytes": 399025210,
        "task66_ref_us": 89.574,
        "task66_target_us": 94.053,
    },
    {
        "stage": "stage2",
        "token": 128,
        "measured_hbm_bytes": 526315059,
        "task66_ref_us": 117.419,
        "task66_target_us": 123.290,
    },
    {
        "stage": "stage2",
        "token": 256,
        "measured_hbm_bytes": 583779091,
        "task66_ref_us": 130.424,
        "task66_target_us": 136.945,
    },
    {
        "stage": "stage2",
        "token": 512,
        "measured_hbm_bytes": 640056467,
        "task66_ref_us": 142.422,
        "task66_target_us": 149.543,
    },
)


def align_bytes(num_bytes: int, tiles_per_thread: int) -> tuple[int, int]:
    block_bytes = BLOCK_THREADS * tiles_per_thread * VEC_DWORDS * 4
    aligned = max(
        block_bytes,
        ((int(num_bytes) + block_bytes - 1) // block_bytes) * block_bytes,
    )
    return aligned, aligned // block_bytes


def build_flydsl_kernel(tiles_per_thread: int, cache_modifier: int):
    tile_stride = BLOCK_THREADS * VEC_DWORDS

    @flyc.kernel
    def bw_kernel(in_ptr: fx.Tensor, out_ptr: fx.Tensor, gate_ptr: fx.Tensor):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        in_rsrc = buffer_ops.create_buffer_resource(in_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        gate_rsrc = buffer_ops.create_buffer_resource(gate_ptr, max_size=True)

        gate_vec = buffer_ops.buffer_load(
            gate_rsrc,
            arith.constant(0, type=T.i32),
            vec_width=VEC_DWORDS,
            dtype=T.i32,
        )
        gate_scalar = vector.extract(gate_vec, [0])
        zero_i32 = arith.constant(0, type=T.i32)
        cond_op = arith.cmpi(1, ArithValue(gate_scalar), zero_i32)
        cond_val = cond_op.ir_value() if hasattr(cond_op, "ir_value") else cond_op

        block_dwords = tile_stride * tiles_per_thread
        base_dw = ArithValue(bid) * block_dwords + ArithValue(tid) * VEC_DWORDS
        vec4_i32 = T.vec(4, T.i32)
        final_acc = arith.constant_vector(0, vec4_i32)

        for t in range_constexpr(tiles_per_thread):
            offset_dw = base_dw + (t * tile_stride)
            data = buffer_ops.buffer_load(
                in_rsrc,
                offset_dw,
                vec_width=VEC_DWORDS,
                dtype=T.i32,
                cache_modifier=cache_modifier,
            )
            final_acc = final_acc ^ data
            if_op = _scf_dialect.IfOp(
                cond_val,
                [],
                has_else=False,
                loc=ir.Location.unknown(),
            )
            if len(if_op.regions[0].blocks) == 0:
                if_op.regions[0].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                buffer_ops.buffer_store(data, out_rsrc, offset_dw)
                _scf_dialect.YieldOp([])

        first_lane = vector.extract(final_acc, [0])
        tail_op = arith.cmpi(0, ArithValue(first_lane), ArithValue(gate_scalar))
        tail_val = tail_op.ir_value() if hasattr(tail_op, "ir_value") else tail_op
        tail_if = _scf_dialect.IfOp(
            tail_val,
            [],
            has_else=False,
            loc=ir.Location.unknown(),
        )
        if len(tail_if.regions[0].blocks) == 0:
            tail_if.regions[0].blocks.append(*[])
        with ir.InsertionPoint(tail_if.regions[0].blocks[0]):
            buffer_ops.buffer_store(final_acc, gate_rsrc, zero_i32)
            _scf_dialect.YieldOp([])

    @flyc.jit
    def launch(
        in_ptr: fx.Tensor,
        out_ptr: fx.Tensor,
        gate_ptr: fx.Tensor,
        grid_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        bw_kernel(in_ptr, out_ptr, gate_ptr).launch(
            grid=(grid_size, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def run_flydsl(call_args: tuple[Any, ...], launcher: Any, key: tuple[int, int]) -> None:
    compiled = _COMPILED_CACHE.get(key)
    if compiled is None:
        compiled = flyc.compile(launcher, *call_args)
        _COMPILED_CACHE[key] = compiled
    else:
        compiled(*call_args)


def measure_flydsl_load(
    num_bytes: int,
    cache_modifier: int,
    tiles_per_thread: int,
    warmup: int,
    iters: int,
    cold: bool,
) -> dict[str, Any]:
    aligned, grid_size = align_bytes(num_bytes, tiles_per_thread)
    num_i32 = aligned // 4
    inp = torch.randint(-128, 127, (num_i32,), dtype=torch.int32, device="cuda")
    out = torch.empty_like(inp)
    gate = torch.zeros(VEC_DWORDS, dtype=torch.int32, device="cuda")
    launcher = build_flydsl_kernel(tiles_per_thread, cache_modifier)
    stream = torch.cuda.current_stream()
    call_args = (inp, out, gate, grid_size, stream)
    key = (tiles_per_thread, cache_modifier)

    run_flydsl(call_args, launcher, key)
    torch.cuda.synchronize()
    for _ in range(warmup):
        run_flydsl(call_args, launcher, key)
    torch.cuda.synchronize()

    def bench_fn() -> None:
        if cold:
            flush_cache(1024)
        run_flydsl(call_args, launcher, key)

    _, report = profile_cuda_kernels(
        bench_fn,
        warmup=warmup,
        iters=iters,
        topk=80,
        enable_print=False,
    )
    time_ms = 0.0
    for kernel_name, kernel_ms in report.items():
        if "bw_kernel" in kernel_name.lower():
            time_ms = float(kernel_ms)
            break
    if time_ms <= 0.0:
        raise RuntimeError(f"bw_kernel not found in profile report: {list(report)}")

    time_us = time_ms * 1000.0
    return {
        "method": "flydsl_load_kernel_profile_cuda_kernels",
        "requested_bytes": int(num_bytes),
        "aligned_bytes": int(aligned),
        "hbm_bytes": int(aligned),
        "time_us": time_us,
        "bandwidth_tb_s": aligned / (time_us * 1e-6) / 1e12,
        "grid_size": grid_size,
        "tiles_per_thread": tiles_per_thread,
    }


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(TASK66_ROWS)
    if args.stage:
        stages = set(args.stage)
        rows = [row for row in rows if row["stage"] in stages]
    if args.tokens:
        tokens = set(args.tokens)
        rows = [row for row in rows if row["token"] in tokens]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def build_result_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for row in select_rows(args):
        result = measure_flydsl_load(
            int(row["measured_hbm_bytes"]),
            cache_modifier=args.cache_modifier,
            tiles_per_thread=args.tiles_per_thread,
            warmup=args.warmup,
            iters=args.iters,
            cold=not args.warm_cache,
        )
        measured_ref_us = float(result["time_us"])
        measured_target_us = measured_ref_us * 1.05
        task66_ref_us = float(row["task66_ref_us"])
        task66_target_us = float(row["task66_target_us"])
        ref_delta_pct = (measured_ref_us / task66_ref_us - 1.0) * 100.0
        target_delta_pct = (measured_target_us / task66_target_us - 1.0) * 100.0
        out = {
            "stage": row["stage"],
            "token": row["token"],
            "measured_hbm_bytes": row["measured_hbm_bytes"],
            "cache_modifier": args.cache_modifier,
            "mode": "loadonly",
            "tiles_per_thread": args.tiles_per_thread,
            "cold_cache": int(not args.warm_cache),
            "task66_ref_us": f"{task66_ref_us:.3f}",
            "measured_ref_us": f"{measured_ref_us:.3f}",
            "ref_delta_pct": f"{ref_delta_pct:+.2f}",
            "task66_target_us": f"{task66_target_us:.3f}",
            "measured_target_us": f"{measured_target_us:.3f}",
            "target_delta_pct": f"{target_delta_pct:+.2f}",
            "aligned_bytes": result["aligned_bytes"],
            "grid_size": result["grid_size"],
            "bandwidth_tb_s": f"{float(result['bandwidth_tb_s']):.3f}",
            "method": result["method"],
        }
        rows.append(out)
        print(
            "[bw] "
            f"{out['stage']} token={out['token']} bytes={out['measured_hbm_bytes']} "
            f"ref={out['measured_ref_us']}us target={out['measured_target_us']}us "
            f"target_delta={out['target_delta_pct']}%",
            file=sys.stderr,
            flush=True,
        )
    return rows


def write_csv_rows(path: Path | None, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "token",
        "measured_hbm_bytes",
        "cache_modifier",
        "mode",
        "tiles_per_thread",
        "cold_cache",
        "task66_ref_us",
        "measured_ref_us",
        "ref_delta_pct",
        "task66_target_us",
        "measured_target_us",
        "target_delta_pct",
        "aligned_bytes",
        "grid_size",
        "bandwidth_tb_s",
        "method",
    ]
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    deltas = [abs(float(row["target_delta_pct"])) for row in rows]
    within_1pct = sum(1 for delta in deltas if delta <= 1.0)
    max_abs = max(deltas) if deltas else 0.0
    lines = [
        "# Task66 Bandwidth Reference Retest",
        "",
        "Method: `measure_flydsl_load(measured_hbm_bytes, "
        f"cache_modifier={args.cache_modifier}, mode=\"loadonly\", "
        f"tiles_per_thread={args.tiles_per_thread}, warmup={args.warmup}, "
        f"iters={args.iters}, cold={not args.warm_cache})`.",
        "",
        "Timing source: `inlined profile_cuda_kernels -> bw_kernel` device time.",
        "",
        "Target formula: `target_us = 1.05 * measured_ref_us`.",
        "",
        "## Summary",
        "",
        "```text",
        f"rows: {len(rows)}",
        f"rows within 1% target delta: {within_1pct}/{len(rows)}",
        f"max absolute target delta: {max_abs:.2f}%",
        "```",
        "",
        "## Rows",
        "",
        "| stage | token | measured_hbm_bytes | task66_ref_us | measured_ref_us | "
        "ref_delta_pct | task66_target_us | measured_target_us | target_delta_pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['token']} | {row['measured_hbm_bytes']} | "
            f"{row['task66_ref_us']} | {row['measured_ref_us']} | "
            f"{row['ref_delta_pct']}% | {row['task66_target_us']} | "
            f"{row['measured_target_us']} | {row['target_delta_pct']}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce proj007 task66 FP8 PTPC bandwidth target_us rows."
    )
    parser.add_argument("--out", type=Path, help="Write CSV output to this path.")
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="Optional markdown summary path.",
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2"],
        nargs="+",
        help="Limit rows to one or more stages.",
    )
    parser.add_argument("--tokens", type=int, nargs="+", help="Limit rows by token.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected rows.")
    parser.add_argument(
        "--cache-modifier",
        type=int,
        choices=[0, 2],
        default=DEFAULT_CACHE_MODIFIER,
        help="Task66 target_us uses cache_modifier=2; cm0 is diagnostic only.",
    )
    parser.add_argument(
        "--tiles-per-thread",
        type=int,
        default=DEFAULT_TILES_PER_THREAD,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Disable per-iteration flush_cache. Not task66 target_us mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_modifier != DEFAULT_CACHE_MODIFIER:
        print(
            "warning: task66 target_us uses cache_modifier=2; this run is diagnostic",
            file=sys.stderr,
        )
    if args.tiles_per_thread != DEFAULT_TILES_PER_THREAD:
        print(
            "warning: task66 target_us uses tiles_per_thread=4; this run is diagnostic",
            file=sys.stderr,
        )
    if args.warm_cache:
        print(
            "warning: task66 target_us uses cold-cache timing with flush_cache",
            file=sys.stderr,
        )
    torch.set_default_device("cuda")
    rows = build_result_rows(args)
    write_csv_rows(args.out, rows)
    if args.markdown_out:
        write_markdown(args.markdown_out, rows, args)


if __name__ == "__main__":
    main()
