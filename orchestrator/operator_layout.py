"""Detection helpers for supported operator directory layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def is_sol_op(op_dir: Path) -> bool:
    """Return whether *op_dir* is a SOL-ExecBench operator."""
    return (op_dir / "definition.json").is_file() and (op_dir / "workload.jsonl").is_file()


def find_atrex_bench_root(op_dir: Path) -> Optional[Path]:
    """Return the canonical Atrex-Bench checkout owning a native shapes operator."""
    for candidate in (op_dir, *op_dir.parents):
        if (
            (candidate / "scripts" / "run_eval.py").is_file()
            and (candidate / "src" / "atrex_bench").is_dir()
        ):
            return candidate
    return None
