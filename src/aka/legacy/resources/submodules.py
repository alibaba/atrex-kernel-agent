"""Resources submodules implementation and callable interface."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from aka.legacy.resources.paths import REPO_ROOT
from aka.legacy.task.hardware import hardware_vendor


def ensure_submodules(platform: str = "", arch: str = "") -> None:
    """Initialize submodules required by the optimization pipeline.

    Covers 3rdparty/ncu-report-skill. Optional plugin skills are installed when available.
    PPU campaigns also require their vendor reference projects: without those
    working trees the framework-baseline catalog silently contains no usable
    PPU implementation sources.
    Idempotent: already-initialized submodules are untouched.
    """
    needed = [
        (
            "3rdparty/ncu-report-skill",
            REPO_ROOT / "3rdparty" / "ncu-report-skill" / "SKILL.md",
        ),
    ]
    if hardware_vendor(platform, arch) == "ppu":
        needed.extend(
            (path, REPO_ROOT / path / "README.md")
            for path in (
                "reference-projects/sailify",
                "reference-projects/hggc-samples",
                "reference-projects/FlashMLA-for-sail",
                "reference-projects/DeepGEMM-for-sail",
                "reference-projects/flash-attention-for-sail",
                "reference-projects/actlize",
                "reference-projects/triton-for-sail",
            )
        )
    # Internal launchers may expose a generated repository view without independent Git
    # metadata. Run submodule commands in the recorded open-source checkout while keeping
    # the current branch's intentionally small required-submodule set.
    submodule_root = REPO_ROOT
    runtime_metadata = REPO_ROOT / ".internal-runtime.json"
    if runtime_metadata.is_file():
        try:
            metadata = json.loads(runtime_metadata.read_text(encoding="utf-8"))
            recorded_root = metadata.get("open_root")
            if not isinstance(recorded_root, str) or not recorded_root.strip():
                raise ValueError("open_root is missing")
            submodule_root = Path(recorded_root).expanduser().resolve()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid internal runtime metadata: {runtime_metadata}: {exc}"
            ) from exc
    to_init = [path for path, marker in needed if not marker.exists()]
    if to_init:
        print(f"[orchestrator] initializing submodules: {to_init}", flush=True)
        cmd = ["git", "submodule", "update", "--init", "--depth", "1", "--"] + to_init
        subprocess.run(cmd, cwd=str(submodule_root), check=True)
        # verify
        for path, marker in needed:
            if not marker.exists():
                raise RuntimeError(
                    f"submodule init failed for {path} — {marker} not found. "
                    "Run `git submodule update --init` manually."
                )
        print("[orchestrator] all submodules ready", flush=True)

