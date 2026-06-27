#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NVIDIA SASS/PTX Extraction and Analysis Tool

Extracts SASS (native GPU assembly) from compiled NVIDIA kernels and analyzes
key instruction patterns. Counterpart to the AMD-side extract_asm.py.

Usage:
    # Extract SASS from .ncu-rep (recommended, especially for CuteDSL kernels)
    python tools/extract_nvidia_asm.py --ncu-rep profiles/v1/ncu.ncu-rep --check-all

    # Extract from a kernel file (Triton backend)
    python tools/extract_nvidia_asm.py <kernel.py> --check-all

    # Extract from an existing cubin / .so
    python tools/extract_nvidia_asm.py --cubin kernel.cubin --check-all --arch sm90

    # Analyze an existing SASS text file
    python tools/extract_nvidia_asm.py --asm-file kernel.sass --check-all

Recommended workflow for CuteDSL:
    1. bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1
    2. python tools/extract_nvidia_asm.py --ncu-rep profiles/v1/ncu.ncu-rep --check-all

    Reason: CuteDSL compiles via NVRTC, and the cubin cache location is not fixed.
    Extracting SASS from .ncu-rep (action.sass_by_pc()) is the most reliable approach.
    To retain PTX, set CUTE_DSL_KEEP_PTX=1 or use cute.compile[cute.KeepPTX()].

Dependencies:
    cuobjdump (CUDA Toolkit) -- extract SASS from cubin/.so
    ncu_report (CUDA Toolkit) -- extract SASS from .ncu-rep (optional, for --ncu-rep mode)

Security:
    The Triton and CuTeDSL backends import and run the target kernel file in
    this interpreter (importlib exec_module) to trigger JIT compilation. This
    executes arbitrary code from `kernel.py`. The tool assumes the kernel file
    is trusted (authored locally or in the workspace) — do not point it at
    untrusted .py files. The --ncu-rep / --cubin / --asm-file modes do NOT
    execute the kernel and are safe for untrusted inputs.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ============================================================
# SASS Extraction
# ============================================================

def extract_sass_from_cubin(cubin_path):
    """Extract SASS from a .cubin file."""
    if not shutil.which("cuobjdump"):
        raise FileNotFoundError(
            "cuobjdump not found. Ensure CUDA Toolkit is installed and on PATH\n"
            "  Common path: /usr/local/cuda/bin/cuobjdump"
        )
    result = subprocess.run(
        ["cuobjdump", "--dump-sass", str(cubin_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {result.stderr}")
    return result.stdout


def extract_sass_from_so(so_path):
    """Extract SASS from a .so file (for nvcc-compiled CUDA/CUTLASS kernels)."""
    return extract_sass_from_cubin(so_path)


def extract_ptxas_stats(cubin_path):
    """Try to get ptxas compilation statistics (register count, etc.) from a cubin.

    Note: ptxas -v output is only available at compile time. For an existing cubin,
    use cuobjdump --dump-resource-usage to get similar information.
    """
    if not shutil.which("cuobjdump"):
        return None
    result = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(cubin_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _find_cubin(cache_dir):
    """Find a .cubin file anywhere under a compilation-cache dir tree.

    Generic over backends — used for both the Triton (TRITON_CACHE_DIR) and the
    CuTeDSL (NVRTC temp-dir) extraction paths, so the name is backend-neutral.
    """
    for root, dirs, files in os.walk(cache_dir):
        for fname in files:
            if fname.endswith(".cubin"):
                return os.path.join(root, fname)
    return None


def _warn_kernel_exec(kernel_file):
    """Emit a runtime warning before importing/executing a user kernel file.

    The Triton and CuTeDSL SASS paths must run the kernel to trigger JIT
    compilation, which executes arbitrary Python from `kernel_file`. Surface that
    at runtime so it is never silent; see the module docstring 'Security' note.
    """
    print(
        f"WARNING: executing '{kernel_file}' to trigger kernel compilation — "
        "this runs arbitrary Python. Use --ncu-rep mode for untrusted inputs.",
        file=sys.stderr,
    )


def extract_sass_triton(kernel_file):
    """Extract SASS from a Triton kernel: set temp cache dir, execute, find cubin, cuobjdump."""
    dump_dir = tempfile.mkdtemp()
    original_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = dump_dir

    try:
        spec = importlib.util.spec_from_file_location("_kernel_mod", kernel_file)
        if spec is None:
            raise ImportError(f"Unable to load module: {kernel_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_kernel_mod"] = module
        # SECURITY: runs arbitrary code from kernel_file; trusted-input only
        # (see module docstring). Use --ncu-rep mode for untrusted kernels.
        _warn_kernel_exec(kernel_file)
        spec.loader.exec_module(module)

        cubin_path = _find_cubin(dump_dir)
        if not cubin_path:
            raise FileNotFoundError(
                f".cubin file not found in Triton cache\n"
                f"  Cache directory: {dump_dir}"
            )
        sass = extract_sass_from_cubin(cubin_path)
        resource = extract_ptxas_stats(cubin_path)
        return sass, resource
    finally:
        sys.modules.pop("_kernel_mod", None)
        if original_cache_dir is not None:
            os.environ["TRITON_CACHE_DIR"] = original_cache_dir
        else:
            os.environ.pop("TRITON_CACHE_DIR", None)
        shutil.rmtree(dump_dir, ignore_errors=True)


def extract_sass_from_ncu_rep(ncu_rep_path):
    """Extract SASS from an .ncu-rep file (Method 1: most reliable).

    The ncu Python API provides action.sass_by_pc(), which can directly extract
    complete SASS instructions from a profile report without needing a cubin file.

    Requirement: first collect .ncu-rep using profile_nvidia.sh or ncu --set full.
    """
    ncu_helpers_dir = _find_ncu_helpers()
    if not ncu_helpers_dir:
        raise RuntimeError(
            f"Unable to extract SASS from {ncu_rep_path}: ncu_helpers/ not found "
            "(need ncu_utils.py; set NCU_HELPERS to override)."
        )

    inserted = ncu_helpers_dir not in sys.path
    if inserted:
        sys.path.insert(0, ncu_helpers_dir)
    try:
        from ncu_utils import load_action  # imports ncu_report lazily on first load
        action = load_action(ncu_rep_path)
        sass_by_pc = action.sass_by_pc()
    except Exception as e:
        # Surface the real cause (e.g. missing/incompatible ncu_report) instead
        # of swallowing it behind a generic message.
        raise RuntimeError(
            f"Unable to extract SASS from {ncu_rep_path}: {type(e).__name__}: {e}\n"
            "  Ensure the ncu_report Python module is available (CUDA Toolkit)."
        ) from e
    finally:
        if inserted:
            try:
                sys.path.remove(ncu_helpers_dir)
            except ValueError:
                pass

    if not sass_by_pc:
        raise RuntimeError(
            f"No SASS found in {ncu_rep_path} (action.sass_by_pc() was empty); "
            "recollect with line info (ncu --set full / --import-source yes)."
        )
    lines = []
    for pc in sorted(sass_by_pc.keys()):
        lines.append(f"        /*{pc:04x}*/ {sass_by_pc[pc]} ;")
    return "\n".join(lines), None


def extract_sass_cutedsl(kernel_file):
    """Extract SASS from a CuteDSL kernel.

    CuteDSL compiles to cubin via cute.compile() using NVRTC.
    Three strategies are provided, sorted by reliability:

    Method 1 (recommended): Use --ncu-rep to extract SASS from an existing .ncu-rep
      - First run profile_nvidia.sh to collect .ncu-rep
      - Then use extract_nvidia_asm.py --ncu-rep profiles/v1/ncu.ncu-rep --check-all
      - The ncu Python API's action.sass_by_pc() directly returns SASS

    Method 2: Use cute.compile[cute.KeepPTX()] to retain PTX
      - Set the environment variable CUTE_DSL_KEEP_PTX=1
      - cute.compile will keep .ptx files in the compilation directory
      - Then compile PTX to cubin with ptxas, and finally extract SASS with cuobjdump

    Method 3 (current implementation): Execute the kernel to trigger compilation, search for .cubin in temp directories
      - Not very reliable, because the cute.compile cache location is not publicly documented
    """
    # Method 3: try searching the compilation cache
    dump_dir = tempfile.mkdtemp()
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"

    try:
        spec = importlib.util.spec_from_file_location("_kernel_mod", kernel_file)
        if spec is None:
            raise ImportError(f"Unable to load module: {kernel_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_kernel_mod"] = module
        # SECURITY: runs arbitrary code from kernel_file; trusted-input only
        # (see module docstring). Use --ncu-rep mode for untrusted kernels.
        _warn_kernel_exec(kernel_file)
        spec.loader.exec_module(module)

        # Search for cubin and ptx ONLY under our private dump_dir. Scanning the
        # shared system temp dir (tempfile.gettempdir()) would pick up stale or
        # other-process .cubin/.ptx files and silently analyze the wrong kernel.
        cubin_path = _find_cubin(dump_dir)
        ptx_path = None
        for root, dirs, files in os.walk(dump_dir):
            for fname in files:
                if fname.endswith(".ptx"):
                    ptx_path = os.path.join(root, fname)
                    break
            if ptx_path:
                break

        if cubin_path:
            sass = extract_sass_from_cubin(cubin_path)
            resource = extract_ptxas_stats(cubin_path)
            return sass, resource

        if ptx_path:
            print(f"Found PTX file: {ptx_path}")
            print("  Tip: use ptxas to compile PTX to cubin, then cuobjdump to extract SASS")
            with open(ptx_path, "r") as f:
                return f"[PTX content - requires ptxas to compile to cubin]\n{f.read()}", None

        raise FileNotFoundError(
            "CuteDSL compilation artifacts (.cubin or .ptx) not found.\n"
            "\n"
            "Recommended approach:\n"
            "  1. First collect .ncu-rep with profile_nvidia.sh:\n"
            "       bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1\n"
            "     Then extract SASS from .ncu-rep:\n"
            "       python tools/extract_nvidia_asm.py --ncu-rep profiles/v1/ncu.ncu-rep --check-all\n"
            "\n"
            "  2. Or use cute.compile with options='--generate-line-info',\n"
            "     and use cute.compile[cute.KeepPTX()] in code to retain PTX.\n"
            "     Reference: CUTLASS PR-3091 examples/python/CuTeDSL/hopper/grouped_gemm.py"
        )
    finally:
        sys.modules.pop("_kernel_mod", None)
        os.environ.pop("CUTE_DSL_KEEP_PTX", None)
        shutil.rmtree(dump_dir, ignore_errors=True)


def _find_ncu_helpers():
    """Detect the ncu report helpers path (bundled copy ships in tools/ncu_helpers/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, "ncu_helpers"),
        os.path.expanduser("~/.claude/skills/ncu-report-skill/helpers"),
        os.path.expanduser("~/.config/opencode/skills/ncu-report-skill/helpers"),
        os.path.expanduser("~/.codex/skills/ncu-report-skill/helpers"),
    ]
    for p in search_paths:
        if os.path.isfile(os.path.join(p, "ncu_utils.py")):
            return os.path.abspath(p)
    return None


def detect_backend(kernel_file):
    """Infer backend type from kernel file content."""
    with open(kernel_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(4096)
    if "cutlass.cute" in content or "cute.compile" in content or "from cutlass" in content:
        return "cutedsl"
    if "import triton" in content or "@triton.jit" in content or "@gluon" in content:
        return "triton"
    return "cuda"


def extract_sass(kernel_file, backend="auto"):
    """Main extraction entry point. Returns (sass_content, resource_usage_or_none)."""
    if backend == "auto":
        backend = detect_backend(kernel_file)
        print(f"Detected backend: {backend}")

    if backend == "triton":
        return extract_sass_triton(kernel_file)
    elif backend == "cutedsl":
        return extract_sass_cutedsl(kernel_file)
    elif backend == "cuda":
        raise NotImplementedError(
            "CUDA/CUTLASS backend requires compiling to .so via compile_cu.py first,\n"
            "  then use --cubin to specify the .so path."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# SASS Analysis Functions
# ============================================================

def analyze_spills(sass_content, resource_content=None):
    """Detect register spill / local memory usage.

    In SASS:
    - STL (store local) / LDL (load local) indicate spills to local memory
    - A large number of STL/LDL means registers are insufficient
    """
    stl_count = len(re.findall(r"\bSTL\b", sass_content))
    ldl_count = len(re.findall(r"\bLDL\b", sass_content))

    # Extract register count and smem from resource usage
    reg_count = None
    smem_bytes = None
    spill_stores = None
    spill_loads = None

    if resource_content:
        reg_match = re.search(r"REG:(\d+)", resource_content)
        if reg_match:
            reg_count = int(reg_match.group(1))

        smem_match = re.search(r"SMEM:(\d+)", resource_content)
        if smem_match:
            smem_bytes = int(smem_match.group(1))

        spill_st_match = re.search(r"spill_store:(\d+)", resource_content, re.IGNORECASE)
        if spill_st_match:
            spill_stores = int(spill_st_match.group(1))

        spill_ld_match = re.search(r"spill_load:(\d+)", resource_content, re.IGNORECASE)
        if spill_ld_match:
            spill_loads = int(spill_ld_match.group(1))

    return {
        "stl_count": stl_count,
        "ldl_count": ldl_count,
        "register_count": reg_count,
        "smem_bytes": smem_bytes,
        "spill_stores": spill_stores,
        "spill_loads": spill_loads,
        "has_spills": stl_count > 0 or ldl_count > 0,
    }


# SM90 (Hopper) expected instructions
SM90_TENSOR_INSTRUCTIONS = [
    "HMMA",     # Half-precision MMA
    "WGMMA",    # Warp-group MMA (Hopper)
]
SM90_ASYNC_INSTRUCTIONS = [
    "LDGSTS",   # Load global, store shared (async)
    "LDSM",     # Load shared matrix (ldmatrix)
    "CPASYNC",  # cp.async
]

# SM100 (Blackwell) expected instructions
# (ref: gpu-wiki/docs/kernel-opt/nvidia/common/blackwell/languages/ptx-sm100.md)
SM100_TENSOR_INSTRUCTIONS = [
    "TCGEN05",  # tcgen05.mma / tcgen05.ld / tcgen05.st / tcgen05.cp
    "GMMA",     # Blackwell GMMA
]
SM100_ASYNC_INSTRUCTIONS = [
    "UTMALDG",  # TMA load (Blackwell SASS encoding)
    "ULDGSTS",  # Bulk async copy
    "LDSM",     # Load shared matrix
]

# Common expected instructions
COMMON_EXPECTED = [
    "MMA",      # Generic MMA
    "TMA",      # Tensor Memory Accelerator
]


def analyze_expected_instructions(sass_content, arch="sm90"):
    """Check whether expected tensor core / async instructions are present."""
    if arch.startswith("sm100") or arch.startswith("sm_100"):
        tensor_expected = SM100_TENSOR_INSTRUCTIONS
        async_expected = SM100_ASYNC_INSTRUCTIONS
    else:
        tensor_expected = SM90_TENSOR_INSTRUCTIONS
        async_expected = SM90_ASYNC_INSTRUCTIONS

    results = {}

    for instr in tensor_expected + async_expected + COMMON_EXPECTED:
        pattern = re.compile(rf"\b{re.escape(instr)}\b", re.IGNORECASE)
        matches = pattern.findall(sass_content)
        results[instr] = len(matches)

    tensor_found = [i for i in tensor_expected if results.get(i, 0) > 0]
    async_found = [i for i in async_expected if results.get(i, 0) > 0]

    return {
        "arch": arch,
        "instruction_counts": results,
        "tensor_found": tensor_found,
        "async_found": async_found,
        "has_tensor_core": len(tensor_found) > 0 or results.get("MMA", 0) > 0,
        "has_async_copy": len(async_found) > 0 or results.get("TMA", 0) > 0,
    }


# Precompiled once and reused per line in analyze_load_width's hot loop
# (avoids re-parsing the same patterns for every SASS line).
_LDW_RE = {
    "ldg128": re.compile(r"\bLDG\.E\.128\b"),
    "ldg64": re.compile(r"\bLDG\.E\.64\b"),
    "ldg": re.compile(r"\bLDG\.E\b"),
    "stg128": re.compile(r"\bSTG\.E\.128\b"),
    "stg64": re.compile(r"\bSTG\.E\.64\b"),
    "stg": re.compile(r"\bSTG\.E\b"),
    "lds128": re.compile(r"\bLDS\.128\b"),
    "lds64": re.compile(r"\bLDS\.64\b"),
    "lds": re.compile(r"\bLDS\b"),
    "ldsm": re.compile(r"\bLDSM\b"),
    "sts128": re.compile(r"\bSTS\.128\b"),
    "sts64": re.compile(r"\bSTS\.64\b"),
    "sts": re.compile(r"\bSTS\b"),
}


def analyze_load_width(sass_content):
    """Analyze global / shared / local load/store instruction width.

    SASS instruction format:
    - LDG.E / LDG.E.64 / LDG.E.128 / LDG.E.SYS  (global load)
    - STG.E / STG.E.64 / STG.E.128                 (global store)
    - LDS / LDS.64 / LDS.128                       (shared load)
    - STS / STS.64 / STS.128                       (shared store)
    """
    results = {
        "global_load": {"32": 0, "64": 0, "128": 0, "other": 0},
        "global_store": {"32": 0, "64": 0, "128": 0, "other": 0},
        "shared_load": {"32": 0, "64": 0, "128": 0, "other": 0},
        "shared_store": {"32": 0, "64": 0, "128": 0, "other": 0},
    }

    for line in sass_content.split("\n"):
        stripped = line.strip()

        # Global loads
        if _LDW_RE["ldg128"].search(stripped):
            results["global_load"]["128"] += 1
        elif _LDW_RE["ldg64"].search(stripped):
            results["global_load"]["64"] += 1
        elif _LDW_RE["ldg"].search(stripped):
            results["global_load"]["32"] += 1

        # Global stores
        if _LDW_RE["stg128"].search(stripped):
            results["global_store"]["128"] += 1
        elif _LDW_RE["stg64"].search(stripped):
            results["global_store"]["64"] += 1
        elif _LDW_RE["stg"].search(stripped):
            results["global_store"]["32"] += 1

        # Shared loads
        if _LDW_RE["lds128"].search(stripped):
            results["shared_load"]["128"] += 1
        elif _LDW_RE["lds64"].search(stripped):
            results["shared_load"]["64"] += 1
        elif _LDW_RE["lds"].search(stripped) and not _LDW_RE["ldsm"].search(stripped):
            results["shared_load"]["32"] += 1

        # Shared stores
        if _LDW_RE["sts128"].search(stripped):
            results["shared_store"]["128"] += 1
        elif _LDW_RE["sts64"].search(stripped):
            results["shared_store"]["64"] += 1
        elif _LDW_RE["sts"].search(stripped):
            results["shared_store"]["32"] += 1

    return results


def analyze_scalar_fallback(sass_content):
    """Detect scalar fallback (excessive FMUL/FADD instead of tensor core)."""
    fmul = len(re.findall(r"\bFMUL\b", sass_content))
    fadd = len(re.findall(r"\bFFMA\b", sass_content))
    hmma = len(re.findall(r"\bHMMA\b", sass_content))
    wgmma = len(re.findall(r"\bWGMMA\b", sass_content))
    mma = len(re.findall(r"\bMMA\b", sass_content))
    tcgen = len(re.findall(r"\bTCGEN05\b", sass_content, re.IGNORECASE))

    tensor_total = hmma + wgmma + mma + tcgen
    scalar_total = fmul + fadd

    return {
        "scalar_fmul": fmul,
        "scalar_ffma": fadd,
        "tensor_hmma": hmma,
        "tensor_wgmma": wgmma,
        "tensor_mma": mma,
        "tensor_tcgen05": tcgen,
        "scalar_total": scalar_total,
        "tensor_total": tensor_total,
        "is_scalar_heavy": scalar_total > tensor_total * 5 and tensor_total > 0,
        "no_tensor_at_all": tensor_total == 0 and scalar_total > 0,
    }


def analyze_instruction_mix(sass_content):
    """Instruction classification breakdown statistics."""
    categories = {
        "compute_tensor": 0,    # HMMA, WGMMA, MMA, TCGEN05
        "compute_scalar": 0,    # FMUL, FFMA, FADD, IMAD, IADD
        "memory_global": 0,     # LDG, STG
        "memory_shared": 0,     # LDS, STS, LDSM
        "memory_local": 0,      # LDL, STL
        "memory_async": 0,      # LDGSTS, CPASYNC, TMA
        "control": 0,           # BRA, JMP, EXIT, SYNC, BAR, RET
        "other": 0,
    }

    instruction_pattern = re.compile(r"/\*[^*]*\*/\s+([A-Z][A-Z0-9_.]+)")

    for line in sass_content.split("\n"):
        m = instruction_pattern.search(line)
        if not m:
            continue
        instr = m.group(1).split(".")[0]

        if instr in ("HMMA", "WGMMA", "MMA", "TCGEN05", "GMMA"):
            categories["compute_tensor"] += 1
        elif instr in ("FMUL", "FFMA", "FADD", "FMNMX", "FSET",
                        "IMAD", "IADD3", "IADD", "ISETP", "IMNMX",
                        "LOP3", "SHF", "SHL", "SHR", "PRMT",
                        "MUFU", "HFMA2", "HMUL2", "HADD2"):
            categories["compute_scalar"] += 1
        elif instr in ("LDG", "STG", "ATOMG", "REDG"):
            categories["memory_global"] += 1
        elif instr in ("LDS", "STS", "LDSM", "ATOMS", "REDS"):
            categories["memory_shared"] += 1
        elif instr in ("LDL", "STL"):
            categories["memory_local"] += 1
        elif instr in ("LDGSTS", "CPASYNC", "UTMALDG", "ULDGSTS"):
            categories["memory_async"] += 1
        elif instr in ("BRA", "JMP", "EXIT", "RET", "SYNC",
                        "BAR", "BSYNC", "BSSY", "YIELD",
                        "WARPSYNC", "NANOSLEEP", "NOP"):
            categories["control"] += 1
        else:
            categories["other"] += 1

    total = sum(categories.values())

    return {
        "counts": categories,
        "total": total,
        "percentages": {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in categories.items()
        },
    }


# ============================================================
# Report Output
# ============================================================

def print_analysis(sass_content, resource_content=None, arch="sm90"):
    """Print the full analysis report."""
    print(f"{'=' * 60}")
    print(f"  NVIDIA SASS Analysis Report (arch: {arch})")
    print(f"{'=' * 60}")

    # Spills
    spills = analyze_spills(sass_content, resource_content)
    print(f"\n--- Register Spill Analysis ---")
    if spills["register_count"] is not None:
        print(f"  Register usage: {spills['register_count']}")
    if spills["smem_bytes"] is not None:
        print(f"  Shared memory: {spills['smem_bytes']} bytes")
    if spills["has_spills"]:
        print(f"  ❌ Spills detected! STL: {spills['stl_count']}, LDL: {spills['ldl_count']}")
    else:
        print(f"  ✅ No spills (STL: 0, LDL: 0)")

    # Expected instructions
    expected = analyze_expected_instructions(sass_content, arch)
    print(f"\n--- Expected Instruction Check ({arch}) ---")
    for instr, count in expected["instruction_counts"].items():
        if count > 0:
            print(f"  ✅ {instr}: {count}")
        else:
            print(f"  ⚠️  {instr}: 0 (not detected)")
    if expected["has_tensor_core"]:
        print(f"  ✅ Tensor core instructions present: {', '.join(expected['tensor_found']) or 'MMA'}")
    else:
        print(f"  ❌ Tensor core instructions not detected")
    if expected["has_async_copy"]:
        print(f"  ✅ Async copy instructions present: {', '.join(expected['async_found']) or 'TMA'}")
    else:
        print(f"  ⚠️  Async copy instructions not detected")

    # Load width
    widths = analyze_load_width(sass_content)
    print(f"\n--- Instruction Width Analysis ---")
    for op_type, counts in widths.items():
        total = sum(counts.values())
        if total > 0:
            print(f"\n  {op_type} ({total} total):")
            for w, c in counts.items():
                if c > 0:
                    marker = "✅" if w == "128" else ("⚠️" if w == "32" else "  ")
                    print(f"    {marker} {w}-bit: {c}")

    # Scalar fallback
    scalar = analyze_scalar_fallback(sass_content)
    print(f"\n--- Scalar Fallback Detection ---")
    print(f"  Tensor core instructions: {scalar['tensor_total']} "
          f"(HMMA:{scalar['tensor_hmma']}, WGMMA:{scalar['tensor_wgmma']}, "
          f"MMA:{scalar['tensor_mma']}, TCGEN05:{scalar['tensor_tcgen05']})")
    print(f"  Scalar arithmetic instructions: {scalar['scalar_total']} "
          f"(FMUL:{scalar['scalar_fmul']}, FFMA:{scalar['scalar_ffma']})")
    if scalar["no_tensor_at_all"]:
        print(f"  ❌ No tensor core instructions at all, entirely scalar path")
    elif scalar["is_scalar_heavy"]:
        print(f"  ⚠️  Scalar instructions far exceed tensor core ({scalar['scalar_total']} vs {scalar['tensor_total']})")
    else:
        print(f"  ✅ Tensor/scalar ratio is normal")

    # Instruction mix
    mix = analyze_instruction_mix(sass_content)
    print(f"\n--- Instruction Classification Breakdown ---")
    print(f"  Total instructions: {mix['total']}")
    for cat, pct in mix["percentages"].items():
        count = mix["counts"][cat]
        if count > 0:
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {cat:<20s} {bar} {pct:5.1f}% ({count})")

    print(f"\n{'=' * 60}")


def print_json(sass_content, resource_content=None, arch="sm90"):
    """Output analysis results in JSON format."""
    result = {
        "arch": arch,
        "spills": analyze_spills(sass_content, resource_content),
        "expected_instructions": analyze_expected_instructions(sass_content, arch),
        "load_width": analyze_load_width(sass_content),
        "scalar_fallback": analyze_scalar_fallback(sass_content),
        "instruction_mix": analyze_instruction_mix(sass_content),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA SASS/PTX extraction and analysis tool")
    parser.add_argument("kernel", nargs="?", help="Kernel source file")
    parser.add_argument("-o", "--output", help="Save SASS to file")
    parser.add_argument("--ptx", action="store_true",
                        help="Also extract PTX (requires cuobjdump --dump-ptx)")
    parser.add_argument("--backend", choices=["cutedsl", "cuda", "triton", "auto"],
                        default="auto", help="Backend type (default: auto)")
    parser.add_argument("--cubin", help="Directly analyze an existing cubin / .so file")
    parser.add_argument("--ncu-rep",
                        help="Extract SASS from .ncu-rep (recommended for CuteDSL kernels, "
                             "first collect with profile_nvidia.sh)")
    parser.add_argument("--asm-file", help="Directly analyze an existing SASS text file (skip extraction)")
    parser.add_argument("--arch", default="sm90",
                        help="Architecture (sm90/sm100), affects expected instruction set (default: sm90)")
    parser.add_argument("--check-spills", action="store_true",
                        help="Check for register spill / local memory")
    parser.add_argument("--check-instructions", action="store_true",
                        help="Check for expected instructions (GMMA/HMMA/CPASYNC, etc.)")
    parser.add_argument("--check-load-width", action="store_true",
                        help="Check load/store instruction width")
    parser.add_argument("--check-scalar", action="store_true",
                        help="Check for scalar fallback")
    parser.add_argument("--check-mix", action="store_true",
                        help="Instruction classification breakdown")
    parser.add_argument("--check-all", action="store_true",
                        help="Run all checks")
    parser.add_argument("--json", action="store_true",
                        help="Output in JSON format")
    args = parser.parse_args()

    # Get SASS content
    resource_content = None

    if args.asm_file:
        with open(args.asm_file, "r") as f:
            sass_content = f.read()
    elif args.ncu_rep:
        sass_content, resource_content = extract_sass_from_ncu_rep(args.ncu_rep)
    elif args.cubin:
        sass_content = extract_sass_from_cubin(args.cubin)
        resource_content = extract_ptxas_stats(args.cubin)
    elif args.kernel:
        sass_content, resource_content = extract_sass(args.kernel, args.backend)
    else:
        parser.error("Must specify a kernel file, --cubin, --ncu-rep, or --asm-file")

    # Save SASS
    if args.output:
        with open(args.output, "w") as f:
            f.write(sass_content)
        print(f"SASS saved to: {args.output}")

    # Save PTX
    if args.ptx and args.cubin:
        if shutil.which("cuobjdump"):
            result = subprocess.run(
                ["cuobjdump", "--dump-ptx", args.cubin],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                ptx_path = (args.output or "kernel") + ".ptx"
                with open(ptx_path, "w") as f:
                    f.write(result.stdout)
                print(f"PTX saved to: {ptx_path}")

    # Analysis
    if args.check_all:
        if args.json:
            print_json(sass_content, resource_content, args.arch)
        else:
            print_analysis(sass_content, resource_content, args.arch)
    else:
        any_check = False
        if args.check_spills:
            any_check = True
            spills = analyze_spills(sass_content, resource_content)
            if args.json:
                print(json.dumps({"spills": spills}, indent=2))
            else:
                print(f"spills: STL={spills['stl_count']}, LDL={spills['ldl_count']}, "
                      f"regs={spills['register_count']}, has_spills={spills['has_spills']}")

        if args.check_instructions:
            any_check = True
            expected = analyze_expected_instructions(sass_content, args.arch)
            if args.json:
                print(json.dumps({"expected_instructions": expected}, indent=2))
            else:
                for instr, count in expected["instruction_counts"].items():
                    print(f"{instr}: {count}")

        if args.check_load_width:
            any_check = True
            widths = analyze_load_width(sass_content)
            if args.json:
                print(json.dumps({"load_width": widths}, indent=2))
            else:
                for op_type, counts in widths.items():
                    total = sum(counts.values())
                    if total > 0:
                        print(f"{op_type}: {counts}")

        if args.check_scalar:
            any_check = True
            scalar = analyze_scalar_fallback(sass_content)
            if args.json:
                print(json.dumps({"scalar_fallback": scalar}, indent=2))
            else:
                print(f"tensor: {scalar['tensor_total']}, scalar: {scalar['scalar_total']}, "
                      f"scalar_heavy: {scalar['is_scalar_heavy']}")

        if args.check_mix:
            any_check = True
            mix = analyze_instruction_mix(sass_content)
            if args.json:
                print(json.dumps({"instruction_mix": mix}, indent=2))
            else:
                for cat, pct in mix["percentages"].items():
                    if mix["counts"][cat] > 0:
                        print(f"{cat}: {pct}% ({mix['counts'][cat]})")

        if not any_check:
            if args.json:
                print_json(sass_content, resource_content, args.arch)
            else:
                print_analysis(sass_content, resource_content, args.arch)


if __name__ == "__main__":
    main()
