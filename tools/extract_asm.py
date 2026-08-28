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
Assembly extraction and analysis tool

Extract AMDGPU assembly from a compiled Gluon kernel and analyze key instruction patterns.

Usage:
    python tools/extract_asm.py <kernel.py> -o kernel.asm
    python tools/extract_asm.py <kernel.py> --check-load-width
    python tools/extract_asm.py <kernel.py> --check-all
"""

import argparse
import sys
import re
import os
import tempfile
import shutil
import importlib.util


def extract_asm(kernel_file, output_file=None):
    """
    Extract AMDGCN assembly from a Gluon kernel.

    Run the kernel with a temporary TRITON_CACHE_DIR,
    then search the compilation cache for .amdgcn files.
    """
    dump_dir = tempfile.mkdtemp()
    original_cache_dir = os.environ.get('TRITON_CACHE_DIR')
    os.environ['TRITON_CACHE_DIR'] = dump_dir

    try:
        spec = importlib.util.spec_from_file_location("_kernel_mod", kernel_file)
        if spec is None:
            raise ImportError(f"Unable to load module spec: {kernel_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["_kernel_mod"] = module
        spec.loader.exec_module(module)

        asm_content = None
        asm_path = None
        for root, dirs, files in os.walk(dump_dir):
            for fname in files:
                if fname.endswith('.amdgcn') or '.amdgcn' in fname:
                    asm_path = os.path.join(root, fname)
                    with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                        asm_content = f.read()
                    break
            if asm_content:
                break

        if not asm_content:
            raise FileNotFoundError(
                f"No .amdgcn file was found in the compilation cache.\n"
                f"Cache directory: {dump_dir}\n"
                f"Confirm that running the kernel file triggers GPU kernel compilation."
            )

        return asm_content

    except Exception as e:
        print(f"❌ Assembly extraction failed:{e}", file=sys.stderr)
        raise

    finally:
        sys.modules.pop("_kernel_mod", None)
        if original_cache_dir is not None:
            os.environ['TRITON_CACHE_DIR'] = original_cache_dir
        else:
            os.environ.pop('TRITON_CACHE_DIR', None)
        shutil.rmtree(dump_dir, ignore_errors=True)


def analyze_load_width(asm_content):
    """Analyze instruction widths for buffer_load/store and ds_read/write"""
    results = {
        "buffer_load": {"dword": 0, "dwordx2": 0, "dwordx4": 0},
        "buffer_store": {"dword": 0, "dwordx2": 0, "dwordx4": 0},
        "ds_read": {"b32": 0, "b64": 0, "b128": 0},
        "ds_write": {"b32": 0, "b64": 0, "b128": 0},
    }

    for line in asm_content.split("\n"):
        line = line.strip()

        if "buffer_load_dwordx4" in line:
            results["buffer_load"]["dwordx4"] += 1
        elif "buffer_load_dwordx2" in line:
            results["buffer_load"]["dwordx2"] += 1
        elif re.search(r"buffer_load_dword\b", line):
            results["buffer_load"]["dword"] += 1

        if "buffer_store_dwordx4" in line:
            results["buffer_store"]["dwordx4"] += 1
        elif "buffer_store_dwordx2" in line:
            results["buffer_store"]["dwordx2"] += 1
        elif re.search(r"buffer_store_dword\b", line):
            results["buffer_store"]["dword"] += 1

        if "ds_read_b128" in line:
            results["ds_read"]["b128"] += 1
        elif "ds_read_b64" in line:
            results["ds_read"]["b64"] += 1
        elif "ds_read_b32" in line:
            results["ds_read"]["b32"] += 1

        if "ds_write_b128" in line:
            results["ds_write"]["b128"] += 1
        elif "ds_write_b64" in line:
            results["ds_write"]["b64"] += 1
        elif "ds_write_b32" in line:
            results["ds_write"]["b32"] += 1

    return results


def analyze_bpermute(asm_content):
    """Count ds_bpermute instructions"""
    count = asm_content.count("ds_bpermute")
    lines = [
        (i + 1, line.strip())
        for i, line in enumerate(asm_content.split("\n"))
        if "ds_bpermute" in line
    ]
    return count, lines


def analyze_scratch(asm_content):
    """Detect scratch operations (register spill)"""
    scratch_loads = len(re.findall(r"scratch_load", asm_content))
    scratch_stores = len(re.findall(r"scratch_store", asm_content))

    spill_match = re.search(r"\.vgpr_spill_count:\s*(\d+)", asm_content)
    vgpr_spill = int(spill_match.group(1)) if spill_match else -1

    vgpr_match = re.search(r"\.vgpr_count:\s*(\d+)", asm_content)
    vgpr_count = int(vgpr_match.group(1)) if vgpr_match else -1

    sgpr_spill_match = re.search(r"\.sgpr_spill_count:\s*(\d+)", asm_content)
    sgpr_spill = int(sgpr_spill_match.group(1)) if sgpr_spill_match else -1

    return {
        "scratch_loads": scratch_loads,
        "scratch_stores": scratch_stores,
        "vgpr_spill_count": vgpr_spill,
        "vgpr_count": vgpr_count,
        "sgpr_spill_count": sgpr_spill,
    }


def analyze_accvgpr_moves(asm_content):
    """Count AGPR-to-VGPR transfer instructions"""
    reads = asm_content.count("v_accvgpr_read_b32")
    writes = asm_content.count("v_accvgpr_write_b32")
    return {"accvgpr_reads": reads, "accvgpr_writes": writes}


def print_analysis(asm_content):
    """Print the full analysis report"""
    print(f"{'='*60}")
    print(f"  Assembly Analysis Report")
    print(f"{'='*60}")

    width = analyze_load_width(asm_content)
    print(f"\n--- Instruction Width Analysis ---")
    for op_type, counts in width.items():
        total = sum(counts.values())
        if total > 0:
            print(f"\n  {op_type} (total {total} items):")
            for w, c in counts.items():
                if c > 0:
                    marker = "✅" if w in ("dwordx4", "b128") else "⚠️"
                    print(f"    {marker} {w}: {c}")

    bpermute_count, bpermute_lines = analyze_bpermute(asm_content)
    print(f"\n--- ds_bpermute Analysis ---")
    if bpermute_count == 0:
        print(f"  ✅ No ds_bpermute instructions")
    else:
        print(f"  ⚠️  Found {bpermute_count} ds_bpermute instructions")

    scratch = analyze_scratch(asm_content)
    print(f"\n--- Register Spill Analysis ---")
    print(f"  VGPR usage: {scratch['vgpr_count']}")
    print(f"  VGPR spills: {scratch['vgpr_spill_count']}")
    if scratch["vgpr_spill_count"] > 0:
        print(f"  ❌ VGPR spills detected!scratch_load: {scratch['scratch_loads']}, scratch_store: {scratch['scratch_stores']}")
    elif scratch["vgpr_spill_count"] == 0:
        print(f"  ✅ No VGPR spills")

    accvgpr = analyze_accvgpr_moves(asm_content)
    print(f"\n--- AGPR Transfer Analysis ---")
    print(f"  v_accvgpr_read_b32:  {accvgpr['accvgpr_reads']}")
    print(f"  v_accvgpr_write_b32: {accvgpr['accvgpr_writes']}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze Gluon kernel assembly")
    parser.add_argument("kernel", help="Kernel source file")
    parser.add_argument("-o", "--output", help="Save assembly to a file")
    parser.add_argument("--check-load-width", action="store_true", help="Check load/store instruction widths")
    parser.add_argument("--check-bpermute", action="store_true", help="Check ds_bpermute instructions")
    parser.add_argument("--check-scratch", action="store_true", help="Check scratch operations")
    parser.add_argument("--check-all", action="store_true", help="Run all checks")
    parser.add_argument("--asm-file", help="Analyze an existing assembly file directly (skip extraction)")
    args = parser.parse_args()

    if args.asm_file:
        with open(args.asm_file, "r") as f:
            asm_content = f.read()
    else:
        asm_content = extract_asm(args.kernel, args.output)

    if args.output and not args.asm_file:
        with open(args.output, "w") as f:
            f.write(asm_content)
        print(f"Assembly saved to: {args.output}")

    if args.check_all:
        print_analysis(asm_content)
    elif args.check_load_width:
        width = analyze_load_width(asm_content)
        for op_type, counts in width.items():
            total = sum(counts.values())
            if total > 0:
                print(f"{op_type}: {counts}")
    elif args.check_bpermute:
        count, lines = analyze_bpermute(asm_content)
        print(f"ds_bpermute: {count} items")
    elif args.check_scratch:
        scratch = analyze_scratch(asm_content)
        print(f"scratch: {scratch}")
    else:
        print_analysis(asm_content)


if __name__ == "__main__":
    main()
