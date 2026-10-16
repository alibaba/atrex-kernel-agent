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

"""Structured, source-correlated SASS (+PTX) for an .ncu-rep kernel.

A Python port of VeloQ's `ncu disasm` verb
(crates/ncu/veloq-ncu/src/disasm.rs + disasm_pipeline/ + scripts/ncu_export.py),
on the `ncu_report` API atrex already uses.

Two tiers, with graceful degradation:

  * Baseline (no external tools): walk the cubin via `action.sass_by_pc()` and
    `action.source_info()` to emit per-instruction
    {address, opcode, operands, source} plus a (file,line) -> [sass_addr]
    source_index. This always works wherever the existing tools work.

  * Full pipeline (when `nvdisasm` + `cuobjdump` are on PATH): extract the CUDA
    ELF cubin from the .ncu-rep, run `nvdisasm --emit-json` to add `predicate`
    and `control_flow` per instruction, and `cuobjdump --dump-ptx` to add the
    PTX listing with its own source attribution.

Design note vs VeloQ: VeloQ disassembles with nvdisasm and *overlays*
ncu_report source onto it. We invert that — the ncu_report walk is the
authoritative base (so addresses and source lines are always correct, the bug
VeloQ's overlay exists to fix), and nvdisasm only *enriches* predicate/
control_flow by matching cubin-relative address. Output fields are equivalent.

Usage:
    python3 disasm.py --run-dir profile/run --report profile/run/ncu.ncu-rep --tag run

Output in <run-dir>/analysis/:
    disasm_<tag>.json   v1 envelope
    disasm_<tag>.txt    readable digest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ncu_utils as U  # noqa: E402
import envelope as E  # noqa: E402
import row_key as K  # noqa: E402

# Cubin-walk geometry lives in ncu_utils (single source of truth); aliased here
# for the local references (length math, nvdisasm address matching, _digest).
INSTRUCTION_STRIDE = U.INSTRUCTION_STRIDE
EM_CUDA = 190


# --- ncu_report baseline walk ------------------------------------------------

def baseline_instructions(action, base):
    """List of {address(cubin-relative), opcode, operands, source|None}."""
    return [
        {"address": rel, "opcode": opcode, "operands": operands, "source": source}
        for rel, opcode, operands, source in U.walk_cubin(action, base)
    ]


# --- CUDA ELF cubin extraction from the .ncu-rep (stdlib only) ----------------

def _elf64_byte_len(buf, off):
    """Exact length of the ELF64 image starting at `off`, or None if not a
    parseable ELF64. Mirrors VeloQ native/cubin.rs extent computation."""
    if buf[off:off + 4] != b"\x7fELF":
        return None
    if off + 64 > len(buf):
        return None
    ei_class = buf[off + 4]
    ei_data = buf[off + 5]
    if ei_class != 2:  # ELFCLASS64
        return None
    endian = "<" if ei_data == 1 else ">"
    # ELF64 header layout after e_ident(16): H H I Q Q Q I H H H H H H
    try:
        (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
         e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = \
            struct.unpack_from(endian + "HHIQQQIHHHHHH", buf, off + 16)
    except struct.error:
        return None
    if e_machine != EM_CUDA:
        return None
    end = e_ehsize
    if e_phoff:
        end = max(end, e_phoff + e_phnum * e_phentsize)
    if e_shoff:
        end = max(end, e_shoff + e_shnum * e_shentsize)
    # walk section headers for the true extent (skip SHT_NOBITS=8)
    for i in range(e_shnum):
        sh = off + e_shoff + i * e_shentsize
        if sh + e_shentsize > len(buf):
            break
        try:
            sh_type = struct.unpack_from(endian + "I", buf, sh + 4)[0]
            sh_offset = struct.unpack_from(endian + "Q", buf, sh + 24)[0]
            sh_size = struct.unpack_from(endian + "Q", buf, sh + 32)[0]
        except struct.error:
            continue
        if sh_type != 8:
            end = max(end, sh_offset + sh_size)
    # walk program headers
    for i in range(e_phnum):
        ph = off + e_phoff + i * e_phentsize
        if ph + e_phentsize > len(buf):
            break
        try:
            p_offset = struct.unpack_from(endian + "Q", buf, ph + 8)[0]
            p_filesz = struct.unpack_from(endian + "Q", buf, ph + 32)[0]
        except struct.error:
            continue
        end = max(end, p_offset + p_filesz)
    if off + end > len(buf):
        end = len(buf) - off
    return end


def extract_cuda_cubins(report_path):
    """Return list of CUDA ELF cubin byte-blobs embedded in the .ncu-rep."""
    try:
        buf = Path(report_path).read_bytes()
    except Exception:
        return []
    out = []
    start = 0
    while True:
        idx = buf.find(b"\x7fELF", start)
        if idx < 0:
            break
        n = _elf64_byte_len(buf, idx)
        if n and n > 0:
            out.append(bytes(buf[idx:idx + n]))
            start = idx + n
        else:
            start = idx + 4
    return out


# --- nvdisasm --emit-json parsing --------------------------------------------

def _run_tool(args, warnings):
    """Run `args` (cubin path last), return stdout or None; collect stderr."""
    try:
        r = subprocess.run(args, capture_output=True, text=True)
    except Exception as e:
        warnings.append(f"{args[0]} spawn failed: {e}")
        return None
    if r.stderr.strip():
        warnings.append(f"{args[0]} stderr: {r.stderr.strip()[:500]}")
    if r.returncode != 0:
        warnings.append(f"{args[0]} exit {r.returncode}")
        return None
    return r.stdout


def parse_emit_json(stdout, warnings):
    """Parse `nvdisasm --emit-json`. Return {function_name: {rel_addr: {predicate,
    control_flow}}}. nvdisasm top-level is an array; kernels are element[1:]."""
    if not stdout:
        return {}
    brace = stdout.find("[")
    if brace < 0:
        return {}
    try:
        doc = json.loads(stdout[brace:])
    except Exception as e:
        warnings.append(f"nvdisasm json decode failed: {e}")
        return {}
    if not isinstance(doc, list):
        warnings.append("nvdisasm top-level not an array")
        return {}
    out = {}
    for kernel in doc[1:]:
        if not isinstance(kernel, dict):
            continue
        fn = kernel.get("function-name")
        if not fn:
            continue
        start = int(kernel.get("start", 0))
        insns = kernel.get("sass-instructions", []) or []
        m = {}
        for i, ins in enumerate(insns):
            addr = start + i * INSTRUCTION_STRIDE
            pred = ins.get("predicate")
            cf = str(ins.get("other-attributes", {}).get("control-flow", "")).lower() == "true"
            m[addr] = {"predicate": pred or None, "control_flow": cf}
        out[fn] = m
    return out


# --- cuobjdump --dump-ptx parsing --------------------------------------------

def parse_ptx(stdout):
    """Return list of {line_number, text, source|None} from cuobjdump PTX.
    Tracks `.file <id> "<path>"` and `.loc <id> <line> [<col>]`."""
    if not stdout:
        return []
    files = {}
    cur = None
    out = []
    for i, line in enumerate(stdout.splitlines(), start=1):
        s = line.strip()
        if s.startswith(".file"):
            # .file <id> "<path>"
            try:
                parts = s.split(None, 2)
                fid = int(parts[1])
                path = parts[2].strip().strip('"')
                files[fid] = path
            except Exception:
                pass
        elif s.startswith(".loc"):
            try:
                parts = s.split()
                fid = int(parts[1])
                ln = int(parts[2])
                cur = {"file": files.get(fid, str(fid)), "line": ln}
            except Exception:
                pass
        elif s.startswith("}"):
            cur = None
        attributable = (s and not s.startswith(".") and s not in ("{", "}"))
        rec = {"line_number": i, "text": line}
        if attributable and cur is not None:
            rec["source"] = {"file": cur["file"], "line": cur["line"]}
        out.append(rec)
    return out


# --- assembly ----------------------------------------------------------------

def build_source_index(instructions, ptx_lines):
    """Inverted (file,line) -> {sass_addresses, ptx_line_numbers}."""
    idx = {}
    for ins in instructions:
        src = ins.get("source")
        if src:
            row = idx.setdefault((src["file"], src["line"]),
                                 {"sass": set(), "ptx": set()})
            row["sass"].add(ins["address"])
    for p in ptx_lines:
        src = p.get("source")
        if src:
            row = idx.setdefault((src["file"], src["line"]),
                                 {"sass": set(), "ptx": set()})
            row["ptx"].add(p["line_number"])
    out = []
    for (file, line) in sorted(idx):
        row = idx[(file, line)]
        out.append({
            "file": file, "line": line,
            "sass_addresses": sorted(row["sass"]),
            "ptx_line_numbers": sorted(row["ptx"]),
        })
    return out


def _sm_label(action):
    def _i(name):
        v = U.safe(action, name, None)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    major = _i("device__attribute_compute_capability_major")
    minor = _i("device__attribute_compute_capability_minor")
    if major is None or minor is None:
        return None
    return f"sm_{major}{minor}"


def run(report, tag, ordinal):
    action = U.load_action(report)
    base = U.cubin_load_base(action)
    warnings = []
    mangled = None
    try:
        nb = getattr(action, "NameBase_MANGLED", None)
        mangled = action.name(nb) if nb is not None else action.name()
    except Exception:
        mangled = None
    fn_name = mangled or U._kernel_demangled(action)

    instructions = baseline_instructions(action, base)
    if not instructions:
        warnings.append("no baseline SASS — recapture with `--set full`/`--set source`")

    ptx_lines = []
    cubin_sha = None
    enriched = 0
    have_nvdisasm = shutil.which("nvdisasm") is not None
    have_cuobjdump = shutil.which("cuobjdump") is not None
    if not have_nvdisasm:
        warnings.append("nvdisasm not on PATH — predicate/control_flow omitted (baseline only)")
    if not have_cuobjdump:
        warnings.append("cuobjdump not on PATH — PTX omitted")

    if have_nvdisasm or have_cuobjdump:
        cubins = extract_cuda_cubins(report)
        if not cubins:
            warnings.append("no embedded CUDA cubin found in report — full pipeline skipped")
        else:
            # pick the cubin that defines this kernel via nvdisasm function-name;
            # fall back to the first cubin.
            chosen = None
            chosen_pred = {}
            with tempfile.TemporaryDirectory() as td:
                for ci, blob in enumerate(cubins):
                    cpath = Path(td) / f"cubin_{ci}.cubin"
                    cpath.write_bytes(blob)
                    if have_nvdisasm:
                        out = _run_tool(["nvdisasm", "--emit-json", str(cpath)], warnings)
                        kmap = parse_emit_json(out, warnings)
                        if fn_name in kmap:
                            chosen, chosen_pred = (blob, cpath), kmap[fn_name]
                            break
                        if chosen is None and kmap:
                            # remember first cubin with any kernel as fallback
                            chosen = (blob, cpath)
                            chosen_pred = next(iter(kmap.values()))
                    elif chosen is None:
                        chosen = (blob, cpath)
                if chosen is not None:
                    blob, cpath = chosen
                    cubin_sha = hashlib.sha256(blob).hexdigest()
                    # enrich predicate/control_flow by matching cubin-relative addr
                    for ins in instructions:
                        e = chosen_pred.get(ins["address"])
                        if e:
                            if e.get("predicate"):
                                ins["predicate"] = e["predicate"]
                            ins["control_flow"] = e["control_flow"]
                            enriched += 1
                    if have_cuobjdump:
                        out = _run_tool(["cuobjdump", "--dump-ptx", str(cpath)], warnings)
                        ptx_lines = parse_ptx(out)
            if have_nvdisasm and enriched == 0 and instructions:
                warnings.append("nvdisasm enrichment matched 0 instructions "
                                "(address bases may differ); predicate/control_flow omitted")

    source_index = build_source_index(instructions, ptx_lines)
    source_present = any(ins.get("source") for ins in instructions)

    row = {
        "key": K.kernel_key(fn_name),
        "function_name": fn_name,
        "start": 0,
        "length": (instructions[-1]["address"] + INSTRUCTION_STRIDE) if instructions else 0,
        "instructions": instructions,
    }
    aux = {
        "row_id": U.launch_identity(action, ordinal),
        "kernel_demangled": U._kernel_demangled(action),
        "cubin_sha": cubin_sha,
        "sm": _sm_label(action),
        "instruction_stride": INSTRUCTION_STRIDE,
        "source_lineinfo_present": source_present,
        "ptx_lines": ptx_lines,
        "source_index": source_index,
        "warnings": warnings,
    }
    return E.envelope("ncu.disasm", report, count=1, total_matched=1,
                      rows=[row], auxiliary=aux)


def _digest(env, max_insns=60):
    d = env["data"]
    r = d["rows"][0]
    a = d["auxiliary"]
    lines = [f"# disasm {r['function_name']}",
             f"# sm={a.get('sm')} stride={a['instruction_stride']} "
             f"insns={len(r['instructions'])} "
             f"source_lineinfo={a['source_lineinfo_present']} "
             f"ptx_lines={len(a['ptx_lines'])} cubin_sha={a.get('cubin_sha')}"]
    for w in a.get("warnings", []):
        lines.append(f"# WARNING: {w}")
    for ins in r["instructions"][:max_insns]:
        pred = (ins.get("predicate") + " ") if ins.get("predicate") else ""
        cf = " [cf]" if ins.get("control_flow") else ""
        src = ins.get("source")
        loc = f"   ; {src['file']}:{src['line']}" if src else ""
        lines.append(f"  0x{ins['address']:04x}  {pred}{ins['opcode']:<14} "
                     f"{ins['operands']}{cf}{loc}")
    if len(r["instructions"]) > max_insns:
        lines.append(f"  ... ({len(r['instructions']) - max_insns} more)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--ordinal", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"[skip] {args.report} does not exist", file=sys.stderr)
        return 1

    env = run(args.report, args.tag, args.ordinal)
    analysis = args.run_dir / "analysis"
    stem = f"disasm_{args.tag}"
    E.write_json(analysis / f"{stem}.json", env)
    (analysis / f"{stem}.txt").write_text(_digest(env))
    n = len(env["data"]["rows"][0]["instructions"])
    print(f"disasm -> {analysis / (stem + '.json')} ({n} instructions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
