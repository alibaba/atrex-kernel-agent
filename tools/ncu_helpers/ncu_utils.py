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

"""Shared helpers for parsing Nsight Compute reports.

Usage:
    from ncu_utils import load_report, safe, dump_all_metrics

The caller may set PYTHONPATH to include ncu_report, e.g.:
    export PYTHONPATH=$PYTHONPATH:$(dirname "$(command -v ncu)")/extras/python

If ncu_report is not importable, we try a small list of common paths. The import
is lazy (see _import_ncu_report): modes that never read a .ncu-rep — e.g. the
caller only uses B200_KEY_METRICS — can import this module even with no
ncu_report installed.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import weakref
from pathlib import Path

# --- Attempt to locate ncu_report --------------------------------------------

def _has_ncu_report(dirpath):
    """True if `dirpath` holds an importable ncu_report — module `.py`, compiled
    extension (`.so`/`.pyd`), or package dir. A `.py`-only check misses
    binary-only Nsight Compute installs."""
    p = Path(dirpath)
    if not p.is_dir():
        return False
    if (p / "ncu_report.py").exists() or (p / "ncu_report" / "__init__.py").exists():
        return True
    for pat in ("ncu_report*.so", "ncu_report*.pyd"):
        if any(p.glob(pat)):
            return True
    return False


def _locate_ncu_report():
    # Ordered candidates, most authoritative first. No CUDA/Nsight version is
    # pinned — we derive the path from the environment or the installed `ncu`
    # binary so it keeps working across toolkit upgrades.
    candidates = []

    # 1. Explicit override.
    env = os.environ.get("NCU_REPORT_PATH")
    if env:
        candidates.append(env)

    # 2. Derive from the `ncu` binary on PATH:
    #    .../nsight-compute-<ver>/ncu  ->  .../nsight-compute-<ver>/extras/python
    ncu_bin = shutil.which("ncu")
    if ncu_bin:
        candidates.append(str(Path(ncu_bin).resolve().parent / "extras" / "python"))

    # 3. Version-agnostic generic symlink.
    candidates.append("/usr/local/cuda/nsight-compute/extras/python")

    # 4. Glob versioned install roots (no hardcoded version number).
    for root in ["/usr/local", "/opt/nvidia", "/opt/cuda"]:
        p = Path(root)
        if not p.is_dir():
            continue
        for pattern in (
            "cuda-*/nsight-compute-*/extras/python",
            "nsight-compute-*/extras/python",
        ):
            candidates.extend(str(s) for s in p.glob(pattern))

    for c in candidates:
        if _has_ncu_report(c):
            return c
    return None

_ncu_report = None


def _import_ncu_report():
    """Import and cache the `ncu_report` module, lazily.

    Done on first use (not at module import) so importing ncu_utils does not
    require ncu_report — only the .ncu-rep loading paths do. Raises ImportError
    if the module cannot be found on PATH or in the located install dir.
    """
    global _ncu_report
    if _ncu_report is not None:
        return _ncu_report
    try:
        import ncu_report
    except ImportError:
        found = _locate_ncu_report()
        if not found:
            raise
        sys.path.insert(0, found)
        import ncu_report
    _ncu_report = ncu_report
    return ncu_report


# --- Loading -----------------------------------------------------------------

# An action reads memory owned by its report, so the report must outlive the
# action. This weak-keyed map anchors report -> action lifetime and releases the
# report automatically when the action is garbage-collected (no unbounded
# growth even across many large reports).
_REPORT_ANCHOR = weakref.WeakKeyDictionary()


def load_report(path):
    """Load a .ncu-rep file and return the first action (= first kernel launch).

    Returns (report, action) tuple — keep the report alive while using the action.
    """
    r = _import_ncu_report().load_report(str(path))
    rng = r.range_by_idx(0)
    action = rng.action_by_idx(0)
    return r, action


def load_action(path):
    """Shortcut when you don't need the report object separately.

    The report owns the memory the action reads from, so we anchor it to the
    returned action to keep it alive for the action's lifetime. Without this the
    report is GC'd as soon as this function returns, leaving the action dangling.
    """
    report, action = load_report(path)
    try:
        _REPORT_ANCHOR[action] = report  # released when `action` is GC'd
    except TypeError:
        action._ncu_report = report  # not weak-referenceable; anchor via attr
    return action


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """Try each candidate name, return first that works. Useful for
    GPU-gen-specific names: some metric names differ on sm_100 vs sm_90."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Value-kind robust accessor (for per-instance data) ---------------------

def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name):
    """Return a list of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every metric name + value to a JSON file for later analysis.

    Returns the number of entries written.
    """
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC → source line mapping ------------------------------------------------

def per_pc_values(action, metric_name):
    """For a source-level metric (with correlation_ids = PCs), return list of (pc, value)."""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            v = 0
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc):
    """Return (file, line) for a given PC, or ('?', 0) if unavailable.

    Requires -lineinfo at compile time.
    """
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Curated metric sets -----------------------------------------------------
#
# These metric names are known to exist and return meaningful values on
# B200 / sm_100 with Nsight Compute 2026.x. For a fuller list and rationale
# see reference/profile_guide.md. Other GPU generations (A100, H100,
# consumer cards) and future ncu releases may need alternate names — always
# verify with action.metric_names() if a metric returns None.

B200_KEY_METRICS = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    # Timing
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Warp divergence (avg active threads per warp-instruction; 32 = no divergence)
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Atomics / reductions (for atomics-contention analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    # Stall reasons — aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Stall reasons — per-PC (requires --set source)
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


# --- SASS / PC mapping primitives (ported from VeloQ's ncu_export.py) --------
#
# VeloQ's source-metrics / disasm / warp-stalls all build on three IAction
# methods that atrex's ncu_report already exposes:
#   action.sass_by_pc(pc)  -> disassembly text; "" = out of cubin,
#                             "N/A" = a mid-instruction byte (real PC, no opcode)
#   action.source_info(pc) -> object with .file_name()/.line(), or None
#   action.timed_warp_samples() -> periodic warp-state samples
# The helpers below wrap them defensively so the higher-level tools stay clean.

# Sentinels returned by sass_by_pc — kept as named constants so callers do not
# sprinkle magic strings around.
SASS_OUT_OF_CUBIN = ""   # pc is not in this kernel's cubin
SASS_MID_INSTR = "N/A"   # pc lands on a non-leading byte of an instruction

# Cubin-walk geometry (canonical home — disasm.py / source_metrics.py alias
# these so the assumption lives in one place).
INSTRUCTION_STRIDE = 16     # SASS instruction width (Volta+), VeloQ's assumption
CUBIN_SCAN_LIMIT = 1 << 20  # 1 MiB safety bound when walking sass_by_pc


def sass_by_pc(action, pc):
    """Return the SASS text at an absolute PC, or SASS_OUT_OF_CUBIN on error."""
    try:
        return action.sass_by_pc(pc)
    except Exception:
        return SASS_OUT_OF_CUBIN


def source_ref(action, pc):
    """Return {"file", "line"} for an absolute PC, or None.

    None means the PC has no DWARF line info (a "DWARF hole" — kernel built
    without -lineinfo, or a compiler-generated instruction).
    """
    try:
        si = action.source_info(pc)
    except Exception:
        return None
    if si is None:
        return None
    try:
        return {"file": si.file_name(), "line": int(si.line())}
    except Exception:
        return None


def correlation_pcs(m):
    """Return the list of per-instance correlation IDs (absolute PCs) for a
    metric, or [] when the metric carries no per-PC instances."""
    try:
        if not m.has_correlation_ids() or m.num_instances() == 0:
            return []
    except Exception:
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(m.num_instances()):
        try:
            out.append(cor.as_uint64(i))
        except Exception:
            try:
                out.append(int(cor.as_double(i)))
            except Exception:
                out.append(None)
    return out


def instance_value_f64(m, i):
    """Numeric per-instance value as a float, or None for string/unreadable
    instances. Thin numeric wrapper over metric_value_at so the value-kind
    handling lives in exactly one place; matches VeloQ's rollup, which drops
    string-valued instances."""
    v = metric_value_at(m, i)
    if isinstance(v, str) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cubin_load_base(action):
    """Lowest in-cubin correlation ID across every per-PC metric.

    This is the anchor for cubin-relative addressing: a sampled PC is in-cubin
    iff sass_by_pc(pc) != "" (including mid-instruction bytes). Returns None if
    the report carries no in-cubin per-PC instances (nothing to attribute).
    Ported from VeloQ ncu_export.py::_cubin_load_base.
    """
    base = None
    try:
        names = action.metric_names()
    except Exception:
        return None
    for name in names:
        try:
            m = action[name]
        except Exception:
            continue
        for pc in correlation_pcs(m):
            if pc is None:
                continue
            if sass_by_pc(action, pc) != SASS_OUT_OF_CUBIN:
                base = pc if base is None else min(base, pc)
    return base


def walk_cubin(action, base, scan_limit=CUBIN_SCAN_LIMIT, stride=INSTRUCTION_STRIDE):
    """Yield (rel_address, opcode, operands, source) per instruction in the cubin.

    Starts at absolute PC `base` and steps `stride` bytes at a time, skipping
    mid-instruction bytes (SASS_MID_INSTR) and stopping at the first
    out-of-cubin PC. `rel_address` is cubin-relative (pc - base); `source` is
    source_ref() for that PC or None. Yields nothing when base is None.

    Single source of truth for the SASS walk shared by disasm.py and
    source_metrics.py. Ported from VeloQ ncu_export.py::_disasm.
    """
    if base is None:
        return
    addr = base
    while addr - base <= scan_limit:
        text = sass_by_pc(action, addr)
        if text == SASS_OUT_OF_CUBIN:
            break
        if text != SASS_MID_INSTR:
            stripped = text.strip()
            if " " in stripped:
                opcode, operands = stripped.split(" ", 1)
            else:
                opcode, operands = stripped, ""
            yield addr - base, opcode, operands.strip(), source_ref(action, addr)
        addr += stride


# --- Warp-state sampling -----------------------------------------------------

def _stall_reason_names(ncu_report):
    """Map StallReason enum int -> lowercased suffix name, from the live module
    so an enum renumber across ncu versions can't corrupt classification."""
    out = {}
    for n in dir(ncu_report):
        if n.startswith("StallReason_"):
            v = getattr(ncu_report, n)
            if isinstance(v, int):
                out[v] = n[len("StallReason_"):].lower()
    return out


def timed_warp_samples(action):
    """Return (samples, reason_names) where samples is a list of
    {"pc", "stall_reason"(int), "not_issued"(bool)} and reason_names maps the
    int code to a lowercase name. Empty list if warp sampling wasn't captured.
    Ported from VeloQ ncu_export.py::_warp_stalls input handling.
    """
    try:
        raw = action.timed_warp_samples()
    except Exception:
        return [], {}
    reason_names = _stall_reason_names(_import_ncu_report())
    out = []
    for s in raw:
        try:
            out.append({
                "pc": s["pc"],
                "stall_reason": s["stall_reason"],
                "not_issued": bool(s.get("not_issued", False)),
            })
        except Exception:
            continue
    return out, reason_names


# --- Metric additivity (gates source-metrics line/file axes) ------------------
#
# Ported from VeloQ additivity.rs. A counter may be summed across source lines
# only if it is "additive" (a raw count). Percentages, ratios, rates, and
# pre-aggregated avg/min/max are non-additive and must not be summed.

_NAME_MAPS = {}


def _enum_name_map(ncu_report, prefixes):
    """int code -> lowercased name, scanning IMetric and the module top-level
    for attributes whose name starts with any of `prefixes`."""
    key = tuple(prefixes)
    if key in _NAME_MAPS:
        return _NAME_MAPS[key]
    out = {}
    scopes = [ncu_report]
    imetric = getattr(ncu_report, "IMetric", None)
    if imetric is not None:
        scopes.append(imetric)
    for scope in scopes:
        for n in dir(scope):
            for p in prefixes:
                if n.startswith(p):
                    v = getattr(scope, n)
                    if isinstance(v, int):
                        out[v] = n[len(p):].lower()
    _NAME_MAPS[key] = out
    return out


def _classify_enum(m):
    """Return (type_name, subtype_name, rollup_name) as lowercase strings or
    'unknown', resolved from the live enum maps."""
    nr = _import_ncu_report()
    tmap = _enum_name_map(nr, ("MetricType_",))
    smap = _enum_name_map(nr, ("MetricSubtype_",))
    rmap = _enum_name_map(nr, ("RollupOperation_", "Rollup_"))

    def _resolve(getter, mp):
        try:
            code = getter()
        except Exception:
            return "unknown"
        if not isinstance(code, int):
            return "unknown"
        return mp.get(code, "unknown")

    t = _resolve(getattr(m, "metric_type", lambda: None), tmap)
    st = _resolve(getattr(m, "metric_subtype", lambda: None), smap)
    ru = _resolve(getattr(m, "rollup_operation", lambda: None), rmap)
    return t, st, ru


def _is_non_additive_by_name(name, unit=""):
    """Name/unit-suffix fallback for additivity (VeloQ is_non_additive)."""
    n = name
    if n.endswith(".pct") or ".pct_" in n or n.endswith("_pct") or "_pct_" in n:
        return True
    for suf in (".per_second", ".per_cycle_active", ".per_cycle_elapsed"):
        if n.endswith(suf):
            return True
    u = (unit or "").lower()
    if "per_" in u or u == "hertz":
        return True
    for suf in (".ratio", ".avg", ".max", ".min"):
        if n.endswith(suf):
            return True
    return False


def is_additive(action, name):
    """True if metric `name` may be summed across source lines.

    Enum classification is authoritative when available; otherwise fall back to
    the name/unit suffix rule. Ported from VeloQ additivity.rs::is_additive_native.
    """
    m = None
    try:
        m = action[name]
    except Exception:
        m = None
    if m is not None:
        t, st, ru = _classify_enum(m)
        if st in ("pct", "ratio", "per_second"):
            return False
        if t in ("ratio", "throughput"):
            return False
        if t == "counter":
            if ru == "sum":
                return True
            if ru != "unknown":
                return False
            # rollup unknown -> fall through to name rule
        # type other/unknown -> fall through to name rule
    unit = ""
    if m is not None:
        try:
            unit = m.unit()
        except Exception:
            unit = ""
    return not _is_non_additive_by_name(name, unit)


# --- Content-derived launch identity (stable cross-capture key prefix) --------
#
# VeloQ keys rows as "launch:<idx>" where <idx> is positional, so the prefix
# breaks if launches are reordered/added between captures. We derive the prefix
# from launch *content* instead so the same kernel launch gets the same prefix
# across two .ncu-rep files. The leaf templates (|line:, |sass:, ...) match
# VeloQ verbatim — see row_key.py.

def _kernel_demangled(action):
    try:
        nr = _import_ncu_report()
        base = getattr(action, "NameBase_DEMANGLED", None)
        if base is not None:
            return action.name(base)
    except Exception:
        pass
    try:
        return action.name()
    except Exception:
        return "?"


def launch_identity(action, ordinal=0):
    """Stable, content-derived launch id for use as a row-key prefix.

    Composed from the demangled kernel name, grid/block dims, and an occurrence
    ordinal (to disambiguate repeated launches of the same kernel within one
    capture). Stable across captures as long as the kernel is launched with the
    same geometry — unlike VeloQ's positional launch:<idx>.
    """
    def _i(metric):
        v = safe(action, metric, None)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    name = _kernel_demangled(action)
    grid = (_i("launch__grid_dim_x"), _i("launch__grid_dim_y"), _i("launch__grid_dim_z"))
    block = (_i("launch__block_dim_x"), _i("launch__block_dim_y"), _i("launch__block_dim_z"))
    return "launch:{name}|grid={g}|block={b}|n={ord}".format(
        name=name,
        g="x".join(str(x) for x in grid),
        b="x".join(str(x) for x in block),
        ord=ordinal,
    )


# --- Convenience: NCU rule results --------------------------------------------

def rule_results(action):
    """Return the NCU rule-engine results as a list of dicts, or [] if unavailable."""
    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


def rule_speedups(action):
    """Return list of (est_speedup_pct, rule_name, message) sorted desc by est_speedup.
    Missing est_speedup becomes 0."""
    out = []
    for rr in rule_results(action):
        est = rr.get("estimated_speedup_pct", None)
        if est is None:
            est = 0.0
        try:
            est = float(est)
        except Exception:
            est = 0.0
        rule = rr.get("rule_name", "?")
        msg = rr.get("message_for_display", "?")
        out.append((est, rule, msg))
    out.sort(key=lambda x: -x[0])
    return out
