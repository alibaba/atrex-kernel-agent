# rocprofv3 Instruction-Level Profile Details

**Last Updated**: 2026-03-09
**Verified Environment**: ROCm 7.0.1, MI300X (gfx942), rocprofv3 1.0.0

---

## Overview

rocprofv3 is the GPU profiling tool for the AMD ROCm platform. This guide focuses on **instruction-level analysis** (Advanced Thread Trace, ATT), used exclusively for identifying performance hotspots in Gluon kernels.

The full feature set is called **Thread Trace**, and the toolchain consists of three components:

| Component | Function | Source |
|------|------|------|
| **rocprofv3** | Captures instruction-level trace data | Bundled with ROCm installation (`/opt/rocm/bin/rocprofv3`) |
| **rocprof-trace-decoder** | Decodes raw .att binaries into an analyzable format | [GitHub releases](https://github.com/ROCm/rocprof-trace-decoder/releases), downloaded in this guide to `tools/rocprof-trace-decoder/` |
| **rocprof-compute-viewer** | Visualizes trace data (GUI, optional) | [ROCm docs](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/how-to/using_compute_viewer.html) |

**This guide focuses on CLI-based instruction-level analysis** and does not rely on GUI visualization tools.

---

## 1. Complete Thread Trace Toolchain

### 1.1 Verified Runtime Commands

The following commands have been verified in the current environment and can be used directly:

```bash
# Complete command — Use input_att.yaml configuration file + local trace decoder library
env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
    rocprofv3 --att \
    --att-library-path ./tools/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 \
    -i tools/input_att.yaml \
    -- python <kernel.py>
```

**Key Points**:
- `--att-library-path` points to the .so library path of `rocprof-trace-decoder`
- `-i input_att.yaml` uses a YAML configuration file to control trace parameters
- `LD_LIBRARY_PATH` must include `/opt/rocm/lib64` and `/opt/rocm/lib`

### 1.2 Configuration File input_att.yaml

The `tools/input_att.yaml` configuration for this guide (verified working):

```yaml
jobs:
    -
        kernel_iteration_range: "[1]"        # Only trace the 2nd dispatch
        output_file: out
        output_directory: tt_test            # ⚠️ Relative path, must run from tools/ directory
        output_format: [json, csv]           # CLI analysis only needs json + csv
        truncate_kernels: true
        sys_trace: false                     # Disable system-level trace
        advanced_thread_trace: true
        att_target_cu: 1
        att_shader_engine_mask: "0xf"        # Collect 1 CU on each of 4 SEs
        att_simd_select: "0xf"               # Collect 4 SIMDs on a single CU
        att_buffer_size: "0x6000000"         # 96MB trace buffer
```

**Configuration Key Points**:
- `att_shader_engine_mask: "0xf"` collects 4 Shader Engines → 4 wave traces (more comprehensive)
- `att_shader_engine_mask: "0x1"` collects only 1 SE → smaller data size, less prone to buffer overflow
- `kernel_iteration_range: "[1]"` only traces the 2nd dispatch (skips initialization/compilation dispatches)
- `att_buffer_size` default 96MB is sufficient for most kernels; complex kernels can be increased to `0x20000000` (512MB)

### 1.3 Configuration Pitfalls (Verified)

The following issues were encountered and confirmed during actual use — **be sure to avoid them**:

| Issue | Impact | Solution |
|------|------|------|
| Placing multiple jobs (ATT + PMC) in YAML | Each job runs the python script completely → **double the time** | Only place 1 ATT job; run PMC separately with `rocprofv3 --pmc` |
| `output_format` includes otf2/pftrace | Generates ~340MB of useless large files + ~10s additional finalization | Only use `[json, csv]` |
| `sys_trace: true` | Additionally collects HIP/HSA/Memory API traces + generates 220MB `out_results.json` | Set to `false` |
| `output_directory` is a relative path | Output goes to the wrong location when CWD is incorrect → **trace lost** | Must run rocprofv3 from the `tools/` directory |

**Before and After Optimization** (measured):

| Metric | Before Optimization | After Optimization |
|------|--------|--------|
| Python execution count | 2 (2 jobs) | 1 |
| Finalization | ~16s | ~2.4s |
| Total runtime | ~22s | ~5.6s |
| Output size | ~470MB | ~68MB |### 1.4 Output File Structure

```
tt_test/ # output_directory(sys_trace=false nonedirectory)
├── out_agent_info.csv # GPU agent
├── out_results.json # trace data(~2MB, sys_trace=false )
│
├── out_<pid>_shader_engine_<SE>_<dispatch>.att # SQTT data
├── out_gfx942_code_object_id_<N>.out # Code object
│
├── stats_ui_output_agent_<agent>_dispatch_<N>.csv # ★ instruction-level(coreanalysisfile)
└── ui_output_agent_<agent>_dispatch_<N>/ # dispatch data
 ├── code.json # + mapping
 ├── filenames.json # filecolumn
 ├── occupancy.json # wave/SIMD
 ├── snapshots.json # path
 ├── se<SE>_sm<SM>_sl<SL>_wv<WV>.json # ★ wave
 ├── wstates<N>.json # wave data
 └── source_<N>_<filename>.py #
```

> **Note**: When `sys_trace: false`, output goes directly under `tt_test/` (without the `pass_1/` subdirectory),
> and files such as `out_kernel_trace.csv`, `out_*_api_trace.csv`, `.pftrace`, `.otf2` will not be generated.
> Locating the target kernel is instead determined by the dispatch_id in the `stats_*.csv` filename along with the file size.

### 1.5 How to Locate the Target Kernel

The dispatch_id corresponding to the target Gluon kernel is determined as follows:

```bash
# Method 1: stats CSV file size — Gluon kernel has most instructions, stats file is largest```

Example: The Gluon kernel is typically the **last dispatch** (with the largest dispatch_id), and the stats CSV file is the largest (~115 KB, while other kernels are typically <35 KB).

After finding the dispatch_id, analyze the corresponding `stats_ui_output_agent_<agent>_dispatch_<N>.csv`.

---

## 2. Analyzing stats_*.csv in a CLI Environment (Instruction-Level Statistics)

### 2.1 Column Definitions

```csv
"CodeObj","Vaddr","Instruction","Hitcount","Latency","Stall","Idle","Source"
11,6400,"s_load_dwordx2 s[2:3], s[0:1], 0x0",16,64,0,0,"chunk_gdn_gluon.py:47"
```

| Column | Meaning | Importance |
|--------|---------|------------|
| **Instruction** | Assembly instruction | Identify operation type |
| **Hitcount** | Execution count (total across all traced waves) | Hotness assessment: hot-loop instruction hitcount ≫ prologue/epilogue |
| **Latency** | Total latency cycles = Stall + Issue/Execute time | **Core metric**: higher value = greater contribution to total execution time |
| **Stall** | Pipeline stall cycles (TCP/LDS backpressure, waiting for resource readiness) | **Bottleneck indicator**: high Stall = pipeline front-end is blocked |
| **Idle** | Idle cycles (register dependencies, icache miss, data dependencies) | Dependency chain signal: high Idle = waiting for prior instruction results |
| **Source** | Source code line number (requires debug info preserved during Gluon/Triton compilation) | Pinpoint Python source code |

**Note**:
- Latency, Stall, and Idle are all **cumulative values** across all traced waves (not averages)
- To obtain the average time per execution: `avg_latency = Latency / Hitcount`
- The Stall value of `s_waitcnt` instructions reflects the wait time of **prior asynchronous operations**

### 2.2 Analysis Commands (Verified)

grep 'ds_bpermute' "$STATS_FILE"                          # layout conversion
grep 'scratch' "$STATS_FILE"                              # Register spill
grep 'v_mfma' "$STATS_FILE"                               # Matrix multiplication
grep 's_barrier' "$STATS_FILE"                            # Barrier synchronization
grep 's_waitcnt' "$STATS_FILE" | sort -t',' -k6 -nr      # Wait instructions (reflect async latency)### 2.3 Python Analysis Script (Recommended)

```python
import csv

stats_file = "tt_test/pass_1/stats_ui_output_agent_33148_dispatch_27.csv"
with open(stats_file) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Only keep instructions with execution data
data = [(r[2], int(r[3]), int(r[4]), int(r[5]), int(r[6]), r[7])
        for r in rows if int(r[3]) > 0]
# Top 10 by Latencyprint("=== Top 10 by Latency ===")
for inst, hit, lat, stall, idle, src in sorted(data, key=lambda x: x[2], reverse=True)[:10]:
    avg = lat // hit if hit else 0
    print(f"  {inst:55s}  hit={hit:5d}  lat={lat:7d}  stall={stall:7d}  idle={idle:7d}  avg={avg:5d}  {src}")

# Count total Stall ratio for each instruction type
```

---

## 3. Functional Unit Utilization Analysis (VALU/MFMA/VMEM Idle Time)

`ui_output_agent_<agent>_dispatch_<N>/se*_sm*_sl*_wv*.json` contains the complete instruction timeline for each wave, which can be used to analyze when each functional unit is idle.

### 3.1 Instruction Timeline Format

`wave['wave']['instructions']` Each element `[timestamp, type, stall, latency, inst_index]`:

| Field | Meaning |
|------|------|
| `timestamp` | GPU cycle at which the instruction begins execution |
| `type` | Instruction type code (see table below) |
| `stall` | Stall cycles for this instruction |
| `latency` | Total latency cycles for this instruction |
| `inst_index` | Corresponding instruction index in code.json |

### 3.2 Type Encoding → Functional Unit Mapping (Verified)

| type | Functional Unit | Instructions Included | Instruction Count (Measured) |
|------|---------|-----------|---------------|
| 6 | **VALU + MFMA** | `v_*` (ALU), `v_mfma_*` (Matrix) | ~38000 |
| 3, 4 | **VMEM** | `buffer_load/store`, `global_load/store` | ~2000 |
| 5 | **LDS** | `ds_read/write`, `ds_bpermute` | ~10000 |
| 1, 2, 7, 9 | **SCALAR** | `s_load`, `s_waitcnt`, `s_barrier`, etc. | ~10000 |

> type=6 includes both VALU and MFMA, which need to be distinguished by the instruction name in `code.json` (`v_mfma_*` = MFMA, other `v_*` = VALU).

### 3.3 Functional Unit Utilization Analysis Script

```python
import json, os

# Load data
UI_DIR = "tt_test/ui_output_agent_<agent>_dispatch_27"
with open(os.path.join(UI_DIR, "code.json")) as f:
    code_list = json.load(f)['code']
with open(os.path.join(UI_DIR, "se0_sm0_sl0_wv0.json")) as f:
    wave = json.load(f)

insts = wave['wave']['instructions']
wave_begin = wave['wave']['begin']
duration = wave['duration']

WINDOW = 1024
num_windows = (duration + WINDOW - 1) // WINDOW
valu_busy = [0] * num_windows
mfma_busy = [0] * num_windows
vmem_busy = [0] * num_windows
lds_busy  = [0] * num_windows

for ts, typ, stall, lat, idx in insts:
    if idx >= len(code_list):
        continue
    w = (ts - wave_begin) // WINDOW
    if w < 0 or w >= num_windows:
        continue
    inst_name = code_list[idx][0]
    if typ == 6:
        if 'v_mfma' in inst_name:
            mfma_busy[w] = min(mfma_busy[w] + 4, WINDOW)
        else:
            valu_busy[w] = min(valu_busy[w] + 4, WINDOW)
    elif typ in (3, 4):
        vmem_busy[w] = min(vmem_busy[w] + 4, WINDOW)
    elif typ == 5:
        lds_busy[w] = min(lds_busy[w] + 4, WINDOW)

# Count idle cyclesidle = {u: sum(WINDOW for w in range(num_windows) if busy[w] == 0)
        for u, busy in [('VALU', valu_busy), ('MFMA', mfma_busy), ('VMEM', vmem_busy), ('LDS', lds_busy)]}

print(f"Wave duration: {duration:,} cycles")
for unit, cycles in idle.items():
    print(f"  {unit} idle: {cycles:>10,} cycles ({cycles*100//duration}%)")
# Print timeline (sample every 10 windows)```

### 3.4 Interpretation of Measured Results (chunk_gated_delta_rule kernel, 2026-03-09)

```
Functional Unit Idle Time:
  VALU idle:     5,120 cycles ( 0%)  ← Almost fully utilized
  MFMA idle:   304,128 cycles (45%)  ← Nearly half the time waiting for data
  VMEM idle:   415,744 cycles (62%)  ← More than half the time idle

Timeline Pattern (alternating):
  MFMA busy + VMEM idle → Compute phase (MFMA doing matrix multiplication, no memory access)
  MFMA idle + LDS busy  → Data preparation phase (ds_write/ds_bpermute writing to LDS)
  VMEM busy + MFMA idle  → Load phase (buffer_load loading data from global memory)
```

**Optimization Direction**: MFMA 45% idle + VMEM 62% idle = **typical insufficient compute-memory overlap**.
Software pipelining should be used to overlap loads and computations, reducing functional unit idle time.

---

## 4. Hardware Counters (PMC)

### Basic Commands

```bash
rocprofv3 --pmc \
    SQ_LDS_BANK_CONFLICT,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,\
    SPI_RA_VGPR_SGPR_FULL_CSN,TCP_TCC_MISS \
    -d ./pmc_output \
    -- python <kernel_script.py>
```

### Key Counter Categories

#### Memory / Vector Operations
| Counter | Description | Used for Diagnosis |
|--------|------|---------|
| `SQ_INSTS_VMEM_RD` | Number of vector memory read instructions | load throughput |
| `SQ_INSTS_VMEM_WR` | Number of vector memory write instructions | store throughput |
| `SQ_INSTS_VMEM` | Total vector memory instructions | Overall memory access pressure |
| `SQ_INSTS_FLAT` | Number of flat instructions | global/scratch/LDS mixed access |

#### LDS (Shared Memory)
| Counter | Description | Used for Diagnosis |
|--------|------|---------|
| `SQ_LDS_BANK_CONFLICT` | Bank conflict stall cycles | swizzle/layout issues |
| `SQ_LDS_ADDR_CONFLICT` | Address conflict stall cycles | address calculation issues |
| `SQ_INSTS_LDS` | Total LDS instructions | LDS operation frequency |

#### Occupancy / Resource Stalls
| Counter | Description | Used for Diagnosis |
|--------|------|---------|
| `SPI_RA_VGPR_SGPR_FULL_CSN` | Insufficient VGPR stall | register spilling |
| `SPI_RA_LDS_CU_FULL_CSN` | Insufficient LDS space | LDS over-limit |
| `SPI_RA_WAVE_SIMD_FULL_CSN` | Insufficient wave slots | occupancy issues |

#### Cache Performance
| Counter | Description | Used for Diagnosis |
|--------|------|---------|
| `TCP_TOTAL_READ` | L1 cache reads | cache utilization |
| `TCP_TCC_MISS` | L2 cache miss | cache efficiency |
| `TCC_HIT` / `TCC_MISS` | L2 hit/miss | L2 hit rate |

---

## 5. Hot Instruction Identification → Optimization Step Mapping

### Quick Diagnosis Table

| Assembly Instruction Pattern | High Stall? | High Idle? | Possible Issue | Optimization Step |
|-------------|----------|---------|---------|---------|
| `buffer_load_dword` (not x4) | — | — | Insufficient load width | 3.1 |
| `buffer_store_dword` (not x4) | — | — | Insufficient store width | 3.1 |
| `ds_read_b32` (not b128) | — | — | Insufficient LDS read width | 3.1 |
| `ds_write_b32` (not b128) | — | — | Insufficient LDS write width | 3.1 |
| `ds_read_*` / `ds_write_*` | ✅ | — | Bank conflict | 3.2 |
| `ds_bpermute_b32` | — | — | Layout conversion overhead | 3.3 |
| `buffer_load/store` + scratch addr | ✅ | — | Register spilling | 3.4 |
| `buffer_load_*` | ✅ | ✗ | Memory access not overlapped with compute | 3.5 |
| `buffer_load_*` | — | ✅ | Data dependency chain too long | 3.5 |
| `v_mfma_*` | — | ✅ | MFMA waiting for data readiness | 3.5 |

### Real Validation Example

The following data comes from an actual trace of the `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` kernel (validated 2026-03-09):

**Top 5 Stall Instructions:**
```buffer_load_dwordx4 (line 280)   hit=2352  stall=507076  ← Memory access stall, not overlapped by compute
buffer_load_dwordx2 (line 290)   hit=2352  stall=440816  ← Narrow load (dwordx2 instead of dwordx4)
buffer_load_dwordx4 (line 275)   hit=2352  stall=373356  ← Memory access stall
s_barrier           (line 301)   hit=2352  stall=311288  ← Barrier synchronization waiting
buffer_load_dwordx4 (line 285)   hit=2352  stall=309156  ← Memory access stall
```

**Optimization Insights:**
- `buffer_load_dwordx2` (line 290) → Step 3.1: Should be optimized to dwordx4
- Multiple `buffer_load` with high Stall → Step 3.5: Need to improve compute-memory overlap
- `s_barrier` with high Stall → May need to reconsider the synchronization strategy

### How to Distinguish Scratch from Global Memory Access

In assembly, scratch operations can be identified as follows:
- Use `scratch_load_*` / `scratch_store_*` instructions (gfx9 syntax)
- Or `buffer_load/store` using a scratch descriptor
- The `.vgpr_spill_count` in TTGIR compilation output is not 0

---

## 6. Complete Profile Workflow

```
Step 1: Measure kernel execution time
  └─ tools/measure_kernel_time.py → Get ms timing

Step 2: Compute compute utilization
  └─ tools/compute_utilization.py → Get utilization %

Step 3: If utilization < 90%, run instruction-level trace
  └─ env LD_LIBRARY_PATH=/opt/rocm/lib64:$LD_LIBRARY_PATH \
       rocprofv3 --att --att-library-path <decoder_path> \
       -i tools/input_att.yaml -- python <kernel.py>

Step 4: Locate target kernel
  └─ Read out_kernel_trace.csv to find target dispatch_id

Step 5: Parse hot instructions (CLI)
  └─ sort/grep stats_ui_output_agent_<agent>_dispatch_<N>.csv → Locate problematic instructions

Step 6: Analyze wave-level timeline (optional, deeper)
  └─ Read ui_output_agent_<agent>_dispatch_<N>/se*_sm*_sl*_wv*.json → Instruction pipeline

Step 7: Collect hardware counters (optional, to confirm diagnosis)
  └─ rocprofv3 --pmc → Confirm bank_conflict / spill / cache_miss

Step 8: Execute corresponding optimization steps (3.1 ~ 3.5) based on diagnosis

Step 9: Clean up trace files
  └─ Delete trace output after analysis completes (~68MB after optimization)
```

### 6.1 Trace File Cleanup

After analysis is complete, you **must clean up** trace output to prevent disk accumulation.

#### Trace Output File Size Distribution (measured, after optimizing configuration)

| File Type | Typical Size | Description |
|-----------|-------------|-------------|
| `ui_output_*/se*_wv*.json` | **~20MB** per dispatch | Wave-level timeline (core analysis data) |
| `out_results.json` | **~2MB** | Trace metadata (very small when sys_trace=false) |
| `*.att` | **~2MB** total | Raw SQTT binary |
| `*.out` | **~1MB** total | Code object binary |
| `stats_*.csv` | **~115KB** per dispatch | Instruction-level statistics (core analysis data) |

#### Cleanup Commands

```bash
TRACE_DIR="tt_test"
# ===== Method 1: Full cleanup (recommended) =====```

#### Cleanup Timing

- **After each trace is completed and analysis conclusions are extracted**, clean up immediately
- **During iterative optimization**: re-trace after each round of optimization; new traces will overwrite old data (same `output_directory`)
- **Note**: when re-running traces with different PIDs, new files will be generated and old files will not be automatically deleted → run `rm -rf` first, then trace

---

## 7. Environment Setup and Common Issues

### 7.1 Prerequisites

| Component | Requirement | Check Command |
|-----------|-------------|---------------|
| **ROCm** | 7.0+ | `rocprofv3 --version` |
| **AQL Profile** | Bundled with ROCm 7.0+, or build from source | `ls /opt/rocm/lib/libhsa-amd-aqlprofile64.so` |
| **ROCprofiler-SDK** | Bundled with ROCm 7.0+ | `which rocprofv3` |
| **ROCprof Trace Decoder** | Download from [GitHub releases](https://github.com/ROCm/rocprof-trace-decoder/releases) | This guide assumes the decoder is available at `tools/rocprof-trace-decoder/` |

### 7.2 Common Issues and Solutions

#### Q: ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA
**Cause**: The AQL Profile version is incompatible with the Trace Decoder version.
**Solution**:
1. Ensure AQL Profile is installed (bundled with ROCm 7.0+)
2. If ROCm < 7.0, build AQL Profile from [source](https://github.com/ROCm/aqlprofile/releases)
3. Ensure the Trace Decoder version matches the ROCm version
# Create symbolic link in /usr/lib64# Need to install both packages#### Q: `-fPIC` error during cmake build (linking against libpython3.10.a)
**Cause**: Need to link the dynamic library .so instead of the static library .a.
**Solution**:
```bash
# Symlink .so from the active Python environment to the CMake search path
ln -s "$CONDA_PREFIX/lib/libpython3.10.so" "${PYTHON_PREFIX:-/usr/local/python3.10}/lib/libpython3.10.so"
```

#### Q: Captured stats_*.csv is empty (only header)
**Cause**:
1. AQL Profile not installed (most common)
2. Kernel has too few waves, not assigned to the CU specified by `att_target_cu`
3. The dispatch count specified by `kernel_iteration_range` exceeds the actual range

**Solution**:
1. Install AQL Profile (ROCm 7.0+ or build from source)
2. Try different `att_target_cu` values (0-15)
3. Increase `att_shader_engine_mask` (e.g., `"0xf"`) to collect more SEs
4. Check `out_kernel_trace.csv` to confirm the dispatch_id of the target kernel

#### Q: Cannot find librccl.so
**Solution**: It is under `/opt/rocm/lib`. Ensure LD_LIBRARY_PATH includes this path.

#### Q: Data Lost / Buffer Full warning
**Cause**: Trace buffer is not large enough.
**Solution**:
1. Increase buffer: `att_buffer_size: "0x20000000"` (512MB)
2. Reduce the number of SEs: `att_shader_engine_mask: "0x1"`
3. Reduce kernel loop count and block/wave count

#### Q: How to capture traces in a bazel environment?
```bash
bazelisk run \
  --config=rocm \
  --jobs=100 \
  --test_output=all \
  --test_env=HIP_VISIBLE_DEVICES=5 \
  --run_under="env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:<decoder_path>:$LD_LIBRARY_PATH \
    rocprofv3 --att -i <input_att.yaml> --" \
  //path/to:target
```
**Note**: When capturing in bazel, the kernel loop count should not be too large, and the block/wave count should also not be too large, otherwise capture is prone to failure.

---

## 8. Reference Documentation

- [rocprofv3 Usage Guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [Thread Trace Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html)
- [ROCprof Compute Viewer](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/how-to/using_compute_viewer.html)
- [ROCprof Trace Decoder](https://github.com/ROCm/rocprof-trace-decoder)
- [AQL Profile Releases](https://github.com/ROCm/aqlprofile/releases)
- [MI300 Performance Counters](https://instinct.docs.amd.com/latest/gpu-arch/mi300-mi200-performance-counters.html)

---

## 9. ATT Parameter Complete Reference

| Parameter | Type | Range | Default | Description |
|------|------|------|--------|------|
| `--att` | flag | — | — | Enable Advanced Thread Trace |
| `--att-target-cu` | int | 0-15 | 1 | Target CU number |
| `--att-shader-engine-mask` | hex | 1-~0u | 0x1 | SE mask (`0xf` = 4 SEs) |
| `--att-simd-select` | hex | 0-0xF | gfx9: 0xF | SIMD mask |
| `--att-buffer-size` | bytes | 1MB-2GB | 96MB | Trace buffer size |
| `--att-activity` | int | 1-16 | — | **Required for MI300**, perfmon streaming level |
| `--att-perfcounter-ctrl` | int | 1-32 | — | SQ performance counter sampling rate |
| `--att-perfcounters` | string | SQ-only | — | Specify SQ counter list |
| `--att-serialize-all` | bool | — | false | Serialize non-traced kernels |
| `--att-consecutive-kernels` | int | ≥0 | — | Profile N consecutive kernels |
| `--att-library-path` | path | — | /opt/rocm/lib | Trace decoder .so path |
| `--att-gpu-index` | int list | — | — | Only profile specified GPUs |
| `--kernel-include-regex` | string | — | — | Filter kernels by name |
| `--kernel-exclude-regex` | string | — | — | Exclude specified kernels |
| `--kernel-iteration-range` | list | — | — | Dispatch iteration range |
| `-i` | path | — | — | YAML/JSON input configuration file |
| `-d` | path | — | — | Output directory |

## Related Documents

- **General rocprofv3 Usage**: [AMD rocprofv3 Profiling Guide](../../common/rocprofv3-profiling-guide.md) — general features such as tracing, counter collection, output formats
- **CDNA4 Instruction-Level Analysis**: [CDNA4 rocprofv3 Profile](../gfx950/profiling_guide.md)
- **NVIDIA Counterpart**: [NCU Profiling Guide](../../../nvidia/common/ncu-profiling-guide.md) — complete NVIDIA Nsight Compute usage
- **Prerequisite Knowledge**: [GPU Instruction-Level Optimization](../../../generic/gpu-instruction-optimization.md) — Roofline analysis principles
