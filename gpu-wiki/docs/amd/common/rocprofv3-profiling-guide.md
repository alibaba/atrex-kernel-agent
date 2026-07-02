# AMD rocprofv3 Profiling Guide

rocprofv3 is a CLI profiling tool on the AMD ROCm platform, installed at `/opt/rocm/bin`. Application tracing and kernel counter collection can be performed without modifying source code.


**Last updated**: 2026-06-30

> This document covers general usage of rocprofv3 (tracing, counter collection, output formats, etc.). For instruction-level analysis (ATT/Thread Trace), refer to the architecture-specific profiling_guide.

## Basic Usage

```bash
# syntax
rocprofv3 <options> -- <application_path>

# English note
rocprofv3 --version

# column counter
rocprofv3 --list-avail
```

## Application Tracing

Tracing provides execution time data for API calls and GPU commands (kernel execution, asynchronous memory copies, barriers).

### Trace Type Quick Reference

| Option | Description | Output File |
|------|------|---------|
| `--kernel-trace` | Kernel dispatch information | `kernel_trace.csv` |
| `--memory-copy-trace` | Memory copies (hipMemcpy) | `memory_copy_trace.csv` |
| `--hip-trace` | HIP Runtime + Compiler API | `hip_api_trace.csv` |
| `--hip-runtime-trace` | HIP Runtime API only | `hip_api_trace.csv` |
| `--hsa-trace` | HSA low-level API (advanced users) | `hsa_api_trace.csv` |
| `--marker-trace` | ROCTx marker annotations | `marker_api_trace.csv` |
| `--rccl-trace` | RCCL communication collectives | `rccl_api_trace.csv` |
| `--memory-allocation-trace` | Memory allocation/deallocation | `memory_allocation_trace.csv` |
| `--scratch-memory-trace` | Scratch memory (register spilling) | `scratch_memory_trace.csv` |
| `--kokkos-trace` | Kokkos framework integration | `marker_api_trace.csv` |

### Aggregate Trace Shortcuts

```bash
# Runtime trace(recommended)
# : HIP runtime API + marker + RCCL + memory ops + kernel dispatch
# : HIP compiler API, HSA API
rocprofv3 --runtime-trace --output-format csv -- ./app

# System trace
# runtime trace + HSA API
rocprofv3 --sys-trace --output-format csv -- ./app
```

### Important Notes

**`--hip-trace` does not automatically enable kernel trace and memory copy trace**. They need to be explicitly combined:

```bash
# Correct: HIP API + kernel + memory copy
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --output-format csv -- ./app

# ordirect --runtime-trace
rocprofv3 --runtime-trace --output-format csv -- ./app
```

### Common Trace Commands

```bash
# Kernel execute(, grid/block size, VGPR/SGPR/LDS )
rocprofv3 --kernel-trace --output-format csv -- ./app

# memorycopyelapsed time
rocprofv3 --memory-copy-trace --output-format csv -- ./app

# Scratch memory (register)
rocprofv3 --scratch-memory-trace --output-format csv -- ./app

# RCCL profiling
rocprofv3 --rccl-trace --output-format csv -- ./app
```

### Kernel Trace Output Fields

| Field | Description |
|------|------|
| Kernel_Name | Kernel name |
| Agent_Id | Agent ID of the executing GPU |
| Queue_Id / Stream_Id | HSA Queue and HIP Stream |
| Dispatch_Id | Dispatch number |
| Start_Timestamp / End_Timestamp | Start and end timestamps |
| LDS_Block_Size | LDS usage |
| Scratch_Size | Scratch memory size |
| VGPR_Count / SGPR_Count | Register usage |
| Workgroup_Size_X/Y/Z | Block size |
| Grid_Size_X/Y/Z | Grid size |

## Counter Collection

### List Available Counters

```bash
rocprofv3 --list-avail
```

Lists all available metrics defined in `counter_defs.yaml`.

### Command-Line Collection

```bash
# or
rocprofv3 --pmc COUNTER1 COUNTER2 -- ./app
rocprofv3 --pmc COUNTER1,COUNTER2 -- ./app
```

**Note**: If the specified counters cannot be collected in a single pass, the command will fail. You need to reduce the number of counters.

### Collection via Input File

```bash
# JSON, YAML,
rocprofv3 -i input.txt -- ./app
rocprofv3 --input config.yaml -- ./app
```

- **JSON/YAML**: Supports all configuration options (tracing + profiling)
- **Plain text**: Only supports HW counter specification

### Additional Counter Definitions

```bash
rocprofv3 --extra-counters extra_counters.yaml -- ./app
```

## Output Format

| Format | Option | Description |
|------|------|------|
| **rocpd** | Default | SQLite3 database, can be converted to other formats using the rocpd module |
| **csv** | `-f csv` | Comma-separated, convenient for script processing |
| **json** | `-f json` | JSON format |
| **pftrace** | `-f pftrace` | Perfetto trace, visualized with [Perfetto UI](https://ui.perfetto.dev/) |
| **otf2** | `-f otf2` | Open Trace Format 2 |

```bash
# output
rocprofv3 --kernel-trace --output-format csv -- ./app

# output
rocprofv3 --sys-trace -f csv json pftrace -- ./app

# outputdirectoryfile
rocprofv3 --kernel-trace -o my_profile -d ./results -f csv -- ./app
```

The output filename is prefixed with the process ID, for example `238_kernel_trace.csv`.

### Filename Macros

```bash
# defaultpath: %hostname%/%pid%
rocprofv3 --kernel-trace -o profile_%hostname% -d ./output -- ./app
```

## Kernel Filtering

### Filter by Name (Regex)

```bash
# kernel
rocprofv3 --kernel-include-regex "matmul_kernel" --pmc COUNTER1 -- ./app

# kernel( include after)
rocprofv3 --kernel-exclude-regex "setup_kernel" --pmc COUNTER1 -- ./app

# use
rocprofv3 --kernel-include-regex "gemm" --kernel-exclude-regex "debug" --pmc COUNTER1 -- ./app
```

### Filter by Iteration Count

```bash
rocprofv3 --kernel-iteration-range 5-10 --pmc COUNTER1 -- ./app
```

### Filter by Time Window

Format: `-P (START_DELAY):(COLLECTION_TIME):(REPEAT)`

```bash
# 10 seconds, 10 seconds, 1
rocprofv3 -P 10:10:1 --sys-trace -- ./app

# 5 seconds, 3 seconds, none
rocprofv3 -P 5:3:0 --sys-trace -- ./app

# secondsunit
rocprofv3 -P 500:1000:2 --collection-period-unit msec --sys-trace -- ./app

# English note
rocprofv3 -P 10:10:1 5:3:0 --sys-trace -- ./app
```

## Statistics and Summary

```bash
# (must tracing )
rocprofv3 --stats --hip-trace --kernel-trace -- ./app

# English note
rocprofv3 -S --sys-trace -- ./app

# by domain
rocprofv3 -D --sys-trace -- ./app

# by
rocprofv3 --summary-groups 'KERNEL_DISPATCH|MEMORY_COPY' --sys-trace -- ./app

# outputfile, definitionunit
rocprofv3 -S --summary-output-file results.txt -u msec --sys-trace -- ./app
```

**Note**: `--stats` does not include default kernel statistics (different from the legacy rocprof).

## Kernel Naming Control

```bash
# mangled
rocprofv3 -M --kernel-trace -- ./app

# demangled (high)
rocprofv3 -T --kernel-trace -- ./app

# ROCTx region kernel
rocprofv3 --kernel-rename --kernel-trace --marker-trace -- ./app
```

## Process Attachment (Dynamic Profiling)

Profile without restarting the application:

```bash
rocprofv3 --attach <PID> --sys-trace
rocprofv3 -p <PID> --kernel-trace
```

## Multi-GPU Profiling

The `Agent_Id` field in the kernel trace output identifies the executing GPU. The `Source_Agent_Id` / `Destination_Agent_Id` fields in the memory copy trace track cross-GPU data movement.

```bash
# by HSA Queue (default HIP Stream )
rocprofv3 --group-by-queue --kernel-trace -f pftrace -- ./app
```

## PC Sampling (Beta)

```bash
rocprofv3 --pc-sampling-beta-enabled \
          --pc-sampling-unit time \
          --pc-sampling-method host_trap \
          --pc-sampling-interval 1000 \
          -- ./app
```

Currently only `time` units and the `host_trap` method are supported.

## Perfetto Integration

```bash
# In-process (default)
rocprofv3 --sys-trace -f pftrace -- ./app

# System (requires traced daemon row)
rocprofv3 --sys-trace -f pftrace --perfetto-backend system -- ./app

# definition
rocprofv3 --sys-trace -f pftrace \
          --perfetto-buffer-size 2097152 \
          --perfetto-buffer-fill-policy ring_buffer \
          -- ./app
```Default buffer 1 GB, default shared memory 64 KB.

## Complete CLI Options Quick Reference

### I/O Control

| Option | Purpose | Default |
|------|------|--------|
| `-i INPUT` | Input config file (JSON/YAML/text) | — |
| `-o OUTPUT_FILE` | Output filename | `%hostname%/%pid%` |
| `-d OUTPUT_DIRECTORY` | Output directory | `%hostname%/%pid%` |
| `-f FORMAT` | Output format (csv/json/pftrace/otf2/rocpd) | rocpd |
| `--output-config` | Generate parsed config file | — |
| `--log-level` | Log level (fatal/error/warning/info/trace) | — |
| `-E FILE` | Extra counter definition YAML | — |

### Tracing Control

| Option | Purpose |
|------|------|
| `-r` / `--runtime-trace` | Recommended: HIP runtime + marker + RCCL + memory + kernel |
| `-s` / `--sys-trace` | Full: runtime trace + HSA API |
| `--hip-trace` | HIP Runtime + Compiler API |
| `--kernel-trace` | Kernel dispatch |
| `--memory-copy-trace` | Memory copy |
| `--memory-allocation-trace` | Memory allocation/deallocation |
| `--scratch-memory-trace` | Scratch memory |
| `--marker-trace` | ROCTx marker |
| `--rccl-trace` | RCCL communication |
| `--hsa-trace` | HSA low-level API |

### Filtering Control

| Option | Purpose |
|------|------|
| `--kernel-include-regex` | Include kernels by regex |
| `--kernel-exclude-regex` | Exclude kernels by regex |
| `--kernel-iteration-range` | By iteration count range |
| `-P START:TIME:REPEAT` | By time window |
| `--collection-period-unit` | Time window unit |

### Counter Collection

| Option | Purpose |
|------|------|
| `--pmc COUNTER [...]` | Specify PMC counter |
| `-L` / `--list-avail` | List available counters |

### Output Control

| Option | Purpose |
|------|------|
| `--stats` | Collect statistics |
| `-S` / `--summary` | Show single-pass summary |
| `-D` / `--summary-per-domain` | Summary by domain |
| `--summary-groups REGEX` | Group summary by regex |
| `-u UNIT` | Summary time unit (sec/msec/usec/nsec) |
| `-M` | Keep mangled kernel names |
| `-T` | Truncate kernel names |
| `--kernel-rename` | Rename using ROCTx regions |

### Process and Device

| Option | Purpose |
|------|------|
| `-p PID` / `--attach PID` | Attach to running process |
| `--group-by-queue` | Group by HSA Queue |
| `--preload LIB` | Preload library |

## Useful Command Quick Reference

```bash
# fast kernel execute
rocprofv3 --kernel-trace --output-format csv -- ./app

# tracing(recommended)
rocprofv3 --runtime-trace --output-format csv -- ./app

# counter
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU -- ./app

# kernel counter
rocprofv3 --kernel-include-regex "gemm" --pmc SQ_WAVES -- ./app

# Perfetto
rocprofv3 --sys-trace -f pftrace -- ./app
# https://ui.perfetto.dev/ .pftrace file

# English note
rocprofv3 -P 30:10:0 --collection-period-unit sec --runtime-trace -f csv -- ./app

# row
rocprofv3 --attach $(pidof my_app) --kernel-trace -f csv

# Scratch memory (register)
rocprofv3 --scratch-memory-trace --kernel-trace --output-format csv -- ./app

# RCCL profiling
rocprofv3 --rccl-trace --kernel-trace --output-format csv -- ./app

# column counter
rocprofv3 --list-avail
```

## Mapping with NVIDIA Nsight Compute

| Function | rocprofv3 | ncu |
|------|-----------|-----|
| Kernel info | `--kernel-trace` | Default output |
| API tracing | `--hip-trace` / `--sys-trace` | N/A (handled by nsight systems) |
| Counter collection | `--pmc` / `-i input` | `--metrics` / `--set` |
| Output format | csv/json/pftrace/otf2/rocpd | ncu-rep/csv |
| Kernel filtering | `--kernel-include-regex` | `-k` / `--kernel-name` |
| Time window | `-P delay:duration:repeat` | `--profile-from-start` + `cu(da)ProfilerStart/Stop` |
| Process attach | `--attach PID` | `--mode attach` |
| Visualization | Perfetto UI | ncu-ui |
| Instruction-level analysis | ATT (see architecture-specific profiling_guide) | Source page (`--page source`) |## Related Documents

- **NVIDIA counterpart**: [NCU Profiling Guide](../../nvidia/common/profiling/ncu-profiling-guide.md) — Full usage of NVIDIA Nsight Compute
- **CDNA3 Instruction-level Analysis**: [rocprofv3 ATT Detailed Guide](../gluon/gfx942/profiling_guide.md) — ATT Thread Trace usage
- **CDNA4 Instruction-level Analysis**: [CDNA4 rocprofv3 Profile](../gluon/gfx950/profiling_guide.md)
- **Prerequisites**: [GPU Instruction-level Optimization](../../generic/gpu-instruction-optimization.md) — Roofline analysis principles


## Related

- [aiter Optimization Techniques in Detail](aiter-optimization-techniques.md)
- [AMD GPU Kernel Optimization Framework Overview](amd-kernel-optimization-frameworks.md)
- [Community AMD GPU Kernel Optimization](amd-kernel-optimization.md)
- [AMD MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md)
- [Composable Kernel (CK) Architecture Overview](ck-architecture-overview.md)
- [NVIDIA Nsight Compute (NCU) Profiling Guide](../../nvidia/common/profiling/ncu-profiling-guide.md)
