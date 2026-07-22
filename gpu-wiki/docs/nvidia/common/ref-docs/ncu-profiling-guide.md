# NVIDIA Nsight Compute (NCU) Profiling Guide

## Overview

NVIDIA Nsight Compute (ncu) is a non-interactive command-line profiler at the CUDA kernel level. It works by injecting a measurement library into the application process, intercepting CUDA driver communication, and collecting GPU performance metrics upon detecting a kernel launch.

## Basic Usage

### Simplest Command

```bash
# Basic profiling, output to terminal
ncu ./app

# Save report file (.ncu-rep)
ncu -o profile ./app

# Collect complete metrics
ncu --set full -o full_profile ./app
```

### Output Formats

| Format | Command | Purpose |
|------|------|------|
| Terminal output | `ncu ./app` | Quick inspection |
| Report file | `ncu -o report ./app` | Analyze in ncu-ui |
| CSV | `ncu --csv --page raw ./app` | Script processing |
| Import existing report | `ncu -i report.ncu-rep` | Offline analysis |

### Report Page Types

- **details** (default): Three-column table in sections (metric name, unit, value) + rule analysis results
- **raw**: All raw metrics, including device properties and launch information
- **source**: SASS/PTX/CUDA source code associated with metrics
- **session**: Launch settings, session information, process information, device properties

```bash
ncu --page source --print-source cuda,sass ./app
ncu --page raw --csv -i report.ncu-rep
```

## Kernel Filtering

### Filter by Name

```bash
ncu -k foo ./app                     # Exact match "foo"
ncu -k regex:foo ./app               # Contains "foo"
ncu -k regex:"foo|bar" ./app         # Match "foo" or "bar"
```

### Filter by Launch Count

```bash
ncu -s 5 -c 10 ./app                 # Skip first 5, profile next 10
ncu --launch-skip-before-match 100 -c 1 ./app  # Skip first 100 total launches
ncu -k myKernel -c 1 --kill yes ./app          # Collect target then terminate program
```

### Filter by Kernel ID

Format: `context-id:stream-id:[name-operator:]kernel-name:invocation-nr`

```bash
ncu --kernel-id ::foo:2 ./app            # 2nd call of foo
ncu --kernel-id 1|5:2::7 ./app           # ctx 1 or 5, stream 2, 7th call
```

### Filter by NVTX Range

```bash
ncu --nvtx --nvtx-include "A_range/" ./app                    # push/pop range
ncu --nvtx --nvtx-include "Domain-A@A_range/" ./app           # Specify domain
ncu --nvtx --nvtx-include "A_range/*/B_range" ./app           # Nested range
```

### Filter by Device

```bash
ncu --devices 0,2 ./app
```

### Filter Mode

| Mode | Description |
|------|------|
| `global` | All launches share skip/count (default) |
| `per-gpu` | Each device counted independently |
| `per-launch-config` | Counted independently by grid/block/shared memory configuration |

### Filter by Python Call Stack

```bash
ncu --call-stack-type python --python-include train.py@forward python app.py
```

## Section and Metric System

### Section Set

| Set | Description |
|-----|------|
| `basic` (default) | High-level utilization, static launch/occupancy data |
| `full` | All available sections, most comprehensive metrics but highest overhead |

### Core Sections

| Section | Purpose |
|---------|------|
| **SpeedOfLight** | Compute and Memory throughput SOL%, starting point for bottleneck analysis |
| **ComputeWorkloadAnalysis** | SM compute resources, IPC, pipeline utilization |
| **MemoryWorkloadAnalysis** | Memory resources: Mem Busy, Max Bandwidth, Mem Pipes Busy |
| **LaunchStats** | Kernel grid/block configuration summary |
| **Occupancy** | Active warps / maximum possible warps |
| **SchedulerStats** | Theoretical/Active/Eligible/Issued Warps |
| **WarpStateStats** | Warp state cycle analysis, latency per instruction |
| **InstructionStats** | SASS instruction distribution and frequency |
| **SourceCounters** | Branch efficiency, warp stall reason sampling |
| **PM Sampling** | Metrics sampled periodically on a timeline |

### Query and Selection

```bash
ncu --list-sets # column set
ncu --list-sections # column section
ncu --query-metrics # column metric
ncu --query-metrics --chips gh100 --query-metrics-mode all #

# section
ncu --section LaunchStats --section Occupancy ./app
ncu --section "regex:.*Stats" ./app

# metric
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./app
ncu --metrics "regex:sm__.*" ./app
ncu --metrics "breakdown:sm__throughput.avg" ./app
```### Metric Naming Conventions

Format: `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)`

Example: `l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_active`

### Metric Types and Suffixes

**Counter** has four roll-up types: `.sum`, `.avg`, `.min`, `.max`, each with sub-metrics:
- `.pct_of_peak_sustained_active` — percentage of peak within active cycles
- `.pct_of_peak_sustained_elapsed` — percentage of peak over total elapsed time
- `.per_cycle_active`, `.per_cycle_elapsed`, `.per_second`

**Ratio**: `.pct`, `.ratio`, `.max_rate`

**Throughput**: `.pct_of_peak_sustained_active`, `.pct_of_peak_sustained_elapsed`

### Peak Rate Concepts

- **Burst rate**: the maximum rate that can be reported in a single clock cycle
- **Sustained rate**: the maximum rate achievable over an infinitely long measurement period
- Burst percentage never exceeds 100%; Sustained may slightly exceed 100% in edge cases

## Replay Modes

Since not all metrics can be collected in a single pass, the kernel may need to be replayed across multiple passes.

### Kernel Replay (Default)

```bash
ncu --replay-mode kernel ./app
```

GPU memory is saved before profilingender, and kernel-written memory is restored between each pass. If only a single pass is needed, no save/restore is performed. Suitable for most scenarios.

### Application Replay

```bash
ncu --replay-mode application ./app
```

Re-runs the entire application multiple times. Requires deterministic execution. No memory save/restore needed. Supports kernels with host dependencies.

Configuration: `--app-replay-match` (name/grid/all), `--app-replay-mode` (strict/balanced/relaxed)

### Range Replay

```bash
ncu --replay-mode range --nvtx --nvtx-include "MyRange/" ./app
```

Captures and replays a complete range of CUDA API calls. Supports concurrent kernel profiling. Defined via `cu(da)ProfilerStart/Stop` or NVTX ranges.

### Application-Range Replay

```bash
ncu --replay-mode app-range ./app
```

Profiles a range by re-running the application, without capturing/replaying API state. Requires deterministic execution.

## Key Performance Metric Interpretation

### SOL% (Speed of Light)

The SpeedOfLight section reports the percentage of theoretical peak achieved for compute and memory. It is the starting point for bottleneck analysis:
- High Compute SOL% → compute-bound
- High Memory SOL% → memory-bound
- Both low → latency or occupancy issues exist

### Roofline Analysis

Uses arithmetic intensity on the X-axis and achieved performance on the Y-axis:
- Below the memory ceiling → needs better data reuse
- Below the compute ceiling → needs algorithm optimization
- Left of the ridge point = memory-bound, right = compute-bound

### Memory Bottleneck Identification

| Metric | Meaning |
|------|------|
| Mem Busy | Whether hardware units are fully utilized |
| Max Bandwidth | Whether inter-unit communication bandwidth is exhausted |
| Mem Pipes Busy | Whether memory instruction issue has reached maximum throughput |

### Memory Hierarchy Key Terminology

| Term | Definition |
|------|------|
| **Sector** | A 32-byte aligned block within a cache line (cache line = 4 sectors = 128 bytes) |
| **Request** | A command to access one or more sectors |
| **Wavefront** | A unique work packet processed in parallel; different wavefronts are serialized |

### Occupancy Analysis

Occupancy = active warps / maximum possible warps per SM.

Limiting factors (viewed via `launch__occupancy_limit_*`):
- `registers` — register usage
- `shared_mem` — shared memory usage
- `warps` — block size
- `blocks` — maximum blocks per SM
- `barriers` — barrier count
High occupancy does not guarantee high performance, but low occupancy always reduces the ability to hide latency.

### Warp Scheduling Analysis

Warp states:
- **Active/Resident**: mapped to a sub-partition
- **Eligible**: ready to issue — instruction decoded, dependencies resolved, functional unit available
- **Stalled**: waiting for instruction fetch, memory dependencies, execution dependencies, or synchronization barriers

Key metrics in SchedulerStats:
- Theoretical Warps: upper bound from the launch configuration
- Active Warps: warps allocated per cycle
- Eligible Warps: active warps that are not stalled
- Issued Warp: the warp selected to issue an instruction

"The more issue slots are skipped, the worse the latency hiding."

### SM Pipeline Utilization

| Pipeline | Purpose |
|----------|------|
| `fma` | FP32 arithmetic, FP16, dot products |
| `alu` | Bit operations, logic, integer (excluding IMAD/IMUL) |
| `fp64` | Double-precision floating point |
| `lsu` | Load/Store/Atomic/Reduction to L1TEX |
| `tensor` / `tc` | Matrix multiply-accumulate (MMA) instructions |
| `xu` | Transcendental functions (sin, cos, rsqrt), type conversions |
| `tex` | Texture and surface instructions |
| `tma` | Tensor Memory Accelerator |

When pipeline utilization is very high, it can limit overall performance.

## Clock and Cache Control

### Ensuring Reproducible Results

```bash
# defaultrow: SM + flush cache
ncu --clock-control base --cache-control all ./app

# nsight systems , use
nvidia-smi -lgc <freq>
ncu --clock-control none ./app

# profiling
ncu --clock-control reset
```

### Notes

- NCU serializes kernel launches by default (except for Range replay)
- NCU and Nsight Systems measure duration differently; NCU typically reports longer times
- It is recommended to enable GPU persistence mode to avoid initialization overhead

## Multi-Process / MPI Profiling

### Single-Node Full Rank Profiling

```bash
ncu --target-processes all -o report mpirun [args] ./app
```

### Per-Rank Independent Reports

```bash
mpirun [args] ncu -o report_%q{OMPI_COMM_WORLD_RANK} ./app
```

### Using a Wrapper to Profile Only Specific Ranks

```bash
#!/bin/bash
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
    ncu -o report_${OMPI_COMM_WORLD_RANK} --target-processes all "$@"
else
    "$@"
fi
```

### Scenarios Requiring Concurrent Kernels (NCCL/NVSHMEM)

Cross-process tree (TCP):
```bash
mpirun -np 4 ncu --communicator tcp --communicator-num-peers 4 \
    --lockstep-kernel-launch -o report ./app
```
Same-process tree (shared memory):
```bash
ncu --communicator shmem --communicator-shmem-num-peers 2 \
    -f -o report -k regex:nccl.* torchrun --nproc_per_node=2 app.py
```

### Multi-Node Slurm

```bash
#!/bin/bash
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname)

srun ncu \
    --communicator=tcp \
    --communicator-tcp-hostname="$head_node_addr" \
    --communicator-tcp-num-peers="$SLURM_NTASKS" \
    --lockstep-kernel-launch \
    ./application
```

### Process Filtering

```bash
ncu --target-processes-filter MatrixMul ./app #
ncu --target-processes-filter regex:Matrix ./app #
ncu --target-processes-filter exclude:launcher.exe ./app #
```

## CUDA Graph Profiling

```bash
ncu --graph-profiling node ./app # by kernel profile(default)
ncu --graph-profiling graph ./app # entire graph workload profile
```

Primary use cases for Graph mode: profiling graphs with mandatory concurrent kernel nodes, and more accurate cross-node caching behavior.

## PM Sampling and Warp Sampling

### PM Sampling

Periodically samples performance counters, providing a timeline view of behavioral changes during kernel execution.

```bash
ncu --metrics "pmsampling:sm__throughput" ./app
ncu --pm-sampling-interval 1000 ./app # sampling(cycles/ns)
```

### Warp Sampling

Periodically samples warp states to identify stall patterns.

```bash
ncu --warp-sampling-interval 10 ./app # sampling = 2^(5+value) cycles
```

## Common CLI Options Quick Reference

### Profiling Control

| Option | Purpose | Default |
|------|------|--------|
| `-k, --kernel-name` | Filter kernels by name | All |
| `-c, --launch-count` | Maximum number of profiling passes | Unlimited |
| `-s, --launch-skip` | Skip the first N matching launches | 0 |
| `--set` | Section set (basic/full) | basic |
| `--section` | Specify a section | basic set |
| `--metrics` | Specify a metric | — |
| `--replay-mode` | kernel/application/range/app-range | kernel |
| `--devices` | Specify GPU device | All |
| `--kill` | Terminate application after collection | no |
| `--profile-from-start` | Whether to start profiling from launch | yes |

### Output Control

| Option | Purpose | Default |
|------|------|--------|
| `-o, --export` | Output report file path | Temporary file |
| `-f, --force-overwrite` | Force overwrite existing files | No |
| `-i, --import` | Import report file | — |
| `--csv` | CSV format output | — |
| `--page` | Report page type | details |
| `--print-summary` | Summary mode (per-kernel/per-gpu) | none |
| `--print-source` | Source type (sass/ptx/cuda) | sass |
| `--open-in-ui` | Open in ncu-ui | — |

### Environment and Control

| Option | Purpose | Default |
|------|------|--------|
| `--clock-control` | base/boost/none/reset | boost |
| `--cache-control` | all/none | all |
| `--nvtx` | Enable NVTX support | false |
| `--target-processes` | application-only/all | all |

### Filename Macros

| Macro | Expands To |
|----|--------|
| `%h` | Hostname |
| `%q{ENV_NAME}` | Environment variable value |
| `%p` | ncu process ID |
| `%i` | Smallest unused positive integer |

## Quick Reference for Useful Commands

```bash
# Basic profiling
ncu -o profile ./app

# Full metrics collection
ncu --set full -o full_profile ./app

# Profile specific kernel, limit count
ncu -k regex:matmul -c 3 -s 2 -o report ./app

# CSV output for script processing
ncu --csv --page raw --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./app

# With source correlation
ncu --page source --print-source cuda,sass --section SpeedOfLight ./app

# Application replay (non-replayable kernels)
ncu --replay-mode application -c 1 ./app

# NVTX range replay
ncu --replay-mode range --nvtx --nvtx-include "training_step/" ./app

# Query specific chip metrics
ncu --query-metrics --chips gh100 --query-metrics-mode all

# Import and re-filter report
ncu -i old.ncu-rep -o filtered.ncu-rep -k regex:conv -c 5

# Per-kernel summary
ncu --print-summary per-kernel ./app

# Reset clock
ncu --clock-control reset
```

## Metric Accuracy Notes

Reasons why results may contain out-of-range values:
- **Asynchronous GPU activity**: Other engines (display, copy, video) accessing shared resources (L2, DRAM, PCIe)
- **Multi-pass collection**: Workload distribution varies across passes, e.g., hit rate hits and queries collected in different passes may have significant errors

Mitigation: Increase workload size (>20 us), reduce concurrent GPU processes, reduce the number of metrics collected simultaneously.

## Remote Profiling

```bash
# Target machine
ncu --mode launch ./app

# Host (attach)
ncu --mode attach --hostname <target-ip>
```

Default port 49152, up to 64 connections.

## Configuration File

Default configuration file `config.ncu-cfg` search order: CWD → `$HOME/.config/NVIDIA Corporation` (Linux).

```ini
[Launch-and-attach]
-c = 1
--section = LaunchStats, Occupancy

[Import]
--open-in-ui
```

Also supports Response File: `ncu @args.txt ./app`

## Related Documentation

- **Prerequisites**: [GPU Instruction-Level Optimization](../../../generic/ref-docs/gpu-instruction-optimization.md) — Roofline analysis principles
- **Hopper Specialization**: [Hopper ncu Profiling Details](../../hopper/ref-docs/gluon/profiling_guide.md) — ncu usage for Gluon kernels
- **AMD Counterpart**: [CDNA3 rocprofv3 Profile](../../../amd/cdna3/ref-docs/gluon/profiling_guide.md) | [CDNA4 rocprofv3 Profile](../../../amd/cdna4/ref-docs/gluon/profiling_guide.md) — AMD profiling
- **Hardware Specifications**: `hardware_specs.md` per architecture — Peak TFLOPS required for SOL% calculation
