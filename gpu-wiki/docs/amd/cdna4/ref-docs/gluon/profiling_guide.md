# ROCm Profiling Guide (CDNA4 / gfx950)

**Last Updated**: 2026-03-28

---

## Prerequisites

### Install rocprofv3

```bash
# rocprofv3 ROCm
# English note
rocprofv3 --version

# ATT (Async Trace Tool) library path
ATT_LIB_PATH="./tools/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"
```

### Configuration File

`tools/input_att.yaml`:
```yaml
options:
  output_directory: "tt_test"
  trace_dispatch_mode: per_kernel
  roctx_trace: true
  kernel_dispatch: true
```

---

## Step 1: Collect Instruction-Level Trace

```bash
env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
    rocprofv3 --att \
    --att-library-path ./tools/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 \
    -i tools/input_att.yaml \
    -- python <kernel.py>
```

**Output**: `tt_test/stats_*.csv`

### CSV Column Description

| Column | Description |
|------|------|
| Instruction | Assembly instruction |
| Hitcount | Execution count |
| Latency | Total latency cycles = Stall + Issue |
| Stall | Pipeline stall cycles (TCP/LDS backpressure) |
| Idle | Idle cycles (register dependency, icache miss) |
| Source | Corresponding source code line number |

---

## Step 2: Collect Hardware Counters

```bash
rocprofv3 --pmc \
    SQ_LDS_BANK_CONFLICT,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,\
    SPI_RA_VGPR_SGPR_FULL_CSN,TCP_TCC_MISS \
    -d ./pmc_output -- python <kernel.py>
```

### Key Counters

| Counter | Description | Optimization Target |
|--------|------|---------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict count | → 0 |
| `SQ_INSTS_VMEM_RD` | VMEM read instruction count | Reduce |
| `SQ_INSTS_VMEM_WR` | VMEM write instruction count | Reduce |
| `SPI_RA_VGPR_SGPR_FULL_CSN` | VGPR spill stall | → 0 |
| `TCP_TCC_MISS` | L2 cache miss | Reduce |

---

## Step 3: Analyze Hotspot Instructions

```bash
# by Latency descending ordercolumn
sort -t',' -k5 -nr ./profile_output/stats_*.csv | head -20

# by Stall descending ordercolumn
sort -t',' -k6 -nr ./profile_output/stats_*.csv | head -20
```

---

## Step 4: Clean Up Trace Files

**⚠️ Important**: A single trace outputs ~400-500MB. Must be cleaned up after analysis!

```bash
rm -rf tt_test
```

---

## Common Diagnostic Patterns

### Pattern 1: buffer_load High Stall + Low Idle

**Symptoms**: `buffer_load_dword` shows high values in the Stall column and low values in the Idle column

**Cause**: Memory access is not overlapped by computation — typical characteristic of a memory-bound kernel

**Fix**:
- §3.5 Software pipelining (prologue/main/epilogue)
- GEMM: §3.6 warp_pipeline_stage + async_copy

### Pattern 2: ds_read/ds_write High Stall

**Symptoms**: LDS operations have extremely high stall cycles

**Cause**: Bank conflict

**Fix**: §3.2 Adjust swizzle parameters

### Pattern 3: Frequent ds_bpermute_b32

**Symptoms**: Large number of `ds_bpermute_b32` instructions

**Cause**: Layout conversion overhead (convert_layout)

**Fix**: §3.3 Eliminate unnecessary layout switches

### Pattern 4: buffer_store to Scratch Addresses

**Symptoms**: buffer_load/store pointing to scratch space

**Cause**: VGPR spill

**Fix**: §3.4 Reduce block size or reduce intermediate variables

---

## NCU vs rocprofv3 Comparison

| Feature | Nsight Compute (NVIDIA) | rocprofv3 (AMD) |
|------|------------------------|-----------------|
| Instruction-level trace | ✅ | ✅ (--att) |
| Hardware counters | ✅ | ✅ (--pmc) |
| SASS analysis | ✅ | ✅ (ISA) |
| Roofline auto analysis | ✅ | ❌ (requires manual calculation) |
| Output format | .ncu-rep | CSV |

---

## Utility Scripts

### profile_kernel.sh

```bash
#!/bin/bash
# Usage: bash tools/profile_kernel.sh <kernel.py> --wrapper-name <wrapper> --output-dir ./profile_output

KERNEL=$1
shift
OUTPUT_DIR="./profile_output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR=$2; shift 2;;
        *) shift;;
    esac
done

mkdir -p $OUTPUT_DIR

rocprofv3 --att \
    --att-library-path ./tools/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 \
    -i tools/input_att.yaml \
    --python $KERNEL

# analysisresult
sort -t',' -k5 -nr tt_test/stats_*.csv | head -20 > $OUTPUT_DIR/top_latency.csv
sort -t',' -k6 -nr tt_test/stats_*.csv | head -20 > $OUTPUT_DIR/top_stall.csv

# cleanup
rm -rf tt_test
```## Related Documentation

- **General rocprofv3 Usage**: [AMD rocprofv3 Profiling Guide](../../../common/ref-docs/rocprofv3-profiling-guide.md) — tracing, counter collection, output formats, and other general features
- **CDNA3 Instruction-Level Analysis**: [CDNA3 rocprofv3 ATT Detailed Guide](../../../cdna3/ref-docs/gluon/profiling_guide.md)
- **NVIDIA Counterpart**: [NCU Profiling Guide](../../../../nvidia/common/ref-docs/ncu-profiling-guide.md) — complete NVIDIA Nsight Compute usage
- **Prerequisite Knowledge**: [GPU Instruction-Level Optimization](../../../../generic/ref-docs/gpu-instruction-optimization.md) — Roofline analysis principles
