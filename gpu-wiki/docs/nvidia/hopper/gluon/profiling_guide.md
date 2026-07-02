# Nsight Compute (ncu) Profiling Guide

**Last updated**: 2026-03-18
**Target Environment**: CUDA 12.x, NVIDIA Hopper (sm_90, H20/H100/H200)

---

## Overview

Nsight Compute (ncu) is NVIDIA's official GPU kernel profiling tool. This guide focuses on **kernel performance analysis in a CLI environment**, usedотор to locate performance hotspots in Gluon kernels.

| Component | Function | Source |
|------|------|------|
| **ncu** (Nsight Compute CLI) | Collect kernel performance data | Bundled with CUDA Toolkit |
| **ncu-ui** (Nsight Compute GUI) | Visual analysis (optional) | Bundled with CUDA Toolkit |
| **cuobjdump** | Extract SASS disassembly | Bundled with CUDA Toolkit |

---

## 1. Basic Profile Commands

### 1.0 Locate the Launch Index of the Target Kernel (Mandatory Step Zero) ⚠️

> **Key Practice**: Before executing the target kernel, Gluon kernel Python scripts typically trigger a significant number of PyTorch internal kernel launches (`torch.randn` random number generation, `torch.cat` copies, elementwise operations, etc.). **If you blindly use a fixed value like `--launch-skip 10`, there is a very high probability that you will profile a PyTorch internal setup kernel rather than the target Gluon kernel.**
>
> **Measured Data**: In a typical Gluon kernel test script, there are **23 PyTorch internal kernel launches** before the target kernel (including `distribution_elementwise_grid_stride_kernel`, `vectorized_elementwise_kernel`, `CatArrayBatchedCopy`, `DeviceScan`, etc.).

```bash
# Step 0: List all kernel launch names and indices in the script
ncu --print-summary per-kernel python <kernel.py>
```

**Sample Output**:
```
==PROF== Profiling "distribution_elementwise_grid..." - 0: ...
==PROF== Profiling "vectorized_elementwise_kernel" - 1: ...
...
==PROF== Profiling "chunk_gated_delta_rule_fwd_ke..." - 23: ...
```

Find the **launch index** of the target Gluon kernel from the output (23 in the example above), and use that index as the value for `--launch-skip` in subsequent steps.

**⚠️ Diagnosing Incorrect Profiling**: If the kernel name in the profile results is not the target Gluon kernel (e.g., `distribution_elementwise_grid_stride_kernel` appears), it means the `--launch-skip` value is incorrect and you must return to this step to re-locate.

### 1.1 Collect Full Profile Data

```bash
# Replace N with the index determined in Step 1.0
# Collect complete data from all sections (most comprehensive)
ncu --set full \
    --launch-skip <N> --launch-count 1 \
    -o profile_output \
    python <kernel.py>
```

**Key Parameters**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--set full` | Collect metrics from all sections | Default usage |
| `--launch-skip N` | Skip the first N kernel launches | **Use the index determined in §1.0** |
| `--launch-count N` | Profile only N launches | 1 |
| `--kernel-name <regex>` | Filter by kernel name (alternative) | Target kernel name |
| `-o <file>` | Output .ncu-rep report file | Must be specified |

### 1.2 Filter Kernel by Name (Alternative)

> When the kernel name is known and unique, `--kernel-name` may be used in place of `--launch-skip`. However, Gluon kernel names may be truncated by the compiler, so **it is recommended to prioritize the index-based location method in §1.0**.

```bash
# Only profile kernels matching the name
ncu --set full \
    --kernel-name "chunk_gated_delta_rule_fwd" \
    --launch-count 1 \
    -o profile_output \
    python <kernel.py>
```

### 1.3 Collect Only Specific Metrics (Faster)

```bash
# Only collect memory and compute throughput
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__occupancy \
    --launch-skip 10 --launch-count 1 \
    python <kernel.py>
```

---

## 2. Viewing Profile Data via CLI

### 2.1 View Report Summary

```bash
# View summary of all sections
ncu --import profile_output.ncu-rep --page raw

# View specific metrics
ncu --import profile_output.ncu-rep --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed
```

### 2.2 Key Metric Categories

#### Compute Throughput

| Metric | Meaning | Used For Diagnosis |
|------|------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput (% of peak) | Overall compute efficiency |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Core activity rate | wgmma utilization |
| `smsp__inst_executed.sum` | Total executed instructions | Instruction throughput |#### Memory Throughput

| Metric | Meaning | Used for Diagnosis |
|------|------|---------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput (% of peak) | HBM bandwidth utilization |
| `dram__bytes_read.sum` | Total DRAM read bytes | Actual data volume |
| `dram__bytes_write.sum` | Total DRAM write bytes | Actual data volume |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/Tex throughput | Shared memory / texture cache |

#### Shared Memory (L1/TEX)

| Metric | Meaning | Used for Diagnosis |
|------|------|---------|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | Shared memory load wavefronts | Number of smem loads |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | Shared memory store wavefronts | Number of smem stores |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory load bank conflicts | Bank conflicts |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | Shared memory store bank conflicts | Bank conflicts |

#### Occupancy / Resources

| Metric | Meaning | Used for Diagnosis |
|------|------|---------|
| `launch__occupancy` | Achieved occupancy (%) | Register / smem pressure |
| `launch__registers_per_thread` | Registers per thread | Register usage |
| `launch__shared_mem_per_block_dynamic` | Dynamic shared memory (bytes) | smem usage |
| `launch__shared_mem_per_block_static` | Static shared memory (bytes) | smem usage |

#### Warp Stall Reasons

| Metric | Meaning | Corresponding Optimization |
|------|------|---------|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | Waiting for global memory | Improve memory access pipeline |
| `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | Waiting for shared memory / L1 | Reduce bank conflicts |
| `smsp__warp_issue_stalled_wait_per_warp_active.pct` | Waiting for synchronization (barrier) | Reduce unnecessary sync |
| `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct` | Math pipeline backpressure | Compute units fully loaded |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` | MIO pipeline backpressure | smem access bottleneck |
| `smsp__warp_issue_stalled_not_selected_per_warp_active.pct` | Warp ready but not scheduled | Insufficient occupancy |

---

## 3. SASS Disassembly Analysis

### 3.1 Extracting SASS

```bash
# Method 1: Extract via ncu report
ncu --import profile_output.ncu-rep --page source --print-source sass

# Method 2: Directly disassemble cubin
cuobjdump -sass <kernel.cubin>

# Method 3: Via nvdisasm
nvdisasm <kernel.cubin>
```

### 3.2 SASS Instruction Identification → Optimization Mapping

| SASS Instruction Pattern | Meaning | Possible Issue | Optimization Step |
|-------------|------|---------|---------|
| `LDG.E.32` | 32-bit global load | Load width insufficient | 3.1 |
| `LDG.E.64` | 64-bit global load | Load width can be optimized | 3.1 |
| `LDG.E.128` | 128-bit global load | ✅ Optimal width | — |
| `STG.E.32` | 32-bit global store | Store width insufficient | 3.1 |
| `STG.E.128` | 128-bit global store | ✅ Optimal width | — |
| `STS.32` | 32-bit shared store | smem store narrow | 3.1 |
| `STS.128` | 128-bit shared store | ✅ Optimal width | — |
| `LDS.32` | 32-bit shared load | smem load narrow | 3.1 |
| `LDS.128` | 128-bit shared load | ✅ Optimal width | — |
| `LDSM` / `LDSM.16.M88.4` | Load shared matrix | wgmma data loading | — |
| `HGMMA.64x*x*.F32.BF16` | Hopper GMMA (Tensor Core) | ✅ wgmma correctly used | — |
| `HMMA` / `WGMMA` | Legacy name (actual SASS as HGMMA) | ✅ Tensor Core usage | — |
| `STL` | Store to local memory | Register spilling ❌ | 3.3 |
| `LDL` | Load from local memory | Register spilling ❌ | 3.3 |
| `LDGSTS` | CP_ASYNC (global→shared) | ✅ async_copy used | — |
| `BAR.SYNC` | Barrier synchronization | Too many may be a bottleneck | 3.4 |
| `DEPBAR` | Dependency barrier | async operation waiting | 3.4/3.5 |---

Step 0: Locate target Gluon kernel launch index ⚠️(Mandatory)
  └─ Find target kernel name and index N from output
  └─ Note: PyTorch setup typically produces 10-30 internal kernel launches

Step 1: Measure kernel runtime
  └─ tools/measure_kernel_time.py → Get ms runtime

Step 2: Calculate compute utilization
  └─ tools/compute_utilization.py → Get utilization %

Step 3: If utilization < 90%, run ncu profile
  └─ ncu --set full --launch-skip <N> --launch-count 1 -o profile python <kernel.py>
  └─ ⚠️ Confirm kernel name in report is target Gluon kernel, not PyTorch internal kernel

Step 4: View key metrics
  └─ ncu --import profile.ncu-rep --metrics <key metric list>

Step 5: Analyze warp stall reasons
  └─ smsp__warp_issue_stalled_* → Locate bottleneck type

Step 6: View SASS disassembly (optional, deeper)
  └─ ncu --import profile.ncu-rep --page source --print-source sass
  └─ Or cuobjdump -sass <kernel.cubin> (get cubin from ~/.triton/cache/)

Step 7: Execute corresponding optimization steps based on diagnosis (3.0 ~ 3.6)

---

## 5. Common Issues and Solutions

### Q: ncu run fails with "Permission denied" or "ERR_NVGPUCTRPERM"
**Cause**: Requires root privileges or profiling permission configuration.
**Solution**:
```bash
# Method 1: Run as root
sudo ncu --set full ...

# Method 2: Set perf permissions
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### Q: Profile data is incomplete (some sections show N/A)
**Cause**: Used `--metrics` instead of `--set full`, or the kernel execution time is too short.
**Solution**:
1. Use `--set full` to collect all sections
2. Increase the kernel's input size or `--launch-count`

### Q: Multiple kernel launches — how to locate the target Gluon kernel?
**Background**: In Gluon kernel Python scripts, PyTorch tensor initialization (`torch.randn`, `torch.cat`, `.cumsum`, etc.) triggers a large number of internal kernel launches before the target kernel. In actual testing, a typical script generated **23 PyTorch internal launches** before the target kernel.
**Solution**:
```bash
# First step (recommended): List all kernel launches and indices, find target kernel
ncu --print-summary per-kernel python <kernel.py>
# Find target kernel name and index in output (e.g., "chunk_gated_delta_rule_fwd_ke..." - 23)

# Second step: Profile with determined index
ncu --set full --launch-skip 23 --launch-count 1 -o profile python <kernel.py>

# Alternative: Filter by name (name may be truncated, need regex match)
ncu --set full --kernel-name "chunk_gated_delta_rule" --launch-count 1 -o profile python <kernel.py>
```
**⚠️ Common Mistake**: Directly using fixed values like `--launch-skip 10` results in profiling PyTorch internal kernels such as `distribution_elementwise_grid_stride_kernel`, yielding data unrelated to the target kernel. See §1.0 for details.

### Q: How to obtain achieved FLOPS / bandwidth?
```bash
ncu --import profile.ncu-rep --metrics \
    sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
    dram__bytes.sum,\
    gpu__time_duration.sum
```

---

## 6. ncu Common Commands Quick Reference

```bash
# Step 0: Locate target kernel launch index (mandatory!)
ncu --print-summary per-kernel python kernel.py

# Complete profile (replace N with index determined in previous step)
ncu --set full --launch-skip <N> --launch-count 1 -o out python kernel.py

# Quick profile (view summary only)
ncu --set basic --launch-skip <N> --launch-count 1 python kernel.py

# View memory and compute throughput only
ncu --section SpeedOfLight --launch-skip <N> --launch-count 1 python kernel.py

# View occupancy limiting factors
ncu --section Occupancy --launch-skip <N> --launch-count 1 python kernel.py

# View warp stall reasons
ncu --section WarpStateStatistics --launch-skip <N> --launch-count 1 python kernel.py

# View SASS source
ncu --import out.ncu-rep --page source --print-source sass

# Import and view all metrics
ncu --import out.ncu-rep --page details

# List all available metrics
ncu --query-metrics
```## 7. Reference Documents

- [Nsight Compute CLI Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Compute Metrics Reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
- [CUDA Profiling Tools Interface](https://docs.nvidia.com/cupti/index.html)
- [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)


## Related

- [Hopper (sm_90) General ISA Optimization Checklist](common_optimizations.md)
- [Fused Attention (Prefill / Paged Attention) Optimization Guide](fused_attention.md)
- [Hopper (sm_90) SASS Instruction Patterns and Optimization Reference](isa_patterns.md)
- [Chunk Linear Attention / Recurrent State Update Optimization Guide](linear_attention.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [rocprofv3 Instruction-Level Profile Details](../../../amd/gluon/gfx942/profiling_guide.md)
- [ROCm Profiling Guide (CDNA4 / gfx950)](../../../amd/gluon/gfx950/profiling_guide.md)
- [NVIDIA Nsight Compute (NCU) Profiling Guide](../../common/profiling/ncu-profiling-guide.md)
- [AMD rocprofv3 Profiling Guide](../../../amd/common/rocprofv3-profiling-guide.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
