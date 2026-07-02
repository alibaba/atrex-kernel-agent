# Profiling Tools Overview

| Tool | Purpose |

**Last updated**: 2026-07-01

|------|---------|
| `rocprof` | Low-level hardware performance counters |
| ROCm Compute Profiler | Automatic counter collection, Roofline analysis |
| ROCm Systems Profiler | End-to-end application tracing |
| PyTorch Profiler | High-level CPU/GPU profiling, Perfetto visualization |
| NPKit | Fine-grained trace of RCCL kernels |

## ISA and MLIR Debugging

```bash
export AMDGCN_ENABLE_DUMP=1    # ISA assembly output
export MLIR_ENABLE_DUMP=1      # MLIR IR output
```

## Memory Debugging

```bash
HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 \
HSA_ENABLE_DEBUG=1 ./my_program
# Use -ggdb -O0 when compiling
```

---

## Related

- **Index**: AMD GPU Kernel Tuning Guide — Complete tuning topic index
- **rocprofv3 Details**: [AMD rocprofv3 Profiling Guide](rocprofv3-profiling-guide.md) — Detailed guide on general rocprofv3 usage
- **CDNA3 Profiling**: [rocprofv3 Instruction-Level Profile Details](../gluon/gfx942/profiling_guide.md) — ATT hotspot analysis
- **CDNA4 Profiling**: [ROCm Profiling Guide (CDNA4)](../gluon/gfx950/profiling_guide.md) — Configuration and usage on CDNA4
- **NVIDIA Comparison**: [NCU Profiling Guide](../../nvidia/common/profiling/ncu-profiling-guide.md) — NVIDIA Nsight Compute profiling
