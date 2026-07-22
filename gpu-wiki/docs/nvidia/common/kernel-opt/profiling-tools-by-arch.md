# Profiling Tools for Different Architectures

| Architecture | Tool | Key Metrics |
|------|------|----------|
| All NVIDIA | Nsight Systems (`nsys`) | Timeline, kernel proportion, transfer overlap |
| CC 7.0+ | Nsight Compute (`ncu`) | SOL%, roofline, warp stall causes |
| CC 9.0 | `ncu` + TMA metrics | TMA utilization, cluster efficiency |

## Key Nsight Compute Sections

```bash
# Basic profile
ncu --set full -o report ./my_app

# Memory metrics only
ncu --section MemoryWorkloadAnalysis ./my_app

# Roofline analysis
ncu --section SpeedOfLight_RooflineChart ./my_app
```

## Related Documentation

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [NVIDIA Nsight Compute (NCU) Profiling Guide](../ref-docs/ncu-profiling-guide.md) — Complete NCU usage guide
- [NCU Profile-Driven Optimization Workflow](../ref-docs/ncu-profile-driven-optimization-workflow.md) — Optimization workflow that converges from report/counter/source/PM sampling/PTX-SASS evidence to a single kernel edit
- [Hopper Profiling In-Depth](../../hopper/ref-docs/gluon/profiling_guide.md) — Hopper-specific ncu profiling
