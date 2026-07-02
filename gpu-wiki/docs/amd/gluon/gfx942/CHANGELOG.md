# Changelog for Preview 0.1.4

Release 0.1.4 aligns with ROCm 7.0

**Last updated**: 2026-06-30

- This is a preview release for 0.1.4

### Resolved issues

- Fixed an issue in gfx12 where s_barrier_wait would fail to parse for waves not in a workgroup.

### Changes

- On RDNA GPUs, global_ and scratch_ are now reported as _VMEM (was _FLAT).
  - This was to keep consistency with MI series, but could be misleading.
- Changed headers to reflect recent rocprofiler-SDK changes
  - The ABI is kept the same, there were only field name changes

### Optimization

- Reduced memory usage for large traces on all gfx.
  - \+ Slight improvement parsing speed.


## Related

- [AMD MI308X (gfx942) GEMM Optimization Techniques Reference](ck_gemm_optimization_reference.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [CDNA3 (gfx942) ISA Instruction Patterns and Optimization Reference](isa_patterns.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
