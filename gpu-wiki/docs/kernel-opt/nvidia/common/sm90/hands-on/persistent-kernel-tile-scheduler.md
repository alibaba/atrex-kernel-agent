# Persistent Kernel and Tile Scheduler (Hopper Specialized)

> **General Basics**: For the principles and algorithms of Persistent Kernel and GROUP_SIZE_M Swizzle, see [Persistent Kernel and Tile Scheduling](../../../../generic/hands-on/persistent-kernel-tile-scheduling.md). This document only records Hopper-specific configurations and experience.

## Hopper-Specific Configurations

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm90.py`

- H100 has **132 SMs**, typical configuration `num_ctas = 132` (or `132 * ctas_per_sm`)
- Swizzle and persistent kernel must be used together; otherwise, adjacent CTAs access distant memory locations
- For small matrices (tile count < SM count), the CTA launch overhead advantage of persistent kernel is particularly significant

---

## Related Documents

- **General Pattern**: [Persistent Kernel and Tile Scheduling](../../../../generic/hands-on/persistent-kernel-tile-scheduling.md) — GROUP_SIZE_M Swizzle algorithm and practical experience
- **General Optimization**: [Hopper Common ISA Optimization Checklist](../../../../../ref-docs/nvidia/gluon/sm90/common_optimizations.md)
- **Hardware Specifications**: [Hopper Hardware Specification Table](../../../../../hardware-specs/hardware_specs_hopper.md) — SM count
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source code
