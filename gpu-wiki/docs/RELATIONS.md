# Architecture and Knowledge Relationships

This page records the relationships that matter when navigating or transferring
GPU optimization knowledge. The physical directory is the authoritative scope;
similar product names do not imply compatible hardware behavior.

## Reading order

```text
generic fundamentals
        ↓
vendor/common
        ↓
architecture-general knowledge
        ↓
product overlay (when present)
        ↓
kernel-opt → ref-docs + pitfalls → reference-kernels
```

Start from [the documentation router](README.md), or use the scoped query:

```bash
python3 gpu-wiki/scripts/query.py <keywords> --arch <target>
```

## NVIDIA families

| Scope | Products / aliases | Relationship |
|---|---|---|
| [Ampere](nvidia/ampere/) | A100, SM80 | Independent architecture scope |
| [Hopper](nvidia/hopper/) | H20, H100, H200, SM90 | Independent architecture scope |
| [Blackwell](nvidia/blackwell/) | SM100 general | Parent of the B200 overlay |
| [B200](nvidia/blackwell/b200/) | B200, GB200 | Inherits Blackwell general knowledge; product evidence stays here |
| [Blackwell Ultra](nvidia/blackwell-ultra/) | B300, GB300, SM103 | May inherit applicable Blackwell concepts, but excludes B200-only evidence |
| [Blackwell GeForce/workstation](nvidia/blackwell-geforce/) | RTX PRO 5000, Pro5000, SM120 | Separate from SM100 and SM103 despite the shared Blackwell brand |

Important transfer boundaries:

- WGMMA and Hopper pipeline examples do not automatically apply to SM100,
  SM103, or SM120.
- SM100 tcgen05/TMEM examples do not establish availability or identical
  behavior on SM120.
- B200 benchmark results and resource assumptions must not be used as B300
  facts unless a document explicitly compares both products.
- RTX PRO 5000 hardware facts come from the SM120 hardware page and official
  workstation sources, not B200/B300 data-center specifications.

## AMD families

| Scope | Products / aliases | Relationship |
|---|---|---|
| [CDNA3](amd/cdna3/) | gfx942 general | Parent of MI300X and MI308X overlays |
| [MI300X](amd/cdna3/mi300x/) | MI300X | Inherits CDNA3 general knowledge; excludes MI308X-only experiments |
| [MI308X](amd/cdna3/mi308x/) | MI308X | Inherits CDNA3 general knowledge; excludes MI300X-only facts |
| [CDNA4](amd/cdna4/) | MI355X, gfx950 | Independent architecture scope |
| [RDNA4](amd/rdna4/) | gfx1250 | Independent architecture scope |

MI300X and MI308X share gfx942/CDNA3 programming concepts, but their compute
resources and product measurements differ. Use the exact product overlay for
hardware facts and measured optimization evidence.

## Cross-vendor concept mapping

| Concept | NVIDIA terminology | AMD terminology | Transfer rule |
|---|---|---|---|
| Matrix instructions | MMA, WGMMA, tcgen05 | MFMA, WMMA | Transfer the tiling idea, then re-derive instruction shapes and resource use |
| On-chip shared storage | Shared memory / SMEM | LDS | Re-check bank count, banking pattern, capacity, and synchronization |
| Execution group | Warp | Wavefront | Re-check group width and layout mapping |
| Profiling | Nsight Compute / NCU | rocprofv3 | Metrics are not directly interchangeable |
| Async movement | cp.async, TMA | architecture/framework-specific mechanisms | Re-derive supported copies and pipeline semantics |

## Role relationships

- `hardware-specs/` supplies factual inputs for roofline and resource analysis.
- `kernel-opt/` provides a concise decision or optimization pattern.
- `ref-docs/` supplies the detailed evidence and implementation journey.
- `pitfalls/` records failed or unsafe approaches and should be read before
  porting a technique across architectures.
- `converter/` maps code structure and APIs; it does not override hardware
  constraints.
- `reference-kernels/` provides implementation examples, not automatic runtime
  dependencies.

## High-value companion groups

### SM120 / RTX PRO 5000

- [SM120 hardware facts](nvidia/blackwell-geforce/hardware-specs/hardware_specs_sm120.md)
- [SM120 CuTeDSL references](nvidia/blackwell-geforce/ref-docs/cutedsl/)
- [SM120 CuTeDSL pitfalls](nvidia/blackwell-geforce/pitfalls/cutedsl/)
- [SM120 reference kernels](../reference-kernels/nvidia/blackwell-geforce/)

### B200

- [B200 hardware facts](nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md)
- [Blackwell general optimization cards](nvidia/blackwell/kernel-opt/)
- [B200-only reports and pitfalls](nvidia/blackwell/b200/)

### MI308X FlyDSL

- [CDNA3 FlyDSL references](amd/cdna3/mi308x/ref-docs/flydsl/)
- [MI308X-only reports and pitfalls](amd/cdna3/mi308x/)
- [MI308X reference kernels](../reference-kernels/amd/cdna3/flydsl/FlyDSL/)

For any cross-architecture transfer, keep the algorithmic idea but re-check
instruction availability, memory hierarchy, occupancy limits, synchronization,
and measurements inside the destination scope.
