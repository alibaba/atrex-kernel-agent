# GPU Wiki Documentation

Document root for the GPU kernel programming and optimization knowledge base, covering two major areas: optimization knowledge and code conversion.

---

| Directory | Description |
|------|------|
| [hardware-specs/](hardware-specs/) | Hardware compute specification tables for all target GPUs: [MI300X](hardware-specs/hardware_specs_mi300x.md), [MI308X](hardware-specs/hardware_specs_mi308x.md), [MI355X](hardware-specs/hardware_specs_mi355x.md), [Hopper](hardware-specs/hardware_specs_hopper.md), [Blackwell](hardware-specs/hardware_specs_sm120.md), and [cross-architecture comparison](hardware-specs/hardware-comparison-cdna3-cdna4.md) |
| [kernel-opt/](kernel-opt/) | GPU optimization knowledge: general theory, AMD (CDNA3/CDNA4), NVIDIA (Hopper/Blackwell) architecture-specific optimization; includes [Blackwell knowledge cards](kernel-opt/nvidia/common/blackwell/) |
| [ref-docs/](ref-docs/) | Framework/architecture reference materials and per-kernel optimization journeys: FlyDSL, CuTeDSL, Gluon; includes [SM120 hardware specs](hardware-specs/hardware_specs_sm120.md) |
| [pitfalls/](pitfalls/) | Non-obvious pitfalls encountered during implementation/porting: trap → symptom → root cause → lesson |
| [converter/](converter/) | Code conversion knowledge: PyTorch→Triton, Triton→Gluon cross-platform conversion rules |
| [RELATIONS.md](RELATIONS.md) | Document relationship diagram: reading paths, cross-architecture comparisons, conflict and discrepancy lists |
