# NVIDIA General Optimization Reference Articles

General optimization reference articles for NVIDIA hardware, covering PTX, NCU profiling, shared memory, pipelining, and community knowledge.

---

| File | Description |
|------|-------------|
| [hierarchical-reduction-memory-bound.md](hierarchical-reduction-memory-bound.md) | Hierarchical Reduction Memory-Bound Optimization |
| [low-precision-accumulation-strategies.md](low-precision-accumulation-strategies.md) | FP8 Low-Precision Accumulation Strategies |
| [ncu-profiling-guide.md](ncu-profiling-guide.md) | Nsight Compute Profiling Guide |
| [ncu-profile-driven-optimization-workflow.md](ncu-profile-driven-optimization-workflow.md) | NCU Profile-Driven Optimization Workflow: collecting reports, exporting counter/source/PM sampling/PTX-SASS evidence, comparing baselines, and converging to a verifiable kernel edit |
| [ncu-rule-est-speedup-meta-rules.md](ncu-rule-est-speedup-meta-rules.md) | Cross-architecture meta-rules: NCU single rule % est. is an architecture upper bound, not a wall-time gain. When it is trustworthy, when it is misleading, and the three-strike convergence signal. Cheap-fix bias > heavyweight. |
| [ncu-measurement-discipline.md](ncu-measurement-discipline.md) | Measurement-trust layer: NCU `Duration` ≠ harness latency (use NCU for ratios only); noise discipline (smoke-test variance, same-process A/B, variance floor) before believing a sub-1× delta; profiling and silent kernel-skip under CUDA graph capture; per-call overhead floor. |
| [nvidia-ptx-mma-instructions.md](nvidia-ptx-mma-instructions.md) | PTX MMA Instruction Evolution |
| [nvidia-ptx-sync-and-async.md](nvidia-ptx-sync-and-async.md) | PTX Synchronization and Asynchronous Primitives |
| [ptx-instruction-set.md](ptx-instruction-set.md) | PTX Instruction Set Reference |
| [ptx-programming-model.md](ptx-programming-model.md) | PTX Programming Model |
| [register-pressure-warp-occupancy.md](register-pressure-warp-occupancy.md) | Register Pressure and Warp Occupancy |
| [smem-swizzling-bank-conflicts.md](smem-swizzling-bank-conflicts.md) | Shared Memory Swizzling and Bank Conflicts |
| [software-pipeline-depth-optimization.md](software-pipeline-depth-optimization.md) | Software Pipeline Depth Optimization |
| [tile-rasterization-l2-locality.md](tile-rasterization-l2-locality.md) | Tile Rasterization and L2 Locality |
| [warp-specialization-design-principles.md](warp-specialization-design-principles.md) | Warp Specialization Design Principles |
| [gpu-architecture-deep-dive.md](gpu-architecture-deep-dive.md) | Community GPU Architecture Deep Dive |
| [nsight-profiling-practice.md](nsight-profiling-practice.md) | Community Nsight Profiling Practice |
| [ptx-sass-programming.md](ptx-sass-programming.md) | Community PTX/SASS Programming |
| [system-level-optimization.md](system-level-optimization.md) | Community System-Level Optimization |

### Subdirectories

| Directory | Description |
|-----------|-------------|
| [sm90/](sm90/) | Hopper (SM90) specific articles |
| [sm120/](sm120/) | Blackwell GeForce / RTX PRO (SM120) hardware specifications and architecture whitepaper |
