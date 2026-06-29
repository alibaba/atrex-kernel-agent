# CuTeDSL / CUTLASS / QuACK Reference Articles

Reference articles on the CuTeDSL programming model, CUTLASS 3.x architecture, and the QuACK high-performance kernel library.

---

| File | Description |
|------|-------------|
| [cutedsl-api-reference-guide.md](cutedsl-api-reference-guide.md) | CuTeDSL API Reference Guide |
| [cutedsl-gemm-allreduce-fusion.md](cutedsl-gemm-allreduce-fusion.md) | CuTeDSL Communication: GEMM and AllReduce Fusion |
| [cutedsl-inline-ptx-patterns.md](cutedsl-inline-ptx-patterns.md) | CuTeDSL Inline PTX Writing Overview |
| [cutedsl-pipeline-patterns.md](cutedsl-pipeline-patterns.md) | CuTeDSL Software Pipeline and Synchronization Patterns |
| [cutedsl-programming-model.md](cutedsl-programming-model.md) | CuTeDSL Programming Model |
| [cutlass-3x-architecture.md](cutlass-3x-architecture.md) | CUTLASS 3.x Architecture |
| [cutlass-4.0-python-support.md](cutlass-4.0-python-support.md) | CUTLASS 4.0 Python Support |
| [cutlass-conv-implicit-gemm.md](cutlass-conv-implicit-gemm.md) | CUTLASS Convolution and Implicit GEMM Implementation Analysis |
| [cutlass-cute-fundamentals.md](cutlass-cute-fundamentals.md) | CUTLASS/CuTe Core Concepts and Layout Algebra |
| [cutlass-epilogue-visitor-tree.md](cutlass-epilogue-visitor-tree.md) | CUTLASS Epilogue Visitor Tree (EVT) |
| [cutlass-fmha-mla.md](cutlass-fmha-mla.md) | Deep Dive into CUTLASS FMHA and MLA Implementations |
| [cutlass-gemm-optimization.md](cutlass-gemm-optimization.md) | CUTLASS GEMM Optimization Strategy |
| [cutlass-quantization-block-scaled.md](cutlass-quantization-block-scaled.md) | CUTLASS Quantization and Block-Scaled GEMM |
| [cutlass-tile-scheduling.md](cutlass-tile-scheduling.md) | CUTLASS Tile Scheduling |
| [nvidia-cutedsl-arch-primitives.md](nvidia-cutedsl-arch-primitives.md) | CuTeDSL Architecture Primitives (NVIDIA General) |
| [quack-architecture-overview.md](quack-architecture-overview.md) | QuACK Architecture Overview |
| [quack-gemm-epilogue.md](quack-gemm-epilogue.md) | QuACK GEMM and Composable Epilogue System |
| [quack-reduction-kernels.md](quack-reduction-kernels.md) | QuACK: CuTeDSL Memory-Bound Reduction Kernels |

### Subdirectories

| Directory | Description |
|-----------|-------------|
| [sm90/](sm90/) | Hopper (SM90) CuTeDSL-specific articles |
| [sm100/](sm100/) | Blackwell (SM100) CuTeDSL-specific articles |
| [sm120/](sm120/) | Blackwell GeForce / RTX PRO 5000 (SM120) optimization reports (GDN decode fp32-state journey; GDN chunk fwd V113 0.531-0.533ms = 1.51× FLA; NVFP4 persistent GEMM; FA epilogue + NVFP4 quant; PipelineTmaAsync notes) |
