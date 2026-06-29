# NVIDIA General Optimization Reference Articles

General optimization reference articles for NVIDIA hardware, covering PTX, NCU profiling, shared memory, pipelining, and community knowledge.

---

| File | Description |
|------|-------------|
| [ai-systems-performance-engineering-01.md](ai-systems-performance-engineering-01.md) | AI Systems Performance Engineering (Part 1) |
| [cuda-mode-lecture-77-gpu-kernel-dsl.md](cuda-mode-lecture-77-gpu-kernel-dsl.md) | CUDA-MODE Lecture 77: DSLs for GPU Kernels |
| [cuda-tile-vs-triton-benchmark-paper.md](cuda-tile-vs-triton-benchmark-paper.md) | CUDA Tile vs Triton: Hopper/Blackwell GPU Benchmark Paper |
| [cutlass-python-blackwell-gemm-peak-performance.md](cutlass-python-blackwell-gemm-peak-performance.md) | Achieving Peak Tensor Core Performance for GEMM on Blackwell via CUTLASS Python |
| [dsl-overview-5090-memory-insight.md](dsl-overview-5090-memory-insight.md) | DSL Deep Dive: The Insight Behind 5090 Memory Bottleneck |
| [flash-attention-1-to-4-gpu-evolution.md](flash-attention-1-to-4-gpu-evolution.md) | FlashAttention 1–4: GPU Generational Evolution |
| [flashinfer-efficient-attention-engine.md](flashinfer-efficient-attention-engine.md) | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference |
| [flashinfer-tensorrt-llm-cubin-integration.md](flashinfer-tensorrt-llm-cubin-integration.md) | FlashInfer Integration of TensorRT-LLM Cubin Kernels |
| [gdn-decode-kernel-no-tensor-core.md](gdn-decode-kernel-no-tensor-core.md) | GDN Kernel Optimization: Why Decode Cannot Use Tensor Core |
| [gpu-architecture-deep-dive.md](gpu-architecture-deep-dive.md) | GPU Architecture Deep Dive |
| [h100-to-b200-gpgpu-scaling-analysis.md](h100-to-b200-gpgpu-scaling-analysis.md) | From H100 to B200: GPGPU and LLM Scaling Deep Analysis |
| [hierarchical-reduction-memory-bound.md](hierarchical-reduction-memory-bound.md) | Memory-Bound Kernel Optimization: Hierarchical Reduction |
| [large-model-communication-hardware-topology.md](large-model-communication-hardware-topology.md) | Large Model Communication: Hardware Topology (MNNVL/NVL72) |
| [low-precision-accumulation-strategies.md](low-precision-accumulation-strategies.md) | Low-Precision Accumulation Strategies |
| [megatron-lm-limitations-next-generation.md](megatron-lm-limitations-next-generation.md) | Megatron-LM Limitations Deep Analysis and Next-Generation Architecture |
| [ncu-measurement-discipline.md](ncu-measurement-discipline.md) | NCU Measurement Discipline — Trusting Your Profile and Timing Numbers |
| [ncu-profile-driven-optimization-workflow.md](ncu-profile-driven-optimization-workflow.md) | NCU Profile-Driven Optimization Workflow |
| [ncu-profiling-guide.md](ncu-profiling-guide.md) | NVIDIA Nsight Compute (NCU) Profiling Guide |
| [ncu-rule-est-speedup-meta-rules.md](ncu-rule-est-speedup-meta-rules.md) | NCU "Estimated Speedup" — When It's a Wall-Time Lever, When It Isn't |
| [nsight-profiling-practice.md](nsight-profiling-practice.md) | Nsight Profiling in Practice |
| [nvfp4-mxfp4-numeric-system-quantization.md](nvfp4-mxfp4-numeric-system-quantization.md) | NVFP4/MXFP4 Numeric System: PTX, CUTLASS, Triton, and Quantization |
| [nvidia-ptx-mma-instructions.md](nvidia-ptx-mma-instructions.md) | PTX MMA Instruction Evolution |
| [nvidia-ptx-sync-and-async.md](nvidia-ptx-sync-and-async.md) | PTX Synchronization and Asynchronous Primitives |
| [nvidia-tensor-core-evolution-history.md](nvidia-tensor-core-evolution-history.md) | NVIDIA Tensor Core Evolution: Volta to Blackwell |
| [ptx-instruction-evolution-a100-h100-b200.md](ptx-instruction-evolution-a100-h100-b200.md) | PTX Instruction Evolution: A100, H100, B200 |
| [ptx-instruction-set.md](ptx-instruction-set.md) | PTX Core Instruction Set |
| [ptx-programming-model.md](ptx-programming-model.md) | PTX Programming Model and Basics |
| [ptx-sass-programming.md](ptx-sass-programming.md) | PTX/SASS Low-Level Programming Practices |
| [python-operator-dsl-overview.md](python-operator-dsl-overview.md) | Emerging Python Operator DSLs: Triton, CuTeDSL, Mojo Overview |
| [pytorch-performance-profiling-tuning.md](pytorch-performance-profiling-tuning.md) | PyTorch Performance Profiling, Tuning, and Scaling |
| [qwen3.5-gdn-prefill-kernel-optimization.md](qwen3.5-gdn-prefill-kernel-optimization.md) | Qwen3.5 GDN Prefill Kernel Optimization |
| [qwen3.5-gdn-principle-code-analysis.md](qwen3.5-gdn-principle-code-analysis.md) | Qwen3.5 GDN (Gated Delta Networks): Principle and Code Analysis |
| [register-pressure-warp-occupancy.md](register-pressure-warp-occupancy.md) | Register Pressure and Warp Occupancy |
| [sglang-hopper-blackwell-backend-selection.md](sglang-hopper-blackwell-backend-selection.md) | Systematic Performance Bottleneck Analysis for LLM Inference Frameworks |
| [sglang-performance-optimization-august-2025.md](sglang-performance-optimization-august-2025.md) | SGLang Performance Optimization Notes: August 2025 |
| [sglang-performance-optimization-september-2025.md](sglang-performance-optimization-september-2025.md) | SGLang Performance Optimization Notes: September 2025 |
| [smem-swizzling-bank-conflicts.md](smem-swizzling-bank-conflicts.md) | Shared Memory Swizzling and Bank Conflict Elimination |
| [software-pipeline-depth-optimization.md](software-pipeline-depth-optimization.md) | Software Pipeline Depth Optimization |
| [system-level-optimization.md](system-level-optimization.md) | System-Level GPU Optimization |
| [tca-51-mfu-8-hidden-performance-loss.md](tca-51-mfu-8-hidden-performance-loss.md) | TCA 51% but MFU Under 8%: Hidden GPU Performance Losses |
| [tensor-core-volta-to-blackwell.md](tensor-core-volta-to-blackwell.md) | Tensor Core from Volta to Blackwell |
| [tile-level-dsl-emergence.md](tile-level-dsl-emergence.md) | Why Tile-Level DSLs Emerged: Triton, TileLang, cuTile and Mojo |
| [tile-rasterization-l2-locality.md](tile-rasterization-l2-locality.md) | Tile Rasterization and L2 Cache Locality |
| [tokenspeed-inference-framework-analysis.md](tokenspeed-inference-framework-analysis.md) | TokenSpeed Inference Framework Architecture |
| [tokenspeed-qwen3.5-peak-throughput.md](tokenspeed-qwen3.5-peak-throughput.md) | TokenSpeed Achieves 580 TPS on Qwen3.5-397B-A17B with Blackwell |
| [triton-cutile-ptx-isa-future-comparison.md](triton-cutile-ptx-isa-future-comparison.md) | Triton vs cuTile vs PTX-ISA: The Future of AI Hardware Programming |
| [triton-to-sass-tma-multicast-warp-specialize.md](triton-to-sass-tma-multicast-warp-specialize.md) | Triton to SASS: TMA, Multicast, and Warp Specialization Debugging |
| [vllm-hybrid-attention-models.md](vllm-hybrid-attention-models.md) | Hybrid Attention Models as First-Class Citizens in vLLM |
| [warp-specialization-design-principles.md](warp-specialization-design-principles.md) | Warp Specialization Design Principles |

### Subdirectories

| Directory | Description |
|-----------|-------------|
| [sm90/](sm90/) | Hopper (SM90) specific articles |
| [sm100/](sm100/) | Blackwell datacenter (SM100) architecture and optimization articles |
| [sm120/](sm120/) | Blackwell GeForce / RTX PRO (SM120) hardware specifications and architecture whitepaper |
