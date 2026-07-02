# GPT-OSS Blackwell Performance Optimization: Pushing the Pareto Frontier

Systematic optimization of gpt-oss-120b (native MXFP4 MoE model) on NVIDIA B200/GB200 through deep vLLM + FlashInfer integration, achieving +38% max throughput and +13% lowest latency across the entire Pareto curve.


**Last updated**: 2026-06-30

---

## 1. Introduction

Optimizing a single metric (maximum throughput or minimum batch latency) is often insufficient for real deployments. The true challenge is optimizing the **Pareto frontier**: the best trade-off between TPS/GPU (TCO) and TPS/User (interactivity). The optimization target is OpenAI's **gpt-oss-120b**, a native 4-bit quantized (MXFP4) MoE model, using deep vLLM + FlashInfer integration on NVIDIA B200/GB200 architecture.

## 2. FlashInfer Integration and torch.compile Fusion

**Key compute kernel integration:**

- **MoE backend:** Enables both trtllm-gen and cutlass backends, selecting optimal per scenario; FlashInfer provides JIT, auto-tuning, and kernel caching.
- **FP8 KV-Cache:** Serves more concurrent requests within the same memory budget; FP8 attention simultaneously reduces compute/memory complexity (via FlashInfer-optimized attention kernels).

**torch.compile automatic fusion:**

- **AR + RMSNorm fusion** (PR20691): Eliminates TP communication bottleneck
- **Pad + Quant and Finalize + Slice** (PR30647): Further slims MoE path, estimated +6%

Rather than hard-coded fusion, infrastructure built on torch.compile enables automatic execution, easing generalization and maintenance.

## 3. Runtime Improvements

Blackwell GPUs are so fast that the CPU (host) becomes the bottleneck, manifesting as inter-kernel gaps.

**Async scheduling (PR23569):** CPU scheduling decoupled from GPU execution — GPU still processing current batch while CPU prepares next. ~10% improvement on H200/B200/GB200, enabled by default in latest vLLM.

**Stream Interval (PR27869):** Buffers generated tokens to reduce HTTP/gRPC response frequency (first token still sent immediately preserving TTFT). gpt-oss-20b @1024 concurrency achieves **+57%** end-to-end, with better TPOT. `--stream-interval <num_tokens>` defaults to 1; set to 10 for high-throughput scenarios.

## 4. Deployment Tuning

Recommended configuration (published on vLLM Recipes page for GPT-OSS):

- Graph capture: `--cuda-graph-capture-size 2048`
- Scheduling: `--api-server-count 20` or `--stream-interval 20`
- MoE backend: `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`

## 5. Results and Next Steps

Since InferenceMax release: **max throughput +38%, lowest latency +13%**, covering the entire Pareto curve.

**In progress:**

- **Disaggregated architecture:** Prefill / Decode separated to different GPUs
- **DEP2** (Attention DP + MoE EP on 2 GPUs): Predicted to outperform TP1/TP2 in TPS/GPU at same latency; currently MoE kernel selection issues cause underperformance vs TP — being fixed
- **TP8 concurrency 8** lowest-latency scenario: RoPE+Q+Cache fusion (already in FlashInfer, vLLM integration in progress), router GEMM and fc_qkv/fc_o_proj GEMM using PDL-aware micro-GEMM

## 6. Notes

- sm120 (RTX 6000 Pro 96GB Blackwell) MXFP4 support is not yet available
- Datacenter Blackwell (sm100/103) is prioritized; GeForce Blackwell (sm120) coverage pending

## References

- vLLM official blog: GPT-OSS Optimizations (2026)
- SemiAnalysis InferenceMAX benchmark
- vLLM InferenceMax blog (2025)


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
- [CK Tile MoE / Norm / Conv / Reduce Operations](../../../amd/common/ck-moe-norm-conv.md)
