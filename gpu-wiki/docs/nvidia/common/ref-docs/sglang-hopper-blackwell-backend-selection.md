# Systematic Performance Bottleneck Analysis for LLM Inference Frameworks

A comprehensive methodology for diagnosing performance bottlenecks in LLM inference frameworks (SGLang, vLLM), covering a seven-layer diagnostic model from hardware to application layer, with practical tooling guidance and decision trees.

---

## 1. The GPU Utilization Illusion

`nvidia-smi` showing 95% GPU utilization is the biggest dashboard deception in inference tuning. The metric only indicates whether the GPU had *any* activity in the past second — it reveals nothing about SM saturation, bandwidth utilization, or Tensor Core engagement. True diagnosis requires opening the black box and understanding the full request pipeline from HTTP ingress to token egress.

## 2. vLLM V1: The 2025 Rewrite

vLLM released V1 Alpha in January 2025 and made V1 the default engine in v0.8.0. V0 was formally deprecated and is being removed starting from v0.10. Tuning advice from before Q1 2025 should be applied with caution.

V1's core architectural change splits AsyncLLM (OpenAI-compatible server, tokenizer, request flow) and EngineCore (Scheduler + ModelExecutor loop) into two processes communicating via ZeroMQ IPC. With Llama-3.1 8B on H100 taking only 5ms per forward pass, V0's single-process design (where Python tokenizer, scheduler, and detokenizer competed for GIL) starved the GPU. Real-world benchmarks show up to 24% improvement on ShareGPT workloads, with larger gains on multimodal.

V1's scheduler auto-enables chunked prefill and multi-step scheduling. Automatic Prefix Caching is on by default with less than 1% throughput loss at zero hits and multi-fold speedup at high hit rates.

## 3. SGLang: Running CPU Ahead of GPU

SGLang's core philosophy is keeping the CPU one step ahead of the GPU. The Zero-Overhead Batch Scheduler (v0.4) prepares metadata for batch N+1 while the GPU executes batch N, using future token placeholders and CUDA events for dependency resolution.

RadixAttention uses a compressed prefix tree to reuse KV cache across completely unrelated requests, with O(n) longest-prefix lookup and LRU eviction enhanced by cache-aware scheduling. Benchmarks show up to 5x speedup over vLLM in few-shot, tree-of-thought, agent, and multi-turn scenarios with zero measurable overhead at zero hits.

## 4. The Seven-Layer Bottleneck Map

A request traverses at least seven layers from ingress to egress. Any slow layer manifests externally as high TTFT or low throughput.

### 4.1 Hardware Layer: Four Metrics That Matter

The four real hardware-layer metrics:

1. **SM Throughput** (`sm__throughput.avg.pct_of_peak_sustained_elapsed`): Prefill GEMM should exceed 70%. Below 30% indicates memory-bound or latency-bound — increasing SM count is pointless; kernel fusion or larger tiles are needed.

2. **Memory Throughput** (`dram__throughput.avg.pct_of_peak_sustained_elapsed`): Decode matvec should exceed 80%. H200 vs H100 bandwidth improves from 3.35 TB/s to 4.8 TB/s (1.43x), matching TensorRT-LLM's 1.45x decode throughput improvement on Llama-2 70B.

3. **L2 Cache Hit Rate**: Decode KV cache reuse should keep L2 hits above 50%. Below 20% indicates incorrect PagedAttention block_size or prefix cache failure.

4. **Achieved Occupancy**: Below 25% indicates register pressure or grid too small. Common in batch=1 decode attention kernels.

**Core physics**: Prefill is compute-bound, decode is memory-bound. H100 SXM5 has 1979 TFLOPS BF16 and 3.35 TB/s HBM3, with a ridge point around 590 FLOP/byte. A 70B BF16 model at batch=1 decode has an absolute floor of approximately 140GB / 3.35 TB/s ≈ 42ms/token — this is physics, not engineering.

**Multi-GPU traps**:
- NCCL falling back to TCP: Check `NCCL_DEBUG=INFO` logs for `[send] via NET/Socket` vs `[send] via NET/IB/GDRDMA`. NVLink 4 H100 TP=8 should exceed 400 GB/s bus bandwidth.
- NUMA affinity: Cross-PCIe root complex TP loses 30%+ bandwidth. Pin with `numactl --cpunodebind=0 --membind=0`.

**Blackwell reality**: B200 without large-scale EP performs roughly equal to H100. The 3.8x prefill / 4.8x decode improvement on GB200 NVL72 requires the full stack: FlashInfer Blackwell CuTe DSL, NVFP4 GEMM, tcgen05.mma, TMA, warp specialization, 2CTA MMA. NVFP4 has precision issues — Llama-3.1-405B-Instruct-FP4 with FlashInfer backend on B200 produces garbled output, requiring Triton backend fallback.

### 4.2 Kernel Layer: Attention Backend, CUDA Graph, torch.compile

FlashAttention v3 on H100 provides 1.5-2x speedup over v2 with native FP8 support. SGLang defaults to FA3 on Hopper, trtllm_mha/trtllm_mla on Blackwell, FlashInfer on A100/Ada/Ampere, and Triton otherwise.

**MLA page size**: FlashInfer MLA page=1, FlashMLA page=64, CUTLASS MLA page=128, TRTLLM MLA page=32 or 64. Mismatched page sizes cause backend fallback to Triton with 2-4x performance loss.

**PagedAttention block_size**: vLLM defaults to 16, but production PD disaggregation uses 128 or 256. For long context (>32K), 32 or 64 is community consensus.

**CUDA Graph pitfalls**: A common issue is DP attention with batch 32 x DP 16 = 512 exceeding `--cuda-graph-max-bs` default (160 or 256), causing fallback to eager mode. Fix: increase `--cuda-graph-max-bs` to 768. Cold-start TTFT of several seconds is lazy graph capture — Piecewise CUDA Graph pre-captures during startup.

**MoE kernels**: DeepGEMM uses contiguous-layout Grouped GEMM (prefill, dynamic shapes, with DeepEP normal dispatch) and masked-layout Grouped GEMM (decode, fixed shapes with mask, CUDA graph compatible with low-latency dispatch). DeepEP normal dispatch: intranode EP8 reaches 153 GB/s; internode EP32 reaches 58 GB/s; low-latency EP128: dispatch 192μs, combine 369μs.

Critical constraint: DP attention, normal dispatch, low-latency dispatch, and CUDA graph cannot coexist simultaneously — this is the fundamental architectural reason for PD disaggregation.

**Diagnostic tools**:
- nsys (Nsight Systems): Timeline view for kernel gaps. Gap > 15% = CPU bottleneck; NCCL > 20% = communication bottleneck.
- ncu (Nsight Compute): Deep kernel analysis. Focus on Speed Of Light, Memory Workload, Occupancy, and Roofline sections.
- torch.profiler with Perfetto: Python-side analysis. Perfetto handles >1GB traces with SQL queries.

### 4.3 Memory Layer: Three Ways KV Cache Dies

**Death 1 — Greedy pre-allocation**: V1 pre-allocates the entire KV pool at startup by `max-model-len × max-num-seqs`. Even one 128K user locks approximately 37GB. Fix: reduce `--max-model-len` to actual P99, reduce `--max-num-seqs`, enable `--kv-cache-dtype fp8_e5m2` (FP8 KV on 8x H200 expands token capacity from 54,560 to 512,000).

**Death 2 — PyTorch allocator fragmentation**: Symptom: `torch.OutOfMemoryError` with sufficient reserved but unallocated memory. Fix: `export PYTORCH_ALLOC_CONF=expandable_segments:True` or increase `--block-size` from 16 to 32.

**Death 3 — mem-fraction-static ignored on ROCm**: SGLang on MI210 with 0.8 fraction actually consumes 61.8 GB of 64 GB. Workaround: reduce to 0.65-0.70 or use `--disable-cuda-graph`.

**Prefix cache health**: vLLM exposes `vllm:gpu_prefix_cache_hit_rate`; SGLang uses `cached_tokens` in logs. Healthy targets: chat >0.3, agent multi-turn >0.6.

### 4.4 Scheduling Layer: Python GIL as Invisible Ceiling

A real incident: SGLang router on Qwen2.5-0.5B + L4 + 32 vCPU + 150 concurrency hit 127% single-core CPU, with 2.4x worse latency than vLLM. Root cause: pure Python router eating GIL. Fix: switch to Rust-based `sglang-router`, or use vLLM's `--api-server-count N`.

**Chunked prefill tuning**: vLLM V1 defaults `max-num-batched-tokens=2048`. Low values favor ITL; high values favor TTFT and throughput. However, Qwen3-30B-A3B MoE + TP2 + PP3 with chunked prefill enabled is 11x slower than disabled — root cause: Triton `prefix_prefill.py` BLOCK_M/BLOCK_N defaults poorly adapted for MoE shapes.

**PD Disaggregation**: Core motivation — prefill is compute-bound, decode is memory-bound. Colocating them causes ITL jitter when long prefills arrive. DistServe (OSDI 2024) proved 7.4x request capacity or 12.6x tight SLO. Mooncake (FAST 2025 Best Paper) achieves 525% long-context throughput improvement in Kimi production.

A classic PD trap: DeepSeek-V3 on H20 in 2P+1D mode, TTFT jumped from 2,338ms to 24,766ms. Root cause: prefill TP ≠ decode TP (non-MLA fallback) plus Mooncake transfer thread pool starvation. Fix: match TP and set `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4`.

### 4.5 Parallelism Layer: TP, PP, EP, DP Selection

- **TP**: Splits attention and FFN by column/row with per-layer all-reduce. Works within NVLink domain (≤8 GPUs); cross-node all-reduce kills throughput.
- **PP**: Splits by layers, fewer all-reduces but has bubbles. Without IB, 2xPP x 8xTP outperforms 16-way TP by 6.6x.
- **DP**: Duplicates model, independent batches. SGLang's DP Attention splits attention by batch, then all-gathers for MoE.
- **EP**: Splits MoE experts, per-layer all-to-all, requires large NVLink domain.

SGLang on 96 H100 running DeepSeek: DP Attention + FFN DP + MoE EP via DeepEP + LM Head DP + TBO + EPLB achieves per-node 52.3k input and 22.3k output tok/s ($0.20/1M output tokens, 5x over vanilla TP).

**EPLB**: Uses redundant experts (e.g., 256 expanded to 288 with 32 replicas) to enable non-power-of-2 parallelism. Elastic EP with Mooncake backend enables 10-second recovery from single-card failure with zero static performance degradation.

**TBO (Two-Batch Overlap)**: Splits batches into micro-batches for compute-communication overlap. Prefill improves 27-35%. Decode at 32 tokens/device: -27%; must exceed 64-128 tokens/device for positive gains.

### 4.6 Speculative Decoding and Application Layer

**Speculative decoding KPI**: Acceptance rate. EAGLE-3 achieves 3.0-6.5x speedup with accept length 4.05-7.5. However, large batch degrades EAGLE (verification overhead scales linearly, accept length sub-linearly). Production recommendation: k=2-3; only k=4 for highly repetitive code.

**MTP (Multi-Token Prediction)**: DeepSeek V3 native feature, second token acceptance 85-90%, 1.8x native speedup.

**Tokenizer bottleneck**: vLLM's `ThreadPoolExecutor(max_workers=1)` for tokenization blocks subsequent requests on long prompts. Fix: `--api-server-count 4`.

**Structured output**: xgrammar (10x faster than outlines) for JSON mode/tool calling. But structured constraints reduce speculative decoding acceptance — draft model does not know constraints.

**Vision encoder**: Qwen2.5-VL-72B's 675M ViT under TP=4 suffers from communication overhead. Fix: vLLM's `--mm-encoder-tp-mode=data` (DP for encoder, 40% throughput gain); SGLang's ViTCudaGraphRunner with bucketed shapes.

### 4.7 Diagnostic Decision Tree

Symptom-based diagnostic flow:

**High P95 TTFT, normal P95 TPOT**:
- Queue deep + GPU <80% → CPU-bound: add `--api-server-count`, check slow tokenizer, py-spy dump
- Queue shallow + high TTFT → prefill-bound: enable chunked prefill/PD disaggregation, increase max-num-batched-tokens
- Only first request slow → cold start: pre-warm `--cuda-graph-max-bs`

**High/jittery P95 TPOT**:
- `num_preempted_requests > 0` or `gpu_cache_usage > 0.95` → KV pressure: increase gpu_memory_utilization, reduce max_num_seqs, enable FP8 KV
- Multi-GPU: grep NCCL logs for `via NET` to confirm IB
- GPU full but slow → ncu: Memory >80% with low SM = expected memory-bound (quantize or increase batch); SM >80% = low occupancy or poor tiling

**Profiling overhead reference**: Prometheus scrape <1%, py-spy <5%, PyTorch profiler (no stack) 3-10%, nsys 5-15%, ncu 10-100x.

## 5. Production Monitoring Recommendations

SLO should use goodput rather than raw throughput:

\[\text{Goodput} = \max \text{RPS s.t. } P_{90}\text{TTFT} < 200\text{ms AND } P_{90}\text{TPOT} < 50\text{ms}\]

vLLM's `vllm bench serve --goodput ttft:3000 tpot:100` directly supports this metric.

## 6. Conclusion

Inference optimization in 2025-2026 has no silver bullets. DeepSeek's $0.20/1M tokens requires the multiplicative combination of DP Attention + DeepEP + DeepGEMM + EPLB + MTP + PD disaggregation + Mooncake + Elastic EP + NVFP4. Hardware upgrades must be paired with full-stack software upgrades and performance validation. The methodology of controlled variables — changing one parameter at a time across 10+ orthogonal parameters — remains the most reliable approach to identifying causal relationships in inference tuning.
