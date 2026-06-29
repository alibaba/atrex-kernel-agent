# SGLang Performance Optimization Notes: August 2025

A compilation of 29 performance optimization techniques applied to the SGLang inference framework during August 2025, covering kernel fusion, communication optimization, CUDA Graph improvements, and backend upgrades.

---

## 1. AllReduce-Add-RMSNorm Fusion for GPT-OSS and DeepSeek-V3/R1

Applies the FlashInfer `trt-llm-allreduce-add-rmsnorm` kernel API to SGLang, achieving performance gains on GPT-OSS and DeepSeek-V3/R1 deployments.

Results: For bs=1, gpt-oss-120b on B200 TP4 yields 8% output throughput improvement; B200 TP8 deploying DeepSeek-V3/R1 yields 14% output throughput improvement. This optimization was later enabled on Hopper as well.

## 2. PDL Support for Quant and RoPE Kernels

Enables Programmatic Dependent Launch (PDL) for quantization and RoPE kernels.

Results: For bs=1, gpt-oss-120b TP4 deployment achieves 3% end-to-end throughput improvement.

## 3. Optimizing CUTLASS Fused MoE Performance on H20

Configures specialized tuning parameters for the `fp8_blockwise_scaled_group_mm` kernel on H20:
- MmaTileShape: 64x128x128
- ClusterShape: 1x2x1
- Uses `KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum` schedule

The overall logic on Hopper:

```cpp
if (is_h20 && tuning_H20_kernel) {
    // H20-specific optimized configuration
    using execute_gemm_config = sm90_fp8_pp_config_64_128_128_1_2_1;
} else {
    if (multiProcessorCount == 78 && a.size(1) > 128) {
        // K > 128: Pingpong schedule (MmaConfig0)
        // MmaTileShape: 64x128x128, ClusterShape: 2x1x1
    } else {
        // K <= 128: Cooperative schedule (MmaConfig1)
        // MmaTileShape: 128x128x128, ClusterShape: 1x2x1
    }
}
```

## 4. FlashInfer CUTLASS MoE DP Communication: FP4 Quantization + Allgatherv

Optimizes the communication bottleneck for FlashInfer CUTLASS MoE under data parallelism (DP) by performing FP4 quantization before allgather to reduce communication volume, and introduces Allgatherv and Reducescatterv collective operations.

Key improvements:
- Allgatherv: TensorRT-LLM allgather implementation via PyNCCL supporting variable-length inputs across ranks
- Reducescatterv: TensorRT-LLM reducescatter implementation via PyNCCL supporting variable-length inputs
- MoE communication flow: uses allgatherv for token dispatch, FP4 quantization before allgather, and reducescatterv to replace all_reduce

Activation condition: Automatically enabled when `--enable-flashinfer-cutlass-moe`, `--enable-dp-attention`, and `dp_size == ep_size` are all satisfied.

Results: On DeepSeek-R1-0528-FP4, end-to-end throughput improves 9.38% (27,763 to 30,367 tok/s).

## 5. Fast Math Optimization for per_token_group_quant_8bit Kernel

Addresses performance discrepancy between CUDA versions by enabling Fast Math compilation for the `per_token_group_quant_8bit` kernel.

Root cause: CUDA 12.4 implicitly enables `-ftz` or `--use_fast_math`, while CUDA 12.8 does not.

```cmake
set_source_files_properties("csrc/gemm/per_token_group_quant_8bit"
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")
```

Results: Performance improves approximately 34%, reducing execution time from 81.02μs to 53.60μs. Key SASS difference: CUDA 12.4 uses `FMNMX.FTZ` while CUDA 12.8 uses standard `FMNMX`.

## 6. FP4 MoE Quantization Kernel: Grid-Stride Layout + Dynamic Launch Configuration

Ported from vLLM, targeting FP4 quantization performance for MoE models on Blackwell GPUs.

Optimization techniques:
- **Grid-stride loop layout**: Replaces per-block row processing for better thread-level parallelism
- **Dynamic launch configuration tuning**: When grid size < SM count and block size is large, automatically doubles grid size and halves block size to improve occupancy
- **Tiered memory access**: Small-scale uses registers for expert offsets; large-scale loads expert offsets to shared memory with binary search

```cpp
int const numBlocksPerSM = 2048 / block.x;
if (grid.x < numSMs && block.x > threshold) {
    grid.x *= 2;
    block.x /= 2;
}
```

## 7. Reduce-Scatter Communication for DeepSeek-V3, Qwen, and Llama4 under DP Attention

Extends reduce-scatter communication optimization to more model architectures, replacing all-reduce with reduce-scatter after MoE/MLP layers when DP attention max padding is enabled.

Mathematical equivalence:
- Traditional: `scatter(all_reduce(X)) = scatter(sum(X_i))`
- Optimized: `reduce_scatter(X) = sum(X_i) / DP_size`

```python
if self.tp_size > 1:
    if skip_all_reduce:
        output = tensor_model_parallel_reduce_scatter(output)
    else:
        output = tensor_model_parallel_all_reduce(output)
```

Results: Qwen3-235B test achieves 12,692 tok/s total token throughput with significantly reduced end-to-end latency.

## 8. FlashAttention-3 Backend for GPT-OSS: Attention Sinks

Adds FA3 backend support for GPT-OSS models with attention sinks functionality.

Results (GPT-OSS-20B, TP1, 4k input/1k output):
- FA3 at concurrency=1: 309.425 tok/s output, TTFT 75.511ms (2.1% throughput gain)
- FA3 at concurrency=32: 3,057.230 tok/s output, TTFT 1,271.047ms (31.7% TTFT reduction)

## 9. FP8 CUTLASS Kernel Tuning on Blackwell: Dynamic Configuration Dispatch

Ports vLLM's FP8 GEMM performance tuning for Blackwell (SM100), using dynamic kernel dispatch based on input matrix M dimension.

Segmented configuration strategy:
- [1, 16]: Lightweight configuration for small matrices
- (16, 64]: Balanced configuration for medium-small matrices
- (64, 256]: Performance configuration for medium matrices
- (256, ∞]: High-throughput configuration for large matrices

```cpp
template<typename T>
auto select_fp8_gemm_config(int M) {
    if (M <= 16) return small_config;
    else if (M <= 64) return medium_small_config;
    else if (M <= 256) return medium_config;
    else return large_config;
}
```

## 10. FlashInfer TensorRT-LLM FP8 Blockscale GEMM Backend

Upgrades SGLang's FlashInfer CUTLASS backend to the TensorRT-LLM FP8 GEMM implementation, focusing on low-latency optimization.

Results on DeepSeek-R1-0528 (TP8+DP8): Request throughput improves 6% (0.83 to 0.88 req/s), first-token latency decreases 9% (10.1s to 9.2s), overall throughput improves 6.7% (7.6k to 8.1k tok/s).

## 11. Custom Set KV Buffer Kernel Fusion

Develops a custom CUDA kernel fusing key and value cache storage operations, identified via nsys profiling as a bottleneck on H100.

```cpp
__global__ void set_kv_buffer_kernel(
    scalar_t* k_cache, scalar_t* v_cache,
    const int64_t* loc,
    const scalar_t* k, const scalar_t* v) {
    // Fused key and value storage
}
```

## 12. RoPE + Set KV Buffer Kernel Fusion

Fuses the KV cache write operation directly into the RoPE kernel, eliminating separate memory operations and kernel launch overhead.

```cpp
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* cos_ptr, const scalar_t* sin_ptr,
    std::optional<scalar_t*> kv_buffer = std::nullopt,
    std::optional<int64_t*> cache_loc = std::nullopt) {
    apply_rotary_embedding(query, key, cos, sin);
    if (kv_buffer.has_value()) {
        write_to_kv_cache(key, value, kv_buffer, cache_loc);
    }
}
```

Results: gpt-oss-120b TP4 achieves 3% end-to-end throughput improvement.

## 13. MoE Padding and Quantization Kernel Fusion for GPT-OSS

Fuses hidden state padding into the quantization kernel for GPT-OSS MoE layers, eliminating redundant memory operations.

```python
# Before: separate pad/unpad
def forward(self, hidden_states):
    hidden_states = pad(hidden_states, target_size)
    x_quant = mxfp8_quantize(hidden_states)
    output = moe_computation(x_quant)
    return unpad(output, original_size)

# After: fused
def forward(self, hidden_states):
    x_quant = mxfp8_quantize(hidden_states, output_hidden_size=target_size)
    return trtllm_fp4_block_scale_moe(x_quant)
```

## 14. Non-Padded Token Count for MoE in DP Scenarios

Fixes inaccurate token counting in MoE computation under data parallelism.

Problem: Original implementation incorrectly uses global DP rank token count, causing overestimation of effective tokens, incorrect MoE routing, and wasted computation.

```python
def get_num_token_non_padded_local(total_tokens, tp_size, tp_rank):
    base = total_tokens // tp_size
    extra = 1 if tp_rank < (total_tokens % tp_size) else 0
    return base + extra
```

Results: DeepSeek-V3-0324 DP deployment throughput improves from 53.05 to 57.07 tok/s (7.6% gain).

## 15. TRT-LLM MLA FP8 Support

Adds FlashInfer's TRT-LLM MLA FP8 KV Cache Backend support.

## 16. MoE Routed Scaling Factor Kernel Fusion

Fuses routed scaling factor computation into `moe_fused_gate` and `select_experts` kernels, reducing standalone operation overhead.

```python
def moe_fused_gate(..., apply_routed_scaling_factor_on_output=False):
    return fused_gate_with_scaling(...) if apply else traditional(...)
```

## 17. GPT-OSS Attention Sinks with TRT-LLM MHA Backend

Adds TRT-LLM multi-head attention (MHA) backend support for GPT-OSS with attention sinks mechanism.

Key modifications:
- Direct invocation of TRT-LLM generated MHA modules with highly optimized kernels
- Attention sinks via `sk` parameter for long-sequence optimization
- `trtllm_mha` added as valid attention backend option

Results (GPT-OSS-20B): TRT-LLM MHA backend achieves 17,151.680 tok/s versus Triton backend at 14,607.150 tok/s.

## 18. TBO Optimization: Two Chunk Overlap

Improves Two Batch Overlap (TBO) by introducing Two Chunk Overlap, which intelligently splits long sequences into two chunks for parallel execution, eliminating idle batches.

```
# Traditional TBO (idle batch problem):
micro_batch0: extend_seq_len = [3072]  # active
micro_batch1: extend_seq_len = [0]     # idle

# Two Chunk Overlap (eliminates idle batch):
micro_batch0: extend_seq_len = [1536], extend_prefix_len = [0]    # chunk 1
micro_batch1: extend_seq_len = [1536], extend_prefix_len = [1536] # chunk 2
```

Results (2x8x H800, DeepSeek-V3-0324):
- Special case (single long request per DP, length 3072): Average 12.56% throughput gain
- General case (variable-length input 30-3072 tokens): Average 5.15% throughput gain

## 19. DP Attention: LayerNorm Before AllGather

Moves LayerNorm before the allgather operation in data parallelism, performing normalization on 1/DP-count tokens to reduce computation. Only enabled when DP==TP to ensure numerical stability.

Results: DeepSeek-R1-0528-FP4 end-to-end throughput improves 3.79% (27,310 to 28,345 tok/s).

## 20. FP8 MoE Kernel Schedule Selection for H100/H200/H800

Resolves performance regression when migrating from H20 to H100/H200/H800 for `fp8_blockwise_scaled_grouped_mm`. The root cause: Tensor Core MMA time reduces to 1/4 on H100, but Pingpong schedule CUDA Core FMA cannot overlap, reducing SM Tensor Pipe throughput.

Solution: Identifies GPU architecture by SM count — H20 (78 SMs) continues using Pingpong schedule; other Hopper architectures use Cooperative schedule to resolve Tensor Core contention.

## 21. FlashInfer MoE Blockscale FP8 Backend for TP MoE

Extends FlashInfer MoE blockscale FP8 backend to tensor parallel (TP) MoE configurations. Adds `FlashInferFusedMoE` class encapsulating optimization logic, decoupling from EP MoE dependency. Removes the forced `enable_ep_moe` requirement; TP MoE can independently use the `trtllm_fp8_block_scale_moe` kernel.

## 22. TRT-LLM Generated MLA Decode Kernel Integration

Integrates TensorRT-LLM generated Multi-Head Latent Attention (MLA) decode kernels for DeepSeek series models, with SM100 architecture compatibility checks.

## 23. Disabling Python GC During CUDA Graph Capture

Disables Python garbage collector during CUDA graph capture using `gc.freeze()` to avoid GC scanning long-lived objects.

Results: CUDA graph capture speed improves 2.3x-3.7x. Llama4 model from 25s to 10s; Qwen3-0.6B from 6s to 1s.

## 24. MRoPE torch.compile Optimization

Adds `torch.compile(dynamic=True)` to `MRotaryEmbedding.forward()`, reducing kernel launch overhead for small VLM models.

Results: On Qwen2.5-VL-3B-Instruct, request throughput improves 28% (2.53 to 3.25 req/s), MRoPE latency reduces 8x, ITL from 5.86ms to 4.48ms.

## 25. GC Freeze for Latency Jitter Reduction

Adds GC freeze functionality via `freeze_gc` API to exclude server warmup objects from garbage collection, avoiding 100ms-300ms pauses from gen2 GC.

Implementation features: New `/freeze_gc` HTTP endpoint, distributed GC management, configurable `gc_warning_threshold_secs`.

```python
def freeze_gc(context: str):
    gc.freeze()  # Move current objects to permanent generation

def configure_gc_warning(warn_threshold_secs):
    def gc_callback(phase, info):
        if phase == "stop":
            duration = time.time() - gc_start_time.get(gen, time.time())
            if duration > warn_threshold_secs:
                logger.warn(f"LONG GC DETECTED | Gen {gen} | {duration:.4f}s")
    gc.callbacks.append(gc_callback)
```

## 26. FlashInfer GPU-CPU Synchronization Optimization

Fixes unnecessary GPU-CPU synchronization in FlashInfer when page_size=1 by directly constructing a `torch.ones` tensor instead of GPU-to-CPU data transfer.

Results: On B200, Qwen2.5-7B at concurrency=1, total throughput improves from 425.01 to 437.64 tok/s (3.0% gain).

## 27. FlashInfer GQA Tensor Core Decode Threshold

Lowers the GQA group size threshold for enabling Tensor Core decode in FlashInfer from >4 to >=4, allowing models like Llama3-8B with 4 GQA groups to benefit from Tensor Core acceleration.

Rationale: FlashInfer fuses the head group dimension with the token dimension, making group size 4 sufficient for Tensor Core benefits.

## 28. CUTLASS 4.2 Upgrade with K-Major Scale Factor Support

Upgrades CUTLASS to 4.2, enabling K-Major Scale Factor for SM90 FP8 Blockwise Group GEMM. Unifies code paths with Blackwell and eliminates `per_group_transpose` format conversion overhead. Also optimizes H20 device detection using ATen interface to avoid `cudaGetDeviceProperties` call overhead.

## 29. FlashInfer/FlashMLA Chunked Prefill Cache Support

Adds MHA Chunked Prefill cache support for FlashInfer and FlashMLA backends, removing the page_size=1 limitation to support larger page sizes for improved memory efficiency.

Accuracy tests show consistent precision across different page size configurations (GSM8K: 0.954-0.955). Significant TTFT reduction observed in benchmarks.
