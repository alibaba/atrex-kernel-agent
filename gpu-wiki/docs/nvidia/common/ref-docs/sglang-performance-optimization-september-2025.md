# SGLang Performance Optimization Notes: September 2025

A compilation of 28 performance optimization techniques applied to the SGLang inference framework during September 2025, covering kernel unification, architecture-specific optimizations, memory management fixes, and distributed inference improvements.

---

## 1. Mooncake Store Metadata Fetch: CPU Tensor to List Acceleration

Optimizes the `get_buffer_meta` method in Mooncake distributed KV cache by pre-converting CPU tensors to Python lists, avoiding repeated PyTorch indexing overhead in loops.

```python
def get_buffer_meta(self, keys, indices, local_rank):
    kv_buffer_data_ptr = self.kv_buffer.data_ptr()
    indices = indices.tolist()  # Core optimization: avoid tensor indexing
    for index in range(0, len(indices), self.page_size):
        k_ptr = kv_buffer_data_ptr + indices[index] * ...
        ptr_list.append(k_ptr)
```

Results: L20: 8ms to 0.7ms (11x improvement); H800: 0.5-3s to 1-2ms (250-1500x improvement).

## 2. GPT-OSS FP8 KV Cache: Disabling Fused Set KV Buffer

Enables FP8 KV cache for GPT-OSS on B200/GB200 by conditionally disabling the fused set kv buffer operation (which only supports bfloat16).

Motivation: KV cache data volume is the GPT-OSS performance bottleneck on B200/GB200, limiting batch size. FP8 KV cache significantly improves batch size (from 630 to 768). The fix adds dtype checking to `_enable_fused_set_kv_buffer`, enabling fusion only for bfloat16.

## 3. DeepSeek-R1 W4AFP8 Quantization: TP Mode Support

Adds TP (Tensor Parallelism) mode support for DeepSeek-R1 W4AFP8 (weight INT4, activation FP8) quantization, offering better first-token latency compared to EP mode.

Results on 8x H20 (DeepSeek-R1-W4AFP8, ISL1000/OSL1000):
- TP8: TTFT median 6,612ms, ITL median 68.05ms, output throughput 1,610 tok/s
- EP8: TTFT median 8,145ms, ITL median 66.38ms, output throughput 1,586 tok/s
- TP8 TTFT is approximately 19% lower than EP8, with 1.5% throughput improvement.

## 4. Nsys Profiling Tool: Automated GPU Kernel Classification

An automated nsys performance analysis tool that classifies, aggregates, and visualizes GPU kernel traces, supporting Llama, DeepSeek, and GPT-OSS models.

Features:
- Automatic kernel classification via regex rules (attention, gemm, MoE, quantization, etc.)
- Non-overlapping GPU kernel execution time calculation (eliminates double-counting from concurrent kernels)
- HTML visualization (stacked bar charts) and CSV output
- Extensible via JSON configuration files

```bash
python3 examples/profiler/nsys_profile_tools/gputrc2graph.py \
    --in_file nsys_res.nsys-rep,sglang,llama,132 \
    --title "Llama-3.1-8B Performance Analysis"
```

## 5. Expert Model Parallel Communication Group Memory: Smart TP Reuse

Reduces redundant communication resource allocation for MoE expert parallelism by reusing existing TP communication groups when EP/ETP sizes match TP size.

```python
global _MOE_EP
if moe_ep_size == tensor_model_parallel_size:
    _MOE_EP = _TP  # Reuse TP group
else:
    _MOE_EP = init_model_parallel_group(...)

global _MOE_TP
if moe_tp_size == tensor_model_parallel_size:
    _MOE_TP = _TP  # Reuse TP group
else:
    _MOE_TP = init_model_parallel_group(...)
```

## 6. DeepSeek-V3/R1 MXFP4 Quantization: Kernel Fusion for Activation Quantization (AMD)

Fuses activation tensor quantization into different operators (activation, layernorm, gemm, flatten) to eliminate standalone quantization kernel overhead for MXFP4 inference.

Key optimizations:
- **Fused Quant-GEMM**: Quantization performed inside GEMM kernel
- **BumpAllocator**: Reuses pre-allocated memory pool for GEMM output buffers
- **MoE Gate fusion**: Applies fused quantization GEMM in gate and shared expert computations

Results on DeepSeek-R1-WMXFP4-Preview (TP8, 512 input/800 output): End-to-end latency decreases approximately 9%, input/output throughput improves approximately 10%.

## 7. Qwen3-MoE: FlashInfer Fused AllReduce

Simplifies Qwen3-MoE model code (removes deepep path, dual stream complexity) to correctly leverage FlashInfer fused allreduce, merging AllReduce+RMSNorm+ResidualAdd into a single kernel.

Results (Qwen3-30B-A3B, TP8): Input throughput improves 2.2%. Kernel fusion GPU time drops from 19.71% to 12.98%.

## 8. DeepSeek-R1 TRT-LLM MLA Backend: Prefill Optimization

Adds prefill support to the TRT-LLM MLA backend using FlashInfer's `trtllm_ragged_attention_deepseek` kernel, with FP8 KV cache support.

Results on DeepSeek-R1 (8k ISL prefill): Prefill throughput improves 2x. Accuracy: 0.961 (no precision loss).

## 9. Per-Token Group Quant 8bit Kernel Unification

Comprehensive refactoring of INT8/FP8 quantization kernels: removes the v2 branch, unifies implementation, and adds MoE-specific optimizations.

Key additions:
- **Fused SiLU and Mul**: SiLU activation and multiplication fused into quantization kernel
- **Masked Layout**: `masked_m` parameter support for MoE EP scenarios
- **Parameterized subwarp GroupReduce**: Supports 1/2/4/8/16 thread subwarp configurations
- **DeepEP fast math**: `fast_pow2` and optimized FP8 scale calculation
- **PTX memory optimization**: `st.global` and `ld.global.nc` assembly instructions

```cpp
// Blackwell-optimized SiLU
__device__ __forceinline__ float silu(const float& val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    float half = 0.5f * val;
    float t = __tanhf(half);
    return half * (1.0f + t);
#else
    return val / (1.0f + __expf(-val));
#endif
}
```

## 10. DeepSeek-V3 Blackwell Optimization: Router GEMM Dtype and Correction Bias

Optimizes Router GEMM output type and correction bias dtype for DeepSeek-V3 on Blackwell, eliminating unnecessary type conversion overhead.

Changes:
- Router GEMM output changed from bfloat16 to float32
- FP4 quantization scenario: correction bias converted to bfloat16
- `TRTLLM_ENABLE_PDL` environment variable flexibility: allows disabling via `TRTLLM_ENABLE_PDL=0`

## 11. MoE Sum Reduce Kernel: 2D Tile Batch Processing

Replaces serial per-token processing with 2D tile batch processing for the MoE sum reduce operation, significantly improving parallelism and memory access efficiency.

```python
# Before: serial per-token
for token_index in range(token_start, token_end):
    accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    for i in range(topk_num):
        tmp = tl.load(input_t_ptr + i * input_stride_1, ...)
        accumulator += tmp

# After: 2D tile batch
accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
for i in range(topk_num):
    tile = tl.load(base_ptrs + i * input_stride_1, ...)
    accumulator += tile.to(tl.float32)
```

Additional changes: num_warps increased from 8 to 16; unified 2D mask handling.

## 12. SM120 Architecture FP8 Blockwise GEMM Support

Adds FP8 blockwise scaled matrix multiplication for the SM120 architecture (next-generation GPU) with specialized tile configurations.

```cpp
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ArchTag = cutlass::arch::Sm120;
using MmaTileShape = Shape<_128, _128, _128>;
using PerSmTileShape = Shape<_128, _128, _128>;
using EpilogueTileShape = Shape<_128, _64>;
```

Requirements: CUDA >= 12.8, CUTLASS SM120 architecture support.

## 13. NVFP4 GEMM Dynamic Configuration: Eliminating Small-Batch Redundancy

Implements M-dimension adaptive ClusterShape and TileShape configuration for NVIDIA FP4 block-scaled GEMM, eliminating redundant computation in small-batch decode scenarios.

| M Range | MMA Tile | Cluster Shape | SM Strategy | Optimization Goal |
|---------|----------|---------------|-------------|-------------------|
| ≤128 | 128x256x256 | (1,4,1) | 1SM | Eliminate M-direction redundancy |
| 128-256 | 256x256x256 | (2,4,1) | 2SM | Balance compute and TMA benefits |
| >256 | 256x256x256 | (4,4,1) | 2SM | Maximize B matrix sharing |

Key insight: When ClusterShapeM > 1, ThreadBlocks in a Cluster share B matrix via TMA Multicast but load different A matrices. For small M, this causes redundant A loading. Setting ClusterShapeM=1 for small M eliminates this while maintaining ClusterShapeN > 1 for B sharing.

## 14. Retract Memory Release Fix: Page Size > 1 OOM

Fixes OOM caused by incorrect memory release during retract operations when page_size > 1, by unifying memory check logic and precise per-request subset memory calculation.

```python
def new_page_count_next_decode(self, selected_indices=None):
    page_size = self.token_to_kv_pool_allocator.page_size
    requests = (self.reqs if selected_indices is None
                else [self.reqs[i] for i in selected_indices])
    if page_size == 1:
        return len(requests)
    return sum(1 for req in requests if req.seqlen % page_size == 0)
```

Root cause: Original implementation assumed every request needs fixed token count for retract, but with page_size > 1, only requests crossing page boundaries need new pages.

## 15. Speculative Decoding Attention Backend Configuration

Adds `--speculative-attention-backend` parameter allowing target verify and draft extend operations to use either prefill or decode backend.

```python
def _select_backend(self, forward_mode):
    if forward_mode.is_decode_or_idle():
        return self.decode_backend
    elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
        return (self.decode_backend
                if self.server_args.speculative_attention_backend == "decode"
                else self.prefill_backend)
    else:
        return self.prefill_backend
```

CUDA graph initialization optimized to only initialize graphs for the actually selected backend.

## 16. MLA K Matrix Concat: Warp-Level Vectorized Memory Access

Implements a highly optimized K matrix concatenation kernel for DeepSeek-V2/V3 MLA architecture using warp-level cooperation and vectorized memory access.

Design:
- Each warp processes one head chunk (16 heads)
- k_nope uses int2 (128-bit) for 4 bfloat16 elements per thread
- k_rope uses int (64-bit), shared across all heads (loaded once per warp)
- `#pragma unroll` for all 16 heads

```cpp
constexpr int NUM_LOCAL_HEADS = 128;
constexpr int HEAD_CHUNK_SIZE = 16;
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;  // 8

// Vectorized access per warp: 
// Read k_nope: 16 heads x 128 dim x 2 bytes = 4,096 bytes
// Read k_rope: 1 x 64 dim x 2 bytes = 128 bytes (shared)
// Write k: 16 heads x 192 dim x 2 bytes = 6,144 bytes
```

## 17. FlashAttention-4 (FA Cute) Support: CUTLASS DSL Implementation

Adds FlashAttention-4 support based on CUTLASS Cute DSL, optimized for both Hopper (Sm90) and Blackwell (Sm100) architectures.

```python
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

def flash_attn_varlen_func(..., ver=3):
    if ver == 4:
        return flash_attn_varlen_func_v4(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale, causal=causal,
            pack_gqa=pack_gqa, learnable_sink=sinks)
```

Limitations: FA4 does not yet support `flash_attn_with_kvcache` (decode scenario). Requires `nvidia-cutlass-dsl==4.1.0`.

## 18. Pipeline Parallelism KV Cache Fix: Cross-Rank Memory Synchronization

Fixes inconsistent KV cache token capacity across different PP ranks by adding an all-reduce synchronization to ensure all ranks use the same minimum capacity.

```python
if self.pp_size > 1:
    tensor = torch.tensor(self.max_total_num_tokens, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN,
        group=get_world_group().cpu_group)
    self.max_total_num_tokens = tensor.item()
```

Problem: Different PP ranks may have different layer counts (e.g., ranks 0-2 with 33 layers, rank 3 with 32 layers), leading to different computed `max_total_num_tokens` and potential out-of-bounds access during cross-rank communication.

## 19. Data Parallel Controller: Orphan Process Prevention

Adds parent process monitoring and fault handling to the DP controller process, preventing orphan processes when the parent unexpectedly exits.

```python
def run_data_parallel_controller_process(server_args, port_args, pipe_writer):
    kill_itself_when_parent_died()  # Auto-terminate if parent exits
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()  # Print stack trace on crash
```

## 20. Qwen2-MoE Dual Stream: Shared Experts Parallel with Router Experts

Parallelizes shared experts and router experts computation using dual CUDA streams for Qwen2-MoE, improving MoE layer efficiency in small-batch scenarios.

```python
DUAL_STREAM_TOKEN_THRESHOLD = 1024

def forward_normal_dual_stream(self, hidden_states):
    current_stream = torch.cuda.current_stream()
    self.alt_stream.wait_stream(current_stream)
    # Main stream: shared experts
    shared_output = self._forward_shared_experts(hidden_states)
    # Alt stream: router experts (parallel)
    with torch.cuda.stream(self.alt_stream):
        router_output = self._forward_router_experts(hidden_states)
    current_stream.wait_stream(self.alt_stream)
    return router_output, shared_output
```

Only enabled when token count ≤ 1024 to avoid overhead exceeding benefit at large batch sizes.

## 21. HiCache Page First Direct Memory Layout

Adds `page_first_direct` memory layout for HiCache distributed KV cache, optimizing host-device data transfer efficiency through page-level direct memory access.

Layout comparison:
- `layer_first`: (2, layer_num, size, head_num, head_dim)
- `page_first`: (2, size, layer_num, head_num, head_dim)
- `page_first_direct`: (2, page_num, layer_num, page_size, head_num, head_dim)

Benefits: Page-aligned access reduces fragmentation, batch transfers reduce kernel launch count, simplified indexing avoids complex offset calculations.

## 22. DP Attention Race Condition Fix: Independent Buffers

Fixes a race condition in DP Attention by allocating independent buffers per LogitsMetadata instead of sharing a global buffer.

```python
# Before: global shared buffer (race condition)
hidden_states = get_global_dp_buffer()

# After: per-metadata independent buffer
hidden_states = logits_metadata.gathered_buffer
```

Buffer size is dynamically allocated based on actual need (logprob requirement vs preset size).

## 23. DP Attention Extend Mode Consistency Fix

Fixes padding mode inconsistency in DP Attention extend mode by forcing SUM_LEN padding strategy for extend operations, ensuring all ranks use identical strategies.

```python
if self.forward_mode.is_extend():
    dp_padding_mode = DpPaddingMode.SUM_LEN  # Fixed for extend
else:
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(global_num_tokens)
```

Root cause: Different ranks could select different modes (MAX_LEN vs SUM_LEN) based on local token distributions, causing all-gather dimension mismatches.

## 24. Dynamic Batch Tokenizer: Async Queue for Concurrency

Introduces an async dynamic batching tokenizer using `asyncio.Queue` to collect concurrent requests and batch-process them, reducing tokenization overhead under high concurrency.

```python
class AsyncDynamicbatchTokenizer:
    def __init__(self, tokenizer, max_batch_size=32, batch_wait_timeout_s=0.002):
        self._queue = asyncio.Queue()

    async def encode(self, prompt, **kwargs):
        result_future = asyncio.get_running_loop().create_future()
        await self._queue.put((prompt, kwargs, result_future))
        return await result_future
```

Usage: `--enable-dynamic-batch-tokenizer --dynamic-batch-tokenizer-batch-size 32 --dynamic-batch-tokenizer-batch-timeout 0.002`

## 25. Generative Score API: Prefill-Only Optimization

Optimizes the Generative Score API for prefill-only scenarios by skipping unnecessary input token logprobs computation, sampling steps, and deferring GPU-to-CPU copies.

Key optimizations:
- Skip input token logprobs (only compute at last position)
- Skip sampling step entirely for scoring requests
- Vectorized batch logprobs extraction (single GPU kernel for all requests)
- Delayed GPU-to-CPU copy (overlap with next batch computation)

Results (Qwen3-0.6B on H100, 300 tokens input, 10 items/request): At 1000 items/s, P99 latency drops from 6220ms to 454ms (13.7x improvement).

## 26. Triton Attention Deterministic Inference: Fixed Tile Size Split-KV

Adds a fixed tile size split-KV strategy for Triton Attention to ensure deterministic inference results.

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend triton \
    --triton-attention-split-tile-size 256
```

Fixed tile size ensures consistent floating-point accumulation order regardless of sequence length, unlike the original fixed split-count strategy. Test: 50 samples produce exactly 1 unique output.

## 27. OpenTelemetry Request Tracing System

Adds distributed request tracing based on OpenTelemetry with fine-grained latency monitoring and Jaeger visualization.

Three-layer trace context design:
- `SglangTraceReqContext` (request-level)
- `SglangTraceThreadContext` (thread-level: scheduler, tokenizer)
- `SglangTraceSliceContext` (slice-level: prefill, decode, tokenize)

```bash
python -m sglang.launch_server --model MODEL \
    --enable-trace --oltp-traces-endpoint localhost:4317
```

## 28. CUTLASS Update and FP8 Blockwise GEMM Schedule Rename

Updates the CUTLASS library version and unifies FP8 blockwise GEMM kernel schedule naming from `FP8BlockScaledAccum` to `FP8Blockwise` for consistency and performance.
