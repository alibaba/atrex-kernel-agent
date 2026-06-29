# When Operators Approach Hardware Limits: Profiling-Driven Analysis of Real-Time Inference on Orin

This document covers compiling Flash Attention 2 on Jetson AGX Orin (SM 8.7), benchmarking FA2 vs SDPA, deep profiling of GEMM bandwidth bottlenecks, and the discovery that 67% of inference time is framework overhead rather than GPU computation.

---

## 1. Background: Why SDPA Was the Only Option

In the previous deployment of wall-x on Orin, Flash Attention 2 failed to compile, forcing the use of SDPA. Profiling revealed that SDPA's Softmax on Orin runs as a standalone `cunn_SoftMaxForward` kernel (746ms, 4.7% of GPU time) — 57x slower than the fused version on RTX 5090. The root cause: HuggingFace's `generate()` always passes `attention_mask`, which forces SDPA to fall back to the math backend since cuDNN and flash_sdp do not support non-null masks.

FA2's core advantage is fusing Q×K^T → Softmax → ×V into a single kernel, handling causal masks internally in CUDA C++ without relying on external `attention_mask`. Compiling FA2 on Orin should eliminate this 57x Softmax penalty.

## 2. Why FA2 Fails to Compile on Orin

### 2.1 No Pre-built Wheel

PyPI only ships x86_64 wheels for flash-attn. Jetson requires source compilation.

### 2.2 SM 8.7 Not in Default Architecture List

FA2 v2.8.3's `setup.py` defaults to:

```python
os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")
```

SM 8.7 is absent. The `add_cuda_gencodes()` function generates only SASS (e.g., `code=sm_80`) without PTX forward-compatible code (`code=compute_80`). SM 8.0 SASS cannot execute on SM 8.7, producing:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### 2.3 Triton Dependency Unsupported on aarch64

FA2 uses `flash_attn.ops.triton.rotary` which depends on Triton's JIT compiler. Triton has no ARM/aarch64 backend support.

## 3. Two Patches for Successful Compilation

**Patch 1: Add SM 8.7 gencode to setup.py**

```cpp
if "87" in cuda_archs():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_87,code=sm_87")
```

**Patch 2: Replace Triton rotary with pure PyTorch fallback**

```python
_HAS_TRITON = False
try:
    import triton
    _HAS_TRITON = True
except ImportError:
    pass
apply_rotary = _apply_rotary_triton if _HAS_TRITON else _apply_rotary_torch
```

Compilation command:

```bash
FLASH_ATTN_CUDA_ARCHS='87' \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=6 NVCC_THREADS=2 \
python setup.py build_ext --inplace
```

73 `.cu` files, ~50 minutes on Orin's ARM CPU.

## 4. Benchmark: FA2 vs SDPA on Orin

Test configuration: wall-x 3B model, bf16, single 640×480 image, VQA prompt, max_new_tokens=64, 3 warmup + 10 timed runs, GPU locked at 1300.5 MHz.

| Metric | SDPA | Flash Attention 2 | Delta |
|--------|------|-------------------|-------|
| Mean latency | 8681 ms (std 109) | 6360 ms (std 25) | FA2 27% faster |
| Throughput | 7.4 tok/s | 10.1 tok/s | +36.5% |
| Peak GPU memory | 8.19 GB | 8.14 GB | Equivalent |
| Latency std dev | 109 ms | 25 ms | FA2 4x more stable |

Cross-platform comparison (by tok/s):

| Platform | Attention | tok/s | vs 5090 |
|----------|-----------|-------|---------|
| RTX 5090 | FA2 | 59.8 | 1.0x |
| Orin + SDPA | cuDNN SDPA | 7.4 | ~8.1x |
| Orin + FA2 | FA2 v2.8.3 | 10.1 | ~5.9x |

## 5. Why FA2 is "Only" 27% Faster

FA2 typically delivers 2-3x speedup on A100/H100. Three factors limit gains on Orin:

### 5.1 Short Sequences Reduce Tiling Benefits

Input sequence length is ~420 tokens. The attention score matrix (420×420 = ~344KB fp16) fits entirely in Orin's 4MB L2 cache, reducing the advantage of FA2's tiling mechanism.

### 5.2 Tile Size Tuned for SM 8.6/8.9, Not Orin

FA2's v2 code path sets tile sizes benchmarked on GPUs with 82-128 SMs and 6-72 MB L2. Orin's 16 SMs + 4MB L2 + 204 GB/s bandwidth is completely outside the design envelope.

NCU measurements confirm this:

| Kernel Type | L2 Read Hit | SM Throughput | Occupancy |
|-------------|-------------|---------------|-----------|
| Prefill (head_dim=96) | 64.9% | 28% | 16.5% |
| ViT (head_dim=96) | 95.9% | 64.2% | 16.9% |
| Decode splitkv | 11.5% | 7.3% | 10% |

Decode's low L2 hit (11.5%) is an inherent characteristic of single-pass KV traversal with no temporal reuse — not fixable by tuning tile size.

### 5.3 Non-contiguous Tensors Require Extra Copies

FA2 requires contiguous input tensors. wall-x's multimodal RoPE produces non-contiguous Q/K, requiring `.contiguous()` copies that add overhead.

## 6. GEMM Profiling: Bandwidth Is the Bottleneck

Detailed breakdown of 1643ms total GEMM time (6264 calls):

| Rank | Kernel | Calls | Time (ms) | % GEMM | Characteristic |
|------|--------|-------|-----------|--------|----------------|
| 1 | bf16 64x64 sliced | 3240 | 869 | 52.9% | Decode batch=1 GEMV |
| 2 | bf16 128x128 | 288 | 204 | 12.4% | Prefill/ViT large GEMM |
| 3 | gemv2T | 30 | 112 | 6.8% | LM head projection |
| 4 | cutlass 256x128 | 128 | 98 | 5.9% | Custom dual_asym_gemm |

The model already uses cuBLAS (`ampere_bf16_s16816gemm`) and CUTLASS — the fastest available bf16 implementations. Measured Orin GPU effective bandwidth:

```
Buffer 64MB:  178.9 GB/s
Buffer 256MB: 168.9 GB/s
Buffer 1GB:   168.1 GB/s
```

Theoretical minimum for one decode step (MoE sparse routing, ~4.3 GB effective weight read):

```
4.3 GB ÷ 168 GB/s = 25.6 ms (bandwidth limit)
Actual GEMM time ≈ 25.7 ms/step (~72% bandwidth utilization)
```

Kernel time ≈ pure memory read time. This is a bandwidth bottleneck, not a compute bottleneck. Only quantization can improve it.

## 7. Quantization Paths on Orin

| Method | Principle | Theoretical Speedup | Feasibility |
|--------|-----------|--------------------:|-------------|
| bf16 (current) | cuBLAS s16816 | 1.0x | In use |
| INT8 | cublasLt INT8, halve bandwidth | ~2x | Viable via cublasLt |
| INT4 (AWQ/GPTQ) | 4-bit weights + fp16 compute | ~3-4x | Needs Marlin kernel |
| FP8 | Tensor Core FP8 | ~2x | SM 8.7 lacks FP8 TC |

All Python-level quantization attempts failed:
- `torch.compile`: only 1.7% faster (custom CUDA ops cause graph breaks)
- `torch._int_mm`: does not support M=1 (decode GEMV)
- `bitsandbytes`: crashes on SM 8.7 (no precompiled kernel)

## 8. The Real Bottleneck: 67% Framework Overhead

```
Per decode step:
  Wall clock:       97.2 ms (100%)
  GPU kernel:       29.7 ms (30.6%)  ← actual computation
  Framework overhead: 67.5 ms (69.4%)  ← Python/HuggingFace idle
```

GPU utilization is only 32.7%. Sources of overhead:
- HuggingFace `generate()` Python dispatch (LogitsProcessor, StoppingCriteria, validation)
- 52,280 kernel launches at 20.7 μs/launch scheduling latency
- Dynamic KV cache memory management
- Python GIL and garbage collection

64 decode steps waste ~4320ms on framework overhead (67.5ms/step × 64) — more than total GEMM time (1643ms).

## 9. TRT-LLM Flash Attention vs Open-Source flash-attn

TRT-LLM ships 180 SM 8.7-specific precompiled cubin files for fused attention, completely independent from open-source flash-attn:

| | Open-source FA2 v2.8.3 | TRT-LLM FMHA |
|---|---|---|
| SM 8.7 handling | Generic sm8x path | 180 dedicated SM 8.7 cubins |
| L2 awareness | None in v2 path | Internally optimized |
| Compilation | User-side nvcc | Precompiled ELF embedded in source |

Isolated attention micro-benchmark (batch=1, 16 heads, head_dim=128, seq=420):

| Backend | Prefill Latency | vs FA2 |
|---------|----------------|--------|
| TRT-LLM FMHA | 0.075 ms | 4.1x faster |
| cuDNN SDPA | 0.127 ms | 2.4x faster |
| Open-source FA2 | 0.306 ms | 1.0x (slowest) |

Key insight: FA2's 27% end-to-end speedup comes from bypassing HuggingFace's attention_mask → math backend degradation path (saving 746ms of standalone softmax), not from a faster attention kernel itself.

Additional finding: bypassing `attention_mask` and calling `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)` enables cuDNN fused attention at 0.076ms — matching TRT-LLM's 0.075ms at zero additional cost.

## 10. Three Flash Attention Routes on Orin

| Route | Prefill Latency | SM 8.7 Adaptation | Integration |
|-------|----------------|-------------------|-------------|
| Open-source FA2 v2.8.3 | 0.306 ms | Generic sm8x path | PyTorch, already in use |
| TRT-LLM FMHA cubin | 0.075 ms | 180 dedicated cubins | CUDA Driver API, high effort |
| FlashInfer | TBD | Unofficial (set `FLASHINFER_CUDA_ARCH_LIST="8.7"`) | Python API, moderate effort |

Since attention only accounts for 1.4% of total GPU time, investing in cubin extraction yields minimal returns (~20ms savings).

## 11. Can TRT-LLM or llama.cpp Run wall-x?

wall-x has four architectural layers:
1. Vision encoder (Qwen2.5 ViT) — standard
2. Text decoder (Qwen2.5 causal decoder + GQA) — standard
3. MoE modification (TokenTypeRouter + custom CUDA kernels) — non-standard
4. Action generation (Flow Matching + ODE Euler integration + KV Cache truncation) — completely non-standard

**TRT-LLM**: VQA text generation is theoretically possible (with MoE plugin development). Flow Action generation is fundamentally impossible — ODE integration requires runtime dynamic tensor replacement that static graph engines cannot support.

**llama.cpp**: Not feasible. No plugin mechanism for 6 custom CUDA ops, no MoE support matching wall-x's TokenTypeRouter, no Flow Matching concept.

## 12. Optimization Priority Order

| Phase | Target | Estimated Savings |
|-------|--------|------------------:|
| 1 | C++ framework (eliminate 67% overhead) | ~4320ms |
| 2 | INT8 quantization (cublasLt in C++) | ~800ms |
| 3 | CUDA Graph (eliminate launch overhead) | ~260ms |

Core conclusion reversal: GEMM computation (78.8% of GPU time) appears to be the bottleneck, but GPU only works 32.7% of wall-clock time. The real bottleneck is 67% framework overhead.

Correct optimization order: C++ runtime → quantization → CUDA Graph.

## 13. Triton Performance on Orin SM 8.7

Triton 3.6.0 compiles and runs on Orin. Benchmark results (warmup 10, measure 100):

| Operator | Size | PyTorch | Triton | Winner |
|----------|------|---------|--------|--------|
| vector_add | 1M elements | 0.039ms | 0.069ms | PyTorch |
| softmax | 2048×2048 | 0.418ms | 0.096ms | Triton 4.3x |
| softmax | 16×420 (decode) | 0.031ms | 0.072ms | PyTorch |
| GEMM | 2048×2048 | 0.642ms | 1.060ms | PyTorch |
| fused add+rmsnorm | 420×2048 | 0.290ms | 0.074ms | Triton 3.9x |

Triton's value on Orin is kernel fusion (3.9-4.3x for fused ops), not replacing cuBLAS/cuDNN for individual operators.

## 14. CPU Load Impact on GPU Inference (Unified Memory)

On Orin's unified memory architecture, CPU high load directly degrades GPU inference:

| Condition | FA2 Latency | Std Dev |
|-----------|------------|---------|
| Idle (load avg < 5) | 6360 ms | 25 ms |
| High load (load avg ~18) | 16289 ms | 1410 ms |

156% degradation due to shared memory bus contention and CPU-side kernel launch doorbell delays. At 52,280 kernel launches per inference, even small per-launch delays compound significantly.

## 15. Conclusion

Key findings in priority order:
1. GPU utilization is only 32.7% — 67% of time is Python/HuggingFace framework overhead
2. GEMM has reached Orin's bandwidth ceiling (cuBLAS at 72% utilization, kernel time ≈ pure memory read time)
3. FA2 provides 27% speedup purely from softmax fusion, despite its tile size being suboptimal for Orin
4. cuDNN SDPA matches TRT-LLM performance (0.076ms vs 0.075ms) when `attention_mask` is bypassed — achievable at zero cost in a C++ framework
5. The optimization path has shifted from operator-level to system-level: C++ runtime, CUDA Graph, and model-set co-scheduling

The fundamental insight: for embodied AI, the bottleneck is not in the model but in the runtime. A VLA model is a lossy physical simulator — increasing refresh rate (1.5 Hz → 8 Hz) matters more than improving single-inference precision.
