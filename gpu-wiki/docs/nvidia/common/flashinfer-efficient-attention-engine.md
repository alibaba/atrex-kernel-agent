# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference

A comprehensive analysis of the FlashInfer system (MLSys 2025), covering its block-sparse KV-Cache format, JIT-compiled attention variants, dynamic load-balanced scheduling, and comparison with FlashAttention.


**Last updated**: 2026-06-30

---

## 1. Paper Overview

FlashInfer is published at MLSys 2025, jointly developed by University of Washington, NVIDIA, Carnegie Mellon University, and Perplexity AI. It addresses the core bottleneck in LLM inference serving — attention computation efficiency — with a systematic solution.

Core contributions span three levels:

1. **Storage:** Unified block-sparse format (BSR) and composable formats solving KV-Cache storage heterogeneity
2. **Compute:** JIT compilation for highly customizable attention templates supporting various attention variants
3. **Scheduling:** Dynamic load-balanced scheduling maintaining CUDA Graph compatibility while managing input dynamism

**Industry adoption:** Integrated into vLLM, SGLang, and MLC-Engine as production-grade LLM serving infrastructure.

## 2. Background and Motivation

### 2.1 Attention Computation Challenges in LLM Inference

Modern LLM inference systems face unprecedented complexity:

| Dimension | Diversity | Technical Challenge |
|-----------|-----------|-------------------|
| Compute patterns | Prefill (full context) / Decode (single token) / Chunked-Prefill | Compute intensity varies 10-1000× |
| KV Storage | PagedAttention / RadixAttention / Contiguous | Different memory access patterns |
| Sequence length | Dynamic range from 1 to 1M+ | Load imbalance reduces GPU utilization |
| Attention variants | GQA, sliding window, Logits SoftCap, ALiBi | Requires customized kernels |
| Batching | Dynamic batching, Continuous Batching | Sequence lengths change in real-time |

Traditional approaches write specialized kernels for each scenario, leading to N×M complexity explosion (N storage formats × M attention variants).

**FlashInfer's core insight:** Use block-sparse matrices as a unified abstraction, solving complexity through compile-time + runtime layered optimization.

### 2.2 FlashAttention Technical Foundation

FlashAttention computes attention in constant on-chip memory via the online-softmax trick, avoiding materializing attention matrices in GPU global memory. FlashAttention-2 and FlashAttention-3 further optimize loop order and pipeline design for Ampere and Hopper GPUs.

FlashAttention's arithmetic intensity: \(O(1/l_{qo} + 1/l_{kv})\), where \(l_{qo}\) and \(l_{kv}\) are query and KV-cache lengths respectively. In LLM serving, this simplifies to \(O(l_{qo})\). Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) boost arithmetic intensity to \(O(g \cdot l_{qo})\), where g is the group size.

Key insight: Decode phase is typically memory-bound (each step generates 1 token but loads entire KV cache), making KV-Cache optimization critical for inference acceleration.

### 2.3 Attention Composability Principle

Block-Parallel Transformer (BPT) reveals that attention outputs from the same query against different key/value sets can be composed by preserving attention outputs and their scales (computed via log-sum-exp).

Attention states can be reduced via a combine operator ⊕ that satisfies both associativity and commutativity, enabling states to be combined in arbitrary order.

This property is exploited by:
- **Ring-Attention:** Distributes long KV across GPUs, each computing local attention states, combined via ⊕. Commutativity ensures rotation order doesn't affect results.
- **Flash-Decoding:** Splits long KV along sequence dimension across SMs for parallel local computation, then ⊕-reduces — transforming serial decode into a parallel reduction tree.

**FlashInfer's design choice:** Attention's canonical output is the state pair (O_unnormalized, lse), with ⊕ as the standard reduction operator (analogous to summation in GEMM).

| Traditional Approach | FlashInfer Approach |
|---------------------|-------------------|
| Must compute serially or fix partition boundaries | Arbitrary partitioning, arbitrary order, post-hoc merge |
| Cannot dynamically reassign when load-imbalanced | Idle SM/GPU can take new KV blocks anytime |
| Cross-device merge requires full softmax results | Only needs to transmit (O, lse) — two small tensors |
| Partition size coupled with correctness | Partition size only affects efficiency, not correctness |

## 3. System Architecture Design

### 3.1 KV-Cache Storage: Unified Block-Sparse Format

FlashInfer uses block-sparse matrices as the unified KV-Cache storage format. This format effectively represents various attention mechanisms including tree attention and importance masks.

KV-Cache is stored in Block-Sparse Row (BSR) format — a block-level extension of CSR (Compressed Sparse Row):

```
BSR representation = (indptr, indices, values)
where:
  - indptr[i] = position of first non-zero column block for query block i
  - indices[j] = column index (physical page number) of the j-th non-zero block
  - values[k] = actual KV data of the k-th non-zero block
```

This unified abstraction encompasses multiple KV-cache formats:

| Storage Scheme | Core Feature | BSR Representation | Typical Use |
|---------------|--------------|-------------------|-------------|
| PagedAttention | Fixed-size page table, non-contiguous physical storage | Block size (Br, page_size), non-zero blocks correspond to valid pages | vLLM-style dynamic memory management |
| RadixAttention | Tree-structured prefix sharing | Hierarchical BSR, supporting intra-node contiguous storage | SGLang dialogue systems |
| Ragged Tensor | Variable-length, compact packing | Block size (Br, 1), row pointers indicate sequence boundaries | Irregular batching |
| Contiguous Tensor | Traditional padded-aligned storage | Dense BSR special case with block size (Br, Bc) | Training scenarios |

### 3.2 Composable Formats

Single BSR has limitations: larger block row sizes improve shared memory/register reuse but increase fragmentation; smaller sizes are more flexible but reduce data reuse.

FlashInfer's composable format allows KV-Cache sparse matrix decomposition based on prior knowledge. For example, when requests share a prefix:
- **Shared prefix** forms a naturally dense sub-matrix (all requests access the same columns)
- **Private suffixes** form diagonal sparse blocks (each request only accesses its own pages)

These two structures have opposing optimal block sizes. The solution: decompose into two sub-matrices, each using its optimal block size independently. Physical KV pages remain untouched — only indptr and indices arrays are recomputed.

Two-phase online softmax ensures mathematical correctness: separate kernel invocations produce partial results merged via exact correction factors (not approximation).

### 3.3 Compute Abstraction

FlashInfer develops CUDA/CUTLASS templates for both dense and block-sparse matrices, compatible with NVIDIA GPUs from Turing to Hopper.

**Global-to-Shared Memory Data Movement:**

| Architecture | Copy Instruction | Use Case | Bandwidth Optimization |
|-------------|-----------------|----------|----------------------|
| Ampere/Ada (sm80-sm89) | LDGSTS (128B) | All KV-Cache types | ~6.6 TB/s at 32KiB in-flight |
| Hopper (sm90) | TMA (Tensor Memory Accelerator) | Contiguous dense KV-Cache | Hardware-accelerated, non-blocking |
| Hopper Fallback | LDGSTS | Sparse/non-contiguous access | When TMA doesn't support non-affine patterns |

**Multiple Tile Size Micro-Kernels:**

FlashInfer provides FA2 kernels with tile sizes (1, 16, 32, 64, 128) × (32, 64, 128), selecting optimal tile size based on hardware resources and workload intensity via heuristics:
1. Determine average query length per batch (fused with head group dimension for GQA)
2. Select smallest query tile size meeting or exceeding that length
3. Formulate register and shared memory constraints as functions of K/V tile size to maximize SM occupancy

**JIT Compiler for Attention Variants:**

Inspired by FlexAttention, FlashInfer designs customizable CUDA templates accepting attention variant specifications as input:

- `QueryTransform` / `KeyTransform` / `ValueTransform`: Transformations applied to Q/K/V before attention computation
- `OutputTransform`: Transformation applied to attention output before returning
- `LogitsTransform`: Transformation applied to logits before softmax
- `LogitsMask`: Mask applied to logits before softmax

The JIT compiler generates CUDA code by inserting variant classes into templates, compiled via PyTorch's JIT compiler and registered as custom operators. This separates the "unchanging high-performance skeleton" from "model-specific small differences."

### 3.4 Dynamic-Aware Runtime

**Load-Balanced Scheduling:**

FlashInfer's scheduling algorithm minimizes SM idle time by uniformly distributing workload across all SMs. The algorithm generates deterministic aggregation order, ensuring consistent output given the same sequence length information.

Runtime flow:
1. CPU reads sequence length information → generates scheduling plan
2. Plan is asynchronously copied to GPU workspace
3. Persistent attention kernel computes local outputs according to plan
4. Contraction kernel aggregates local outputs into final output per the reduction map

The attention kernel does not directly produce final output — long KVs are split into multiple blocks, with final output being the contraction (via attention combine operator) of all blocks' local outputs.

**Why deterministic aggregation order matters:** Attention block merging involves floating-point operations that don't satisfy strict associativity. Fixed aggregation order ensures reproducible, stable outputs.

### 3.5 Programming Interface

FlashInfer provides a two-phase API:

- **plan():** CPU-side scheduling planning (not captured by CUDA Graph)
- **run():** GPU-side execution according to plan (captured by CUDA Graph)

Kernels are JIT-compiled at initialization and cached for reuse. For composable formats, multiple attention wrappers with different block sizes are created and captured into different CUDA Graphs. At runtime, the serving framework selects the most appropriate graph based on current KV-Cache configuration.

## 4. FlashAttention vs FlashInfer

These are not "same-layer" competing products:

| Dimension | FlashInfer | FlashAttention |
|-----------|-----------|----------------|
| Project positioning | Inference kernel library + generator, serving-oriented | Exact attention official implementation, algorithm/operator focused |
| Primary scenario | LLM inference serving: prefill/decode/mixed batching | Training + inference efficient attention |
| API style | wrapper/plan/run/workspace, stateful | flash_attn_func-style functional interface |
| Coverage | Attention + GEMM + MoE + sampling + communication + page API | Primarily attention family |
| KV-cache | Paged/ragged/cascade/MLA as first-class citizens | Has flash_attn_with_kvcache but not the organizational center |
| Backend strategy | Multi-backend automatic selection | Is the attention kernel implementation itself |
| Customization | Custom attention variants + JIT emphasized | More fixed function set |
| Training backward | Not primary focus | Explicitly supports forward/backward |

**One-sentence summary:**
- FlashAttention: Make scaled dot-product attention into an extremely efficient core operator
- FlashInfer: Make the "attention/KV-cache/runtime/kernel combination problem needed for LLM inference" into a service-oriented kernel platform

## 5. vLLM Backend Comparison

### 5.1 FlashInfer Backend in vLLM

Acts as a "dispatch controller" — a multi-backend attention execution entry point:

1. Input validation and profiling/CUDA Graph dummy run handling
2. Compute and cache quantization scaling coefficients
3. Write current K/V to paged KV cache if needed
4. Split tokens into prefill and decode portions based on metadata
5. Route prefill to FlashInfer or TRT-LLM kernel
6. Route decode to FlashInfer or TRT-LLM kernel
7. Handle distributed context parallel (DCP) cross-device gather/reduce if needed
8. Return padded output tensor for CUDA Graph compatibility

### 5.2 FlashAttention Backend in vLLM

Acts as a "FlashAttention kernel adaptation layer" — organizing vLLM's batch/paged KV/varlen metadata into FlashAttention's expected format:

1. Version validation, disable unsupported fusion
2. Handle encoder attention as special path
3. Write current K/V to paged cache
4. Call `flash_attn_varlen_func(...)` for standard path
5. Call `cascade_attention(...)` for cascade path

### 5.3 Key Differences

| Dimension | FlashInfer Version | FlashAttention Version |
|-----------|-------------------|----------------------|
| Overall role | Multi-backend attention execution entry | FlashAttention backend adapter |
| Primary dispatch axis | prefill / decode / mixed | encoder / decoder(cross) + cascade / non-cascade |
| Core execution | wrapper.run(...) / trtllm_* | flash_attn_varlen_func(...) / cascade_attention(...) |
| TRT-LLM branch | Yes | No |
| Output quantization fusion | Supported (TRT-LLM only) | Not supported |
| Explicit prefill/decode split | Yes | No |
| Orientation | Serving runtime | Kernel invocation adapter |

## 6. Summary

FlashInfer represents a paradigm shift from "one kernel per scenario" to "unified abstraction + JIT specialization + runtime scheduling." Its block-sparse format provides storage-agnostic KV-Cache representation, the JIT compiler enables O(1) effort for new attention variants, and the dynamic scheduler ensures GPU utilization across heterogeneous workloads — all while maintaining CUDA Graph compatibility required for production serving.


## Related

- [Async Global-to-Shared Memory Copy (CC 8.0+)](async-global-to-shared-copy.md)
- [FlashAttention 1–4: GPU Generational Evolution](flash-attention-1-to-4-gpu-evolution.md)
- [GPU Architecture Deep Dive](gpu-architecture-deep-dive.md)
- [Memory-Bound Kernel Optimization: Hierarchical Reduction](hierarchical-reduction-memory-bound.md)
- [L2 Cache Persistence Control (CC 8.0+)](l2-cache-persistence.md)
- [Composable Kernel (CK) Architecture Overview](../../amd/common/ck-architecture-overview.md)
