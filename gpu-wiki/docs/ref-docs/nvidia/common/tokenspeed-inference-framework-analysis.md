# TokenSpeed Inference Framework Architecture

A comprehensive technical analysis of the TokenSpeed LLM inference engine, covering its five-layer architecture, C++ scheduler FSM, kernel registry selection algorithm, PD disaggregation with RDMA transfer, and core mechanisms including continuous batching, paged attention, CUDA Graph, and speculative decoding.

---

## 1. Overview

TokenSpeed is a high-performance LLM inference engine targeting TensorRT-LLM-level performance with vLLM-level usability. The project consists of four core sub-packages:

```
tokenspeed/
├── python/tokenspeed/          # Main runtime (core inference engine)
├── tokenspeed-kernel/          # GPU kernel registry and dispatch framework
├── tokenspeed-scheduler/       # C++ high-performance scheduler (pybind11)
├── tokenspeed-mla/             # Blackwell GPU specialized MLA attention kernel
├── docker/                     # Docker deployment
├── docs/                       # VitePress documentation
└── test/                       # Integration tests and benchmarks
```

## 2. Five-Layer Architecture

### 2.1 Layer 1: API Layer (Main Process)

| Component | File | Responsibility |
|-----------|------|----------------|
| CLI | `cli.py` | `tokenspeed serve/bench/env` command entry |
| HTTP Server | `entrypoints/http_server.py` | FastAPI + Uvicorn, OpenAI-compatible API |
| Engine | `entrypoints/engine.py` | Python SDK entry, process management |
| AsyncLLM | `engine/async_llm.py` | Async frontend with tokenizer, IPC, request state |

### 2.2 Layer 2: Scheduling Layer (Subprocess)

| Component | File | Responsibility |
|-----------|------|----------------|
| EventLoop | `engine/event_loop.py` | Main scheduling loop (overlap/non-overlap modes) |
| C++ Scheduler | `tokenspeed-scheduler/csrc/` | High-performance FSM + Radix Tree prefix cache + page management |
| RequestHandler | `engine/request_handler.py` | Request parsing, grammar compilation, queuing |
| OutputProcessor | `engine/generation_output_processor.py` | Post-processing: sampling results to token stream |

### 2.3 Layer 3: Execution Layer (GPU)

| Component | File | Responsibility |
|-----------|------|----------------|
| ModelExecutor | `execution/model_executor.py` | Forward dispatch, CUDA Graph capture, sampling |
| ModelRunner | `execution/model_runner.py` | Model loading (weights), model.forward() invocation |
| Attention Backend | `layers/attention/` | KV Cache Pool + multi-backend attention |
| CudaGraphWrapper | `execution/cuda_graph_wrapper.py` | CUDA Graph capture and replay |
| MemoryExecutor | `cache/executor/` | L2(Host)/L3(Storage) KV Cache tier management |

### 2.4 Layer 4: Model and Kernel Layer

| Component | Location | Responsibility |
|-----------|----------|----------------|
| Models | `runtime/models/` | DeepSeek V3/V4, Kimi K2.5, Qwen3/3.5, Llama, MiniMax |
| Kernel Registry | `tokenspeed-kernel/registry.py` | Unified kernel registration/selection (priority system) |
| Ops | `tokenspeed-kernel/ops/` | gemm/, moe/, attention/, quantization/, communication/ |
| Thirdparty | `tokenspeed-kernel/thirdparty/` | DeepGemm, TRT-LLM, FlashInfer, Triton |

### 2.5 Layer 5: Platform Specialization and Disaggregated Inference

| Component | Location | Responsibility |
|-----------|----------|----------------|
| TokenSpeed MLA | `tokenspeed-mla/` | Blackwell SM100/103 specialized MLA kernel (CuTe DSL) |
| PD Disaggregation | `runtime/pd/` | Prefill-Decode disaggregated deployment (RDMA KV Transfer) |

## 3. Request Lifecycle

```
User HTTP Request (POST /v1/chat/completions)
  → http_server.py: FastAPI route handler
  → OpenAIServingChat.handle_request() → GenerateReqInput
  → AsyncLLM.generate_request(obj)
  → InputProcessor.tokenize_one_request()
  → ZMQ IPC (send_to_scheduler.send_pyobj)
  → EventLoop.event_loop()
    → _process_new_requests()
    → scheduler.next_execution_plan()  [C++ Scheduler]
    → model_executor.execute_forward_op()
    → ModelExecutionResult (next_token_ids, logprobs)
  → OutputProcessor.post_process_forward_op()
  → ZMQ (send_to_tokenizer)
  → AsyncLLM._result_dispatcher
  → HTTP StreamingResponse
```

## 4. C++ Scheduler FSM

### 4.1 State Definitions

```cpp
using State = std::variant<
    Bootstrapping, Submitted, Prefetching, PrefetchDone,
    Aborting, Prefilling, PrefillDone, Decoding,
    Draining, WritingBack, Retracting, Retracted, Finished>;
```

### 4.2 State Descriptions

| State | Meaning | Held Resources |
|-------|---------|----------------|
| Bootstrapping | PD mode: awaiting remote prefill | TokenContainer, page_size |
| Submitted | Queued, waiting for scheduling | TokenContainer, page_size |
| Prefetching | L3/Host cache hit, loading to device | host_pages, host_lock |
| PrefetchDone | Prefetch complete, awaiting scheduling | TokenContainer |
| Prefilling | Executing prefill (may span multiple chunks) | DeviceNodeRef + LocalKVAllocator + ReqPoolIndex |
| PrefillDone | All chunks dispatched, awaiting decode transition | Same + reserve tokens |
| Decoding | Token-by-token generation | DeviceNodeRef + LocalKVAllocator + ReqPoolIndex |
| Draining | Generation complete, preparing host writeback | pages_to_transfer |
| WritingBack | Async device-to-host transfer in progress | DeviceNodeRef + HostNodeRef (RAII lock) |
| Retracting | OOM eviction to host in progress | Same + local_kv_allocator |
| Retracted | Evicted to host, awaiting loadback | HostNodeRef + local_kv_allocator |
| Finished | Request complete | None |

### 4.3 Core Scheduling Logic (NextExecutionPlan)

```cpp
ExecutionPlan Scheduler::NextExecutionPlan() {
    // 1. Generate WriteBack ops (completed requests KV → Host)
    write_back_ops = newWriteBackOperation(requests_);
    // 2. Clean up Finished requests
    std::erase_if(requests_, [](auto& req) { return req->Is<Finished>(); });
    // 3. Collect schedulable candidates
    candidates = collect_candidates();
    // 4. Generate Forward operations (select batch composition)
    auto [fwd_ops, cache_ops] = newForwardOperation(candidates);
    // 5. Assemble execution plan
    return plan.With(FlatForwardOperation{fwd_ops})
               .With(CacheOperation{write_back_ops})
               .With(CacheOperation{load_back_ops});
}
```

Decision logic: Prefill prioritized over decode (chunked prefill strategy); budget control via `max_scheduled_tokens`; OOM triggers retraction; Radix Tree prefix matching skips already-computed portions.

### 4.4 Radix Tree Prefix Cache

Located in `tokenspeed-scheduler/csrc/resource/radix_tree/`:
- Each node represents a token sequence segment and its corresponding KV cache pages
- `Match(token_pages)` returns longest matching prefix (device and host matched separately)
- Reference counting (DeviceNodeRef/HostNodeRef) prevents eviction of active nodes
- Eviction policy: LRU from leaf nodes

## 5. Kernel Registry Selection Algorithm

### 5.1 Three-Dimensional Scoring System

Selection uses lexicographic ranking (oracle, objective, priority) — highest score wins:

| Dimension | Source | Range | Weight (lexicographic) |
|-----------|--------|-------|----------------------|
| Oracle | Per-family expert scorer | [0, 20) | Highest |
| Objective | Kernel tags match user objective | 0 or 1 | Middle |
| Priority | Declared at kernel registration | [0, 20) | Lowest |

### 5.2 Priority Band System

```python
class Priority(IntEnum):
    REFERENCE = 0      # Correctness verification only
    PORTABLE = 4       # Generic (e.g., default Triton)
    PERFORMANT = 8     # Optimized (e.g., Triton for Hopper+)
    SPECIALIZED = 12   # Highly optimized (e.g., CuTe DSL FP8 for Blackwell)
    PLUGIN = 16        # Plugin reserved (third-party override)
```

### 5.3 Override Priority Chain

```
Environment variable (highest)
  → Context Manager: kernel_override("gemm", "mm", "deep_gemm")
  → Config file: ~/.config/tokenspeed-kernel/overrides.yaml
  → Heuristic selection (normal scoring)
```

### 5.4 Platform Capability Matching

```python
@dataclass(frozen=True)
class CapabilityRequirement:
    min_arch_version: ArchVersion | None  # e.g. SM90
    vendors: frozenset[str]               # e.g. {"nvidia"}
    required_features: frozenset[str]     # e.g. {"tensor_core:f8", "tcgen05"}
```

Kernels not satisfying platform requirements are filtered before scoring.

## 6. PD Disaggregation: RDMA Transfer Flow

### 6.1 Architecture

```
Prefill Node                          Decode Node
  User Request → Prefill                Bootstrap ← AsyncLLM
    → Model Forward (all layers)          Wait for KV from Prefill
    → KV Cache in GPU VRAM                Receive KV via RDMA
    → MooncakeKVSender ─── RDMA ────────→ MooncakeKVReceiver
    → Send first token via ZMQ            Continue Decode (token by token)
```

### 6.2 Transfer Backend: Mooncake Transfer Engine

Uses GPU Direct RDMA (GDR) for GPU-to-GPU zero-copy transfer. Supports P2P handshake protocol.

### 6.3 Transfer Phases

**Phase 1 — Bootstrap**: Prefill node starts ZMQ Bootstrap Server; Decode node connects and exchanges RDMA connection information (remote memory addresses, rkey).

**Phase 2 — KV Transfer**: Sender transmits specified KV cache pages via RDMA write. Receiver allocates page slots and initiates RDMA receive.

**Phase 3 — Layerwise Transfer (Pipeline Optimization)**: When enabled, prefill starts transmitting layer N's KV immediately after computation, overlapping GPU compute with network transfer.

### 6.4 Synchronization

TP-group synchronization uses Gloo all_reduce to ensure all ranks observe consistent transfer completion state before entering decode.

## 7. Core Mechanisms

### 7.1 Continuous Batching

Each EventLoop iteration can insert new requests and remove completed ones. `FlatForwardOperation` places prefills first, then decodes, enabling mixed-batch single-forward execution.

### 7.2 Paged Attention

KV cache organized by pages (blocks), each storing a fixed number of tokens. Block tables provide logical-to-physical mapping. Supports prefix sharing via Radix Tree (shared physical pages for common prefixes).

### 7.3 CUDA Graph

Decode phase has relatively fixed batch shapes — CUDA Graph eliminates kernel launch overhead. Warmup captures graphs for sizes [1, 2, 4, 8, 16, 32, 64, 128, ...]. Runtime pads to nearest captured size.

### 7.4 ForwardMode

```python
class ForwardMode(IntEnum):
    EXTEND = auto()          # Prefill (extend KV cache)
    DECODE = auto()          # Token-by-token decode
    MIXED = auto()           # Chunked prefill + Decode mixed
    IDLE = auto()            # DP attention idle rank
    TARGET_VERIFY = auto()   # Speculative: target model verification
    DRAFT_EXTEND = auto()    # Speculative: draft model extension
```

## 8. Advanced Mechanism Interactions

### 8.1 CUDA Graph and DP Interaction

All DP ranks synchronize via CPU `all_gather` exchanging `global_num_tokens` and `global_batch_size`. Idle ranks execute `execute_idle_forward()` with zero tokens but replay the same CUDA graph (padded to minimum captured size) because MoE all-to-all and TP all-reduce are collectives requiring all ranks to participate.

### 8.2 DeepEP NVLink vs RDMA

NVLink and RDMA buffers coexist rather than being mutually exclusive:
- **NVLink buffer**: Allocated only in normal mode for intra-node high-bandwidth transfer
- **RDMA buffer**: Allocated in both normal and low-latency modes for inter-node or GPU Direct RDMA

| Mode | Trigger | Buffer Used | Use Case |
|------|---------|-------------|----------|
| Normal | ForwardMode != DECODE | NVL + RDMA mixed | Prefill (large batch) |
| Low-latency | ForwardMode == DECODE | Pure RDMA (IBGDA) | Decode (batch < 256) |

### 8.3 Speculative Decoding

For topk ≤ 1 (linear chain, current production path): each request's draft tokens form a contiguous query sequence for causal attention. Tree attention (topk > 1) code exists as scaffolding but is currently abandoned — would require custom_mask construction, tree verify kernel, and spec_info integration with CUDA graph wrapper.

### 8.4 Three-Level Cache

```
RadixTree (shared structure)
├── DeviceResource (L1 GPU Pages)
├── HostResource (L2 Host Pages)
└── page_hashes_ (L3 Storage Keys via SHA256 hash chain)
```

Request lifecycle: Match prefix → Allocate device pages → LoadBack if host > device → Prefill → Decode → Finish → Insert new pages → WriteBack (GPU→Host) → Finished.

### 8.5 Mamba Hybrid Cache

TreeNode holds both KV pages (attention layers) and `mamba_slot_` (SSM state snapshot index). `MambaChunkAllocator` manages `conv_state` and `ssm_state` slots shared across all Mamba layers.

### 8.6 DP Budget Scheduler

Uses a "water-fill" algorithm: loads each rank's request count, fills lower ranks to match the next-lowest level, producing a budget queue. Each incoming request pops from the queue. Falls back to round-robin when queue is empty.

### 8.7 Context Parallelism (CP)

Uses zigzag splitting for balanced compute across CP ranks:
```
Original: [B0, B1, B2, B3, B4, B5, B6, B7]
cp_rank0: [B0, B7]  (first+last for balance)
cp_rank1: [B1, B6]
cp_rank2: [B2, B5]
cp_rank3: [B3, B4]
```

Each rank performs local attention with reordered positions, then `token_all_gather` with inverse zigzag reorder restores global sequence. Constraints: forces `attn_tp_size=1`, `max_num_seqs=1`, only implemented for DeepSeek V3/NextN models.

## 9. Key Design Decisions Summary

| Dimension | Design Choice |
|-----------|---------------|
| DP granularity | Whole-request dispatch (no single-request token splitting) |
| DP synchronization | CPU all_gather for MoE collective + CUDA graph alignment |
| Radix Tree edges | Page-aligned token vectors; child lookup by first page token |
| L1 eviction | LRU (leaf nodes with ref_count==0) |
| L1→L2 writeback | Async copy on request completion (DeviceNodeRef + HostNodeRef dual lock) |
| L3 key | Chained SHA-256 (supports incremental hashing for unmatched suffix) |
| Kernel selection cache | O(1) dict lookup after first selection |
| PD transfer | Layerwise for compute-network overlap |
| Multi-process | Main process (API + Tokenizer) + Subprocess (Scheduler + GPU Executor) via ZMQ IPC |
