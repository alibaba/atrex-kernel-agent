# TokenSpeed Achieves 580 TPS on Qwen3.5-397B-A17B with Blackwell

How TokenSpeed achieves peak single-user agentic throughput of 580 tokens/second for Qwen3.5-397B-A17B on Blackwell GPUs through systematic elimination of memory copies, deep kernel fusion, and fully overlapped CPU-GPU execution.

---

## 1. Introduction

Qwen3.5 uses **hybrid attention** (most layers are GDN linear attention, with full attention + KV cache every N layers) to significantly reduce long-sequence inference complexity. TokenSpeed, open-sourced by LightSeek Foundation under MIT license, targets TensorRT-LLM-level performance with vLLM-level usability; the underlying implementation uses native SPMD + static compilation.

## 2. Runtime Design

### 2.1 GDN/Mamba Prefix Caching

Layered responsibilities:

- **C++ side:** Radix tree matching, page IDs, eviction, Mamba slot lifecycle management
- **Python side:** GPU KV pages / Mamba conv_state / ssm_state / stream ordering / copy-on-write / zeroing / snapshot copying

Each active Mamba request holds two types of slots:

- **Working slot:** Mutable state used by current forward pass
- **Checkpoint slot:** Snapshot target, published to prefix tree only after alignment boundary + Python writes are clean

Matching flow: First match KV via radix → find nearest Mamba checkpoint node → scheduler returns `mamba_cow_src_index` → Python COWs to request's working slot before forward. Cache tree slots are never overwritten.

**Core correctness invariant:** `MambaChunkAllocator` does not clear GPU memory when dispensing slots. The only two safe paths for newly allocated working slots are: COW from a known-clean checkpoint, or Python explicitly zeroes before use. Checkpoints are only published at alignment boundaries.

### 2.2 Scheduler: Dual Resource Pools

- Each request simultaneously holds **KV cache block indices** + **mamba_pool_indices**
- State lifecycle: arrival allocation → prefill filling (or prefix load) → decode in-place update → completion/preemption release
- Speculative decoding: scheduler maintains `spec_cache`, per-step Conv/SSM snapshots, rollback on verification failure
- `HybridLinearAttnBackend` routes by `layer_id` to full-attention / linear-attention backend, initializing respective metadata types

### 2.3 GDN Prefill-Decode Disaggregation

**Unified state transfer:** Dual tensor pools (conv_state + ssm_state) pre-allocated as contiguous GPU memory; at registration, Prefill/Decode exchange buffer descriptors (base address, per-slot size, physical buffer → global layer ID mapping). Mamba state and KV use the same RDMA path, differing only in addressing: KV via page table, Mamba via flat slot ID.

**Cross-layer unified heartbeat:** A unified step counter increments after each layer's forward; transfer thread sends data by layer window ("which buffers map to layers 4–7? Send when counter reaches 7"). Decode side mirrors: each layer's forward only blocks if that layer hasn't arrived yet, overlapping network receive with early-layer computation.

**Three-phase handshake:** (1) Last layer group's KV+Mamba completes transfer (barrier waits for forward) → (2) Prefill forward completes producing first output token, event loop notifies transfer thread → (3) Transfer thread sends token notification via side channel to Decode; Decode fires "remote prefill complete" event only after receiving everything. Mamba state / KV / bootstrap token arrive as a logical atomic unit.

## 3. Performance Optimizations

### 3.1 Mamba State Update: Index Indirection Eliminates Copying

**Before:** `fused_mamba_state_scatter_with_mask` copies across `num_layers × state_dim` full tensors every decode step.

**After:** Move pointers, not data.

- State buffer extends a "scratch area" after scheduler-allocated base slot; each request gets a private scratch row slice indexed by `req_pool_index`
- Lightweight `current_input_indices` records which physical row currently holds canonical state per request
- During target-verify: kernel reads initial state from `current_input_indices`-pointed row (no data movement, just index lookup); `output_state_indices` tells kernel where each step's output writes (slot 0 = working, slot 1..N = scratch rows)
- After verification: simply `current_input_indices[req] = accepted last scratch row` — O(1) integer write instead of O(L·D) tensor copy

### 3.2 Overlap Is Everything (CUDA Multi-Stream)

- **Shared expert ∥ Routed experts:** `StreamFork` — main stream runs TopK + dispatch + MoE GEMM, side stream runs shared expert (gate_up → SiLU → down) + sigmoid gating, event-synchronized merge
- **GDN input projection dual-stream:** `in_proj_qkvz` and `in_proj_ba` parallel (activated only during CUDA Graph capture); smaller `in_proj_ba` completely hidden behind larger `in_proj_qkvz`

### 3.3 More Fusion, Less Latency

**Gemma AllReduce fusion:** `GemmaRMSNorm` uses `x*(1+w)` instead of standard `x*w`, previously preventing TRT-LLM AR+Residual+RMSNorm fusion. TokenSpeed pre-computes `gemma_weight = weight + 1.0` as gamma passed to standard fusion kernel, auto-enabled on SM90+ single-node TP.

**`fused_qk_rmsnorm_rope_gate`** (single Triton kernel replacing 5 launches):

| Step | Operation | Original HBM Read | Original HBM Write |
|------|-----------|-------------------|-------------------|
| 1 | Q RMSNorm | q | q_normed |
| 2 | K RMSNorm | k | k_normed |
| 3 | Q RoPE | q_normed | q_rotated |
| 4 | K RoPE | k_normed | k_rotated |
| 5 | Gate split + contiguous copy | q_gate | gate |

After fusion: all intermediate values stay in registers.

**`fused_gate_sigmoid_mul_add`** (MoE shared expert 5 launches → 1): `final += σ(x·w) * shared` — dot-product reduction + sigmoid + broadcast multiply + accumulate all within one threadblock, intermediates never leave registers.

### 3.4 Death by a Thousand Synchronizations

- **Eliminate D2H round-trips:** Replace `.item()` with initialization-time worst-case bounds; H2D: CPU-side max lets GPU tensor + bound arrive simultaneously; speculative decoding uses GPU sentinels + downstream kernel boundary checks to skip invalid entries, keeping all decisions on device
- **torch.compile annotated dispatch indices:** 10–14 independent launches fused into 1–2 elementwise kernels
- **Async everything:** Pinned memory + non-blocking copy; transfer system polls pinned host counter instead of `synchronize()`; per-layer event barriers only wake layers needing data; CPU prepares next batch while current is still processing

### 3.5 FA4 Support

Qwen3.5 defaults to `head_dim=256`; upstream FA4 support has been contributed. Native FA4 for Qwen3.5 in TokenSpeed is under development.

## 4. Benchmarks (EvalScope, B200, NVFP4)

**Baseline:** Attn TP8 + MoE TP8 / Attn TP8 + MoE EP8, bs=1 with MTP enabled: **+100% to +159%**; high-concurrency gains correlate with output length — long output (>4096 tok) at bs=32/64 still achieves +38% to +90%; short output (1024 tok) at bs=64 gains near zero or slightly negative.

**Agentic (50K first turn + 800 tok/turn × 10–15 turns):** All 4 parallel configurations (TP4 / TP4EP4 / TP8 / TP8EP8) at bs=1 ≥ 500 tok/s, **TP8 peak ~580 tok/s**; concurrent=16: TP4 series ~2K tok/min/GPU system throughput, TP8 series ~1K tok/min/GPU. **Multi-turn KV hit rate >90%**.

**Long context NIAH 1M (TP8):** 128K → ~530, 256K → ~495, 1M → ~445 tok/s/user; end-to-end degradation only **~16%**.

## 5. Conclusion

TokenSpeed pushes Qwen3.5-397B-A17B agentic single-user throughput to 580 tps on B200 through: dual resource pool scheduling + GDN prefix caching + unified PD state transfer + index-indirect Mamba state update + multi-stream overlap + extensive operator fusion + inter-graph CPU overhead control.
