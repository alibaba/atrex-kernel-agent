# Achieving Peak Tensor Core Performance for GEMM on Blackwell via CUTLASS Python

Technical analysis of the GTC 2026 presentation demonstrating progressive GEMM optimization on B200 GPU using CUTLASS Python, from baseline to near-peak Tensor Core throughput.


**Last updated**: 2026-06-30

---

## 1. Introduction and Motivation

This document analyzes the GTC 2026 presentation "Achieve Peak Tensor Core Performance for GEMM on Blackwell via CUTLASS Python" by NVIDIA engineers Linfeng Zheng and Albert Di. It systematically demonstrates progressive GEMM optimization on Blackwell architecture (B200 GPU) using CUTLASS Python.

### 1.1 Why CUTLASS Python?

Reaching SOL (Speed of Light) performance on modern GPUs requires low-level abstractions for fine-grained hardware control. Blackwell introduces features like PDL (Programmatic Dependent Launch) that demand such control.

**Why CUTLASS:**
- Day-0 availability on new hardware with full hardware control
- Expressive abstractions delivering high performance across scenarios
- Modular, extensible design validated across multiple GPU generations

**Why Python interface:**
- C++ template metaprogramming is developer-hostile and produces obscure code
- C++ template compilation times severely impact iteration speed
- Python ecosystem dominates deep learning; Python interfaces integrate naturally

### 1.2 Performance Advantages

On 8K×8K×8K GEMM:
- **Runtime performance**: achieves extremely high TFLOPS throughput
- **Compile speed**: >100x faster than C++ implementation
- JIT compilation + caching mechanism achieves C++-equivalent runtime performance after first compilation

### 1.3 Architecture Overview

CUTLASS Python is built on CuTe DSL (Domain Specific Language) with a layered architecture:

**Bottom Layer — CuTe Atoms:**
- TMEM store/load (Tensor Memory operations)
- TMA (Tensor Memory Accelerator)
- MMA SM90/SM100 (Matrix Multiply-Accumulate instructions)

**Middle Layer — Tiled Copy & Tiled MMA:**
- Blocked copy and blocked matrix multiply abstractions
- Collective operations (SM90 TMA SS, SM100 TMA, SM100 ATMEM)

**Collective Layer:**
- SM90 GEMM TMA Warp Specialized (Ping-pong/Cooperative)
- SM100 GEMM TMA Warp Specialized

**Device Layer (top):**
- Universal Interfaces: Dense GEMM, Grouped GEMM

### 1.4 Ecosystem Status

- Peak single-day downloads: 200K
- Cumulative downloads: 5.6 million
- Adopted by: FAv4 (Flash Attention v4), QuACK, TRTLLM (TensorRT-LLM), FlashInfer

### 1.5 Tutorial GEMM Contributions

Progressive tutorials, each 200-300 lines of code:

| Tutorial | Optimization Technique | Problem Solved |
|----------|----------------------|----------------|
| fp16gemm0.py | Basic GEMM | Baseline implementation |
| fp16gemm1.py | Enable tcgen05.2cta | Insufficient latency hiding |
| fp16gemm2.py | Warp Specialization + TMA Store | Intra-wave optimization |
| fp16gemm3/3_1.py | Static/Dynamic Persistent | Inter-wave switching overhead |
| fp16gemm4.py | Preferred/Fallback Clusters | Maximize SM utilization |
| fp16gemm5.py | PDL | Kernel overlap |
| fp16gemm6.py | TMA L2 Prefetch | DRAM latency |

---

## 2. Basic GEMM Overview

### 2.1 Workflow

The basic GEMM workflow comprises 7 steps:

1. **Create MMA Tile view** (`cute.local_tile()`): Partition GMEM matrices by MMA tile size. Each CTA computes one tile.

2. **TiledMMA partition** (`thr_mma.partition_A()`): Assign tile elements to specific threads for cooperative MMA.

3. **TMA partition SMEM/GMEM** (`tma_partition()`): Prepare source/destination memory partitions for TMA operations.

4. **TMA GMEM → SMEM** (`cute.copy()`): Async load data to shared memory buffers using TMA hardware. Multi-buffering enables overlapping load of next K block with current computation.

5. **Create MMA Fragment** (`tiled_mma.make_fragment_A()`): Create data fragments from SMEM for MMA. On Blackwell, these reside in TMEM.

6. **Execute MMA** (`cute.gemm()`): Invoke Tensor Core for matrix multiply-accumulate. Results stored in TMEM accumulator. K-dimension iterates through K blocks.

7. **Store results**: TMEM → RMEM → GMEM (`cute.copy()`): Transfer final results through register memory to global memory.

### 2.2 Tiling Parameters

- **MMA Tile M/N/K**: Defines single MMA operation size
- **K Tiles**: Number of K-dimension blocks (main loop iterations)
- **ab_stages**: Number of SMEM buffers (software pipeline depth)

### 2.3 Baseline Performance

Configuration: Non-Warp-Specialization, MMA tiler 128×256×64, ab_stages=4, Epilogue: TMEM→RMEM→GMEM.

B200 FP16 peak depends on input data type (DVFS):

| Input Data | Frequency | PFLOPS |
|-----------|-----------|--------|
| Zeros | ~1.8 GHz | ~2.18 |
| Random Int | ~1.6 GHz | ~1.94 |
| Random Float | ~1.35 GHz | ~1.64 |

Baseline performance: **1.64 PFLOPS (1640 TFLOPS)** with random float data.

---

## 3. Progressive Optimization — Compute-Bound Scenarios

### 3.1 Tutorial 0: Software Pipeline and Latency Hiding

DRAM load latency is hundreds to thousands of cycles. Without hiding, Tensor Core idles waiting for data.

**Multi-buffering mechanism:**
- Multiple SMEM buffers (ab_stages) alternate storing TMA-loaded data
- Limited by SMEM capacity

**Latency hiding capacity** = (ab_stages - 1) × one_stage_mma_inst_time

```python
for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
    # Issue TMA loads
    ab_empty = ab_producer.acquire_and_advance()
    cute.copy(...)
    # Execute one K-block worth of MMA instructions
    ab_full = ab_consumer.wait_and_advance()
    cute.gemm(...)
    # Signal buffer consumed
    ab_full.release()
```

**Bottleneck**: With 128×256×64 tile, ab_stages max = 4. Latency hiding: (4-1) × 512 = 1536 cycles. Actual load latency: ~1800 cycles. Since 1536 < 1800, latency cannot be fully hidden. NCU shows intra-wave compute throughput at only ~72%.

### 3.2 Tutorial 1: Enable 2CTA (tcgen05.2cta)

**Blackwell 2CTA**: A CTA pair cooperates across 2 SMs in the same TPC to execute one MMA:
- 2×1 cluster = 1 CTA pair; 4×4 cluster = 8 CTA pairs
- Leader CTA issues MMA instruction (tcgen05.mma) for both SMs
- A/B matrices and accumulator split across 2 SMs
- B matrix is **shared** between SMs (key optimization)
- MMA dimension: M=256 (spanning 2 SMs)

**SMEM savings from B-sharing:**

| Config | 1 CTA SMEM (KB) | 2 CTA SMEM (KB) |
|--------|-----------------|-----------------|
| one_stage_A | 16 | 16 |
| one_stage_B | 32 | 16 |
| one_stage_all | 48 | 32 |
| num_ab_stages | 227÷48 = 4 | 227÷32 = 7 |

Latency hiding: (7-1) × 512 = **3072 cycles** — well above 1800 cycles.

**Performance results:**
- 2K³: 0.96x (sync overhead dominates)
- 4K³: 1.05x
- 8K³: 1.08x (latency hiding advantage visible)
- 16K³: 1.02x (L2 cache becomes bottleneck)

### 3.3 Tutorial 2: Epilogue Optimization — TMA Store + Warp Specialization

**Problem**: RMEM → GMEM (STG) produces non-coalesced memory access, wasting bandwidth.

**Solution**: STG → TMA S2G + Sub-tiling
- Step 1: TMEM → RMEM (T2R)
- Step 2: RMEM → SMEM (R2S)
- Step 3: SMEM → GMEM (TMA S2G, coalesced)

Sub-tiling splits epilogue output into multiple sub-blocks, pipelining T2R → R2S → TMA S2G.

**Warp Specialization** assigns different warp roles:
```python
epilogue_warp_ids = (0, 1, 2, 3)  # 4 warps handle epilogue
mma_warp_id = 4                    # 1 warp handles MMA
tma_warp_id = 5                    # 1 warp handles TMA
threads_per_cta = 32 * 6           # 6 warps = 192 threads
```

**Performance (large K: KxKxK):**
- 2K³: 1.18x | 4K³: 1.11x | 8K³: 1.03x | 16K³: 1.00x

**Performance (small K: KxKx256):**
- 2K³: 1.51x | 4K³: 1.51x | 8K³: 1.53x | 16K³: 1.57x

Key finding: When K is small, epilogue dominates execution time; Warp Specialization + TMA Store yields >1.5x speedup. Highly relevant for LLM inference prefill (K = hidden_dim).

### 3.4 Tutorial 3: Persistent Kernel — Inter-Wave Overlap

**Problem**: Non-persistent mode exposes every wave's prologue and epilogue on the critical path. Wave transitions leave Tensor Core idle.

**Solution**: CTAs remain resident on SMs, processing multiple tiles sequentially:
- Saves (num_waves - 1) prologues
- Wave i-1 epilogue overlaps with Wave i MMA (via Warp Specialization)
- Only first prologue and last epilogue are exposed

**Critical**: Warp Specialization + Persistent must be combined for maximum effect.

**Performance (small K: KxKx256):**
- 4K³: 1.51x | 8K³: 1.74x | 16K³: **1.84x**

### 3.5 Tutorial 4: Preferred and Fallback Clusters

**Problem**: 16K³ shows no gain from Tutorial 3. Root cause: L2 hit rate only 32.89% (working set >> 120MB L2).

**Large cluster advantages:**
- **Programmatic Multicast**: One TMA load stores to multiple CTAs' shared memory within the same cluster
- **CTA Rasterization**: Improves spatial locality for L2

**Quantization problem**: 2×2 cluster on 18 SMs → 16 used, 2 idle.

**Blackwell solution — Dual cluster strategy:**
- Preferred Cluster (large): higher TMA multicast benefit
- Fallback Cluster (small): fills SM gaps when preferred cluster alignment fails

**L2 hit rate improvement:**

| MxNxK | Without | With Preferred Cluster |
|-------|---------|----------------------|
| 8K³ | 46.17% | 73.63% |
| 16K³ | 32.89% | 51.13% |

**Performance**: 8K³: 1.09x | 16K³: **1.26x** (reaching ~1597 TFLOPS)

---

## 4. Additional Optimization Scenarios

### 4.1 Tutorial 3-1: Cluster Launch Control (CLC) — Dynamic Persistent

| Feature | Legacy | Static Persistent | Dynamic Persistent |
|---------|--------|-------------------|-------------------|
| Save wave-switch overhead | No | Yes | Yes |
| Preemption | No | No | Yes |
| Load balancing | No | No | Yes |

CLC enables dynamic scheduling:
- Each SM can cancel unexecuted thread blocks
- Idle SMs can "steal" tasks from busy SMs (work stealing)
- Adapts to runtime resource availability

**Multi-stream parallel scenario** (background kernel occupies 20 SMs):
- 4K³: 1.36x | 8K³: 1.29x | 16K³: 1.34x

Critical for LLM inference where multiple kernels run concurrently (e.g., communication-computation overlap).

### 4.2 Tutorial 6: Programmatic Dependent Launch (PDL)

PDL enables overlapping dependent kernels within the same stream.

**Implementation (3 steps):**

Step 1 — Primary kernel declares launch readiness:
```python
@cute.kernel()
def primary_kernel(...):
    # prologue & mainloop ...
    cute.arch.griddepcontrol_launch_dependents()
    # epilogue ...
```

Step 2 — Secondary kernel waits for predecessor:
```python
@cute.kernel()
def secondary_kernel(...):
    # prologue ...
    cute.arch.griddepcontrol_wait()
    # mainloop & epilogue ...
```

Step 3 — Enable PDL at launch:
```python
kernel(...).launch(grid=grid, block=[threads, 1, 1], stream=stream, use_pdl=True)
```

**Application**: Dequantization + GEMM pipeline. GEMM prologue overlaps with Dequant epilogue.

**Performance (Dequant + GEMM):**
- 256×16K×64: 1.12x | 256×16K×128: 1.16x | 256×16K×256: 1.13x | 256×16K×512: 1.10x

---

## 5. Summary

### 5.1 Compute-Bound Optimization Summary (Baseline: 1640 TFLOPS @ 1.35 GHz)

| Technique | Principle | TFLOPS Achieved |
|-----------|-----------|-----------------|
| 2CTA + Multicast | Reduce B SMEM for larger ab_stages; TMA multicast reduces L2 traffic | 1393 |
| Warp Specialization + TMA STG | Intra-CTA warp task parallelism; efficient instruction issue | 1429 |
| Persistent Scheduler | Persistent cluster processes multiple tiles; hides prologue/epilogue | 1468 |
| Preferred & Fallback Clusters | Large cluster → higher TMA multicast; small cluster fills device | 1597 |

### 5.2 Additional Scenarios

- **Dynamic Persistent Scheduler (CLC)**: Up to 1.36x for multi-stream parallel
- **PDL**: Up to 1.16x for kernel overlap (Dequant + GEMM)

### 5.3 GEMM Core Properties

1. **Parallelizable**: Each output element C[i,j] computable independently
2. **Data-reusable**: Row i of C shares A[i,:]; column j shares B[:,j]
3. **Block-friendly**: Groups of elements processable together via MMA
4. **Bottleneck-flexible**: Different problem sizes have different bottlenecks requiring different strategies


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
