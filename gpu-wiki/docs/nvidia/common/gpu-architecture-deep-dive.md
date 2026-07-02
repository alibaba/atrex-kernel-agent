# GPU Architecture Deep Dive

> A synthesis of multiple technical articles from the Zhihu community, covering SM microarchitecture, memory hierarchy, SIMT execution model, warp scheduling mechanisms, thread block scheduling strategies, and more.


**Last updated**: 2026-07-01

---

## 1. SM Microarchitecture Analysis

### 1.1 SM Internal Composition

The SM (Streaming Multiprocessor) is the core compute unit of a GPU. Taking the A100 (Ampere architecture) as an example, one SM contains:

| Component | Count (per SM) | Function |
|-----------|---------------|----------|
| FP32 CUDA Core | 64 | Single-precision floating-point operations |
| FP64 CUDA Core | 32 | Double-precision floating-point operations |
| INT32 Core | 64 | Integer operations |
| Tensor Core | 4 (3rd gen) | Matrix multiply-accumulate (WMMA/WGMMA) |
| LD/ST Unit | 32 | Memory load/store |
| SFU | 16 | Special functions (sin/cos/exp/rcp/rsqrt) |
| Warp Scheduler | 4 | Warp scheduler |
| Dispatch Unit | 4 | Instruction dispatch unit |
| Register File | 65536×32-bit | Register file |
| L1 Cache / Shared Memory | 192 KB (configurable ratio) | On-chip cache and shared memory |

**Partition Structure**: Each SM is divided into 4 processing partitions, each with its own independent warp scheduler and dispatch unit, but sharing the L1 cache and shared memory.

### 1.2 Architectural Generational Evolution

| Architecture | SM Count | Warp/SM | Block/SM | Shared Mem/SM | Key Features |
|-------------|----------|---------|----------|---------------|--------------|
| Fermi (SM20) | 16 | 48 | 8 | 48 KB | First-gen GPGPU architecture |
| Kepler (SM35) | 15 | 64 | 16 | 48 KB | 4 warp scheduler |
| Pascal (SM60) | 60 | 64 | 32 | 64 KB | NVLink 1.0 |
| Volta (SM70) | 80 | 64 | 32 | 96 KB | Tensor Core V1 |
| Ampere (SM80) | 108 | 64 | 32 | 164 KB | Tensor Core V3, cp.async |
| Hopper (SM90) | 132 | 64 | 32 | 228 KB | TMA, WGMMA, Thread Block Cluster |
| Blackwell (SM100) | 148 | - | - | - | tcgen05, TMEM, CLC |

### 1.3 A100 Full-Chip Architecture

```
A100 GPU
├── 8 GPC (Graphics Processing Cluster)
│   ├── Each GPC contains 8 TPC (Texture Processing Cluster)
│   │   └── Each TPC contains 2 SM
│   └── Total 128 SM (actually available: 108)
├── L2 Cache: 40 MB
├── HBM2e: 80 GB, bandwidth 2039 GB/s
└── NVLink 3.0: 600 GB/s
```

---

## 2. Memory Hierarchy

### 2.1 Memory Hierarchy with Bandwidth/Latency

The GPU memory hierarchy ranges from registers to HBM, with decreasing bandwidth and increasing latency:

```
Register
    ↓ ~1 cycle, ~19.5 TB/s (per SM)
L1 Cache / Shared Memory
    ↓ ~5 cycles, ~several TB/s
L2 Cache
    ↓ ~tens of cycles, ~4-6 TB/s
Global Memory (HBM)
    ~500 cycles, ~2-3 TB/s
```

Specific data for the H100 SXM as an example:

| Level | Capacity | Bandwidth | Latency |
|-------|----------|-----------|---------|
| Register File | 256 KB/SM × 132 SM | ~19.5 TB/s/SM | 1 cycle |
| L1 / SMEM | 228 KB/SM | ~33 TB/s (aggregate) | ~5 cycles |
| L2 Cache | 50 MB | ~12 TB/s | ~tens of cycles |
| HBM3 | 80 GB | 3.35 TB/s | ~hundreds of cycles |

### 2.2 Characteristics and Optimization Directions for Each Memory Type

**Global Memory**:
- Physical implementation: HBM (High Bandwidth Memory)
- Optimization direction: Coalesced access, aligned access, reducing transaction count
- Coalescing rule: When the addresses of all 32 threads within a warp fall within the same 128B cache line, they are combined into a single transaction
- Worst-case scenario: stride-128B access, each thread hits a different cache line → 32 transactions

**Shared Memory**:
- Physical implementation: On-chip SRAM, shared with L1
- Optimization direction: Eliminating bank conflicts
- Bank structure: 32 banks, each bank 4B wide
- Bank conflict: Multiple threads within the same warp accessing different addresses in the same bank → serialization
- Solution: Add padding (e.g., `float smem[32][33]`), use swizzle mode

**L2 Cache**:
- Globally shared, accessible by all SMs
- L2 Cache Persistence (Ampere+): Hot data can be pinned in L2
- `cudaAccessPolicyWindow` API controls the L2 persistence region

**Constant Memory**:
- 64 KB constant cache, suitable for broadcast mode where all threads in a warp read the same address
- If threads within a warp read different addresses, access is serialized
- The compiler automatically places `__constant__` variables in constant memory

**Registers**:
- Thread-private, fastest access
- A100 has 65536 32-bit registers per SM
- Maximum of 255 registers per thread (hardware limit)
- Register spill: data spills to local memory (physically on HBM), dramatically increasing latency

### 2.3 Little's Law and Bytes-in-Flight

The GPU memory system follows Little's Law:

```
bytes_in_flight = bandwidth × latency
```To achieve ~90% peak bandwidth, the required in-flight bytes per SM are:
- H100: ~32 KB/SM
- H200: ~64 KB/SM
- B200: ~64 KB/SM

Estimation formula:

```
estimated_bytes_in_flight_per_SM
    = (#loads/thread) × (#bytes/load) × (#threads/block) × (#blocks/SM)
```

Simple vector addition estimation: 2 × 4 × 256 × 8 = 16 KB/SM → insufficient to saturate bandwidth on H100.

**Three methods to increase in-flight bytes**:
1. **ILP (Instruction-Level Parallelism)**: Loop unrolling (`#pragma unroll`), each thread processes multiple elements
2. **DLP (Data-Level Parallelism)**: Vectorized loads (`float4`/`uint4`), 128-bit transactions
3. **Asynchronous Copy**: cp.async / TMA, prefetch data into shared memory, double/triple buffering

Measured results (element-wise multiply, 4 GiB data, vec4 vs scalar baseline):
- V100: ~1-2% improvement
- A100: ~5-6% improvement
- H100: ~12-13% improvement
- H200: ~21-25% improvement
- B200: ~18-23% improvement

Conclusion: Vectorization and unrolling are increasingly critical on newer architectures.

---

## 3. SIMT Execution Model

### 3.1 Warp: The Basic Unit of GPU Execution

**Warp Composition**: 32 threads form a warp, executing in SIMT (Single Instruction, Multiple Threads) fashion—all threads execute the same instruction but operate on different data.

**Warp State Machine**:

```
                 ┌─────────────┐
                 │ Active Warps│ ← From schedule start to execution finish
                 │  (on SM)    │
                 └──────┬──────┘
                        │
              ┌─────────┴─────────┐
              │                   │
      ┌───────┴───────┐   ┌──────┴──────┐
      │ Eligible Warps│   │Stalled Warps│
      │ (ready to run)│   │ (waiting)   │
      └───────┬───────┘   └─────────────┘
              │
      ┌───────┴───────┐
      │ Selected Warps│ ← Each scheduler selects 1
      │  (executing)  │
      └───────────────┘
```

- **Device Limit**: Maximum number of warps/SM supported by hardware (A100 = 64 warps/SM)
- **Theoretical Occupancy**: Upper bound of theoretical occupancy determined by block configuration
- **Active Warps**: Warps resident on the SM and being processed
- **Eligible Warps**: Warps whose next instruction is ready and can be scheduled
- **Stalled Warps**: Warps paused for various reasons
- **Selected Warps**: Warps chosen by the scheduler Hedge to issue instructions

### 3.2 Warp Scheduler Workflow

Each cycle, the warp scheduler performs the following steps:

1. **Check Scoreboard**: See which warps have their next instruction ready (all operands available)
2. **Select Warp**: Choose one from eligible warps according to scheduling policy
3. **Issue Instruction**: Issue the instruction to the corresponding execution unit
4. **Context Switch**: If the current warp encounters a long-latency operation (e.g., global memory access), automatically switch to another ready warp—this is the core mechanism by which GPUs hide latency

### 3.3 Branch Divergence

When threads within a warp diverge on a conditional branch, branch divergence occurs:

```cpp
if (threadIdx.x < 16) {
    // Branch A: first 16 threads
    path_A();
} else {
    // Branch B: last 16 threads
    path_B();
}
```

Handling mechanism:
- The scheduler constructs a 32-bit warp mask, marking active threads for each branch
- Pushes the mask onto the branch synchronization stack
- Executes branch A first (only the first 16 threads active), then executes branch B (only the last 16 threads active)
- Execution time of the two branches is additive (not parallel), with worst-case performance degradation of 50%

Optimization suggestions:
- Make threads within a warp take the same branch
- Group threads with similar workloads in the same block/warp

---

## 4. Instruction Issue and Control Mechanisms

### 4.1 SASS Control Bits

NVIDIA GPU SASS instructions embed hardware scheduling information, with every 3 instructions sharing a 128-bit control word. The control fields corresponding to each instruction include:

| Field | Bits | Description |
|------|------|------|
| Stall Count | 4 bits | Number of cycles to wait before issue (0-15) |
| Yield Flag | 1 bit | Whether to yield warp execution rights |
| Write Barrier (WB) | 3 bits | Write barrier index (marks the start of a long-latency operation) |
| Read Barrier (RB) | 3 bits | Read barrier index (waits for a long-latency operation to complete) |
| Wait Barrier Mask | 6 bits | Which barriers to wait for completion |

### 4.2 Dependency Counters

GPUs use 6 dependency counters (Scoreboard Barriers 0-5) to track long-latency operations:

```
                    ┌── Barrier 0 ──┐
LDG R2, [R4]  ──►  │  counter: 1   │  ← load starts, counter +1
    ...             │               │
    ...             │  counter: 0   │  ← load completes, counter -1
IADD R5, R2, R3 ◄──┘               │  ← wait for barrier 0 to zero
```

**Short Latency vs Long Latency**:
- Short-latency instructions (register operations ~5 cycles): directly delayed via Stall Count
- Long-latency instructions (global memory load ~500 cycles): asynchronously waited on via dependency counters

This mechanism allows the compiler to determine instruction dependencies at compile time, so the hardware only needs to check whether the counter has reached zero, without requiring complex runtime dependency analysis.

### 4.3 Meaning of Stall Counters

Through reverse engineering, researchers discovered that stall counters in SASS have special meanings:

- **Stall = 1**: Minimum latency, indicating that instructions can be issued back-to-back (provided there is no data dependency)
- **Stall > 1**: Indicates additional wait cycles are required
- **Yield = 1**: Advises the scheduler to switch to another warp after this warp stalls

The stall count is automatically calculated by the ptxas compiler based on instruction latency and dependencies, but it can be manually adjusted using CuAsm tools (the CuAsmRL method has demonstrated optimization potential here).

---

## 5. Thread Block Scheduling Strategy

### 5.1 "Largest Space Policy"

Through empirical research on Pascal, Volta, and Turing architectures, the actual behavior of NVIDIA's thread block scheduler was revealed—it is not simple Round-Robin, but rather a **Largest Space Policy**:

> Thread blocks are scheduled onto the SM that can currently support the largest number of thread blocks for that kernel, and only one block is dispatched to that SM at a time.

**How it works**:
1. The scheduler evaluates the remaining resources (threads, shared memory, registers, block limit) of each SM
2. It finds the SM that can accommodate the most blocks for the target kernel
3. It assigns one block to that SM
4. It re-evaluates and assigns the next block

**Arbitration mechanism**: When multiple SMs have the same capacity, selection follows a device-specific fixed order:
- Pascal: SM ID ascending (0, 1, 2, 3, ...)
- Turing: "Even first, odd second" (0, 2, 4, ..., 66, 1, 3, 5, ..., 67)

### 5.2 Performance Impact of Concurrent Kernels

The Largest Space Policy can lead to counterintuitive performance changes in concurrent kernel scenarios:

**Case Study** (Turing GPU, 68 SMs):
- Kernel A: 67 blocks × 512 threads, occupying SM0-66
- Kernel B Version 1: 8 blocks × 32 threads → limiting factor is SM block count limit → blocks are distributed across multiple SMs (co-located scenario)
- Kernel B Version 2: 8 blocks × 33 threads → limiting factor becomes thread count → all blocks concentrated on the idle SM67 (isolated scenario)

**Result**: Simply by having 1 more thread per block (32→33), kernel B's execution time changed by a factor of 3.58×.

**Root cause**:
- In the co-located scenario, the two kernels share the SM's L1 cache → increased cache contention → 1.24-1.33× performance degradation
- In the isolated scenario, each kernel has exclusive access to its SM → no cache contention → performance on par with serial execution

**Practical insights**:
- Performance of concurrent kernels is influenced by multiple factors including scheduling policy, resource contention, and launch timing
- Minor differences in block configuration can trigger entirely different scheduling decisions
- In single-kernel scenarios, the Largest Space Policy is nearly equivalent to a Round-Robin policy (since all blocks have the same configuration)

---

## 6. Memory System Evolution: From V100 to B200

### 6.1 Challenges of New Architectures

As GPU bandwidth continues to grow, but the number of SMs grows even faster, the bandwidth per SM is actually increasing. Simple kernels find it increasingly difficult to saturate bandwidth:

| Architecture | Total Bandwidth | SM Count | Bandwidth/SM | Simple Kernel BWUtil |
|------|--------|-------|---------|-------------------|
| V100 | 900 GB/s | 80 | 11.25 GB/s | ~98% |
| A100 | 2039 GB/s | 108 | 18.9 GB/s | ~95% |
| H100 | 3350 GB/s | 132 | 25.4 GB/s | ~88% |
| H200 | 4800 GB/s | 132 | 36.4 GB/s | ~79% |
| B200 | 8000 GB/s | 148 | 54.1 GB/s | ~75% |

### 6.2 Evolution of Asynchronous Load Technologies

| Technology | Architecture | Alignment Requirements | Characteristics |
|------|------|---------|------|
| Synchronous Load | All | None | Transfers via registers |
| cp.async | Ampere+ | 4/8/16B | Global→Shared, bypasses registers |
| cp.async.bulk | Hopper+ | 16B | Single thread initiates large transfers |
| TMA (Tensor Memory Accelerator) | Hopper+ | SMEM 128B, GMEM 16B | Supports 2D/3D tensors, swizzle, out-of-bounds handling |

### 6.3 Kernel Launch Latency Optimization

For small-scale problems (<100MB), the performance bottleneck is not memory bandwidth but kernel launch latency:

| Optimization Technique | Principle | Effect (Small Scale) |
|----------|------|---------------|
| CUDA Graph | Batch-submit kernels, reduce CPU-GPU interaction | ~50% improvement |
| PDL (Programmatic Dependent Launch) | Kernel launched early before predecessor completes | Cumulative ~70% improvement |
| Early Exit Signal | Block sends completion signal before fully finishing | Cumulative ~75% improvement |
| All Three Combined | Fully minimize launch gaps | Up to 3× speedup (10KB data) |

Decision workflow:
- Large-scale problems (>100MB): Directly benefit from hardware upgrades
- Small-scale problems (<100MB): Must reduce launch latency through software optimization (CUDA Graph + PDL)

---

## 7. GPU Optimization Hierarchy Summary

From bottom to top, GPU optimization spans multiple levels:

```
┌─────────────────────────────────────┐
│ System level: multi-GPU communication, data loading pipeline │
├─────────────────────────────────────┤
│ Graph compiler level: operator fusion, memory planning       │
├─────────────────────────────────────┤
│ Kernel level: GEMM tiling, pipeline design                   │
├─────────────────────────────────────┤
│ Instruction level: PTX/SASS optimization, instruction scheduling │
├─────────────────────────────────────┤
│ Hardware level: architecture features, Tensor Core, TMA      │
└─────────────────────────────────────┘
```

Optimization at each level requires a sufficient understanding of the level below. Understanding GPU architecture is the foundation of all optimization work.

---

## Related

- [NVIDIA Compute Capabilities](nvidia-compute-capabilities.md) — Detailed table of SM resource limits for each architecture
- [NVIDIA Architecture-Specific Optimization](nvidia-arch-specific-optimization.md) — Optimization techniques specific to each generation of architecture
- [Occupancy Tuning](occupancy-tuning-by-arch.md) — Occupancy calculation and tuning strategies for each architecture
- [Async Global-to-Shared Memory Copy](async-global-to-shared-copy.md) — Detailed explanation of cp.async and TMA technologies
- [L2 Cache Persistence](l2-cache-persistence.md) — Usage and configuration of L2 Persistence
- [Thread Block Cluster](thread-block-cluster.md) — Clustered thread blocks in Hopper
- [PTX Programming Model](ptx/ptx-programming-model.md) — Understanding PTX state space and memory model
