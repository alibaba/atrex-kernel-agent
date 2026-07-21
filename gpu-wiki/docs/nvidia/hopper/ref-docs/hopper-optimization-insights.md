# Community Hopper/Blackwell Architecture Optimization Insights

A synthesis of kernel optimization insights on Hopper (SM90) and Blackwell (SM100) architectures from Zhihu community. Supplements the hands-on documentation in [sm90/hands-on/](../../../generic/kernel-opt/hands-on).

> **Source Note**: This document synthesizes core knowledge from approximately 3 related Zhihu articles, consolidated through deduplication, filtering, and structured organization.

---

## 1. DeepSeek R1 Optimization on Blackwell B200

NVIDIA's official throughput optimization of DeepSeek R1 on Blackwell GPUs increased TPS/GPU from approximately 2,000 to **4,600** (ISL/OSL 1K/2K), achieving a **2.3×** improvement. Optimizations span three areas: MLA layers, MoE layers, and runtime.

### 1.1 Parallelism Strategy: ADP + EP Replaces TP

For throughput scenarios, Tensor Parallelism (TP) is replaced with **Attention Data Parallelism (ADP) + Expert Parallelism (EP)**:

- **Key Benefit of ADP**: TP for MQA replicates the KV cache on every GPU (8× replication at TP8), severely limiting concurrency. ADP raises global concurrency from 500 to 4,000 (on an 8-GPU system), delivering an **end-to-end 400% speedup**.
- **Key Benefit of EP**: DeepSeek R1 has 256 small experts, and GEMM problem sizes are small. EP is more efficient than TP on small GEMMs emulsifying and only sends tokens to the corresponding active experts, saving communication bandwidth. End-to-end **speedup of 142%**.

### 1.2 Precision Strategy: FP4 Weights + FP8 KV Cache

- MoE layer weights are quantized to **FP4**, fully leveraging Blackwell's 5th-generation Tensor Cores, reducing memory usage by approximately half (640 GB → 400 GB), freeing more space for KV cache.
- **FP8 KV Cache + FP8 Attention**: KV cache halved, end-to-end throughput improved by **6%**, with no significant accuracy degradation on GSM8K.
- **FP4 AllGather**: Replacing BF16 AllGather with an FP4 version improves communication efficiency by about **3×**, delivering an end-to-end **4%** speedup.

### 1.3 Kernel-Level Optimization Highlights

| Optimization | End-to-End Speedup | Technical Details |
|--------------|--------------------|-------------------|
| MLA Attention Kernel | +20% | Blackwell Tensor Core 5th-gen MMA 2CTA group variant, interleaved tile implementation enabling MLA+softmax overlap |
| Top-K Fusion | +7.4% | 18 PyTorch operations fused into 2 kernels, kernel time 252 μs → 15 μs |
| Multi-Stream Optimization | +5.3% | Shared/routed expert dual-stream parallelism, small operations like Q/KV norms parallelized |
| CUDA Graph | +22% | Reduces launch overhead of numerous small kernels; CUDA Graph padding balances hit rate versus waste |
| Overlap Scheduler | +4% | Overlapped computation and communication, hiding data transfer latency |

### 1.4 Key Takeaways

- For large-scale MoE models, parallelism strategy choices have a far greater performance impact than individual kernel optimizations.
- Low-precision quantization (FP4/FP8) not only accelerates computation but, more importantly, frees memory for KV cache to support higher concurrency.
- Kernel fusion yields significant returns when the number of operators is large and individual operators are small (Top-K reduced from 18 to 2).

---

## 2. Architectural Evolution of Software Pipeline / Warp Specialization / Persistent Kernel

### 2.1 Three Generations of Data Movement Evolution

| Feature | Pre-Ampere (Sync) | Ampere (cp.async) | Hopper (TMA / cp.async.bulk) |
|---------|-------------------|-------------------|-------------------------------|
| Max Copy Limit | 16 bytes per thread (.v4) | 16 bytes per thread | Arbitrary batch size (up to SMEM limit) |
| Execution Model | Threads block and stall | 32 threads each initiate | Single leader thread initiates entire batch |
| Address Computation | Software | Software | TMA hardware auto-computes from tensormap |
| Synchronization | __syncthreads() | wait_group | mbarrier (tracks exact byte count) |
| Cluster Support | None | None | Supports multicast to multiple SMs |

Three hardware-level benefits of TMA:
1. **Register Efficiency**: Hardware automatically handles multi-dimensional address computation and stride.
2. **Hardware-Level OOB Handling**: Automatic OOB fill (returns 0 or NaN), no software branching needed.
3. **Automatic Swizzling**: Tensor Map specifies data layout to prevent bank conflicts.

### 2.2 Synchronization Model Innovation: mbarrier and Phase Flip

Hopper replaces wait_group with **mbarrier**, representing a fundamental shift: **from tracking thread arrival to tracking data volume**.

- Producer warp initiates TMA load and configures mbarrier with the expected byte count.
- Consumer warp suspends on mbarrier Tremors only released after the exact byte count arrives.
- **Phase Flip** (sense reversing barrier): Hardware uses a parity bit to track state, allowing mbarrier to be reused indefinitely and preventing a fast producer from overwriting a slow consumer's buffer.

### 2.3 Memory Proxies and Cross-Proxy Synchronization

TMA introduces three proxy paths: Generic, Async, and Tensormap. Standard thread operations (Generic Proxy) and TMA (Async Proxy) use different paths and require `fence.proxy.async` for explicit cross-proxy synchronization.

### 2.4 History and Architectural Adaptation of Warp Specialization

SWP (Software Pipelining) is a scheduling concept; WASP (Warp Specialization) is the programming paradigm that implements it:

| Architecture | Dominant Paradigm | Reason |
|--------------|-------------------|--------|
| Pre-Ampere | WASP | Sync instructions cannot overlap across iterations |
| Ampere | CTA-aligned SWP | cp.async supports async; WASP's extra warp register overhead cannot be recovered (uniform allocation) |
| Hopper | WASP Returns | setmaxnreg supports differentiated register allocation; TMA single-thread launch saves registers; mbarrier naturally fits producer-consumer |
| Blackwell | WASP Deepens | Both TMA and TCGen05 use single-thread launch; TMEM decouples accumulator; Epilogue independently allocates registers |

### 2.5 Persistent Kernel: Software Pipelining at the Macro Level

Persistent Kernel elevates SWP from intra-tile loops to the multi-tile lifecycle:

- **Amortized Overhead**: Launch and prologue setup occur only once per SM
- **Epilogue-Computation Overlap**: CUTLASS Warp-Specialized Persistent Ping-Pong design — while Consumer Group A executes the epilogue, Group B is already processing Tensor Core computation for the next tile

**Distinguishing Cooperative vs Pingpong**:
- **Cooperative**: Two consumers process different parts of the same output tile, each using half the registers
- **Pingpong**: Two consumers interleave processing of their respective complete tiles (MMA and Epilogue interleaved); FA3/FA4 use this mode

### 2.6 New Changes in Blackwell

- TCGen05 results no longer use registers (they use TMEM), enabling complete decoupling of MMA from Epilogue
- Register pressure is concentrated in the Epilogue (which requires loading from TMEM to registers), thus Epilogue warps need 4/8
- Producer/MMA/Epilogue three roles are fully independent

---

## 3. TMA Practical Experience and Pitfalls

### 3.1 TMA Is Not a Silver Bullet: The Occupancy Trade-off

A beginner's real-world case reveals a key lesson: using TMA on elementwise kernels actually **slows them down by 10%-30%**. Root cause analysis:

1. **Occupancy Drop**: TMA requires shared memory for buffering, leading to reduced occupancy
2. **High-occupancy kernels don't need TMA**: When occupancy is high, the SM can naturally overlap computation and memory access by switching warps, achieving results superior to hand-written TMA software pipelines
3. **TMA's sweet spot**: When occupancy cannot be increased (e.g., GEMM using many registers), TMA is the only way to implement software pipelining

**Core insight**: TMA suits low-occupancy, compute-intensive kernels (e.g., GEMM/Attention), not high-occupancy simple kernels.

### 3.2 QuCo: Automated TMA Configuration Research (HPCA'26)

Configuring TMA/ATT is extremely challenging — tile size, queue slots, LDS allocation, and barrier placement all require manual tuning for each kernel and each architecture. Migrating configurations across kernels causes up to **1.2x** performance degradation, and up to **1.4x** across architectures.

**QuCo (Queue Configurator)** core design:

- **Hardware approach**: A lightweight RISC-V microcontroller embedded on the GPU die (5-stage pipeline, 8KB ROM, 2KB data buffer, 256B GPU specification table)
- **Automated configuration flow**:
  1. **Tile size selection**: Based on merit factor (tile processing time / transfer time), adjusted by compute intensity (CI) — low CI enlarges the tileuru to improve bandwidth utilization; high CI shrinks the tile to balance overlap
  2. **Queue slot count**: A variant of Little's Law, determining queue depth based on the transfer/computation time ratio, corrected by active CU count to avoid memory pressure
  3. **LDS allocation**: Automatic partitioning and allocation

- **Performance**: Within 1.04% of the optimal configuration found by exhaustive search; average **1.15x** speedup on full models like Whisper-Tiny
- **Portability**: The same compiled binary achieves near-optimal performance on three different GPU architectures, and up to **2x** speedup on the resource-constrained Radeon 530
- **DVFS adaptation**: The hardware solution consistently outperforms static software solutions, with up to **17%** speedup in later layers

**Design philosophy**: Many "configurational optimizations" on GPUs deserve to be absorbed by hardware control. QuCo aligns with the trend of increasingly delegating non-compute logic to dedicated microcontrollers on Blackwell.

---

## Related Documents

- [Hopper SM90 Optimization Practices](../../../generic/kernel-opt/hands-on) — 13 practical documents covering TMA, WGMMA, mbarrier pipeline, warp specialization, and more
- [Blackwell SM100 Optimization Practices](../../../generic/kernel-opt/hands-on) — 11 practical documents covering tcgen05/TMEM, three-role warp specialization, CLC, 2CTA, and more
- [NVIDIA General Optimization Knowledge](../../) — 13 documents covering Compute Capability, PTX ISA, NCU profiling, L2 persistence, and more
